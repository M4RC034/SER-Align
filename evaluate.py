import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import glob
import json 
import numpy as np 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from functools import partial
from transformers import AutoTokenizer, AutoProcessor

try:
    from utils import (
        RoBERTaTextEmbedder,
        Wav2VecAudioEmbedder,
        CrossModalAttention,
        ForgetGateFusion,
        FusionTransformer,
        EmotionClassifier,
    )
except ImportError:
    print("Error: Cannot find utils.py. Please ensure it is in the same directory as evaluate.py.")
    exit()

# --- 1. Import model definition, Dataset, and collate_fn (copied from train.py) ---

class FullMultimodalModel(nn.Module):
    def __init__(self, 
                 embed_dim: int = 768, 
                 num_heads: int = 8, 
                 num_classes: int = 4, 
                 dropout: float = 0.1,
                 roberta_model_name: str = "roberta-base",
                 wav2vec_model_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.text_embedder_module = RoBERTaTextEmbedder(model_name=roberta_model_name)
        self.audio_embedder_module = Wav2VecAudioEmbedder(model_name=wav2vec_model_name)
        self.cross_attention = CrossModalAttention(embed_dim, num_heads, dropout)
        self.forget_gate_text = ForgetGateFusion(embed_dim)
        self.forget_gate_audio = ForgetGateFusion(embed_dim)
        self.fusion_transformer = FusionTransformer(embed_dim, num_heads, dropout=dropout)
        self.classifier = EmotionClassifier(embed_dim, num_classes)
        print("FullMultimodalModel initialized for evaluation (fine-tuning architecture).")

    def forward(self, 
                text_input_ids: torch.Tensor, 
                text_attention_mask: torch.Tensor,
                audio_input_values: torch.Tensor,
                audio_attention_mask: torch.Tensor | None = None
               ) -> torch.Tensor:
        text_hidden_states = self.text_embedder_module(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embeds_pooled = self.text_embedder_module.mean_pool(text_hidden_states, text_attention_mask)
        text_embeds = text_embeds_pooled.unsqueeze(1)

        audio_hidden_states = self.audio_embedder_module(input_values=audio_input_values, attention_mask=audio_attention_mask)
        if audio_attention_mask is None or audio_attention_mask.shape[1] != audio_hidden_states.shape[1]:
            audio_attention_mask_for_pool = torch.ones(
                audio_hidden_states.shape[0], audio_hidden_states.shape[1],
                device=audio_hidden_states.device, dtype=torch.long
            )
        else:
            audio_attention_mask_for_pool = audio_attention_mask

        audio_embeds_pooled = self.audio_embedder_module.mean_pool(audio_hidden_states, audio_attention_mask_for_pool)
        audio_embeds = audio_embeds_pooled.unsqueeze(1)

        a_t2a, a_a2t = self.cross_attention(text_embeds, audio_embeds)
        h_ta = self.forget_gate_text(text_embeds, a_t2a)
        h_at = self.forget_gate_audio(audio_embeds, a_a2t)
        fused_out = self.fusion_transformer(h_ta, h_at, key_padding_mask=None)
        mean_pooled = fused_out.mean(dim=1)
        logits = self.classifier(mean_pooled)
        return logits

class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir: str):
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if not self.file_list:
            raise ValueError(f"No .pt files found in {data_dir}. Ensure you ran the correct preprocessing script.")
        print(f"Found {len(self.file_list)} preprocessed samples (raw data) in {data_dir}.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        data = torch.load(filepath, weights_only=False)
        sentence_text = data['sentence_text']
        audio_waveform = data['audio_waveform']
        label = data['label']
        return sentence_text, audio_waveform, label

def collate_fn(batch, text_tokenizer, audio_processor, device, target_sr=16000):
    texts = [item[0] for item in batch]
    audio_waveforms_np = [item[1] for item in batch]
    labels = torch.tensor([item[2].item() for item in batch], dtype=torch.long)

    tokenized_texts = text_tokenizer(
        texts, padding="longest", truncation=True, max_length=512, return_tensors="pt"
    )
    processed_audio = audio_processor(
        audio_waveforms_np, sampling_rate=target_sr, padding="longest", return_tensors="pt"
    )
    batch_data = {
        "text_input_ids": tokenized_texts.input_ids.to(device),
        "text_attention_mask": tokenized_texts.attention_mask.to(device),
        "audio_input_values": processed_audio.input_values.to(device),
        "labels": labels.to(device)
    }
    if "attention_mask" in processed_audio and processed_audio.attention_mask is not None:
        batch_data["audio_attention_mask"] = processed_audio.attention_mask.to(device)
    else:
        batch_data["audio_attention_mask"] = None
    return batch_data

# --- Main evaluation function ---
def evaluate_model(model_path, test_data_dir, batch_size, num_classes, embed_dim, device):
    print(f"Loading model from: {model_path}")
    model = FullMultimodalModel(embed_dim=embed_dim, num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return
    model.eval()

    print("Initializing Tokenizer and Processor for evaluation...")
    text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

    print(f"Loading test data from: {test_data_dir}")
    test_dataset = IEMOCAPDataset(test_data_dir)

    num_workers_eval = 0
    custom_collate_fn_with_tools = partial(collate_fn,
                                           text_tokenizer=text_tokenizer,
                                           audio_processor=audio_processor,
                                           device=device,
                                           target_sr=16000)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers_eval,
                             collate_fn=custom_collate_fn_with_tools)

    all_predictions = []
    all_labels = []
    detailed_predictions = []
    file_index_offset = 0

    print("\n--- Starting Evaluation ---")
    progress_bar = tqdm(test_loader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for batch_data in progress_bar:
            outputs = model(
                text_input_ids=batch_data["text_input_ids"],
                text_attention_mask=batch_data["text_attention_mask"],
                audio_input_values=batch_data["audio_input_values"],
                audio_attention_mask=batch_data["audio_attention_mask"]
            )
            labels = batch_data["labels"]
            _, predicted_indices = torch.max(outputs.data, 1)

            all_predictions.extend(predicted_indices.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_size_current = labels.size(0)
            for i in range(batch_size_current):
                original_pt_path = test_dataset.file_list[file_index_offset + i]
                wav_id = os.path.splitext(os.path.basename(original_pt_path))[0].rsplit('_seg_', 1)[0]

                detailed_predictions.append({
                    "wav_id": wav_id,
                    "pt_path": original_pt_path,
                    "true_label": labels[i].item(),
                    "predicted_label": predicted_indices[i].item()
                })
            file_index_offset += batch_size_current

    # Save detailed predictions to a JSON file
    predictions_output_path = "evaluation_predictions.json"
    print(f"\nSaving detailed predictions to {predictions_output_path}...")
    with open(predictions_output_path, "w", encoding='utf-8') as f:
        json.dump(detailed_predictions, f, indent=4)
    print("Predictions saved.")

    print("\n--- Evaluation Results (Classification Metrics) ---")
    target_names = ['hap', 'ang', 'sad', 'neu']
    print(f"Accuracy: {accuracy_score(all_labels, all_predictions):.4f}")
    print(f"Weighted F1-Score: {f1_score(all_labels, all_predictions, average='weighted', zero_division=0):.4f}")
    print(f"Macro F1-Score: {f1_score(all_labels, all_predictions, average='macro', zero_division=0):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0))

# --- Main execution ---
if __name__ == "__main__":
    MODEL_PATH = "best_emotion_model.pth"
    TEST_DATA_DIR = "./preprocessed_iemocap/test"
    BATCH_SIZE = 16
    NUM_CLASSES = 4
    EMBED_DIM = 768

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        exit()

    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Test data directory '{TEST_DATA_DIR}' not found.")
        exit()

    evaluate_model(MODEL_PATH, TEST_DATA_DIR, BATCH_SIZE, NUM_CLASSES, EMBED_DIM, device)
