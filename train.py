import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset  # TensorDataset may no longer be needed unless for special cases
from tqdm import tqdm
import numpy as np
import os
import glob
from collections import Counter
from functools import partial  # For passing extra args to collate_fn
from transformers import AutoTokenizer, AutoProcessor  # For loading tokenizer and processor

# Assumes all your classes are defined in utils.py
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
    print("Error: Cannot find utils.py. Please ensure it is in the same directory as train.py.")
    print("utils.py should include RoBERTaTextEmbedder and Wav2VecAudioEmbedder.")
    exit()

# --- 1. Define full multimodal model (fine-tunable version) ---
class FullMultimodalModel(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 num_heads: int = 8,
                 num_classes: int = 4,
                 dropout: float = 0.1,
                 roberta_model_name: str = "roberta-base",
                 wav2vec_model_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        
        # Initialize embedders as part of the model (will be fine-tuned)
        self.text_embedder_module = RoBERTaTextEmbedder(model_name=roberta_model_name)
        self.audio_embedder_module = Wav2VecAudioEmbedder(model_name=wav2vec_model_name)

        # Ensure embed_dim matches actual hidden sizes
        actual_text_embed_dim = self.text_embedder_module.hidden_size
        actual_audio_embed_dim = self.audio_embedder_module.hidden_size
        
        if actual_text_embed_dim != embed_dim or actual_audio_embed_dim != embed_dim:
            print(f"Warning: Provided embed_dim ({embed_dim}) does not match actual embedder dims "
                  f"(Text: {actual_text_embed_dim}, Audio: {actual_audio_embed_dim}). "
                  f"Ensure downstream layers expect these dimensions if they differ, "
                  f"or adjust model architecture if they must be {embed_dim}.")
        # For simplicity, assume both match embed_dim or downstream can handle them

        self.cross_attention = CrossModalAttention(embed_dim, num_heads, dropout)
        self.forget_gate_text = ForgetGateFusion(embed_dim)
        self.forget_gate_audio = ForgetGateFusion(embed_dim)
        self.fusion_transformer = FusionTransformer(embed_dim, num_heads, dropout=dropout)
        self.classifier = EmotionClassifier(embed_dim, num_classes)
        print("FullMultimodalModel initialized with fine-tunable embedders.")

    def forward(self,
                text_input_ids: torch.Tensor,
                text_attention_mask: torch.Tensor,
                audio_input_values: torch.Tensor,
                audio_attention_mask: torch.Tensor | None = None
               ) -> torch.Tensor:
        
        # 1. Get text embeddings (with fine-tuning)
        text_hidden_states = self.text_embedder_module(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embeds_pooled = self.text_embedder_module.mean_pool(text_hidden_states, text_attention_mask)
        text_embeds = text_embeds_pooled.unsqueeze(1)  # (B, 1, D)

        # 2. Get audio embeddings (with fine-tuning)
        audio_hidden_states = self.audio_embedder_module(input_values=audio_input_values, attention_mask=audio_attention_mask)
        
        # Prepare attention mask for audio pooling
        if audio_attention_mask is None or audio_attention_mask.shape[1] != audio_hidden_states.shape[1]:
            audio_attention_mask_for_pool = torch.ones(
                audio_hidden_states.shape[0], audio_hidden_states.shape[1],
                device=audio_hidden_states.device, dtype=torch.long
            )
        else:
            audio_attention_mask_for_pool = audio_attention_mask
        
        audio_embeds_pooled = self.audio_embedder_module.mean_pool(audio_hidden_states, audio_attention_mask_for_pool)
        audio_embeds = audio_embeds_pooled.unsqueeze(1)  # (B, 1, D)

        # 3. Fusion and classification
        a_t2a, a_a2t = self.cross_attention(text_embeds, audio_embeds)
        h_ta = self.forget_gate_text(text_embeds, a_t2a)
        h_at = self.forget_gate_audio(audio_embeds, a_a2t)
        fused_out = self.fusion_transformer(h_ta, h_at, key_padding_mask=None)
        mean_pooled = fused_out.mean(dim=1)
        logits = self.classifier(mean_pooled)
        return logits


# --- 2. Define dataset that loads raw data ---
class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir: str):
        self.file_list = glob.glob(os.path.join(data_dir, "*.pt"))
        if not self.file_list:
            raise ValueError(f"No .pt files found in {data_dir}. Did you run preprocessing to save raw data?")
        print(f"Found {len(self.file_list)} preprocessed samples (raw data) in {data_dir}.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        data = torch.load(filepath, weights_only=False)
        
        sentence_text = data['sentence_text']      # str
        audio_waveform = data['audio_waveform']    # np.ndarray (1D)
        label = data['label']                      # torch.Tensor (scalar)
        
        return sentence_text, audio_waveform, label


# --- 3. Define custom collate_fn ---
def collate_fn(batch, text_tokenizer, audio_processor, device, target_sr=16000):
    texts = [item[0] for item in batch]
    audio_waveforms_np = [item[1] for item in batch]  # list of 1D numpy arrays
    labels = torch.tensor([item[2].item() for item in batch], dtype=torch.long)

    # Process text
    tokenized_texts = text_tokenizer(
        texts,
        padding="longest", 
        truncation=True,
        max_length=512, 
        return_tensors="pt"
    )

    # Process audio
    # Wav2Vec2Processor expects a list of raw 1D numpy arrays
    processed_audio = audio_processor(
        audio_waveforms_np, 
        sampling_rate=target_sr,
        padding="longest", 
        return_tensors="pt"
    )

    # Move everything to the correct device
    batch_data = {
        "text_input_ids": tokenized_texts.input_ids.to(device),
        "text_attention_mask": tokenized_texts.attention_mask.to(device),
        "audio_input_values": processed_audio.input_values.to(device),
        "labels": labels.to(device)
    }

    # Safely assign audio_attention_mask
    if "attention_mask" in processed_audio and processed_audio.attention_mask is not None:
        batch_data["audio_attention_mask"] = processed_audio.attention_mask.to(device)
    else:
        batch_data["audio_attention_mask"] = None  # Or create a mask of all ones if needed

    return batch_data



# --- 4. Define training and validation functions (modified to handle dict input) ---
def train_one_epoch_finetune(model, dataloader, criterion, optimizer, device):
    model.train() 
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    
    for batch_data in progress_bar: 
        optimizer.zero_grad()
        
        outputs = model(
            text_input_ids=batch_data["text_input_ids"],
            text_attention_mask=batch_data["text_attention_mask"],
            audio_input_values=batch_data["audio_input_values"],
            audio_attention_mask=batch_data["audio_attention_mask"]
        )
        labels = batch_data["labels"]
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.2f}")
        
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

def validate_finetune(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
    with torch.no_grad():
        for batch_data in progress_bar:
            outputs = model(
                text_input_ids=batch_data["text_input_ids"],
                text_attention_mask=batch_data["text_attention_mask"],
                audio_input_values=batch_data["audio_input_values"],
                audio_attention_mask=batch_data["audio_attention_mask"]
            )
            labels = batch_data["labels"]
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())
            
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

# --- 5. Main entry point ---
def main():
    # --- Hyperparameters ---
    BATCH_SIZE = 8  # For fine-tuning, reduce batch size due to larger model
    LEARNING_RATE_FUSION = 1e-4  # Higher LR for fusion and classifier layers
    LEARNING_RATE_EMBEDDERS = 2e-5  # Lower LR for embedding layers (fine-tuning)
    EPOCHS = 20
    NUM_CLASSES = 4 
    EMBED_DIM = 768
    PREPROCESSED_PATH = "./preprocessed_iemocap"
    MODEL_DROPOUT = 0.2
    OPTIMIZER_WEIGHT_DECAY = 1e-2
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    EARLY_STOPPING_PATIENCE = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Tokenizer and Processor ---
    print("Initializing Tokenizer and Processor...")
    text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    TARGET_SR_COLLATE = 16000  # Must match what processor expects
    print("Tokenizer and Processor initialized.")

    TRAIN_DIR = os.path.join(PREPROCESSED_PATH, 'train')
    TEST_AS_VAL_DIR = os.path.join(PREPROCESSED_PATH, 'test') 
    if not (os.path.exists(TRAIN_DIR) and os.path.exists(TEST_AS_VAL_DIR)):
        print(f"Error: Cannot find preprocessed data directories 'train' and 'test' under '{PREPROCESSED_PATH}'.")
        print("Please ensure you’ve run preprocess.py to generate .pt files with raw text and waveform.")
        exit()
    else:
        train_dataset = IEMOCAPDataset(TRAIN_DIR)
        val_dataset = IEMOCAPDataset(TEST_AS_VAL_DIR)

    NUM_WORKERS = 0  # For on-the-fly processing, 0 is often more stable (especially on Windows)
    print(f"Using num_workers: {NUM_WORKERS} for DataLoader")

    # Use partial to pass tokenizer, processor, device, target_sr to collate_fn
    custom_collate_fn_with_tools = partial(
        collate_fn,
        text_tokenizer=text_tokenizer,
        audio_processor=audio_processor,
        device=device,
        target_sr=TARGET_SR_COLLATE
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Set False since tensors are moved in collate_fn
        collate_fn=custom_collate_fn_with_tools
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        collate_fn=custom_collate_fn_with_tools
    )

    model = FullMultimodalModel(
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        dropout=MODEL_DROPOUT
    ).to(device)

    # --- Set up optimizer with differential learning rates ---
    print("Setting up optimizer with differential learning rates...")
    optimizer_grouped_parameters = [
        {"params": model.text_embedder_module.parameters(), "lr": LEARNING_RATE_EMBEDDERS},
        {"params": model.audio_embedder_module.parameters(), "lr": LEARNING_RATE_EMBEDDERS},
        {"params": model.cross_attention.parameters(), "lr": LEARNING_RATE_FUSION},
        {"params": model.forget_gate_text.parameters(), "lr": LEARNING_RATE_FUSION},
        {"params": model.forget_gate_audio.parameters(), "lr": LEARNING_RATE_FUSION},
        {"params": model.fusion_transformer.parameters(), "lr": LEARNING_RATE_FUSION},
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE_FUSION}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, weight_decay=OPTIMIZER_WEIGHT_DECAY)
    print("Optimizer setup complete.")

    # --- Define loss function ---
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    print("\n--- Starting Training ---")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # Print current learning rates
        lr_text_embed = optimizer.param_groups[0]['lr']
        lr_audio_embed = optimizer.param_groups[1]['lr']
        lr_fusion_parts = optimizer.param_groups[2]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS} | LR Embedders: {lr_text_embed:.1e}/{lr_audio_embed:.1e}, LR Fusion: {lr_fusion_parts:.1e}")
        
        train_loss, train_acc = train_one_epoch_finetune(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc = validate_finetune(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model to best_emotion_model.pth")
            torch.save(model.state_dict(), "best_emotion_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break

    print("\n--- Training Finished ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (saved as best_emotion_model.pth)")

if __name__ == "__main__":
    main()
