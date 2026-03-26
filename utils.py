import torch
import torch.nn as nn
import torchaudio
import json
import os
import numpy as np
from typing import List, Union, Tuple 
import re
import string
import nltk
import torch.nn.functional as F
import math
from tqdm import tqdm
from transformers import (
    RobertaModel,
    RobertaTokenizerFast,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)

# --- NLTK ---
# Ensure NLTK's 'punkt' tokenizer models are downloaded
# This will attempt to download if not found. If it still fails here,
# the user needs to resolve the NLTK download manually.
try:
    nltk.sent_tokenize("This is a test sentence.") # A quick test to trigger punkt loading/check
    print("NLTK 'punkt' resource appears to be available.")
except LookupError:
    print("NLTK 'punkt' resource not found by test, attempting download...")
    try:
        nltk.download('punkt')
        nltk.sent_tokenize("This is a test sentence.") # Test again after download
        print("NLTK 'punkt' downloaded and available.")
    except Exception as e:
        print(f"Failed to download or use NLTK 'punkt'. Please ensure it's correctly installed. Error: {e}")
        # You might want to exit() here if punkt is absolutely critical and download fails
# --- NLTK ---

class TimestampAligner:
    """
    A robust aligner that groups words into segments based on speaker and pause duration.
    This version avoids fragile text matching and is more resilient to ASR errors.
    """
    def __init__(self, max_pause_seconds: float = 1.0):
        """
        Initializes the Aligner with a time-based grouping strategy.

        Args:
            max_pause_seconds (float): The maximum pause between words to be considered 
                                       part of the same conversational turn.
        """
        print("TimestampAligner initialized with robust, time-based grouping strategy.")
        self.max_pause_seconds = max_pause_seconds

    def _load_whisperx_result(self, whisperx_json_path: str):
        """Loads the JSON output from whisperx (after speaker assignment)."""
        print(f"Loading WhisperX output from: {whisperx_json_path}")
        with open(whisperx_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data or not isinstance(data, list) or 'words' not in data[0]:
             print(f"Warning: The input file at {whisperx_json_path} does not seem to contain "
                   f"word-level timestamps and speakers. Results may be suboptimal.")
             return data
        return data

    def _group_words_into_turns(self, all_words: list):
    # """Groups a flat list of words into turns based on speaker and pause duration."""
        if not all_words:
            return []

        turns = []
        current_turn = [all_words[0]]

        for i in range(1, len(all_words)):
            prev_word = current_turn[-1]
            curr_word = all_words[i]

            # Ensure 'speaker' and timestamp keys exist; skip this word if any are missing
            if 'speaker' not in prev_word or 'speaker' not in curr_word or \
            'start' not in curr_word or 'end' not in prev_word:
                continue

            speaker_change = curr_word.get('speaker') != prev_word.get('speaker')
            pause = curr_word['start'] - prev_word['end']
            is_long_pause = pause > self.max_pause_seconds

            if speaker_change or is_long_pause:
                # End the current turn
                turns.append(current_turn)
                # Start a new turn
                current_turn = [curr_word]
            else:
                # Continue the current turn
                current_turn.append(curr_word)
        
        # Append the final turn being processed
        if current_turn:
            turns.append(current_turn)

        return turns

    def _turn_to_segment(self, turn_of_words: list):
        """Converts a list of word dictionaries (a turn) into a single segment dictionary."""
        if not turn_of_words:
            return None
        
        # Use the 'word' key to join text and strip whitespace
        text = " ".join([word.get('word', '').strip() for word in turn_of_words])
        
        # Safely extract timestamps
        start_times = [w['start'] for w in turn_of_words if 'start' in w and w['start'] is not None]
        end_times = [w['end'] for w in turn_of_words if 'end' in w and w['end'] is not None]
        
        if not start_times or not end_times:
            return None

        return {
            "start": min(start_times),
            "end": max(end_times),
            "speaker": turn_of_words[0].get('speaker', 'UNKNOWN'),  # Speaker should be consistent within a turn
            "sentence": text.strip()  # Use 'sentence' key for compatibility with downstream logic
        }


    def align(self, whisperx_json_path: str, diarization_json_path: str = None, output_txt_path: str = None):
        """
        Main alignment method.
        Note: diarization_result_path is now ignored as speaker info is expected in whisperx_result.
        """
        # Step 1: Load ASR results and create a flat list of all words
        whisperx_segments = self._load_whisperx_result(whisperx_json_path)
        if not whisperx_segments:
            return []
            
        all_words = []
        for segment in whisperx_segments:
            if 'words' in segment and isinstance(segment['words'], list):
                all_words.extend(segment['words'])
        
        if not all_words:
            print("Error: No word-level information found in the input file. Cannot perform alignment.")
            return []

        # Step 2: Group words into "turns" based on speaker and pause duration
        turns = self._group_words_into_turns(all_words)

        # Step 3: Convert each "turn" into a final segment format
        final_segments = []
        for turn in turns:
            segment = self._turn_to_segment(turn)
            if segment:
                final_segments.append(segment)

        # (Optional) Write to a text file for inspection
        if output_txt_path:
            print(f"Writing aligned output to: {output_txt_path}")
            with open(output_txt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(final_segments):
                    f.write(f"{i+1} {seg['speaker']} ({seg['start']:.2f}s - {seg['end']:.2f}s): {seg['sentence']}\n")
            print("Output writing complete.")

        return final_segments

# --- Example Usage ---
if __name__ == "__main__":
    # Define paths based on the previous step's output
    whisperx_file = "whisperx_output.json" 
    diarization_file = "diarization_output.json"
    aligned_output_file = "aligned_transcript.txt"
    aligned_output_json_file = "aligned_transcript.json"

    # Create dummy input files if they don't exist (for testing)
    if not os.path.exists(whisperx_file):
        print(f"Creating dummy {whisperx_file} for testing...")
        dummy_whisperx = [
            {'text': 'Hello, this is speaker one.', 'start': 0.5, 'end': 2.8, 'words': [{'word': 'Hello,', 'start': 0.5, 'end': 0.9}, {'word': 'this', 'start': 1.0, 'end': 1.2}, {'word': 'is', 'start': 1.3, 'end': 1.4}, {'word': 'speaker', 'start': 1.5, 'end': 2.1}, {'word': 'one.', 'start': 2.2, 'end': 2.8}]},
            {'text': 'How are you speaker two?', 'start': 3.1, 'end': 5.5, 'words': [{'word': 'How', 'start': 3.1, 'end': 3.3}, {'word': 'are', 'start': 3.4, 'end': 3.5}, {'word': 'you', 'start': 3.6, 'end': 3.8}, {'word': 'speaker', 'start': 4.0, 'end': 4.6}, {'word': 'two?', 'start': 4.7, 'end': 5.5}]}
        ]
        with open(whisperx_file, 'w') as f: json.dump(dummy_whisperx, f, indent=4)

    if not os.path.exists(diarization_file):
        print(f"Creating dummy {diarization_file} for testing...")
        dummy_diarization = [
            {'start': 0.4, 'end': 2.9, 'speaker': 'SPEAKER_01'},
            {'start': 3.0, 'end': 5.6, 'speaker': 'SPEAKER_02'}
        ]
        with open(diarization_file, 'w') as f: json.dump(dummy_diarization, f, indent=4)
        
    # --- Run Alignment ---
    aligner = TimestampAligner()
    aligned_data = aligner.align(
        whisperx_json_path=whisperx_file,
        diarization_json_path=diarization_file,
        output_txt_path=aligned_output_file
    )

    # --- Save as JSON (Recommended for next steps) ---
    if aligned_data:
        print(f"Saving structured output to: {aligned_output_json_file}")
        with open(aligned_output_json_file, 'w', encoding='utf-8') as f:
            json.dump(aligned_data, f, indent=4, ensure_ascii=False)
        print("Done.")
        # print("\nFinal Aligned Data:")
        # for item in aligned_data:
        #     print(item)


# --- Embedder Classes (Wav2VecAudioEmbedder)
class RoBERTaTextEmbedder(nn.Module):
    def __init__(self,
                 model_name: str = "roberta-base",
                 device: torch.device = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"RoBERTa using device: {self.device} (FINE-TUNING ENABLED)")
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)  # Not moved to device here; handled externally by FullModel
        self.hidden_size = self.model.config.hidden_size
        # Removed self.model.eval()

    # forward now receives outputs from the tokenizer
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Removed @torch.no_grad()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1)  # (B, S) -> (B, S, 1)
        summed = (hidden * mask_expanded).sum(dim=1)
        lengths = mask_expanded.sum(dim=1).clamp(min=1)
        return summed / lengths

    # .encode() may no longer be used directly by FullModel in this workflow


class Wav2VecAudioEmbedder(nn.Module):
    def __init__(self,
                 model_name: str = "facebook/wav2vec2-base",
                 device: torch.device = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Wav2Vec2 using device: {self.device} (FINE-TUNING ENABLED)")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)  # Not moved to device here
        self.hidden_size = self.model.config.hidden_size
        # Removed self.model.eval()

    # forward now receives outputs from the processor
    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Removed @torch.no_grad()
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Ensure the mask matches the dimensions of hidden
        # hidden: (B, S_proc, D), mask: (B, S_proc)
        mask_expanded = mask.unsqueeze(-1)  # (B, S_proc) -> (B, S_proc, 1)
        summed = (hidden * mask_expanded).sum(dim=1)
        lengths = mask_expanded.sum(dim=1).clamp(min=1)
        return summed / lengths

    # .encode() may no longer be used directly by FullModel in this workflow


class Wav2VecAudioEmbedder(nn.Module):
    def __init__(self,
                 model_name: str = "facebook/wav2vec2-base",
                 device: torch.device = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Wav2Vec2 using device: {self.device} (FINE-TUNING ENABLED)")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)  # Model is not moved to device here
        self.hidden_size = self.model.config.hidden_size
        # Removed self.model.eval()

    # forward now receives output from the processor
    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Removed @torch.no_grad()
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Ensure mask matches hidden’s shape
        # hidden: (B, S_proc, D), mask: (B, S_proc)
        mask_expanded = mask.unsqueeze(-1)  # (B, S_proc) -> (B, S_proc, 1)
        summed = (hidden * mask_expanded).sum(dim=1)
        lengths = mask_expanded.sum(dim=1).clamp(min=1)
        return summed / lengths

    # .encode() might no longer be called directly from FullModel
    def mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed / lengths

    def encode(self,
               waveforms: Union[torch.Tensor, List[np.ndarray], List[torch.Tensor]],
               sampling_rate: int = 16_000) -> torch.Tensor:
        # forward now returns (last_hidden, attention_mask_for_pooling)
        last_hidden, attention_mask_for_pooling = self.forward(waveforms, sampling_rate)
        pooled = self.mean_pool(last_hidden, attention_mask_for_pooling)
        return pooled.unsqueeze(1)


# --- Feature Extractor Class (NEW) ---

# Place this in your utils.py file
# (Ensure the top of the file includes from typing import Tuple, List
#  and import torch, torchaudio, json, os)

class FeatureExtractor:
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Assumes RoBERTaTextEmbedder and Wav2VecAudioEmbedder are defined in utils.py
        self.text_embedder = RoBERTaTextEmbedder(device=self.device)
        self.audio_embedder = Wav2VecAudioEmbedder(device=self.device)
        self.target_sr = 16_000
        print(f"FeatureExtractor initialized. Target SR: {self.target_sr}, Device: {self.device}")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Loads and resamples audio to target_sr."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"    - Loading audio from: {audio_path}")
        waveform, sr = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:  # Use the first channel, ensure mono audio
            waveform = waveform[0, :].unsqueeze(0)

        if sr != self.target_sr:
            print(f"      Resampling from {sr} Hz to {self.target_sr} Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            
        return waveform.to(self.device)  # Ensure waveform is on the correct device

    def _slice_audio(self, waveform: torch.Tensor, start_sec: float, end_sec: float) -> torch.Tensor:
        """Slices a waveform based on start and end seconds. Assumes waveform is already on self.device."""
        start_sample = int(start_sec * self.target_sr)
        end_sample = int(end_sec * self.target_sr)
        
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)  # waveform shape: (1, num_samples)

        if start_sample >= end_sample:
            print(f"      Warning: Empty/invalid slice requested ({start_sec:.2f}s - {end_sec:.2f}s). Returning small silence on device {self.device}.")
            return torch.zeros(100, device=self.device)  # Return silent tensor on self.device

        # waveform is (1, num_samples), return a 1D slice tensor on the same device
        return waveform[0, start_sample:end_sample]

    def _load_aligned_data(self, json_path: str) -> list:
        """Loads aligned data from a JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Aligned JSON file not found: {json_path}")
            
        print(f"  - Loading aligned data from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def extract(self, aligned_json_path: str, audio_path: str) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Extracts text and audio embeddings for all segments.

    Args:
        aligned_json_path (str): Path to the aligned transcript JSON file.
        audio_path (str): Path to the original audio file.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor | None]: A tuple containing:
            - text_embeddings: Tensor of shape (batch_size, 1, 768) or None if error.
            - audio_embeddings: Tensor of shape (batch_size, 1, 768) or None if error.
    """
    try:
        aligned_data = self._load_aligned_data(aligned_json_path)
        if not aligned_data:
            print(f"    - Warning: No aligned data found in {aligned_json_path}. Skipping feature extraction for this file.")
            return None, None
                
        full_waveform = self._load_audio(audio_path)  # full_waveform is on self.device

        texts = []
        audio_slices_on_device = [] 

        print(f"    - Preparing text and audio slices for {os.path.basename(audio_path)}...")
        for i, segment in enumerate(aligned_data):
            texts.append(segment['sentence'])
            slice_tensor = self._slice_audio(full_waveform, segment['start'], segment['end'])
            audio_slices_on_device.append(slice_tensor)
            # --- DEBUG ---
            print(f"      DEBUG Extractor: Slice {i} for seg start {segment['start']:.2f} has shape {slice_tensor.shape}, dtype {slice_tensor.dtype}, device {slice_tensor.device}")
            if slice_tensor.numel() == 0:
                print(f"      WARNING Extractor: Slice {i} is empty (numel=0)!")
            # --- END DEBUG ---
        
        if not texts:
            print(f"    - Warning: No text sentences found in aligned data for {os.path.basename(audio_path)}. Skipping feature extraction.")
            return None, None
        
        if not audio_slices_on_device:
            print(f"    - Warning: No audio slices prepared for {os.path.basename(audio_path)}. Skipping feature extraction.")
            return None, None
        
        for i, s_tensor in enumerate(audio_slices_on_device):
            if s_tensor.dim() != 1:
                print(f"    - CRITICAL WARNING Extractor: Slice {i} is not 1D! Shape: {s_tensor.shape}. This will likely cause errors.")

        # --- Move audio slices to CPU and convert to NumPy arrays for the processor ---
        print(f"    - DEBUG Extractor: Number of audio slices to process: {len(audio_slices_on_device)}")
        audio_slices_for_processor = [s.cpu().numpy() for s in audio_slices_on_device]
        # --------------------------------------------------------------------

        print(f"    - Extracting embeddings for {len(texts)} segments...")
        text_embeddings = self.text_embedder.encode(texts)  # (num_segments, 1, D)
        
        # Wav2VecAudioEmbedder's encode method calls the processor,
        # which expects a list of 1D NumPy arrays or 1D CPU Tensors
        audio_embeddings = self.audio_embedder.encode(audio_slices_for_processor, sampling_rate=self.target_sr)
        print(f"    - Embedding extraction complete for {os.path.basename(audio_path)}.")

        return text_embeddings, audio_embeddings

    except Exception as e:
        import traceback
        print(f"    - Error during Feature Extraction for {os.path.basename(audio_path)}: {e}")
        traceback.print_exc()
        return None, None


# --- Example Usage ---
if __name__ == "__main__":
    # Assumed alignment output and original audio file
    aligned_file = "aligned_transcript.json" 
    audio_file = "your_audio.wav"  # ← Replace with your actual audio path

    # --- Create dummy files for testing (if missing) ---
    if not os.path.exists(aligned_file):
        print(f"Creating dummy {aligned_file} for testing...")
        dummy_aligned = [
            {'start': 0.5, 'end': 2.8, 'speaker': 'SPEAKER_01', 'sentence': 'Hello, this is speaker one.'},
            {'start': 3.1, 'end': 5.5, 'speaker': 'SPEAKER_02', 'sentence': 'How are you speaker two?'}
        ]
        with open(aligned_file, 'w') as f: json.dump(dummy_aligned, f, indent=4)

    if not os.path.exists(audio_file):
        print(f"Creating dummy {audio_file} for testing...")
        # Generate 6 seconds of 16kHz dummy audio (sine wave + noise)
        sr = 16000
        duration = 6
        frequency = 440
        time = torch.linspace(0., duration, int(sr * duration))
        waveform = torch.sin(frequency * time * 2 * torch.pi) * 0.5 + torch.randn_like(time) * 0.01
        torchaudio.save(audio_file, waveform.unsqueeze(0), sr)
        print(f"Dummy audio file {audio_file} created. Please replace it with your actual audio.")
    # --- Dummy file creation complete ---

    try:
        # Create FeatureExtractor instance
        extractor = FeatureExtractor()

        # Extract features
        text_embeds, audio_embeds = extractor.extract(
            aligned_json_path=aligned_file,
            audio_path=audio_file
        )

        # Check output
        if text_embeds is not None and audio_embeds is not None:
            print(f"\nOutput Text Embedding Shape: {text_embeds.shape}")
            print(f"Output Audio Embedding Shape: {audio_embeds.shape}")
            
            # These two tensors (text_embeds, audio_embeds) are your input
            # to the next step, e.g., a Cross Attention Block!
            # For example: text_embeds[0] and audio_embeds[0] are the first segment's features.

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure both the aligned JSON file and the audio file exist.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")



class CrossModalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        batch_first: bool = True
    ):
        """
        Bidirectional cross‐attention:
          - text queries audio   → a_{t→a}
          - audio queries text   → a_{a→t}
        """
        super().__init__()
        # audio as Key/Value, text as Query
        self.attn_t2a = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        # text as Key/Value, audio as Query
        self.attn_a2t = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        # optional layernorms to stabilize
        self.norm_t2a = nn.LayerNorm(embed_dim)
        self.norm_a2t = nn.LayerNorm(embed_dim)

    def forward(
        self,
        text_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          text_feats:  (batch, T_text,  embed_dim)
          audio_feats: (batch, T_audio, embed_dim)
          text_mask:   (batch, T_text) attention mask (1=keep, 0=pad)
          audio_mask:  (batch, T_audio)

        Returns:
          a_t2a:  (batch, T_text,  embed_dim)  — text‐query → audio‐K/V
          a_a2t:  (batch, T_audio, embed_dim)  — audio‐query → text‐K/V
        """
        # MultiheadAttention expects masks in the form (batch, seq) → key_padding_mask (False=keep)
        key_padding_audio = (audio_mask == 0) if audio_mask is not None else None
        key_padding_text  = (text_mask  == 0) if text_mask  is not None else None

        # text queries audio
        # Query = text_feats, Key=Value=audio_feats
        attn_out_t2a, _ = self.attn_t2a(
            query=text_feats,
            key=audio_feats,
            value=audio_feats,
            key_padding_mask=key_padding_audio,
            need_weights=False
        )
        a_t2a = self.norm_t2a(attn_out_t2a + text_feats)

        # audio queries text
        attn_out_a2t, _ = self.attn_a2t(
            query=audio_feats,
            key=text_feats,
            value=text_feats,
            key_padding_mask=key_padding_text,
            need_weights=False
        )
        a_a2t = self.norm_a2t(attn_out_a2t + audio_feats)

        return a_t2a, a_a2t

class ForgetGateFusion(nn.Module):
    def __init__(self, embed_dim: int = 768):
        """
        A forget‐gate fusion block:
          - W_g : gate linear  (2*D → D)
          - W_v : value linear (D   → D)
        """
        super().__init__()
        self.gate_proj  = nn.Linear(embed_dim * 2, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          z: Tensor of shape (B, T, D)  — base features (e.g. text)
          a: Tensor of shape (B, T, D)  — attended features (e.g. audio)
        Returns:
          h: Tensor of shape (B, T, D)
        """
        # 1) compute gate = σ( W_g [z; a] )
        concat = torch.cat([z, a], dim=-1)           # → (B, T, 2D)
        gate   = torch.sigmoid(self.gate_proj(concat))  # → (B, T, D)

        # 2) optionally project a
        a_proj = self.value_proj(a)                 # → (B, T, D)

        # 3) fuse: ReLU(z + gate * a_proj)
        h = F.relu(z + gate * a_proj)                # → (B, T, D)
        return h
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # (max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)    # even dims
        pe[:, 1::2] = torch.cos(position * div_term)    # odd dims
        pe = pe.unsqueeze(0)                            # (1, max_len, D)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch_size, seq_len, d_model)
        Returns:
          x + positional encoding (same shape)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class FusionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_first: bool = True
    ):
        """
        A Transformer‐Encoder fusion block:
          - concatenates h_ta and h_at along the time axis,
          - adds positional encodings,
          - runs through a stack of TransformerEncoder layers.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)

    def forward(
        self,
        h_ta: torch.Tensor,
        h_at: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
          h_ta: (B, T_text,  embed_dim)
          h_at: (B, T_audio, embed_dim)
          key_padding_mask: optional (B, T_fused) mask where True=PAD
        Returns:
          fused_out: (B, T_text+T_audio, embed_dim)
        """
        # 1) concatenate along the sequence dimension
        fused = torch.cat([h_ta, h_at], dim=1)  # (B, T_text+T_audio, D)

        # 2) add positional encodings
        fused = self.pos_encoder(fused)

        # 3) run through transformer encoder
        fused_out = self.transformer(fused, src_key_padding_mask=key_padding_mask)
        return fused_out

class EmotionClassifier(nn.Module):
    """
    Final Linear Classifier for Emotion Prediction.
    Maps fused embeddings to emotion class logits.
    """
    def __init__(self, input_dim: int = 768, num_classes: int = 4):
        """
        Args:
            input_dim (int): Dimension of the input features (e.g., 768).
            num_classes (int): Number of emotion classes (e.g., 4 for Happy, Angry, Sad, Neutral).
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.num_classes = num_classes
        print(f"EmotionClassifier initialized: Input Dim={input_dim}, Num Classes={num_classes}")

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to get logits.
        This is typically used during TRAINING.

        Args:
            fused_features (torch.Tensor): Input features, expected shape (B, D) 
                                           after mean pooling.

        Returns:
            torch.Tensor: Logits for each class, shape (B, num_classes).
        """
        # Ensure input is (B, D)
        if fused_features.dim() > 2:
            # Assuming input might be (B, T, D), apply mean pooling here
            # or ensure it's done before calling this.
            # Example: fused_features = fused_features.mean(dim=1)
            print(f"Warning: Input dimension is {fused_features.dim()}, "
                  f"expected 2. Applying mean pooling on dim=1.")
            fused_features = fused_features.mean(dim=1)

        logits = self.linear(fused_features) # (B, D) -> (B, num_classes)
        return logits

    def predict(self, fused_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward pass and returns probabilities and predicted class index.
        This is typically used during INFERENCE or EVALUATION.

        Args:
            fused_features (torch.Tensor): Input features, shape (B, D) or (B, T, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - probabilities: Probabilities for each class, shape (B, num_classes).
                - predicted_indices: Index of the predicted class, shape (B,).
        """
        logits = self.forward(fused_features) # Get logits (B, num_classes)
        probabilities = F.softmax(logits, dim=1) # Apply SoftMax (B, num_classes)
        predicted_indices = torch.argmax(probabilities, dim=1) # Apply Argmax (B,)
        return probabilities, predicted_indices