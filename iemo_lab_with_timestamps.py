import os
import json
import re
from collections import Counter
from tqdm import tqdm
import glob

# --- Paths ---
iemo_root = 'IEMOCAP_full_release/' 
output_dir = './preprocessed_iemocap/'

def load_transcriptions(iemo_root_path):
    """
    Step 1: Traverse through the transcription folder
    """
    print("Step 1/3: Loading transcriptions")
    transcription_dict = {}
    trans_paths = glob.glob(os.path.join(iemo_root_path, "Session*", "dialog", "transcriptions", "*.txt"))
    
    for trans_path in trans_paths:
        with open(trans_path, 'r', encoding='latin-1') as f:
            for line in f:
                match = re.match(r'^(\w+)\s+\[.+\]:\s+(.+)$', line.strip())
                if match:
                    utt_id = match.group(1).strip()  # Only keep ID part, e.g., 'Ses01F_impro01_F000'
                    text = match.group(2).strip()    # Only keep the text after the colon
                    transcription_dict[utt_id] = text
                    
    print(f"Loaded. Found {len(transcription_dict)} transcription entries.")
    return transcription_dict

def create_final_labels(iemo_root_path, transcription_dict):
    """
    Step 2: Traverse all 'EmoEvaluation' folders to extract emotion, timestamps, and merge with transcription.
    """
    print("Step 2/3: Parsing emotion labels and timestamps, merging with transcriptions...")
    utterance_data = {}
    EMOTION_MAP = {'hap': 'hap', 'exc': 'hap', 'sad': 'sad', 'ang': 'ang', 'neu': 'neu'}

    emo_eval_paths = glob.glob(os.path.join(iemo_root_path, "Session*", "dialog", "EmoEvaluation", "*.txt"))
    
    for label_path in tqdm(emo_eval_paths, desc="Processing EmoEvaluation files"):
        wav_id = os.path.splitext(os.path.basename(label_path))[0]
        if wav_id not in utterance_data:
            utterance_data[wav_id] = []
        
        with open(label_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith('['):
                    match = re.match(r'^\[(\d+\.\d+) - (\d+\.\d+)\]\s+(\w+)\s+(\w+)\s+\[.+\]$', line.strip())
                    if match:
                        utt_id = match.group(3)
                        mapped_emo = EMOTION_MAP.get(match.group(4))
                        
                        transcription_text = transcription_dict.get(utt_id)
                        
                        if mapped_emo and transcription_text:
                            utterance_data[wav_id].append({
                                "id": utt_id,
                                "start": float(match.group(1)),
                                "end": float(match.group(2)),
                                "emotion": mapped_emo,
                                "transcription": transcription_text 
                            })

    final_data = {k: v for k, v in utterance_data.items() if v}
    print(f"Merging complete. Processed {len(final_data)} WAV files with valid utterances.")
    total_utterances = sum(len(v) for v in final_data.values())
    print(f"Generated {total_utterances} utterances with target emotion and transcription.")
    return final_data

def save_data(data, output_filepath):
    """Step 3: Save the final JSON file"""
    print(f"Step 3/3: Saving results to: {output_filepath}")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print("Saved.")

if __name__ == "__main__":
    if not os.path.exists(iemo_root):
        print(f"Error: IEMOCAP path '{iemo_root}' not found. Please verify the path.")
    else:
        all_transcriptions = load_transcriptions(iemo_root)
        final_label_data = create_final_labels(iemo_root, all_transcriptions)
        output_path = os.path.join(output_dir, 'iemocap_labels_with_timestamps.json')
        save_data(final_label_data, output_path)
