import torch
import torchaudio
import os
import glob
import json
import re
import pandas as pd
from tqdm import tqdm
import whisperx
import traceback

# Import from utils.py
try:
    from utils import TimestampAligner 
except ImportError:
    print("Error: Cannot find utils.py or TimestampAligner class. Please ensure utils.py is in the same directory.")
    exit()

# --- 1. Load models (one-time) ---
def setup_models(hf_token, vad_options=None):
    """
    Load ASR and Diarization models.
    VAD-related settings can be passed in here and will be used when loading the ASR model.
    """
    print("Setting up ASR and Diarization models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if vad_options:
        print(f"WhisperX VAD is ENABLED with options: {vad_options}")
    else:
        print("WhisperX VAD will use library defaults.")

    try:
        asr_model = whisperx.load_model(
            "large-v2", 
            device=device, 
            compute_type="float16",
            vad_options=vad_options
        )
        print("WhisperX ASR model loaded.")
        
        diar_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        print("WhisperX DiarizationPipeline loaded.")
        
        return asr_model, diar_model, device
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        exit()

# --- 2. Run ASR and Diarization ---
def run_whisperx_pyannote_fast(audio_path, whisper_out_path, diar_out_path, asr_model, diar_model_whisperx, device):
    current_basename = os.path.basename(audio_path)
    print(f"  - Running ASR/Diarization for {current_basename}...")
    result_with_speakers = {"segments": []}
    diar_segments_for_log = [] 

    try:
        audio_data_np = whisperx.load_audio(audio_path)
        
        # VAD is configured inside asr_model, force language to English
        result_asr_initial = asr_model.transcribe(audio_data_np, batch_size=8, language="en")
        
        print(f"    - Initial ASR done. Language forced to 'en'.")

        if not result_asr_initial or "segments" not in result_asr_initial or not result_asr_initial["segments"]:
            print(f"    - Warning: Initial ASR produced no segments.")
            result_aligned_asr = {"segments": []}
        else:
            result_aligned_asr = None
            try:
                align_model_ws, metadata_ws = whisperx.load_align_model(language_code="en", device=device)
                print(f"    - Performing word-level alignment...")
                result_aligned_asr = whisperx.align(result_asr_initial["segments"], align_model_ws, metadata_ws, audio_data_np, device, return_char_alignments=False)
                print(f"    - Word-level alignment done.")
                del align_model_ws
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception as e_align_phoneme:
                print(f"    - Error during whisperx phoneme-level alignment: {e_align_phoneme}")
                result_aligned_asr = result_asr_initial 

        if not result_aligned_asr or "segments" not in result_aligned_asr or not result_aligned_asr["segments"]:
             result_with_speakers = result_asr_initial
             if "segments" not in result_with_speakers: result_with_speakers["segments"] = []
        else:
            print(f"    - Performing diarization (min_speakers=2, max_speakers=2)...")
            diarization_df = diar_model_whisperx(audio_path, min_speakers=2, max_speakers=2) 

            if isinstance(diarization_df, pd.DataFrame) and diarization_df.empty:
                print(f"    - Warning: Diarization produced no segments.")
                result_with_speakers = result_aligned_asr
            elif hasattr(diarization_df, 'labels') and not diarization_df.labels():
                 print(f"    - Warning: Diarization produced no segments.")
                 result_with_speakers = result_aligned_asr
            else:
                print(f"    - Diarization done. Assigning speakers...")
                try:
                    result_with_speakers = whisperx.assign_word_speakers(diarization_df, result_aligned_asr)
                except Exception as e_assign:
                    print(f"    - Error during whisperx.assign_word_speakers: {e_assign}")
                    result_with_speakers = result_aligned_asr 
            
            if isinstance(diarization_df, pd.DataFrame) and not diarization_df.empty:
                for index, row in diarization_df.iterrows(): diar_segments_for_log.append({'start': row['start'], 'end': row['end'], 'speaker': str(row['speaker'])})
            elif hasattr(diarization_df, 'itertracks'):
                 for turn, _, speaker in diarization_df.itertracks(yield_label=True): diar_segments_for_log.append({'start': turn.start, 'end': turn.end, 'speaker': str(speaker)})

        segments_to_save = result_with_speakers.get("segments", [])
        with open(whisper_out_path, "w", encoding='utf-8') as f:
            json.dump(segments_to_save, f, indent=4, ensure_ascii=False)
        print(f"    - WhisperX output saved to {whisper_out_path}")

        if segments_to_save and isinstance(segments_to_save, list) and len(segments_to_save) > 0:
            if segments_to_save[0].get("words"): print(f"    - DEBUG: First saved segment HAS 'words' key.")
            else: print(f"    - DEBUG: First saved segment DOES NOT HAVE 'words' key. Keys: {segments_to_save[0].keys()}")
        else: print(f"    - DEBUG: No ASR segments were saved to {whisper_out_path}.")

        with open(diar_out_path, "w", encoding='utf-8') as f:
            json.dump(diar_segments_for_log, f, indent=4, ensure_ascii=False)
        print(f"    - Diarization log saved to {diar_out_path}")
        return True
    except Exception as e:
        print(f"    - Overall Error during ASR/Diarization for {current_basename}: {e}")
        traceback.print_exc()
        return False

# --- Utility Functions ---
def load_ground_truth_utterances(json_path):
    print(f"Loading ground truth data from: {json_path}")
    if not os.path.exists(json_path): return None
    with open(json_path, 'r', encoding='utf-8') as f: return json.load(f)

def get_label_for_segment(wav_filename_base, segment_start, segment_end, all_ground_truth):
    gt_utterances = all_ground_truth.get(wav_filename_base, [])
    if not gt_utterances: return None
    best_overlap, best_emotion = 0.0, None
    for gt_utt in gt_utterances:
        overlap_start = max(segment_start, gt_utt['start'])
        overlap_end = min(segment_end, gt_utt['end'])
        overlap_duration = max(0, overlap_end - overlap_start)
        if overlap_duration > best_overlap:
            best_overlap, best_emotion = overlap_duration, gt_utt['emotion']
    return best_emotion if best_overlap > 0.1 else None

# --- Main Preprocessing Pipeline ---
def preprocess_iemocap(iemocap_root_path, output_path, hf_token):
    LABEL_MAP = {'hap': 0, 'ang': 1, 'sad': 2, 'neu': 3}
    TARGET_SR = 16000
    
    VAD_ONSET_THRESHOLD = 0.4
    VAD_OPTIONS = {"vad_onset": VAD_ONSET_THRESHOLD}
    
    asr_model, diar_model_whisperx, device = setup_models(hf_token, vad_options=VAD_OPTIONS)
    
    labels_json_path = os.path.join(output_path, 'iemocap_labels_with_timestamps.json')
    all_ground_truth = load_ground_truth_utterances(labels_json_path) 
    if all_ground_truth is None: return

    aligner = TimestampAligner()

    aligned_transcripts_txt_dir = os.path.join(output_path, "aligned_transcripts_txt")
    aligned_transcripts_json_dir = os.path.join(output_path, "aligned_transcripts_json")
    train_data_dir = os.path.join(output_path, 'train')
    test_data_dir = os.path.join(output_path, 'test')
    for dir_path in [aligned_transcripts_txt_dir, aligned_transcripts_json_dir, train_data_dir, test_data_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    temp_whisper_json = "temp_whisperx.json"
    temp_diar_json = "temp_diarization.json"

    processed_files_count, skipped_files_count, total_segments_saved = 0, 0, 0

    for session_id in range(1, 6): 
        session_path = os.path.join(iemocap_root_path, f"Session{session_id}", "dialog", "wav")
        if not os.path.exists(session_path): continue
            
        wav_files = sorted(glob.glob(os.path.join(session_path, "*.wav")))

        print(f"\nProcessing Session {session_id} ({len(wav_files)} files)...")
        for audio_file in tqdm(wav_files, desc=f"Session {session_id}"):
            basename = os.path.basename(audio_file)
            wav_id = os.path.splitext(basename)[0]

            asr_diar_success = run_whisperx_pyannote_fast(
                audio_file, temp_whisper_json, temp_diar_json,    
                asr_model, diar_model_whisperx, device
            )

            if not asr_diar_success:
                print(f"  - ASR/Diarization failed for {basename}, skipping file.")
                skipped_files_count += 1
                continue

            aligned_data, permanent_aligned_json_path = None, None
            try:
                permanent_aligned_txt_path = os.path.join(aligned_transcripts_txt_dir, f"{wav_id}_aligned.txt")
                aligned_data = aligner.align(temp_whisper_json, temp_diar_json, permanent_aligned_txt_path)
                if not aligned_data: 
                    print(f"  - TimestampAligner produced no data for {basename}, skipping file.")
                    skipped_files_count += 1
                    continue
                permanent_aligned_json_path = os.path.join(aligned_transcripts_json_dir, f"{wav_id}_aligned.json")
                with open(permanent_aligned_json_path, 'w', encoding='utf-8') as f:
                    json.dump(aligned_data, f, indent=4, ensure_ascii=False)
                print(f"    - Aligned transcript (JSON) saved to {permanent_aligned_json_path}")
            except Exception as e_align:
                print(f"  - Error during Timestamp Alignment for {basename}: {e_align}")
                skipped_files_count += 1
                continue
            
            try:
                full_waveform_tensor, sr = torchaudio.load(audio_file)
                if full_waveform_tensor.shape[0] > 1: full_waveform_tensor = full_waveform_tensor[0, :]
                if sr != TARGET_SR:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
                    full_waveform_tensor = resampler(full_waveform_tensor)
                max_val = torch.max(torch.abs(full_waveform_tensor))
                if max_val > 0.01: full_waveform_tensor = full_waveform_tensor / max_val * 0.95
            except Exception as e_load_audio:
                print(f"  - Error loading or resampling audio {audio_file}: {e_load_audio}")
                skipped_files_count += 1
                continue

            segments_saved_for_this_file = 0
            if aligned_data:
                for i, segment_data in enumerate(aligned_data):
                    label_str = get_label_for_segment(wav_id, segment_data['start'], segment_data['end'], all_ground_truth)
                    if label_str and label_str in LABEL_MAP:
                        label_idx = LABEL_MAP[label_str]
                        start_sample = int(segment_data['start'] * TARGET_SR)
                        end_sample = int(segment_data['end'] * TARGET_SR)
                        audio_slice_tensor = full_waveform_tensor[start_sample:end_sample]

                        if audio_slice_tensor.numel() < 100: continue
                        
                        data_to_save = {
                            'sentence_text': segment_data['sentence'], 
                            'audio_waveform': audio_slice_tensor.cpu().numpy(), 
                            'label': torch.tensor(label_idx, dtype=torch.long)
                        }
                        
                        save_dir = test_data_dir if session_id == 5 else train_data_dir
                        pt_filename = os.path.join(save_dir, f"{wav_id}_seg_{i}.pt")
                        torch.save(data_to_save, pt_filename)
                        segments_saved_for_this_file += 1
                
                if segments_saved_for_this_file > 0:
                    processed_files_count += 1
                    total_segments_saved += segments_saved_for_this_file
                    print(f"    - Saved {segments_saved_for_this_file} segments for {basename}")
                else:
                    print(f"    - No valid labeled segments saved for {basename}.")
            else:
                print(f"    - No aligned data to process for {basename}")
                skipped_files_count +=1

    print("\nCleaning up top-level temporary files...")
    for f_temp in [temp_whisper_json, temp_diar_json]:
        if os.path.exists(f_temp): 
            try:
                os.remove(f_temp)
                print(f"  - Removed {f_temp}")
            except Exception as e_remove:
                print(f"  - Error removing {f_temp}: {e_remove}")
        
    print(f"\nPreprocessing finished.")
    print(f"  Successfully processed and saved segments from {processed_files_count} WAV files.")
    print(f"  Total .pt segments saved: {total_segments_saved}.")
    print(f"  Skipped/Failed {skipped_files_count} WAV files during major processing steps.")


# --- Run Preprocessing ---
if __name__ == "__main__":
    IEMOCAP_PATH = "./IEMOCAP_full_release" 
    OUTPUT_PATH = "./preprocessed_iemocap"  
    HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"  # <--- !!! Replace with your actual token !!!

    if not os.path.exists(IEMOCAP_PATH):
        print(f"Error: IEMOCAP path '{IEMOCAP_PATH}' not found. Please check the path.")
    elif HF_TOKEN == "YOUR_HUGGINGFACE_TOKEN" or not HF_TOKEN:
         print(f"Error: Please set a valid Hugging Face Token in preprocess.py.")
    else:
        print("NOTE: Please make sure you've already run iemo_lab_with_timestamps.py (or another label preparation script)")
        print(f"      to generate the label JSON file at: {os.path.join(OUTPUT_PATH, 'iemocap_labels_with_timestamps.json')}")
        preprocess_iemocap(IEMOCAP_PATH, OUTPUT_PATH, HF_TOKEN)
