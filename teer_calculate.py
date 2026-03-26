import json
import os
import glob
import re
from collections import defaultdict
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm

# --- Global Configurations ---
PREPROCESSED_PATH = "./preprocessed_iemocap"
EMOTION_MAP_INV = {0: 'hap', 1: 'ang', 2: 'sad', 3: 'neu'}  # Inverse label map for display

def load_json(filepath):
    """Safely load a JSON file"""
    if not os.path.exists(filepath):
        print(f"Error: File not found -> {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        print(f"Warning: Failed to read {filepath} with utf-8. Trying latin-1.")
        with open(filepath, 'r', encoding='latin-1') as f:
            return json.load(f)

def get_ground_truth_speaker(utt_id):
    """Extract speaker gender (M/F) from utterance ID"""
    match = re.search(r'_(M|F)\d*$', utt_id)
    if match:
        return match.group(1)
    return "UNKNOWN"

def main():
    print("--- Starting TEER and sTEER Evaluation (Target Emotions Only) ---")
    
    LABEL_MAP = {'hap': 0, 'ang': 1, 'sad': 2, 'neu': 3}
    TARGET_EMOTIONS = set(LABEL_MAP.keys())

    print("Loading data...")
    gt_labels_path = os.path.join(PREPROCESSED_PATH, "iemocap_labels_with_timestamps.json")
    ground_truth_data = load_json(gt_labels_path)
    
    hypothesis_aligned_dir = os.path.join(PREPROCESSED_PATH, "aligned_transcripts_json")

    predictions_path = "evaluation_predictions.json"
    predictions_data = load_json(predictions_path)

    if not all([ground_truth_data, predictions_data]):
        print("Missing required input files. Aborting.")
        return

    predictions_by_pt_path = {pred['pt_path']: pred for pred in predictions_data}
    
    # Metric accumulators
    total_duration, total_ms, total_fa = 0.0, 0.0, 0.0
    total_conf_emo, total_conf_emo_spk = 0.0, 0.0
    total_intersection_emo_duration = 0.0
    total_intersection_emo_spk_duration = 0.0
    
    test_aligned_files = glob.glob(os.path.join(hypothesis_aligned_dir, "Ses05*.json"))

    print(f"Evaluating {len(test_aligned_files)} audio files from the test set...")

    for aligned_json_path in tqdm(sorted(test_aligned_files), desc="Calculating Metrics"):
        wav_id = os.path.splitext(os.path.basename(aligned_json_path))[0].replace("_aligned", "")
        aligned_segments = load_json(aligned_json_path)
        emotion_preds = [p for p in predictions_data if p['wav_id'] == wav_id]

        # Reference (ground truth) annotations
        ref_speech = Annotation()
        ref_emotion = Annotation()
        ref_emo_spk = Annotation()
        
        gt_utterances = ground_truth_data.get(wav_id, [])
        
        # Only include target emotions in reference annotations
        for utt in gt_utterances:
            gt_emo = utt['emotion']
            if gt_emo not in TARGET_EMOTIONS:
                continue

            segment = Segment(utt['start'], utt['end'])
            gt_spk = get_ground_truth_speaker(utt['id'])

            ref_speech[segment] = "speech"
            ref_emotion[segment] = gt_emo
            ref_emo_spk[segment] = f"{gt_emo}_{gt_spk}"

        if not ref_speech:
            print(f"Notice: File {wav_id} contains no ground-truth segments with target emotions. Skipping.")
            continue

        # Hypothesis annotations
        hyp_speech = Annotation()
        hyp_emotion = Annotation()
        hyp_speaker = Annotation()
        hyp_emo_spk = Annotation()
        
        pt_dir = os.path.join(PREPROCESSED_PATH, 'test')
        if aligned_segments is None:
            continue
        
        for i, seg in enumerate(aligned_segments):
            segment = Segment(seg['start'], seg['end'])
            hyp_spk = seg['speaker']
            hyp_speech[segment] = "speech"
            hyp_speaker[segment] = hyp_spk

            pt_filename = os.path.join(pt_dir, f"{wav_id}_seg_{i}.pt").replace('\\', '/')
            prediction_info = predictions_by_pt_path.get(pt_filename)
            
            hyp_emo = 'UNK'
            if prediction_info:
                hyp_emo_idx = prediction_info['predicted_label']
                hyp_emo = EMOTION_MAP_INV.get(hyp_emo_idx, 'UNK')

            hyp_emotion[segment] = hyp_emo
            hyp_emo_spk[segment] = f"{hyp_emo}_{hyp_spk}"

        # Compute metrics with pyannote
        evaluator = DiarizationErrorRate()
        uem = ref_speech.get_timeline().extent() | hyp_speech.get_timeline().extent()

        speech_components = evaluator.compute_components(ref_speech, hyp_speech, uem=uem, detailed=True)
        emo_components = evaluator.compute_components(ref_emotion, hyp_emotion, uem=uem, detailed=True)
        emo_spk_components = evaluator.compute_components(ref_emo_spk, hyp_emo_spk, uem=uem, detailed=True)

        total_duration += speech_components['total']
        total_ms += speech_components['missed detection']
        total_fa += speech_components['false alarm']
        total_conf_emo += emo_components['confusion']
        total_conf_emo_spk += emo_spk_components['confusion']

        total_intersection_emo_duration += emo_components['correct'] + emo_components['confusion']
        total_intersection_emo_spk_duration += emo_spk_components['correct'] + emo_spk_components['confusion']

    # Final TEER/sTEER reporting
    print("\n--- TEER & sTEER Calculation Finished ---")
    if total_duration > 0:
        teer = (total_ms + total_fa + total_conf_emo) / total_duration
        steer = (total_ms + total_fa + total_conf_emo_spk) / total_duration
        eera = total_conf_emo / total_intersection_emo_duration if total_intersection_emo_duration > 0 else 0
        seera = total_conf_emo_spk / total_intersection_emo_spk_duration if total_intersection_emo_spk_duration > 0 else 0
        
        print("\n--- Overall System Performance (Target Emotions Only) ---")
        print(f"Total Target Speech Duration: {total_duration:.2f} seconds")
        print("-" * 45)
        print(f"Missed Speech (MS):              {total_ms:.2f} s")
        print(f"False Alarm (FA):                {total_fa:.2f} s")
        print(f"Emotion Confusion (CONF_emo):    {total_conf_emo:.2f} s")
        print(f"Emotion+Speaker Conf. (CONF_emo+spk): {total_conf_emo_spk:.2f} s")
        print("-" * 45)
        print(f"TEER  = (MS + FA + CONF_emo) / TOTAL  = {teer:.4f} ({teer*100:.2f}%)")
        print(f"sTEER = (MS + FA + CONF_spk) / TOTAL  = {steer:.4f} ({steer*100:.2f}%)")

        print("\n--- Classifier Performance (Overlap Regions Only) ---")
        print(f"Overlap Duration for EERa:         {total_intersection_emo_duration:.2f} s")
        print(f"Emotion Error on Agreement (EERa): {eera:.4f} ({eera*100:.2f}%)")
        print(f"Emo+Spk Error on Agreement (sEERa):{seera:.4f} ({seera*100:.2f}%)")
    else:
        print("No ground-truth reference speech with target emotions found. Cannot compute TEER/sTEER.")

if __name__ == "__main__":
    main()

"""
@inproceedings{wu23_interspeech,
author={Wen Wu and Chao Zhang and Philip C. Woodland},
title={{Integrating Emotion Recognition with Speech Recognition and Speaker Diarisation for Conversations}},
year=2023,
booktitle={Proc. INTERSPEECH 2023},
pages={3607--3611},
doi={10.21437/Interspeech.2023-293}
}
"""
