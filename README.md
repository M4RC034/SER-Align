# SER-Align

📄 Paper for **Enhancing Speech Emotion Recognition Leveraging Aligning Timestamps of ASR Transcripts and Speaker Diarization**  
🗓️ Accepted at **IALP 2025**

This repository contains the full pipeline for our multimodal speech emotion recognition (SER) system. It integrates **word-level ASR transcripts**, **speaker diarization**, and **emotion prediction** into a cohesive and timestamp-aware evaluation framework.

## 🔧 Project Overview

Our work explores the impact of **alignment quality**—specifically, the temporal synchronization of ASR and speaker diarization—on downstream SER performance.

We propose a **fine-tuned RoBERTa + Wav2Vec2** architecture with cross-modal attention and fusion, and evaluate performance using both standard metrics and **TEER** (*Time-weighted Emotion Error Rate*), proposed by Wu et al. (Interspeech 2023).

## 📂 Folder Structure
```bash
.
├── iemo_lab_with_timestamps.py # Generate timestamped emotion labels
├── preprocess.py # Extract audio/text features, save as .pt
├── train.py # Fine-tune multimodal SER model
├── evaluate.py # Evaluate classification performance
├── teer_calculate.py # Compute TEER, sTEER, EERa, sEERa
├── utils.py # Model modules and helper classes
├── preprocessed_iemocap/ # .pt files and aligned transcripts
└── evaluation_predictions.json # Saved output after evaluation
```


## 🚀 Pipeline Execution

Run the following scripts **in order**:

1. **Generate Emotion Labels with Timestamps**
    ```bash
    python iemo_lab_with_timestamps.py
    ```

2. **Preprocess Audio and Text**
    ```bash
    python preprocess.py
    ```

3. **Train the Multimodal Emotion Classifier**
    ```bash
    python train.py
    ```

4. **Evaluate Classification Performance**
    ```bash
    python evaluate.py
    ```

5. **Calculate TEER/sTEER Metrics**
    ```bash
    python teer_calculate.py
    ```

## 🧠 Model Architecture

Our model integrates the following components (see `utils.py`):

- `RoBERTaTextEmbedder` for textual embeddings
- `Wav2VecAudioEmbedder` for acoustic embeddings
- `CrossModalAttention` for multi-stream interaction
- `ForgetGateFusion` for adaptive fusion
- `FusionTransformer` + `EmotionClassifier` for output

## 📊 Evaluation Metrics

We report:

- **Accuracy**
- **Weighted / Macro F1**
- **Confusion Matrix**
- **TEER** (Time-weighted Emotion Error Rate)
- **sTEER** (Speaker-attributed TEER)

### TEER Formulas

Let `MS` be missed speech, `FA` be false alarms, and `CONF` be confusion error:

- `TEER  = (MS + FA + CONF_emo) / TOTAL`
- `sTEER = (MS + FA + CONF_emo+spk) / TOTAL`


## 📚 Citation for Evaluation Metric
TEER and sTEER are computed using an extension of the diarization error metrics introduced by:
```bash
Wu, W., Zhang, C., & Woodland, P. C. (2023).
Integrating Emotion Recognition with Speech Recognition and Speaker Diarisation for Conversations.
In Proceedings of Interspeech 2023, pp. 3607–3611.
https://doi.org/10.21437/Interspeech.2023-293
```

### Results are stored in result.txt for your reference.

# 💬 Contact
For questions or collaborations, please open an issue or contact [40921126l@ntnu.edu.tw].

