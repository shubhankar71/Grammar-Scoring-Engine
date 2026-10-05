# SHL Grammar Scoring Engine

An end-to-end speech-based machine learning system for predicting spoken-English grammar scores from audio recordings.

This project was developed for the **SHL Hiring Assessment 2026** Kaggle challenge, where the objective was to predict a grammar score between 0 and 5 from approximately one-minute spoken responses.

---

## Overview

Evaluating spoken grammar from audio requires more than simply analyzing the acoustic signal. The final system combines:

- Acoustic characteristics of speech
- Automatic speech recognition (ASR)
- Transcript-derived linguistic features
- ASR confidence and speech-quality information
- Tree-based ensemble regression

The final system combines **143 features** and uses an **ExtraTrees Regressor** to predict the grammar score.

### Final Results

| Evaluation | RMSE |
|---|---:|
| 5-Fold Cross-Validation | **0.656376** |
| Public Kaggle Leaderboard | **0.5131** |

---

## Problem Statement

Given a spoken-English audio recording, predict its grammar score:

```text
Grammar Score ∈ [0, 5]
```

The competition uses **Root Mean Squared Error (RMSE)** as the evaluation metric.

Lower RMSE indicates better performance.

---

## Dataset

The competition dataset contains:

- **769 labeled training recordings**
- **216 test recordings**
- WAV audio
- Approximately 45–60 seconds per recording
- Grammar scores ranging from 0 to 5

The training target distribution contains discrete and half-point scores:

```text
0.0, 1.0, 1.5, 2.0, 2.5,
3.0, 3.5, 4.0, 4.5, 5.0
```

No external training data was used.

---

# Methodology

The final pipeline combines two complementary sources of information.

```text
                    Spoken Audio
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Acoustic Analysis        Whisper ASR
              │                     │
              │              Transcript + Metadata
              │                     │
              ▼                     ▼
       127 Acoustic Features   16 ASR/Text Features
              │                     │
              └──────────┬──────────┘
                         ▼
                 143 Combined Features
                         │
                         ▼
                ExtraTrees Regressor
                         │
                         ▼
                 Grammar Score [0,5]
```

---

## 1. Acoustic Feature Engineering

The audio recordings were processed using `librosa` to capture speech and voice characteristics.

A total of **127 acoustic features** were extracted.

### Feature Categories

#### Energy

- RMS energy statistics
- Low-energy ratio

#### Voice Activity

- Speech ratio
- Pause statistics
- Long-pause count
- Number of detected segments

#### Spectral Characteristics

- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Spectral contrast

#### Temporal Characteristics

- Zero-crossing rate
- Onset rate

#### MFCCs

- MFCC 0–19
- MFCC mean/std statistics
- Delta MFCC statistics

#### Fundamental Frequency

- F0 statistics
- F0 range

These features capture information about speech dynamics, energy distribution, temporal characteristics, and acoustic properties of the spoken response.

---

## 2. Automatic Speech Recognition

The recordings were transcribed using **Whisper Small**.

The generated transcripts were then converted into numerical linguistic features.

ASR metadata was also retained to capture information related to transcription confidence and speech characteristics.

---

## 3. Transcript and ASR Features

A total of **16 transcript/ASR features** were used.

### Linguistic Features

- Character count
- Word count
- Unique word count
- Lexical diversity
- Sentence count
- Average words per sentence
- Repetition statistics
- Filler-word statistics
- Long-word statistics
- Grammar-marker count

### ASR Features

- Number of ASR segments
- Mean log probability
- Mean no-speech probability

These features provide information that cannot be captured by acoustic features alone.

---

## 4. Combined Feature Representation

The final model uses:

```text
127 acoustic features
+
16 transcript/ASR features
=
143 total features
```

The acoustic and transcript features were combined using the collision-safe identifier:

```text
split / filename
```

This was necessary because identical filenames can occur in both the training and test splits.

The final feature matrix contains:

```text
985 recordings
143 model features
```

with no missing feature values.

---

# Model Development

## 5. Model Selection

Multiple regression models were evaluated using the same frozen 5-fold cross-validation split.

Tree-based ensemble models performed substantially better than the linear baseline on the acoustic features.

Adding transcript and ASR features further improved performance.

The final model was:

### ExtraTreesRegressor

```python
ExtraTreesRegressor(
    n_estimators=1000,
    max_features=1.0,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

Predictions were clipped to the valid target range:

```text
[0, 5]
```

---

## 6. Cross-Validation

A frozen **5-fold StratifiedKFold** setup was used.

Configuration:

```text
n_splits = 5
shuffle = True
random_state = 42
```

The target was divided into score ranges before stratification to preserve the distribution of low, medium, and high grammar scores across folds.

### Fold Results

| Fold | RMSE |
|---:|---:|
| 0 | 0.7076 |
| 1 | 0.6724 |
| 2 | 0.6095 |
| 3 | 0.6645 |
| 4 | 0.6229 |

### Overall Validation

| Metric | Score |
|---|---:|
| **Pooled RMSE** | **0.656376** |
| Mean Fold RMSE | 0.655379 |
| Fold RMSE Std | 0.035373 |
| MAE | 0.512478 |
| R² | 0.718986 |

The pooled OOF RMSE is used as the primary internal validation metric.

---

# Interpretability

## 7. Feature Importance

The final ExtraTrees model provides feature importance estimates.

The contribution was approximately:

| Feature Group | Importance |
|---|---:|
| Acoustic | **71.93%** |
| Transcript / ASR | **28.07%** |

### Top Features

| Rank | Feature |
|---:|---|
| 1 | `unique_words` |
| 2 | `mean_logprob` |
| 3 | `zcr_p10` |
| 4 | `mean_no_speech` |
| 5 | `centroid_p10` |
| 6 | `rolloff_p10` |
| 7 | `zcr_mean` |
| 8 | `dmfcc1_std` |
| 9 | `mfcc1_std` |
| 10 | `zcr_med` |

The feature importance analysis indicates that both speech acoustics and linguistic/ASR information contribute meaningful predictive signal.

---

# Final Training and Prediction

## 8. Final Model

After model selection and cross-validation, the final ExtraTrees model was trained using all **769 labeled training samples**.

The model generated predictions for all **216 test recordings**.

### Test Prediction Statistics

| Statistic | Value |
|---|---:|
| Minimum | 2.2616 |
| Maximum | 4.8231 |
| Mean | 3.3164 |
| Standard Deviation | 0.5575 |

The final submission was validated to ensure:

- 216 test rows
- Unique filenames
- No missing predictions
- Predictions within `[0, 5]`
- Original test-file order preserved

---

# Results

## Internal Validation

### 5-Fold Cross-Validation RMSE

**0.656376**

## Kaggle Public Leaderboard

### Public RMSE

**0.5131**

The Kaggle leaderboard score is the external evaluation result on the competition's held-out test data.

The in-sample training RMSE was **0.064608**, but this is not used as the primary estimate of generalization because the model was evaluated on the same samples it was trained on.

---

# Repository Structure

```text
shl-grammar-scoring-engine/
│
├── grammar_scoring_engine.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

### `grammar_scoring_engine.ipynb`

Contains the final feature preparation, validation, model training, feature importance analysis, prediction generation, and competition report.

### `requirements.txt`

Lists the primary Python dependencies used by the project.

### `.gitignore`

Prevents datasets, audio files, generated caches, credentials, and temporary files from being committed.

---

# Reproducibility

The notebook documents the final modeling and evaluation workflow used for the submitted solution.

The final experiment uses validated precomputed feature caches for the acoustic and ASR-derived features.

The original competition audio dataset is not included in this repository.

This repository does not contain:

- Competition audio files
- Private credentials
- API keys
- Large generated cache files

The competition dataset should be obtained through the official Kaggle competition environment.

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Librosa
- SoundFile
- PyTorch
- Torchaudio
- Whisper
- Jupyter
- Kaggle

---

# Key Takeaways

1. **Acoustic features provide the majority of the predictive signal.**
2. **Transcript and ASR-derived features provide complementary information.**
3. Combining both feature groups improved performance over acoustic-only modeling.
4. ExtraTrees provided a strong fit for the combined numerical feature representation.
5. Frozen cross-validation was used to make model comparisons consistent.
6. The final system achieved **0.656376 RMSE in 5-fold CV** and **0.5131 RMSE on the public Kaggle leaderboard**.

---

## Competition

**SHL Hiring Assessment 2026 — Grammar Scoring Engine**

This project was developed as part of the **SHL Research Intern hiring assessment**.
