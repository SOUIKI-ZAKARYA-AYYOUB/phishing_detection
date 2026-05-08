# Phishing Email Detection

A two-model pipeline that detects phishing emails by fusing **structural** (12 hand-crafted features) and **linguistic** (Word2Vec embeddings) signals into a single, stronger prediction.

> **Demo:** [Watch the project presentation on YouTube](https://youtu.be/od55u1Tha_Q)

---

## Prerequisites

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) and place all 7 CSV files in `data/raw/` before running any notebook.

`Remark` If you want to download the pre-trained models to save time [click here](https://drive.google.com/drive/folders/1AdJhUR52Fu_OJIspRQuRf8qT5KInoRGi?usp=sharing)
After you finish put the models/ folder in the root folder of the project as states in the following presentation.


---

## Project Structure

```
project/
├── data/
│   ├── raw/                        # Original 7 heterogeneous CSVs
│   ├── preprocessed/
│   │   └── processed_data.csv      # Unified dataset (label, raw_text, cleaned_text, …)
│   ├── model_a/
│   │   ├── extracted_features/
│   │   │   └── features.csv        # Structural feature matrix
│   │   └── splits/                 # Train / val / test splits for Model A
│   └── model_b/
│       ├── vectorized_data/        # Word2Vec document vectors
│       └── splits/                 # Train / val / test splits for Model B
├── models/
│   ├── best_model_a.joblib         # Best structural classifier
│   ├── best_model_b.joblib         # Best linguistic classifier
│   ├── word2vec_150d.joblib        # Trained Word2Vec model
│   └── fusion_weights.json         # Optimal fusion weights (w_A, w_B)
└── website/
    ├── index.html
    └── server.py                   # Local API server
```

---

## Pipeline Overview

### 1. Preprocessing — `01_data_preprocessing.ipynb`
Scans all CSVs in `data/raw/`, auto-detects label/subject/body columns, and normalises labels to binary (0 = legitimate, 1 = phishing). Produces `raw_text` (subject + body) and `cleaned_text` (lowercased, HTML-stripped, stopwords removed), then saves everything to `data/preprocessed/processed_data.csv`.

### 2. Model B: Linguistic Expert — `02_model_b_embedding.ipynb`
Trains a **150-dimensional Skip-gram Word2Vec** model on `cleaned_text`. Each email is represented by the average of its token vectors, producing a fixed-size document embedding saved alongside the Word2Vec model.

### 3. Model A: Structural Expert — `03_model_a_features.ipynb`
Extracts 12 hand-crafted features from `raw_text`:

| # | Feature |
|---|---------|
| 1 | Number of URLs |
| 2 | Number of email addresses |
| 3 | Number of IP addresses |
| 4 | Number of HTML tags |
| 5 | Number of exclamation marks |
| 6 | Number of question marks |
| 7 | Uppercase ratio |
| 8 | Total word count |
| 9 | Average word length |
| 10 | Suspicious keyword count |
| 11 | Reply/forward marker count (`RE:` / `FWD:`) |
| 12 | Total subdomains in URLs |

### 4. Model Training — `04_model_training_a.ipynb` & `05_model_training_b.ipynb`
Both models follow the same workflow: split data 60/20/20 (stratified), train **Random Forest**, **Logistic Regression**, and **Naive Bayes**, then select the best by F1 score on the test set. Winners are saved to `models/`.

### 5. Fusion — `06_model_fusion.ipynb`
Loads both best models and computes a weighted probability:

$$p_{\text{fused}} = w_A \cdot p_A + w_B \cdot p_B$$

Sweeps multiple weight combinations, thresholds at 0.5, and saves the optimal `(w_A, w_B)` to `models/fusion_weights.json`.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt   # or: pip install pandas numpy scikit-learn gensim matplotlib seaborn joblib

# 2. Run notebooks in order
01_data_preprocessing.ipynb
02_model_b_embedding.ipynb
03_model_a_features.ipynb
04_model_training_a.ipynb
05_model_training_b.ipynb
06_model_fusion.ipynb
```

---

## Results

| Model | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| Model A — Structural | 0.822 | — | — |
| Model B — Linguistic | 0.941 | — | — |
| **Fused (A + B)** | **0.857** | — | **1.000** |

---

## Local Demo Website

```bash
cd website
python server.py
# Open http://127.0.0.1:8000
```

> **Note:** Open the URL in a browser — do **not** open `index.html` directly as a file. Predictions require a live call to the `/api/analyze` endpoint.

---

## Extending the Project

- Add richer structural features (e.g., SPF/DKIM headers, attachment count).
- Replace Word2Vec with a lightweight sentence transformer (e.g., `all-MiniLM-L6-v2`).
- Try gradient-boosting classifiers (XGBoost, LightGBM).
- Replace fixed fusion weights with a trained meta-learner.
- Tune the decision threshold beyond the default 0.5.
