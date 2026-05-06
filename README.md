# Phishing Email Detection – Dual‑Model System

A two‑model pipeline that detects phishing emails using **structural** (hand‑crafted) and **linguistic** (Word2Vec) features, then fuses their predictions into a single, stronger output.

---

## Project Structure

```
.
├── data/
│   ├── raw/                         # Original 7 heterogeneous CSVs
│   ├── preprocessed/
│   │   └── processed_data.csv       # Unified dataset (label, raw_text, cleaned_text, …)
│   ├── model_a/
│   │   ├── features.csv             # Structural feature matrix
│   │   └── splits/                  # train / val / test splits for Model A
│   └── model_b/
│       └── vectorized_data/         # train / val / test splits for Model B
│       └── splits/                  # train / val / test splits for Model B
├── models/
│   ├── best_model_a.joblib          # Best structural classifier
│   ├── best_model_b.joblib          # Best linguistic classifier
│   └── fusion_weights.json          # Optimal fusion weights (w_A, w_B)
│   └── word2vec_150d.joblib         # Optimal fusion weights (w_A, w_B)
```

---

## How It Works

### 1. Preprocessing (`01_data_preprocessing.ipynb`)
- Scans all CSVs in `data/raw/`.
- Detects label, subject, and body columns using flexible candidate lists.
- Normalises labels to binary (0 = legitimate, 1 = phishing).
- Creates `raw_text` (subject + `\n` + body) and `cleaned_text` (lowercased, HTML‑escaped, stopwords removed).
- Saves the unified, cleaned dataset to `data/preprocessed/processed_data.csv`.

### 2. Model B – Linguistic Expert (`02_model_b_embedding.ipynb`)
- Loads `processed_data.csv`.
- Trains a **Word2Vec** model (150‑d, Skip‑gram) on the cleaned text.
- Averages word vectors to produce a fixed‑size document embedding for each email.
- Saves the feature matrix and model.

### 3. Model A – Structural Expert (`03_model_a_features.ipynb`)
- Extracts **12 hand‑crafted features** from `raw_text`:
  - Number of URLs, email addresses, IP addresses, HTML tags
  - Number of exclamation & question marks
  - Uppercase ratio, average word length, total words
  - Count of suspicious phishing keywords
  - Number of reply/forward markers (`RE:` / `FWD:`)
  - Total subdomains found in URLs
- Saves the feature matrix to `data/model_a/features.csv`.

### 4. Model Training (`04_model_training_a.ipynb`, `05_model_training_b.ipynb`)
- Both notebooks follow the same logic:
  - Split data into **train / val / test** (60/20/20, stratified).
  - Train three lightweight classifiers: **Random Forest**, **Logistic Regression**, **Naive Bayes**.
  - Select the best model based on **F1 score** on the test set.
  - Save the winner to `models/`.

### 5. Fusion (`06_model_fusion.ipynb`)
- Loads both best models and the aligned test splits.
- Computes phishing probabilities \(p_A\) and \(p_B\).
- Tries multiple weight combinations: \(p_{fused} = w_A \cdot p_A + w_B \cdot p_B\).
- Thresholds at 0.5.
- Reports confusion matrices and F1 scores for each combination.
- Saves the optimal weights to `models/fusion_weights.json`.

---

## Quick Start

1. Clone the repository and place your raw CSVs in `data/raw/`.
2. Set up a Python environment:
   ```bash
   pip install pandas numpy scikit-learn gensim matplotlib seaborn joblib
   ```
3. Run the notebooks in order:
   - `01_data_preprocessing.ipynb`
   - `02_model_b_embedding.ipynb`
   - `03_model_a_features.ipynb`
   - `04_model_training_a.ipynb`
   - `05_model_training_b.ipynb`
   - `06_model_fusion.ipynb`
4. The final fusion configuration will be in `models/fusion_weights.json`.

---

## Key Results (example placeholders – update after running)

| Model | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| Model A (Structural) | 0.xxxx | 0.xxxx | 0.xxxx |
| Model B (Linguistic) | 0.xxxx | 0.xxxx | 0.xxxx |
| **Fused (A+B)** | **0.xxxx** | **0.xxxx** | **0.xxxx** |

---

## Extending the Project
- Add more structural features (e.g., SPF/DKIM headers, attachment count).
- Swap Word2Vec for a lightweight sentence transformer (e.g., `all-MiniLM-L6-v2`) for richer embeddings.
- Experiment with different classifiers or gradient‑boosting models.
- Tune the fusion threshold or use a meta‑learner instead of fixed weights.