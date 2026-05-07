# Phishing Email Detection

A two‑model pipeline that detects phishing emails using **structural** (hand‑crafted) and **linguistic** (Word2Vec) features, then fuses their predictions into a single, stronger output.

`Remark` before you start the project you should download the data from https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset after that put the 7 csv files in data/raw as presented in the following structure:

---

## Project Structure

```
project folder
├── data/
│   ├── raw/                         # Original 7 heterogeneous CSVs
│   ├── preprocessed/
│   │   └── processed_data.csv       # Unified dataset (label, raw_text, cleaned_text, …)
│   ├── model_a/
│   │   ├── extracted_features
|   |   |   ├──features.csv          # Structural feature matrix
│   │   └── splits/                  # contains csv files for train / val / test splits for Model A
│   └── model_b/
│       └── vectorized_data/         # csv file of vectorized data for Model B
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
  If you want to use conda environment:
   ```bash
   pip install -r requirements.txt
   ```
   `Note` before runing the command ensure you are in the root folder of the project

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

# Email Phishing Detection

This project contains a two-expert phishing detector and a local website for testing one email at a time.

- `model_a/` trains the structural expert on 12 handcrafted email features.
- `model_b/` trains the text expert on 150-dimensional average Word2Vec document vectors.
- `models/` stores the live artifacts used by the website: `best_model_a.joblib`, `best_model_b.joblib`, `word2vec_150d.joblib`, and `fusion_weights.json`.
- Fusion combines both phishing probabilities: `w_a * p_a + w_b * p_b`, then compares the result with the saved threshold.

## Run the Website

From the website folder:

```powershell
python server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The website is model-backed. It will not produce real predictions if `index.html` is opened directly as a file, because the browser must call the local `/api/analyze` endpoint.

## Model A Features

Model A expects these 12 numeric features in order:

1. `num_urls`
2. `num_emails`
3. `num_ips`
4. `num_html_tags`
5. `num_exclamations`
6. `num_questions`
7. `uppercase_ratio`
8. `total_words`
9. `avg_word_length`
10. `suspicious_word_count`
11. `num_replies_forwards`
12. `num_subdomains`

## Model B Text Flow

Model B uses the same text preparation as the notebooks:

1. Combine subject and body.
2. Decode HTML entities and remove HTML tags.
3. Lowercase and keep alphanumeric tokens.
4. Remove English stop words.
5. Split into tokens.
6. Average known token vectors from `word2vec_150d.joblib`.
7. Send the resulting 150-dimensional vector to `best_model_b.joblib`.

## Reports

Current report highlights:

- Model A test F1: `0.822`
- Model B validation F1: `0.941`
- Fusion test F1: `0.857`
- Fusion test recall: `1.000`

The live website always uses the artifacts currently saved in `models/`.