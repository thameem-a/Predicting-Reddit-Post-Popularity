# Predicting Reddit Post Popularity

This project predicts Reddit post upvote counts at the exact moment of publication, before any user engagement occurs. The central research question is whether incorporating post body text (beyond the title) improves prediction accuracy over metadata alone.

**Colab notebook:** https://colab.research.google.com/drive/1LCAJoAdfrhui3pZgdSs5bf_VLkwW6QTJ?usp=sharing

---

## Problem Setup

Upvote counts follow a heavily right-skewed distribution, with over 53% of posts receiving zero upvotes. To stabilize training, the target variable is `log(1 + score)` rather than raw score. The task is framed as regression.

---

## Dataset

**Source:** Reddit Pushshift corpus (`RS_2026-01.zst` via Academic Torrents), processed with a custom streaming pipeline.

**Scale:** 450,000 posts total, split into train (315,000), validation (67,500), and test (67,500).

Sampling is stratified by score bin to prevent low-score posts from dominating:

| Bin | Score Range | Cap |
|-----|-------------|-----|
| zero | <= 0 | 150,000 |
| low | 1 to 10 | 150,000 |
| mid | 11 to 100 | 100,000 |
| high | > 100 | 50,000 |

**Features:**

| Feature | Description |
|---------|-------------|
| `title` | Post title (primary text input) |
| `content` | Post body text (present in ~47% of posts) |
| `has_content` | Binary flag indicating whether body text exists |
| `nsfw` | NSFW flag |
| `is_self` | Text post vs. link post |
| `hour` | Hour of day submitted (UTC) |
| `day_of_week` | Day of week (0 = Monday) |
| `is_weekend` | Weekend flag |
| `title_length` | Word count of title |
| `char_length` | Character count of title |
| `uppercase_ratio` | Fraction of uppercase characters in title |
| `num_exclamations` | Count of `!` in title |
| `num_questions` | Count of `?` in title |
| `content_length` | Body text word count |

---

## Models

Five models of increasing complexity are evaluated.

### Baseline Models (metadata only)

**Dummy Regressor** predicts the training mean for every post. This is the performance floor.

**Ridge Regression** fits a linear model over 12 metadata features with L2 regularization. Alpha is swept over [0.001, 0.01, 0.1, 1, 10, 100, 1000]. Features are scaled with `StandardScaler` to prevent data leakage.

**Random Forest** performs grid search over 24 configurations: `n_estimators` (100, 200), `max_depth` (10, 20), `min_samples_split` (2, 5), and `max_features` (sqrt, 0.5, 1.0).

### Hybrid DistilBERT Models (text + metadata)

Both BERT models share the same `HybridModel` architecture:

- **DistilBERT encoder** (`distilbert-base-uncased`) with only the final transformer layer (layer 5) unfrozen, giving approximately 7.3M trainable parameters out of 67M total. Freezing all layers caused immediate plateau; partial unfreezing enabled continuous learning.
- **Mean pooling** over all non-padding token states (768-dim), which outperforms CLS-only pooling.
- **Concatenation** of the pooled text representation with 12 normalized metadata features, producing a 780-dim combined input.
- **Regression head:** Linear -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear -> scalar output.

**Model 1 (Title + Metadata)** tokenizes the post title with `max_length=80`, which covers more than 99% of Reddit titles.

**Model 2 (Content-Aware)** concatenates title and body text with a `[SEP]` separator at `max_length=128`. When body text is absent (about 53% of posts), it falls back to title-only input.

**Training configuration for both BERT models:**
- Differential learning rates: BERT layer 5 at `2e-5`, regression head at `1e-3`
- `weight_decay=1e-4` on weights, 0.0 on biases and LayerNorm parameters
- `warmup_steps=500`, `fp16=True`, `num_train_epochs=10`
- Early stopping with patience 2; best checkpoint by validation RMSE is restored

---

## Results

Primary metric is RMSE on `log(1 + score)`.

| Model | Val RMSE | Test RMSE |
|-------|----------|-----------|
| Dummy Regressor | 1.9398 | ~1.93 |
| Ridge Regression | 1.6918 | ~1.68 |
| Random Forest | 1.4650 | ~1.46 |
| BERT (title + metadata) | 1.3891 | ~1.37 |
| BERT (content-aware) | **1.3717** | **~1.35** |

**In plain terms:** The best model's typical prediction is within ~3.9× of the true upvote count; the metadata-only Random Forest is off by ~4.3×, and a mean-prediction baseline by ~7×.

**Key findings:**

- The title-only BERT model reduces RMSE by 5.3% compared to Random Forest, showing that language understanding adds meaningful signal beyond metadata.
- The content-aware model reduces RMSE by a further ~0.02, confirming that body text helps but modestly (the gain is concentrated in the 47% of posts that have body content).
- Both BERT models generalize cleanly to the held-out test set, with no signs of overfitting.
- Both models underestimate highly viral posts (log_score > 8), which is expected given the long tail and external virality factors the model cannot observe.

**Random Forest feature importances:**
- `hour` (0.278) and `day_of_week` (0.204) are the strongest metadata signals. Timing of submission dominates.
- `is_self` (0.195) ranks third.
- Stylistic features (`uppercase_ratio`, `char_length`) contribute modestly in the tree model but become near-zero in BERT because BERT reads these signals directly from the text.

---

## How to Run

The notebook is designed for **Google Colab with a T4 or L4 GPU**. CPU execution works but is roughly 2 to 2.5 times slower.

1. Open `reddit-popularity-prediction.ipynb` in Google Colab and set the runtime to GPU (T4 or L4).
2. Run the imports cell to install dependencies (`torch`, `transformers`, `gdown`, `sklearn`, etc.).
3. Choose a dataset size by editing one line in the dataset loading cell:

```python
train_df, val_df, test_df = load_dataset("medium")  # "small", "medium", or "full"
```

| Size | Training Posts | Approx. Time per BERT Model |
|------|---------------|-----------------------------|
| small | 78,750 | 10 to 25 minutes |
| medium | 157,500 | 30 to 50 minutes (used for all reported results) |
| full | 315,000 | 60 to 90 minutes |

4. Run all cells top to bottom. Preprocessed JSONL datasets are downloaded automatically from Google Drive via `gdown`. You do not need to re-run the processing pipeline.

Results are reproducible to 4 decimal places on the same GPU hardware. Minor variance of approximately +/-0.01 is expected between T4 and L4 due to fp16 floating-point differences.

---

## Repository Structure

```
project/
  reddit-popularity-prediction.ipynb   # Main notebook: models, training, evaluation
  processing_pipline.ipynb             # Data pipeline: raw .zst dump to train/val/test JSONL
```

The processing pipeline streams the compressed Reddit dump without loading it into RAM, applies stratified bin-capped sampling, engineers all metadata features, computes `log_score`, and writes the three JSONL splits. Medium (50%) and small (25%) subsets are created by random sampling from the full split.
