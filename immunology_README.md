# TCR Source Classifier

Single-label classifier predicting the source category of a T cell receptor (TCR):
**viral · bacterial · cancer · autoimmune**

Built for the Bhargava Systems Intern Competition (Immunology).

## What this does

Given a TCR's CDR3β amino acid sequence (and optionally its Vβ and Jβ gene annotations),
this model predicts which of 4 disease source categories the TCR is associated with.

## Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Data

Place the following in the `data/` folder:
- `TCR-Processed-Raw.csv` — raw training data (from the repo)
- `test_set.csv` — test sequences (from Kaggle data tab)

## How to train

```bash
python train.py
```

This will:
1. Load and clean the raw training CSV (map 25 pathology labels → 4 classes, drop ambiguous ones)
2. Extract sequence features (physicochemical + dipeptide k-mers)
3. One-hot encode Vβ/Jβ gene annotations
4. Train an ExtraTrees classifier with balanced class weights
5. Save the trained model to `model/tcr_classifier.pkl`

## How to run predictions

**Sequence-only mode:**
```bash
python predict.py --mode sequence --input test_set.csv --output submission.csv
```

**Sequence + gene mode** (uses Vβ/Jβ columns if present):
```bash
python predict.py --mode genes --input test_set.csv --output submission.csv
```

## Repository structure

```
├── data/
│   ├── TCR-Processed-Raw.csv     # raw training data
│   └── test_set.csv              # Kaggle test set
├── model/
│   └── tcr_classifier.pkl        # saved trained model
├── tcr_classifier.py             # full pipeline (train + predict)
├── figures/
│   ├── imm_figure_data_overview.png
│   ├── imm_figure_confusion.png
│   └── imm_figure_results.png
└── README.md
```

## Approach

### Data cleaning
The raw dataset had 10,564 rows with 31 unique `Pathology` labels.
We mapped 25 of these to the 4 target classes based on biology:

| Class | Examples |
|-------|----------|
| viral | Influenza, CMV, EBV, HIV, COVID-19, Yellow fever, HTLV-1, HCV, HSV2, HEV |
| bacterial | M. tuberculosis |
| cancer | Melanoma, TAA, Colorectal cancer, Neoantigen, Leukemia, Lung/Breast/Ovarian cancer |
| autoimmune | Multiple sclerosis, Diabetes Type 1, Psoriatic arthritis, Toxic epidermal necrolysis |

Dropped 752 rows with ambiguous pathologies (ARDS, Alzheimer's, Parkinson's, Allergy, etc.).
Final training set: **9,812 TCRs**.

### Feature extraction
Each CDR3β sequence → **863 numerical features**:

- **Physicochemical (19):** length, hydrophobicity stats, charge, polarity, zone averages
- **Amino acid composition (20):** fraction of each of the 20 amino acids
- **Dipeptide k-mers (400):** count of every overlapping 2-letter pair
- **V/J gene one-hot encoding (424):** TRBV, TRBJ, V-family, J-family — one column per unique gene

### Model
**ExtraTreesClassifier** — 400 trees, `class_weight='balanced'`, no max depth limit.

ExtraTrees differs from Random Forest in that it splits nodes using random thresholds
rather than the optimal threshold, making it faster and often more robust to overfitting.
`class_weight='balanced'` ensures autoimmune (176 examples) is not drowned out by viral (7,831).

### Why ExtraTrees over Random Forest?
Our cross-validation showed ExtraTrees scored **0.763 Macro F1** vs **0.718** for Random Forest
on the same feature set, while training ~3× faster. The randomized splits act as additional
regularization on a dataset where viral class heavily dominates.

## Results (3-fold cross-validation)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| viral | 0.90 | 0.99 | 0.94 | 7,831 |
| bacterial | 0.88 | 0.64 | 0.74 | 1,084 |
| autoimmune | 0.99 | 0.68 | 0.81 | 176 |
| cancer | 0.95 | 0.42 | 0.58 | 721 |
| **Macro avg** | **0.93** | **0.68** | **0.77** | **9,812** |

## Limitations & assumptions

- Only one bacterial pathology (M. tuberculosis) is in the dataset — bacterial class may not generalize
- Cancer recall is low (0.42) — the model struggles to detect cancer TCRs, likely due to high within-class diversity
- Autoimmune has very few examples (176) — high precision but low recall suggests the model is conservative
- V/J gene one-hot encoding will produce zeros for any gene not seen in training (handled gracefully via `handle_unknown='ignore'`)
- Duplicate CDR3 sequences with conflicting labels (256 pairs) were kept — this adds noise

## What I would improve next

1. Add 3-mer features for richer sequence context
2. Try gradient boosting (XGBoost/LightGBM) which often outperforms tree ensembles on tabular data
3. Collect more autoimmune and cancer TCRs — data quantity is the main bottleneck
4. Use a pretrained TCR language model (e.g. TCR-BERT) for richer sequence embeddings
5. Ensemble sequence-only and sequence+gene models for robustness
