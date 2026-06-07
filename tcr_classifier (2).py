# TCR Classification Project — Full Pipeline
# Run this in Google Colab, one cell at a time

# ============================================================
# CELL 1 — Install libraries
# ============================================================
# !pip install scikit-learn pandas numpy matplotlib seaborn

# ============================================================
# CELL 2 — Import everything we need
# ============================================================
import pandas as pd
import numpy as np
from collections import Counter
from itertools import product

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded!")

# ============================================================
# CELL 3 — Load your data
# ============================================================
# If running in Colab, upload TCR-Processed-Raw.csv and test_set.csv first
# (click the folder icon on the left → upload)

train = pd.read_csv('TCR-Processed-Raw.csv')
test  = pd.read_csv('test_set.csv')

print(f"Training rows: {len(train)}")
print(f"Test rows:     {len(test)}")
print(f"\nTraining columns: {train.columns.tolist()}")

# ============================================================
# CELL 4 — Map messy Pathology labels to 4 classes
# ============================================================
# Each pathology is mapped to: viral, bacterial, cancer, autoimmune
# Anything not in this map gets DROPPED (too ambiguous to use)

pathology_map = {
    # --- VIRAL ---
    "Influenza":                          "viral",
    "Cytomegalovirus (CMV)":              "viral",
    "Epstein Barr virus (EBV)":           "viral",
    "Human immunodeficiency virus (HIV)": "viral",
    "Yellow fever virus":                 "viral",
    "HTLV-1":                             "viral",
    "Hepatitis C virus":                  "viral",
    "COVID-19":                           "viral",
    "Herpes simplex virus 2 (HSV2)":      "viral",
    "Hepatitis E virus infection (cHEV)": "viral",

    # --- BACTERIAL ---
    "M. tuberculosis":                    "bacterial",

    # --- CANCER ---
    "Melanoma":                           "cancer",
    "Tumor associated antigen (TAA)":     "cancer",
    "Colorectal cancer":                  "cancer",
    "Neoantigen":                         "cancer",
    "Lung cancer":                        "cancer",
    "Breast Cancer":                      "cancer",
    "Leukemia":                           "cancer",
    "Acute myeloid leukemia":             "cancer",
    "Carcinoma":                          "cancer",
    "Epithelial ovarian cancer":          "cancer",

    # --- AUTOIMMUNE ---
    "Multiple sclerosis (MS)":            "autoimmune",
    "Diabetes Type 1":                    "autoimmune",
    "Psoriatic arthritis":                "autoimmune",
    "Toxic epidermal necrolysis":         "autoimmune",

    # DROPPED (too ambiguous):
    # "ARDS", "Allergy", "Alzheimer's disease", "Parkinson disease",
    # "Calcified Aortic Stenosis disease", "Biliary atresia"
}

train['label'] = train['Pathology'].map(pathology_map)

# Show what got dropped
dropped = train[train['label'].isna()]['Pathology'].value_counts()
print("Dropped pathologies:")
print(dropped)

# Keep only mapped rows
train = train.dropna(subset=['label']).copy()
print(f"\nRows after cleaning: {len(train)}")
print("\nClass distribution:")
print(train['label'].value_counts())

# ============================================================
# CELL 5 — Handle duplicates
# ============================================================
# Some CDR3 sequences appear more than once with different pathologies
# We keep them all — the model can handle this — but we flag it

dups = train[train['CDR3.beta.aa'].duplicated(keep=False)]
print(f"Duplicate CDR3 sequences: {len(dups)}")
print("(Keeping all — duplicates with different labels are real biological data)")

# ============================================================
# CELL 6 — Feature extraction
# ============================================================
# We need to turn each CDR3 amino acid sequence into numbers
# Strategy: k-mers (overlapping chunks of the sequence)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# --- k-mer features ---
def get_kmers(seq, k=3):
    """Split sequence into overlapping chunks of length k"""
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def kmer_counts(seq, k=3):
    """Count how often each possible k-mer appears in the sequence"""
    kmers = get_kmers(seq, k)
    c = Counter(kmers)
    # All possible k-mers from the 20 amino acids
    all_kmers = [''.join(p) for p in product(AMINO_ACIDS, repeat=k)]
    return [c.get(km, 0) for km in all_kmers]

# --- physicochemical features ---
# Properties of each amino acid (charge, hydrophobicity, size)
HYDROPHOBICITY = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
    'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
    'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2
}
CHARGE = {
    'R':1,'K':1,'H':0.1,'D':-1,'E':-1,
    'A':0,'N':0,'C':0,'Q':0,'G':0,'I':0,'L':0,'M':0,
    'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0
}

def physicochemical_features(seq):
    """Compute simple summary stats from the sequence"""
    hydro = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    charge = [CHARGE.get(aa, 0) for aa in seq]
    return [
        len(seq),                     # sequence length
        np.mean(hydro),               # average hydrophobicity
        np.std(hydro),                # hydrophobicity variation
        sum(charge),                  # total charge
        seq.count('C') / len(seq),    # cysteine fraction
        seq.count('G') / len(seq),    # glycine fraction (flexible)
        hydro[len(seq)//2],           # hydrophobicity at center
    ]

def encode_sequence(seq):
    """Combine all features for one sequence"""
    return physicochemical_features(seq) + kmer_counts(seq, k=2)

print("Extracting features from training sequences...")
print("(This may take 1-2 minutes for 10,000 sequences)")

X_train_seq = np.array([encode_sequence(str(s)) for s in train['CDR3.beta.aa']])
X_test_seq  = np.array([encode_sequence(str(s)) for s in test['CDR3.beta.aa']])

print(f"Feature matrix shape: {X_train_seq.shape}")
print(f"  → {X_train_seq.shape[1]} features per sequence")

# ============================================================
# CELL 7 — Add V and J gene features (optional, handles missing)
# ============================================================
# V and J gene names get converted to numbers (label encoding)
# "unknown" values are filled with -1 so the model knows they're missing

le_v = LabelEncoder()
le_j = LabelEncoder()

# Fit on combined train + test so no unseen labels
all_v = list(train['TRBV'].fillna('unknown')) + list(test['TRBV'].fillna('unknown'))
all_j = list(train['TRBJ'].fillna('unknown')) + list(test['TRBJ'].fillna('unknown'))
le_v.fit(all_v)
le_j.fit(all_j)

train_v = le_v.transform(train['TRBV'].fillna('unknown')).reshape(-1,1)
train_j = le_j.transform(train['TRBJ'].fillna('unknown')).reshape(-1,1)
test_v  = le_v.transform(test['TRBV'].fillna('unknown')).reshape(-1,1)
test_j  = le_j.transform(test['TRBJ'].fillna('unknown')).reshape(-1,1)

# Full feature matrix: sequence features + V gene + J gene
X_train_full = np.hstack([X_train_seq, train_v, train_j])
X_test_full  = np.hstack([X_test_seq,  test_v,  test_j])

print(f"Full feature matrix shape (with V/J): {X_train_full.shape}")

# ============================================================
# CELL 8 — Encode labels
# ============================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train['label'])

print("Class mapping:")
for i, cls in enumerate(label_encoder.classes_):
    count = (y == i).sum()
    print(f"  {i} = {cls:15s} ({count} examples)")

# ============================================================
# CELL 9 — Train model with cross-fold validation
# ============================================================
# Cross-fold validation: split data into 5 chunks
# Train on 4 chunks, test on 1 — repeat 5 times
# This gives us an honest estimate of how well the model will do

print("Training with 5-fold cross-validation...")
print("(Each fold trains a fresh model — takes ~2-3 minutes total)\n")

model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_full, y, cv=cv, scoring='f1_macro', n_jobs=-1)

print(f"Cross-validation Macro F1 scores: {[round(s,3) for s in cv_scores]}")
print(f"Mean: {cv_scores.mean():.3f}  ±  {cv_scores.std():.3f}")
print()
print("What this means:")
print(f"  Your model scores about {cv_scores.mean():.1%} Macro F1 on unseen data")
print(f"  (1.0 = perfect, 0.25 = random guessing across 4 classes)")

# ============================================================
# CELL 10 — Train final model on ALL training data
# ============================================================
print("Training final model on all training data...")
model.fit(X_train_full, y)
print("Done!")

# Training accuracy (will be high — that's normal)
train_preds = model.predict(X_train_full)
train_f1 = f1_score(y, train_preds, average='macro')
print(f"Training Macro F1: {train_f1:.3f}")

# ============================================================
# CELL 11 — Evaluate: confusion matrix
# ============================================================
# Show a heatmap of what the model gets right and wrong
# (on training data — for a real test, use the CV scores above)

cm = confusion_matrix(y, train_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (training data)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Saved as confusion_matrix.png")

# Per-class report
print("\nPer-class performance:")
print(classification_report(y, train_preds, target_names=label_encoder.classes_))

# ============================================================
# CELL 12 — Predict on test set & save for Kaggle
# ============================================================
test_pred_nums = model.predict(X_test_full)
test_pred_labels = label_encoder.inverse_transform(test_pred_nums)

submission = pd.DataFrame({
    'ID':    test['ID'],
    'label': test_pred_labels
})

submission.to_csv('submission.csv', index=False)
print("Saved submission.csv!")
print(f"\nPrediction breakdown:")
print(pd.Series(test_pred_labels).value_counts())
print("\nFirst 5 rows of your submission:")
print(submission.head())
print("\nNEXT STEP: Download submission.csv and upload it to Kaggle!")

# ============================================================
# CELL 13 — Feature importance plot (bonus)
# ============================================================
importances = model.feature_importances_
# Just show top 15 features
top_idx = np.argsort(importances)[-15:]

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(range(15), importances[top_idx], color='steelblue')
ax.set_yticks(range(15))
ax.set_yticklabels([f'feature_{i}' for i in top_idx])
ax.set_xlabel('Importance')
ax.set_title('Top 15 most important features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("Saved feature_importance.png")
print("\nDone! You now have:")
print("  submission.csv     → upload to Kaggle")
print("  confusion_matrix.png → use in your presentation")
print("  feature_importance.png → use in your presentation")
