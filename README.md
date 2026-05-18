# TCR Source Classification Project

Build a multi-class TCR source classifier.

Each TCR belongs to one of four source categories based on what its cognate antigen is associated with. The data in this repository is the public training set only; final evaluation uses a held-out TCR set that is not included here.

## Data

The training data is provided as a single CSV file at `data/TCR-Processed-Raw.csv`. The file is mostly raw and uncleaned. It contains inconsistent formatting, a long-tail distribution of source pathologies, and many labels that may or may not map to one of the four target classes. Cleaning the data, deciding which pathologies belong to which class, handling ambiguous or duplicate labels, and choosing what to drop are part of the task.

The relevant columns are:

- `CDR3β` — the CDR3β amino acid sequence
- `Vβ` — the Vβ gene (may be missing)
- `Jβ` — the Jβ gene (may be missing)
- `Pathology` — the source pathology label (raw, uncleaned)

The cognate antigen sequence itself is **not** provided.

Target classes:

- `viral`
- `bacterial`
- `cancer`
- `autoimmune`

## Challenge

Train a model that predicts the source category for each TCR.

The submitted model must be able to make predictions in both input modes:

- **sequence only**, when only the CDR3β sequence is provided
- **sequence plus genes**, when the CDR3β sequence is provided alongside Vβ and/or Jβ gene annotations

The model should handle missing V and/or J gene information gracefully, since a substantial fraction of the training data is missing one or both.

There is no required architecture. Handcrafted physicochemical features, classical machine learning, position-specific scoring, k-mer encodings, pretrained protein or TCR language model embeddings, sequence models, and hybrid approaches are all valid if the result is reproducible and scientifically justified.

You may not use any model or dataset that was trained directly on labels overlapping the held-out test set. Any additional TCR-epitope database used by your final model must be disclosed and justified.

## Evaluation

Evaluation is four-class classification on a held-out test set. The held-out set is restricted to TCRs whose source pathology clearly maps to one of the four target classes, so you do not need to handle an "other" category at inference time. Each submission will be evaluated separately in both input modes: sequence only and sequence plus genes.

Submissions should produce one score or probability per class for each TCR in either mode.

Primary metric:

- macro F1 across classes

Secondary metrics:

- micro F1
- per-class F1
- accuracy

Predicted probabilities are preferred so decision thresholds can be applied consistently across submissions.
