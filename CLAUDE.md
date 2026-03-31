# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University coursework (COMP1804) for a machine learning module. Text classification on UK parliamentary Hansard motions data using scikit-learn.

**Two tasks:**
- **Task 1 (Supervised):** Classify motions into 6 parliamentary topic categories using TF-IDF + LinearSVC with GridSearchCV tuning. Client constraints: >=86% accuracy, overfitting gap <10%, >=4 classes with <=13% misclassification, <9% error on Scotland predictions.
- **Task 2 (Semi-Supervised):** Predict a custom `motion_priority` label (`important`/`not_important`, manually annotated in `labels_comp1804.csv`). Uses LogisticRegression with pseudo-labelling on unlabelled data. Compares majority baseline, supervised-only, and semi-supervised approaches.

## Repository Structure

- `code/code_comp1804.ipynb` — **Primary notebook** with the full pipeline (Task 1 + Task 2). This is the main deliverable.
- `code/labels_comp1804.csv` — Manual annotations for Task 2 (motion_id → motion_priority).
- `code/comp1804_coursework_dataset_25-26_hansard_motions_full_revision.csv` — Full source dataset.
- `code/motion_importance_comp1804.xlsx` — Final output with predicted labels.
- `code/coursework.ipynb`, `code/test.ipynb` — Scratch/experimental notebooks.
- `dataset/` — Original dataset snapshot and earlier figures.
- `notebook/` — Report documents (Word, Excel).

## Running

Requires Python with: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `openpyxl`.

```bash
jupyter notebook code/code_comp1804.ipynb
```

Run cells sequentially — Task 2 depends on data loaded in Task 1. The notebook exports `motion_importance_comp1804.xlsx` as the final deliverable.

## Notebook Structure

Cells are numbered with a `T{task}.{step}` scheme (e.g., T1.6, T2.8). Key sections:

**Task 1 (cells 0–30):** Load CSV → filter 6 classes → clean entity columns → train/test split → TF-IDF encoding → feature combination → GridSearchCV → evaluate against client conditions → generate figures.

**Task 2 (cells 31–end):** Load manual labels from CSV → merge with dataset → TF-IDF encoding → majority baseline → hyperparameter tuning → supervised LogisticRegression → pseudo-labelling on unlabelled rows → compare models → export xlsx.

## Key Technical Details

- **Text features:** Word-level TF-IDF (unigrams+bigrams) + character-level TF-IDF + structural entity columns, combined via `scipy.sparse.hstack`.
- **Task 1 tuning:** `GridSearchCV` over LinearSVC `C` values with 5-fold stratified CV. Selected C is hardcoded after tuning.
- **Task 2 pseudo-labelling:** High-confidence predictions (threshold tuned) from LogisticRegression on unlabelled rows are added to training set, then model is retrained.
- **Figures:** 9 PNG files in `code/` (prefixed `task1_` or `task2_`) are generated inline by the notebook — do not edit them manually.
