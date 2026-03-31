# Parliamentary Motion Classifier

Text classification of UK House of Commons motions using NLP and machine learning (scikit-learn).

## Overview

This project implements two ML pipelines on a dataset of 592 parliamentary motions from Hansard records:

### 1. Topic Classification (Supervised)
Classifies motions into 6 policy categories using **LinearSVC** with combined TF-IDF features:
- Terrorism and security
- Welfare and benefits
- Local government
- Scotland devolution
- Climate and environment
- Education (schools autonomy)

**Results:** 88.33% test accuracy | 5.76% overfitting gap | Macro F1: 0.88

### 2. Priority Classification (Semi-Supervised Prototype)
Predicts whether a motion should be prioritised for qualitative analysis using **Logistic Regression** with pseudo-labelling:
- 120 motions manually labelled (important / not_important)
- Semi-supervised pseudo-labelling extends training with high-confidence predictions on 472 unlabelled motions

**Results:** 73.33% test accuracy | -2.2% overfitting gap | Beats 66.7% majority baseline

## Features

- **Text encoding:** Word-level TF-IDF (unigrams + bigrams) + character-level TF-IDF
- **Structured features:** Entity presence flags (date, person, event/organisation)
- **Hyperparameter tuning:** GridSearchCV with stratified k-fold cross-validation
- **Semi-supervised learning:** Confidence-based pseudo-labelling on unlabelled data
- **Evaluation:** Confusion matrices, classification reports, per-class misclassification analysis

## Project Structure

```
code/
  parliamentary_motion_classifier.ipynb   # Main notebook (full pipeline)
  hansard_motions_dataset.csv             # Source dataset (592 motions)
  manual_labels.csv                       # Manual annotations for priority task
  motion_predictions.xlsx                 # Final output with predicted labels
```

## Getting Started

### Requirements

```
python >= 3.8
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
openpyxl
```

### Run

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy openpyxl
jupyter notebook code/parliamentary_motion_classifier.ipynb
```

Run all cells sequentially. The priority classification section depends on data loaded in the topic classification section.

## Key Findings

**Topic Classification:**
- LinearSVC outperformed Logistic Regression, Random Forest, and Naive Bayes
- C=0.05 selected over CV-best C=1.0 to satisfy all per-class performance requirements
- 4 of 6 classes achieve less than 13% misclassification rate
- 0% error on Scotland devolution predictions

**Priority Classification:**
- Semi-supervised pseudo-labelling matched supervised accuracy (73.3%) but with significantly lower overfitting (-2.2% vs 10.0%)
- Macro F1 (0.68) recommended over accuracy due to class imbalance
- TF-IDF outperformed pre-trained Sentence-BERT embeddings on this small dataset

## Tech Stack

Python | scikit-learn | pandas | matplotlib | seaborn | scipy
