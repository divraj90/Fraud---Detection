# Transaction Fraud Detection

This project explores a machine learning workflow for detecting fraudulent financial transactions. It includes exploratory data analysis, feature preparation, model training, evaluation, saved preprocessing objects, a saved model, and a small API structure.

## Project Objective

Fraud datasets are usually highly imbalanced, which means fraudulent transactions are rare compared with legitimate transactions. The goal of this project is to train and evaluate models that can identify suspicious transactions while tracking metrics such as precision, recall, F1 score, and balanced accuracy.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost / LightGBM style ML workflow
- Jupyter Notebook
- Flask-style API structure
- Joblib for saved model artifacts

## Repository Structure

```text
.
├── api/
├── data/raw/
├── functions/
├── models/
├── notebooks/
├── reports/figures/
├── requirements.txt
└── README.md
```

## Workflow

1. Load and inspect transaction data
2. Clean and prepare the dataset
3. Analyze transaction patterns and fraud distribution
4. Encode categorical features and scale numerical features
5. Train multiple classification models
6. Compare model metrics
7. Save model and preprocessing artifacts
8. Prepare an API structure for inference

## Key Results

The notebook compares multiple models and selects a stronger fraud classifier based on balanced performance. For an imbalanced dataset, recall and precision are more important than raw accuracy because a model can appear accurate while missing most fraud cases.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/transaction-fraud-detection-cycle1.ipynb
```

## What I Learned

- Why fraud detection is an imbalanced classification problem
- Why accuracy alone is not enough for fraud models
- How preprocessing artifacts are saved and reused
- How model metrics like precision, recall, F1, and balanced accuracy differ
- How a trained model can be prepared for API-based inference

## Limitations

- The dataset and trained artifacts should be reviewed before production use.
- The API structure needs testing and deployment work.
- The model should be validated on fresh unseen data before real use.

## Future Improvements

- Add a simple API usage example
- Add a `predict.py` script for local inference
- Add confusion matrix and ROC/PR curve screenshots
- Add tests for preprocessing functions
- Add a smaller sample dataset for quick demo runs

