# Machine Learning Study Project: Stroke Risk Classification

Public portfolio version of a study project. Personal data, internal course material and non-essential administrative content have been removed.

This notebook analyses the **Stroke Prediction Dataset** from Kaggle. The project was completed as a group assignment in the course *Maschinelles Lernen* in the BSc Applied Digital Life Sciences at Zurich University of Applied Sciences (ZHAW).

The objective of the project is to build and evaluate binary classification models that distinguish between *stroke* and *no stroke* based on demographic, health and lifestyle variables. The analysis includes data cleaning, exploratory data analysis, feature engineering, dimensionality reduction, model training, hyperparameter tuning and threshold optimisation.

The project focuses on the full machine learning workflow rather than on medical diagnosis. The models are evaluated in the context of a strongly imbalanced classification problem, where identifying potential stroke cases is prioritised over overall accuracy.

## Project Objectives

The project demonstrates how machine learning can be applied to a real-world health-related classification task. In particular, it focuses on:

- preparing and cleaning structured tabular health data
- identifying missing values and inconsistent categories
- exploring distributions, class imbalance and feature relationships
- engineering additional risk-related features
- building reproducible preprocessing pipelines with scikit-learn
- comparing logistic regression and random forest classifiers
- handling class imbalance using stratified splitting and class weights
- evaluating models with recall, precision, F-scores, ROC-AUC and PR-AUC
- tuning decision thresholds on a validation set
- interpreting model behaviour and discussing practical limitations

## Main Topics

- Binary classification
- Imbalanced data
- Exploratory data analysis
- Feature engineering
- Missing-value handling
- One-hot encoding and scaling
- PCA and LDA for exploratory dimensionality reduction
- Logistic regression
- Random forest classification
- Cross-validation and hyperparameter tuning
- Threshold tuning for screening-oriented classification
- Model evaluation with ROC and precision-recall metrics
- Feature importance analysis

## Tools and Libraries

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- scikit-learn

## Notebook

The main analysis is contained in the following notebook:

```text
male_stroke_risk_classification.ipynb
```

## Data Note

The notebook expects the dataset to be located at:

```text
data/healthcare-dataset-stroke-data.csv
```

The dataset used in this project is the publicly available **Stroke Prediction Dataset** from Kaggle:

```text
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
```

The dataset contains 5,110 observations and includes variables such as age, gender, hypertension, heart disease, average glucose level, BMI, smoking status and stroke outcome.

The column `id` is removed during preprocessing because it has no predictive meaning for the classification task.

Before reusing or redistributing the dataset, the original Kaggle source and usage rights should be checked.

## Selected Results

The analysis shows that the target variable is strongly imbalanced: only a small share of observations corresponds to stroke cases. This makes accuracy unsuitable as the main evaluation metric.

Exploratory analysis indicates that **age** is one of the most relevant variables associated with stroke occurrence. Average glucose level and cardiovascular risk indicators such as hypertension and heart disease also contribute meaningful information, while BMI appears to be less strongly associated in this dataset.

Two classification models are compared:

1. **Logistic Regression**
2. **Random Forest**

Both models are embedded in full preprocessing pipelines to avoid data leakage. The models are evaluated using stratified cross-validation and a separate validation/test workflow. Threshold tuning is applied because the default threshold of 0.5 is not appropriate for a screening-oriented task with strong class imbalance.

The final results show that both models perform clearly better than random classification. However, the models still produce many false positives when recall is prioritised. This is an important limitation and reflects the difficulty of predicting rare health events from a limited tabular dataset.

## Repository Structure

```text
male_stroke_risk_classification.ipynb

data/
  healthcare-dataset-stroke-data.csv

README.md
requirements.txt
.gitignore
```

## How to Run

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure the dataset is stored as:

```text
data/healthcare-dataset-stroke-data.csv
```

Then open and run the notebook:

```text
male_stroke_risk_classification.ipynb
```

Run the notebook cells from top to bottom.

## Limitations

This project is exploratory and educational. It must not be interpreted as a clinical diagnostic tool.

The analysis is based on an observational public dataset and does not establish causal relationships. Stroke risk is influenced by many complex medical, genetic, behavioural and environmental factors that are not fully captured in the dataset.

Because stroke cases are rare in the dataset, the models face a strong class imbalance. Optimising for high recall helps identify more potential stroke cases, but it also increases the number of false positives. In a real medical setting, such a model would require extensive validation, clinical oversight and ethical review.

## Portfolio Note

This repository is intended as a public portfolio version of a group study project. Personal contact details, internal course material and non-essential administrative content have been removed.

The project demonstrates applied Python skills in data preprocessing, exploratory analysis, feature engineering, machine learning pipelines, model evaluation and the interpretation of imbalanced classification problems.
