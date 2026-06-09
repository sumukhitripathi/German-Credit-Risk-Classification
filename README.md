# German Credit Risk Classification

A small machine learning project that classifies German credit risk using a logistic regression model. The pipeline loads credit data from `credit_data.csv`, encodes categorical features, trains a classifier, and prints evaluation metrics.

## Project Structure

```text
.
|-- credit_data.csv   # German credit risk dataset
|-- main.py           # Data loading, preprocessing, training, and evaluation pipeline
`-- README.md
```

## Dataset

The dataset contains 1,000 credit records with customer and loan-related attributes such as:

- Checking account status
- Loan duration
- Credit history
- Credit amount
- Savings account status
- Employment duration
- Age
- Job category
- Credit risk status

The target column is `status`, where the model learns to classify credit risk from the other columns.

## Model Workflow

The script performs the following steps:

1. Loads the CSV dataset with pandas.
2. Encodes categorical columns using `LabelEncoder`.
3. Splits the data into training and test sets.
4. Scales features using `StandardScaler`.
5. Trains a `LogisticRegression` classifier.
6. Evaluates the model with a confusion matrix and accuracy score.

## Requirements

Install the required Python packages:

```bash
pip install pandas scikit-learn
```

## How to Run

From the repository root, run:

```bash
python main.py
```