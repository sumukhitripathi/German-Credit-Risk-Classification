import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


class CreditClassifier:

    def __init__(self, filepath):
        """
        Initialize classifier.

        Tasks:
        - Load CSV dataset
        """
        print("Loading dataset... please wait")

        # TODO: Load dataset
        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded with {len(self.df)} records")


    def encode_features(self):
        """
        Encode categorical columns.

        Return encoded DataFrame.
        """
        print("Encoding categorical features...")

        # TODO: Encode categorical columns
        df = self.df.copy()
        encoder = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = encoder.fit_transform(df[col])
        return df


    def train_model(self, df):
        """
        Train classification model.

        Steps:
        - Separate features and target
        - Split dataset
        - Scale numeric features
        - Train Logistic Regression

        Return:
        model, X_test, y_test
        """
        print("Training classifier...")

        # TODO: Implement training logic
        target = "status"
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.fit_transform(X_test)

        model=LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        return model, X_test, y_test


    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance.

        Return:
        confusion matrix and accuracy percentage
        """
        print("Evaluating model...")

        # TODO: Compute predictions
        predictions = model.predict(X_test)

        cm = confusion_matrix(y_test, predictions)

        acc = accuracy_score(y_test, predictions)*100

        return cm, acc


def run_pipeline(filepath):

    classifier = CreditClassifier(filepath)

    df = classifier.encode_features()

    model, X_test, y_test = classifier.train_model(df)

    cm, acc = classifier.evaluate_model(model, X_test, y_test)

    return {
        "confusion_matrix": cm,
        "accuracy": acc
    }


if __name__ == "__main__":

    filepath = "credit_data.csv"

    results = run_pipeline(filepath)

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print(f"\nAccuracy: {results['accuracy']:.2f}%")