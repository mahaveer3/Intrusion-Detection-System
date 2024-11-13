import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib


def preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("The specified file was not found.")
        return None

    # Check if 'Attack' column exists in the original dataset
    if "Attack" not in df.columns:
        print("'Attack' column is missing from the original dataset.")
        return None

    # Handle missing values
    df = df.dropna()

    # Encode categorical features
    for column in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    categorical_features = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "PROTOCOL", "L7_PROTO"]
    label_encoders = {
        feature: LabelEncoder().fit(df[feature]) for feature in categorical_features
    }

    # Save the encoders
    joblib.dump(label_encoders, "label_encoders.pkl")

    # Cap extreme values
    def cap_extreme_values(df, max_value=1e10):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].apply(lambda x: min(x, max_value) if x != np.inf else x)
        return df

    df = cap_extreme_values(df)

    # Check if 'Attack' column is still present
    if "Attack" not in df.columns:
        print("'Attack' column is missing after preprocessing.")
        return None

    # Separate features and target
    X = df.drop(
        columns=["Label", "Attack"], errors="ignore"
    )  # Ensure 'Attack' is not dropped
    y = df.get("Attack")  # Ensure 'Attack' is accessed

    if y is None:
        print("'Attack' column could not be found after dropping columns.")
        return None

    # Feature selection using RandomForestClassifier with a fixed random seed
    model = RandomForestClassifier(n_estimators=25, random_state=42)
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_
    important_features = X.columns[
        np.argsort(importances)[-16:]
    ]  # Top 16 important features

    # Keep only the important features and the target column
    df = df[list(important_features) + ["Attack"]]  # Retain target column

    return df


if __name__ == "__main__":
    df = preprocess_data("NF-UQ-NIDS-v2.csv")
    if df is not None:
        df.to_csv("processed_data.csv", index=False)