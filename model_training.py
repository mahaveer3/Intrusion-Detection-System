import joblib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_model(file_path):
    df = pd.read_csv(file_path)

    X = df.drop("Attack", axis=1)
    y = df["Attack"]

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "feature_names.pkl")  # Save the feature names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=45
    )  # Set random_state for reproducibility

    model = XGBClassifier(
        tree_method="hist",
        device="gpu",
        max_depth=3,
        n_estimators=1000,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.5,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,  # Set random_state for reproducibility
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")

    joblib.dump(model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    evals_result = model.evals_result()
    epochs = len(evals_result["validation_0"]["mlogloss"])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, evals_result["validation_0"]["mlogloss"], label="Train")
    ax.plot(x_axis, evals_result["validation_1"]["mlogloss"], label="Validation")
    ax.legend()
    plt.ylabel("Log Loss")
    plt.title("XGBoost Log Loss")
    plt.show()


if __name__ == "__main__":
    train_model("processed_data.csv")