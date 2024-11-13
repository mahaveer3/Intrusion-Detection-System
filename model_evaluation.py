import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split


def evaluate_model(file_path):
    df = pd.read_csv(file_path)

    X = df.drop("Attack", axis=1)
    y = df["Attack"]

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")

    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )  # Ensure same random_state

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    y_bin = label_binarize(y_test, classes=model.classes_)
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(model.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 7))
    for i in range(len(model.classes_)):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (area = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

    classification_errors = [1 - accuracy_score(y_test, model.predict(X_test))]
    plt.figure(figsize=(10, 7))
    plt.plot(
        range(len(classification_errors)),
        classification_errors,
        label="Classification Error",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Classification Error")
    plt.title("Classification Error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_model("processed_data.csv")