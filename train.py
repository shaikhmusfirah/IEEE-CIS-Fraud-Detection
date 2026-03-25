import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

MODEL_FILE = Path("data/processed/fraud_model.joblib")


def train_model(X, y):

    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, preds)

    print(f"Validation ROC-AUC: {auc:.4f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")

    plot_feature_importance(model, X.columns)

    return model


def plot_feature_importance(model, feature_names):

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    ).head(20)

    plt.figure(figsize=(10,7))

    sns.barplot(
        x="importance",
        y="feature",
        data=importance,
        palette="viridis"
    )

    plt.title("Top Features Driving Fraud Detection")

    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    plt.savefig("outputs/plots/model_feature_importance.png")

    plt.close()