import json
import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .config import CATEGORICAL_COLUMNS, MODEL_DIR, PLOT_DIR, RANDOM_STATE, RESULT_DIR, TARGET, TEST_SIZE
from .data import load_modeling_data, split_features_target
from .models import build_models
from .visualization import (
    save_confusion_matrices, save_feature_importance, save_metric_comparison,
    save_roc_curve, save_target_distribution,
)


def _feature_importance(name, model, numerical, categorical):
    estimator = model.named_steps["model"]
    if name == "CatBoost":
        feature_names = numerical + categorical
    else:
        feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": estimator.feature_importances_,
    }).sort_values("Importance", ascending=False).head(20).sort_values("Importance")


def run_training():
    for directory in (MODEL_DIR, RESULT_DIR, PLOT_DIR):
        directory.mkdir(exist_ok=True)

    data = load_modeling_data()
    X, y = split_features_target(data)
    categorical = [column for column in X if column in CATEGORICAL_COLUMNS]
    numerical = [column for column in X if column not in CATEGORICAL_COLUMNS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    records = []
    probabilities = {}
    importances = {}
    for name, model in build_models(numerical, categorical).items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        train_probabilities = model.predict_proba(X_train)[:, 1]
        test_probabilities = model.predict_proba(X_test)[:, 1]
        predicted = (test_probabilities >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
        records.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, train_probabilities >= 0.50),
            "Test Accuracy": accuracy_score(y_test, predicted),
            "Accuracy Gap": accuracy_score(y_train, train_probabilities >= 0.50) - accuracy_score(y_test, predicted),
            "Train ROC-AUC": roc_auc_score(y_train, train_probabilities),
            "Test ROC-AUC": roc_auc_score(y_test, test_probabilities),
            "ROC-AUC Gap": roc_auc_score(y_train, train_probabilities) - roc_auc_score(y_test, test_probabilities),
            "Balanced Accuracy": balanced_accuracy_score(y_test, predicted),
            "Precision": precision_score(y_test, predicted, zero_division=0),
            "Recall": recall_score(y_test, predicted, zero_division=0),
            "F1-Score": f1_score(y_test, predicted, zero_division=0),
            "PR-AUC": average_precision_score(y_test, test_probabilities),
            "Training Time (seconds)": elapsed,
            "True Negative": int(tn), "False Positive": int(fp),
            "False Negative": int(fn), "True Positive": int(tp),
        })
        probabilities[name] = test_probabilities
        if name != "Logistic Regression":
            importances[name] = _feature_importance(name, model, numerical, categorical)
        joblib.dump(model, MODEL_DIR / f"structured_{name.lower().replace(' ', '_')}.pkl")
        print(f"{name}: {elapsed:.2f}s")

    results = pd.DataFrame(records).sort_values("Test ROC-AUC", ascending=False)
    results.to_csv(RESULT_DIR / "structured_model_results.csv", index=False)
    metadata = {
        "dataset": "data/raw/ecommerce_customer_churn_dataset.csv",
        "rows_after_filters": len(data), "features": X.shape[1], "target": TARGET,
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "numeric_features": numerical, "categorical_features": categorical,
        "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
        "imputation": "fit inside each model pipeline on training data only",
    }
    (RESULT_DIR / "structured_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name, importance in importances.items():
        importance.to_csv(RESULT_DIR / f"structured_{name.lower().replace(' ', '_')}_importance.csv", index=False)

    save_roc_curve(y_test, probabilities, results, PLOT_DIR / "structured_roc_curves.png")
    save_confusion_matrices(results, PLOT_DIR / "structured_confusion_matrices.png")
    save_feature_importance(importances, PLOT_DIR / "structured_feature_importance.png")
    save_metric_comparison(results, PLOT_DIR / "structured_metric_comparison.png")
    save_target_distribution(y, PLOT_DIR / "structured_target_distribution.png")
    print(results.to_string(index=False))
