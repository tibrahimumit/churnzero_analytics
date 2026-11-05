from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_roc_curve(y_test, probabilities, results, output_path: Path):
    figure, axis = plt.subplots(figsize=(9, 7))
    for name, values in probabilities.items():
        from sklearn.metrics import roc_curve
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, values)
        auc_value = results.loc[results["Model"] == name, "Test ROC-AUC"].iloc[0]
        axis.plot(false_positive_rate, true_positive_rate, label=f"{name} ({auc_value:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    axis.set(title="E-commerce Churn ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_confusion_matrices(results, output_path: Path):
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    for axis, row in zip(axes.flat, results.to_dict("records")):
        matrix = [[row["True Negative"], row["False Positive"]], [row["False Negative"], row["True Positive"]]]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set(title=row["Model"], xlabel="Predicted", ylabel="Actual")
    axes.flat[-1].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_feature_importance(importances, output_path: Path):
    figure, axes = plt.subplots(2, 2, figsize=(16, 13))
    for axis, (name, importance) in zip(axes.flat, importances.items()):
        axis.barh(importance["Feature"], importance["Importance"], color="#2F6690")
        axis.set_title(f"{name}: Top 20 Features")
        axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_metric_comparison(results: pd.DataFrame, output_path: Path):
    metric_columns = [
        "Test Accuracy", "Balanced Accuracy", "Precision", "Recall",
        "F1-Score", "Test ROC-AUC", "PR-AUC",
    ]
    chart_data = results.set_index("Model")[metric_columns]
    figure, axis = plt.subplots(figsize=(15, 8))
    chart_data.plot(kind="bar", ax=axis, colormap="viridis")
    axis.set_title("E-commerce Churn Model Comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_target_distribution(target: pd.Series, output_path: Path):
    counts = target.value_counts().sort_index()
    labels = [f"Churned = {value}" for value in counts.index]
    figure, axis = plt.subplots(figsize=(7, 6))
    bars = axis.bar(labels, counts.values, color=["#4C78A8", "#E45756"])
    axis.set_title("Target Distribution")
    axis.set_ylabel("Customer Count")
    for bar, count in zip(bars, counts.values):
        axis.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
