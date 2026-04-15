import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


def compute_metrics(y_true, y_pred, target_names=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    report = classification_report(y_true, y_pred, target_names=target_names)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy":        accuracy,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "report":          report,
        "confusion_matrix": cm,
    }


def print_metrics(metrics):
    print("=" * 40)
    print("Evaluation Results")
    print("=" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("-" * 40)
    print(metrics["report"])


# ---- Confusion Matrix Plot----

def plot_confusion_matrix(y_true, y_pred, target_names=None, title="Confusion Matrix"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if target_names is None:
        target_names = ["Not Spam", "Spam"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ---- Explainability - Top words per class from Naive Bayes ----

def get_top_features(model, vectorizer, class_labels=None, n=20):
    """
    Returns the top-n words with highest log-probability for each class.
    Uses model.feature_log_prob_ which holds log P(word | class).
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    log_probs = model.feature_log_prob_

    if class_labels is None:
        class_labels = ["Not Spam", "Spam"]

    result = {}
    for i, label in enumerate(class_labels):
        top_idx = np.argsort(log_probs[i])[-n:][::-1]
        df = pd.DataFrame({
            "word":     feature_names[top_idx],
            "log_prob": log_probs[i][top_idx],
        })
        result[label] = df

    return result


def plot_top_features(model, vectorizer, class_labels=None, n=15):
    top_feats = get_top_features(model, vectorizer, class_labels, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (label, df) in zip(axes, top_feats.items()):
        ax.barh(df["word"], df["log_prob"], color=sns.color_palette("viridis", len(df)))
        ax.set_xlabel("Log Probability")
        ax.set_title(f"Top {n} Words – {label}")
        ax.invert_yaxis()

    plt.tight_layout()
    return fig


# ---- Helper plots ----

def plot_class_distribution(y, target_names=None, title="Class Distribution"):
    unique, counts = np.unique(y, return_counts=True)
    labels = target_names if target_names else [str(u) for u in unique]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, counts, color=["#4CAF50", "#F44336"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_f1_comparison(results_df, metric_col="mean_f1", param_col="params",
                       title="F1 Score Comparison"):
    df = results_df.sort_values(metric_col, ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df[param_col].astype(str), df[metric_col],
            color=sns.color_palette("coolwarm", len(df)))
    ax.set_xlabel(metric_col)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ---- Full evaluation ----

def evaluate_full(model, vectorizer, x_test, y_test):
    y_pred = model.predict(x_test)

    metrics = compute_metrics(y_test, y_pred, target_names=["Not Spam", "Spam"])
    print_metrics(metrics)

    fig_cm   = plot_confusion_matrix(y_test, y_pred)
    fig_feat = plot_top_features(model, vectorizer)

    return metrics, fig_cm, fig_feat