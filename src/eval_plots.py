import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---- Custom Metrics (no sklearn) ----

def compute_metrics(y_true, y_pred, target_names=None):
    """
    Computes accuracy, precision, recall, F1, confusion matrix,
    and a classification report string — all from scratch using numpy.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    if target_names is None:
        target_names = [str(c) for c in classes]

    # --- Confusion matrix (manual) ---
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t]][class_to_idx[p]] += 1

    # --- Binary metrics for the Spam class (pos_label=1) ---
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # --- Per-class metrics for the report string ---
    report_lines = [
        f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
        ""
    ]
    for i, (cls, name) in enumerate(zip(classes, target_names)):
        row = cm[i]
        col = cm[:, i]
        tp_i  = cm[i, i]
        fp_i  = col.sum() - tp_i
        fn_i  = row.sum() - tp_i
        prec_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
        rec_i  = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
        f1_i   = (2 * prec_i * rec_i) / (prec_i + rec_i) if (prec_i + rec_i) > 0 else 0
        sup_i  = row.sum()
        report_lines.append(
            f"{name:>12} {prec_i:>10.2f} {rec_i:>10.2f} {f1_i:>10.2f} {sup_i:>10}"
        )

    report_lines.append("")
    report_lines.append(
        f"{'accuracy':>12} {'':>10} {'':>10} {accuracy:>10.2f} {len(y_true):>10}"
    )
    report = "\n".join(report_lines)

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


# ---- Confusion Matrix Plot (manual cm, no sklearn) ----

def plot_confusion_matrix(y_true, y_pred, target_names=None, title="Confusion Matrix"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if target_names is None:
        target_names = ["Not Spam", "Spam"]

    classes = [0, 1]
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t]][class_to_idx[p]] += 1

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