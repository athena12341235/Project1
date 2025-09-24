# ---------------------------------------------------------
# This script:
# 1. Loads a dataset of text reviews and labels (1 = positive, 0 = negative).
# 2. Splits the data into training (80%) and testing (20%).
# 3. Converts text into numeric features using TF–IDF with unigrams + bigrams.
# 4. Trains three classifiers: Logistic Regression, Naive Bayes, and Linear SVM.
# 5. Evaluates each model: accuracy, precision, recall, F1, confusion matrix.
# 6. Extracts and logs most important features (words/phrases).
# 7. Saves results in a .txt log file.
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)
import platform  # system info

# -----------------------
# Load and prepare data
# -----------------------
reviews_df = pd.read_csv("review_polarity_clean.csv")
reviews_df = reviews_df[["clean_text", "label"]].dropna()
reviews_df["clean_text"] = reviews_df["clean_text"].astype(str)
reviews_df["label"] = reviews_df["label"].astype(int)

X = reviews_df["clean_text"]
y = reviews_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------
# TF–IDF
# -----------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
feature_names = np.array(vectorizer.get_feature_names_out())

# -----------------------
# Models
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=42),
    "Naive Bayes (MultinomialNB)": MultinomialNB(),
    "Linear SVM (LinearSVC)": LinearSVC(random_state=42)
}

# -----------------------
# Helpers
# -----------------------
def top_features_linear(coef, feature_names, top_n=15):
    if coef.ndim == 2 and coef.shape[0] == 1:
        weights = coef[0]
    elif coef.ndim == 2 and coef.shape[0] == 2:
        weights = coef[1]
    else:
        weights = coef
    top_pos_idx = np.argsort(weights)[-top_n:][::-1]
    top_neg_idx = np.argsort(weights)[:top_n]
    return (
        list(zip(feature_names[top_pos_idx], weights[top_pos_idx])),
        list(zip(feature_names[top_neg_idx], weights[top_neg_idx]))
    )

def top_features_nb(clf, feature_names, top_n=15):
    classes = list(clf.classes_)
    pos_idx = classes.index(1)
    neg_idx = classes.index(0)
    diff = clf.feature_log_prob_[pos_idx] - clf.feature_log_prob_[neg_idx]
    top_pos_idx = np.argsort(diff)[-top_n:][::-1]
    top_neg_idx = np.argsort(diff)[:top_n]
    return (
        list(zip(feature_names[top_pos_idx], diff[top_pos_idx])),
        list(zip(feature_names[top_neg_idx], diff[top_neg_idx]))
    )

def format_top_feats(pairs, width=30):
    return "\n".join(f"{term:<{width}} {weight: .4f}" for term, weight in pairs)

# -----------------------
# Train/evaluate
# -----------------------
summary_rows = []
model_results = []

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["True 0 (Neg)", "True 1 (Pos)"], columns=["Pred 0 (Neg)", "Pred 1 (Pos)"])

    try:
        if hasattr(model, "coef_"):
            pos_feats, neg_feats = top_features_linear(model.coef_, feature_names, top_n=5)
        elif isinstance(model, MultinomialNB):
            pos_feats, neg_feats = top_features_nb(model, feature_names, top_n=5)
        else:
            pos_feats, neg_feats = [], []
    except Exception:
        pos_feats, neg_feats = [], []

    model_results.append({
        "name": name,
        "accuracy": acc,
        "report": report,
        "confusion": cm_df,
        "pos_feats": pos_feats,
        "neg_feats": neg_feats,
        "precision": prec_bin,
        "recall": rec_bin,
        "f1_pos": f1_bin,
        "f1_macro": f1_macro
    })

    summary_rows.append({
        "Model": name,
        "Accuracy": acc,
        "Precision_Pos(1)": prec_bin,
        "Recall_Pos(1)": rec_bin,
        "F1_Pos(1)": f1_bin,
        "F1_Macro": f1_macro
    })

# -----------------------
# Logging
# -----------------------
logfile = "results.txt"
with open(logfile, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("SENTIMENT CLASSIFICATION RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write("Task: Predict review sentiment (1 = Positive, 0 = Negative)\n")
    f.write(f"Python: {platform.python_version()} | pandas: {pd.__version__}\n\n")

    # Dataset summary
    total = len(reviews_df)
    train_n, test_n = len(X_train), len(X_test)
    label_counts = reviews_df["label"].value_counts().to_dict()
    pct_pos = 100 * label_counts.get(1, 0) / total if total else 0
    pct_neg = 100 * label_counts.get(0, 0) / total if total else 0
    f.write("-" * 80 + "\n")
    f.write("DATASET SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total samples: {total}\n")
    f.write(f"Train size: {train_n} | Test size: {test_n}\n")
    f.write(f"Label distribution: {label_counts} (Pos={pct_pos:.1f}%, Neg={pct_neg:.1f}%)\n\n")

    # Overall summary
    f.write("=" * 80 + "\n")
    f.write("OVERALL MODEL COMPARISON\n")
    f.write("=" * 80 + "\n")
    summary_df = pd.DataFrame(summary_rows).sort_values(by=["F1_Pos(1)", "Accuracy"], ascending=False)
    f.write(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n\n")

    # Per-model details
    for res in model_results:
        f.write("-" * 80 + "\n")
        f.write(f"{res['name']}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {res['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(res["report"] + "\n")
        f.write("Confusion Matrix:\n")
        f.write(res["confusion"].to_string() + "\n")
        if res["pos_feats"] and res["neg_feats"]:
            f.write("\nTop features (class 1 = Positive):\n")
            f.write(format_top_feats(res["pos_feats"]) + "\n")
            f.write("\nTop features (class 0 = Negative):\n")
            f.write(format_top_feats(res["neg_feats"]) + "\n")
        else:
            f.write("\n(Feature importance not available for this model.)\n")
        f.write("\n")

print(f"\nResults saved to {logfile}")
