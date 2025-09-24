import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

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
# Train/evaluate Logistic Regression
# -----------------------
model = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=42)
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

pos_feats, neg_feats = top_features_linear(model.coef_, feature_names, top_n=5)

# -----------------------
# Logging
# -----------------------
logfile = "results.txt"
with open(logfile, "w", encoding="utf-8") as f:
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

    # Results
    f.write("=" * 80 + "\n")
    f.write("LOGISTIC REGRESSION RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(cm_df.to_string() + "\n")
    f.write("\nTop features (class 1 = Positive):\n")
    f.write(format_top_feats(pos_feats) + "\n")
    f.write("\nTop features (class 0 = Negative):\n")
    f.write(format_top_feats(neg_feats) + "\n")

print("\nResults saved to", logfile)
print("\nConfusion Matrix:\n")
print(cm_df)
