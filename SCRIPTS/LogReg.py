# ---------------------------------------------------------
# Logistic Regression Sentiment Classifier
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
# Train/evaluate Logistic Regression
# -----------------------
model = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=42)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary", zero_division=0
)

report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# -----------------------
# Plot confusion matrix
# -----------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred 0 (Neg)", "Pred 1 (Pos)"],
            yticklabels=["True 0 (Neg)", "True 1 (Pos)"])
plt.title("Confusion Matrix - Logistic Regression", fontsize=14)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# -----------------------
# Logging (optional)
# -----------------------
logfile = "results.txt"
with open(logfile, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("SENTIMENT CLASSIFICATION RESULTS (Logistic Regression Only)\n")
    f.write("=" * 80 + "\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(pd.DataFrame(cm,
                         index=["True 0 (Neg)", "True 1 (Pos)"],
                         columns=["Pred 0 (Neg)", "Pred 1 (Pos)"]).to_string())
