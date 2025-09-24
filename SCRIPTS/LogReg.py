"""
This script trains and evaluates a Logistic Regression model
for sentiment classification on the Movie Review Polarity dataset.
It expects a CSV file named 'review_polarity_clean.csv' with two columns:
    - clean_data: preprocessed movie review text
    - label: binary sentiment label (1 = positive, 0 = negative)
"""

import pandas as pd
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


# 1. Load dataset
reviews_df = pd.read_csv("review_polarity_clean.csv")
X = reviews_df["clean_text"].astype(str)
y = reviews_df["label"]

# 2. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. TF–IDF vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=2,             # ignore very rare words
    max_df=0.9,           # ignore overly common words
    sublinear_tf=True,    # dampen term frequency
    norm="l2"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# 4. Train SVM
model = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=42)
model.fit(X_train_tfidf, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative","Positive"]))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])


# 6. Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for SVM Classifier")
plt.show()
