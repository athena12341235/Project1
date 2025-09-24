"""
This script trains and evaluates a Logistic Regression model
for sentiment classification on the Movie Review Polarity dataset.
It expects a CSV file named 'review_polarity_clean.csv' with two columns:
    - clean_data: preprocessed movie review text
    - label: binary sentiment label (1 = positive, 0 = negative)
"""

"""
Logistic Regression Sentiment Classifier
----------------------------------------
This script trains and evaluates a Logistic Regression model
for sentiment classification on the Movie Review Polarity dataset.

Input:
    - 'review_polarity_clean.csv' with two columns:
        * clean_text: preprocessed movie review text
        * label: binary sentiment label (1 = positive, 0 = negative)

Output:
    - Console printouts of accuracy and classification report
    - Confusion matrix visualization

Steps in the script:
1. Load dataset
2. Split into training/testing sets
3. Convert text into TF–IDF vectors
4. Train Logistic Regression model
5. Evaluate with accuracy, precision, recall, F1
6. Plot confusion matrix
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

# 1. Load dataset from CSV
# Assumes file is in the same directory
reviews_df = pd.read_csv("review_polarity_clean.csv")

# Extract features (text reviews) and labels (0 or 1 sentiment)
X = reviews_df["clean_text"].astype(str)   # predictor variable
y = reviews_df["label"]                    # target variable

# 2. Split into training and test sets (80% / 20%)
# stratify=y ensures balanced class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF–IDF vectorization
# Converts raw text into numerical features based on word importance
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # include unigrams and bigrams
    min_df=2,             # ignore words that appear in <2 docs
    max_df=0.9,           # ignore words that appear in >90% of docs
    sublinear_tf=True,    # scale term frequency logarithmically
    norm="l2"             # normalize feature vectors
)
X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocab + transform
X_test_tfidf  = vectorizer.transform(X_test)       # transform test set

# 4. Train Logistic Regression classifier
# max_iter=1000 ensures the solver converges
# saga solver works well for large/sparse text data
model = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=42)
model.fit(X_train_tfidf, y_train)

# 5. Evaluate model on test data
y_pred = model.predict(X_test_tfidf)

# Print accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print detailed precision/recall/F1 for each class
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative","Positive"]))

# 6. Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Logistic Regression Classifier")
plt.show()
