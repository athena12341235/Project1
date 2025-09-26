"""
Lingear SGD Sentiment Classifier
--------------------------------
This script trains and evaluates a linear classifier using Stochastic
Gradient Descent (SGD) for sentiment classification on the Movie Review Polarity
Dataset.

Input:
    - 'Input:
    - 'review_polarity_clean.csv' with two columns:
        * clean_text: preprocessed movie review text
        * label: binary sentiment label (1 = positive, 0 = negative)

Output:
    - Console printout of accuracy and classification report
    - Confusion matrix heatmap

Steps in the script:
1. Load dataset
2. Split into training/testing sets (80/20 split)
3. Convert text into TF–IDF vectors (unigrams + bigrams)
4. Train SGDClassifier
5. Evaluate with accuracy, classification report, and confusion matrix
6. Visualize confusion matrix
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("/Users/athenavo/PycharmProjects/Project1/DATA/review_polarity_clean.csv")

# Extract features (text reviews) and labels (0/1 sentiment)
X = df["clean_text"].astype(str)
y = df["label"].astype(int)

# 2. Split into train/test sets (80% train, 20% test)
# stratify=y ensures both sets have balanced class distribution
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Convert text into TF-IDF features (unigrams+bigrams are strong for sentiment)
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9,
                        sublinear_tf=True, norm="l2")

Xtr = tfidf.fit_transform(X_tr)
Xte = tfidf.transform(X_te)

# 4. Train Linear SGDClassifier
# (hinge ≈ linear SVM; weight averaging to improve generalization, early stopping)
sgd = SGDClassifier(loss="hinge", alpha=1e-4, random_state=42, average=True, early_stopping=True, validation_fraction=0.1)
sgd.fit(Xtr, y_tr)

# 5. Evaluate Model
yp = sgd.predict(Xte)
print("Accuracy:", accuracy_score(y_te, yp))
print("\nClassification Report:\n",
      classification_report(y_te, yp, target_names=["Negative","Positive"]))
cm = confusion_matrix(y_te, yp, labels=[0,1])

# 6. Visualize Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for SGD Classifier")
plt.show()
