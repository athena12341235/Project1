"""
This script trains a multi-layer perceptron (MLP) neural network on movie reviews.

Steps:
1. Cleaned review text is transformed into TF-IDF features.
2. TruncatedSVD reduces the high-dimensional TF-IDF vectors to a dense lower-dimensional representation (LSA features).
3. An MLPClassifier (feedforward neural network) is trained on these reduced features to predict sentiment (positive=1, negative=0).
4. Model performance is evaluated with accuracy, classification report, and a confusion matrix heatmap.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from CSV
df = pd.read_csv("/Users/athenavo/PycharmProjects/Project1/DATA/review_polarity_clean.csv")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"].astype(str), df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# Convert text into TF-IDF features (unigrams + bigrams)
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, max_features=50000, sublinear_tf=True, norm="l2")
Xtr_sparse = tfidf.fit_transform(X_train)
Xte_sparse = tfidf.transform(X_test)

# Reduce to dense, low-dim features (e.g., 300 dims)
svd = TruncatedSVD(n_components=300, random_state=42)
Xtr = svd.fit_transform(Xtr_sparse)
Xte = svd.transform(Xte_sparse)

# Train MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(256, 64), activation="relu", alpha=1e-4,
                    learning_rate_init=1e-3, max_iter=200, early_stopping=True, n_iter_no_change=8,
                    validation_fraction=0.1, random_state=42)
clf.fit(Xtr, y_train)

# Evaluate Model
pred = clf.predict(Xte)
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred, target_names=["Negative","Positive"]))

# Plot confusion matrix
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for MLP Classifier")
plt.show()
