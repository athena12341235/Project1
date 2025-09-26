"""
LightGBM Sentiment Classifier
-----------------------------
This script trains and evaluates a LightGBM gradient boosting model
for sentiment classification on the Movie Review Polarity dataset.

Input:
    - 'review_polarity_clean.csv' with two columns:
        * clean_text: preprocessed movie review text
        * label: binary sentiment label (1 = positive, 0 = negative)

Output:
    - Console printout of accuracy and classification report
    - Confusion matrix visualization

Steps in the script:
1. Load dataset
2. Split into training/testing sets
3. Convert text into TF–IDF vectors
4. Train LightGBM classifier
5. Evaluate with accuracy, precision, recall, F1
6. Plot confusion matrix
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

# 1. Load dataset from CSV (from DATA folder)
df = pd.read_csv(os.path.join("..", "DATA", "review_polarity_clean.csv"))

# Split features (text) and labels (0/1)
X = df["clean_text"].astype(str)
y = df["label"]

# 2. Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF–IDF vectorization
# Convert raw text into numerical features
tfidf = TfidfVectorizer(
    ngram_range=(1,2),   # include unigrams + bigrams
    min_df=2,            # ignore very rare words
    max_df=0.9,          # ignore overly common words
    max_features=50000,  # cap vocab size for efficiency
    sublinear_tf=True,
    norm="l2"
)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# 4. Train LightGBM classifier
# n_estimators=400: number of boosting iterations
# learning_rate=0.05: shrinkage factor
# num_leaves=31: controls tree complexity
clf = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,        # row sampling
    colsample_bytree=0.9, # column sampling
    random_state=42
)
clf.fit(Xtr, y_train)

# 5. Evaluate predictions on test set
y_pred = clf.predict(Xte)

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print precision, recall, F1
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative","Positive"]))

# 6. Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for LightGBM Classifier")
plt.show()
