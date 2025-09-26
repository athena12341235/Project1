"""
Support Vector Machine (SVM) Sentiment Classifier
-------------------------------------------------
This script trains and evaluates a Support Vector Machine (SVM) model
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
3. Convert text into TF–IDF features
4. Train a linear SVM classifier
5. Evaluate with accuracy, precision, recall, F1
6. Plot confusion matrix
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset from CSV file (from DATA folder)
df = pd.read_csv(os.path.join("..", "DATA", "review_polarity_clean.csv"))

# Separate input features (text) and target labels (0 or 1)
X = df["clean_text"].astype(str)   # review text (predictors)
y = df["label"]                    # sentiment labels (targets)

# 2. Split dataset into train (80%) and test (20%) sets
# stratify=y ensures label distribution is preserved
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Vectorize text using TF–IDF
# Converts text into numerical representation
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),   # include unigrams and bigrams
    min_df=2,            # ignore terms appearing in fewer than 2 docs
    max_df=0.9,          # ignore terms appearing in more than 90% of docs
    sublinear_tf=True,   # logarithmic scaling of term frequencies
    norm="l2"            # normalize feature vectors
)
X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocab + transform
X_test_tfidf  = vectorizer.transform(X_test)       # transform using learned vocab

# 4. Train Support Vector Machine classifier
# LinearSVC = linear kernel SVM, good for high-dimensional sparse data like text
svm_clf = LinearSVC(random_state=42)
svm_clf.fit(X_train_tfidf, y_train)

# 5. Evaluate model performance on test data
y_pred = svm_clf.predict(X_test_tfidf)

# Accuracy = percentage of correct predictions
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report = precision, recall, F1 per class
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative","Positive"]))

# 6. Plot confusion matrix for visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for SVM Classifier")
plt.show()
