"""
Multinomial Naive Bayes Sentiment Classifier
--------------------------------------------
This script trains and evaluates a Multinomial Naive Bayes (NB) model
for sentiment classification on the Movie Review Polarity dataset.

Input:
    - 'review_polarity_clean.csv' with two columns:
        * clean_text: preprocessed movie review text
        * label: binary sentiment label (1 = positive, 0 = negative)

Output:
    - Console printout of accuracy and classification report

Steps in the script:
1. Load dataset
2. Split into training/testing sets
3. Convert text into TF–IDF vectors
4. Train Multinomial Naive Bayes classifier
5. Evaluate with accuracy, precision, recall, F1
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset from CSV
df = pd.read_csv("review_polarity_clean.csv")

# Extract features (text reviews) and labels (0/1 sentiment)
X = df["clean_text"].astype(str)   # predictor variable
y = df["label"]                    # target variable

# 2. Split into train/test sets (80% train, 20% test)
# stratify=y ensures both sets have balanced class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Convert text into TF–IDF vectors
vectorizer = TfidfVectorizer(
    ngram_range=(1,1),   # use only unigrams
    min_df=2,            # ignore rare words
    max_df=0.9,          # ignore overly common words
    sublinear_tf=True,   # logarithmic scaling
    norm="l2"            # normalize vectors
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# 4. Train Naive Bayes classifier
# MultinomialNB is suitable for word counts/TF–IDF features
nb_clf = MultinomialNB()
nb_clf.fit(X_train_tfidf, y_train)

# 5. Evaluate on test set
y_pred = nb_clf.predict(X_test_tfidf)

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print precision, recall, F1 for each class
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative","Positive"]))
