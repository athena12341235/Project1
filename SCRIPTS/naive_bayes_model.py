"""
This script trains and evaluates a Multinomial Naive Bayes (NB) model
for sentiment classification on the Movie Review Polarity dataset.
It expects a CSV file named 'review_polarity_clean.csv' with two columns:
    - clean_text: preprocessed movie review text
    - label: binary sentiment label (1 = positive, 0 = negative)

The pipeline uses TF–IDF vectorization to convert text into features
and applies the Naive Bayes classifier to predict sentiment.
Model performance is evaluated using accuracy, precision, recall, and F1-score.
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("review_polarity_clean.csv")
X = df["clean_text"].astype(str)
y = df["label"]

# 2. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1,1),  # unigrams
    min_df=2,           # ignore rare words
    max_df=0.9,         # ignore overly common words
    sublinear_tf=True,
    norm="l2"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Naive Bayes
nb_clf = MultinomialNB()
nb_clf.fit(X_train_tfidf, y_train)

# 5. Evaluate
y_pred = nb_clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative","Positive"]))