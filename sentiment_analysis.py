# sentiment_analysis.py
# --- 1. Import libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources (only first run)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# --- 2. Load dataset (make sure review_polarity.csv is uploaded in Colab) ---
df = pd.read_csv("review_polarity.csv")

print("Preview of dataset:")
print(df.head())
print("\nDataset size:", len(df))
print("Class distribution:\n", df['label'].value_counts())

# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation/numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords + lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join back into a single string
    return " ".join(tokens)

# Apply preprocessing
df['clean_text'] = df['review'].apply(preprocess)
df[['clean_text', 'label']].to_csv("review_polarity_clean.csv", index=False)
print("✅ Cleaned dataset saved as review_polarity_clean.csv")

print("Original Review:\n", df['review'][0])
print("\nCleaned Review:\n", df['clean_text'][0])

# -----------------------------
# Train/Test Split
# -----------------------------
"""X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# -----------------------------
# TF–IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Baseline Logistic Regression
# -----------------------------
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_tfidf, y_train)
y_pred_lr = log_reg.predict(X_test_tfidf)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=["Negative", "Positive"]))

# -----------------------------
# Support Vector Machine
# -----------------------------
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

print("\nSVM Results")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, target_names=["Negative", "Positive"]))"""