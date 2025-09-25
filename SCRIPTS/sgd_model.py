import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/Users/athenavo/PycharmProjects/Project1/DATA/review_polarity_clean.csv")
X = df["clean_text"].astype(str)
y = df["label"].astype(int)

# 80/20 split (stratified)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF features (unigrams+bigrams are strong for sentiment)
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9,
                        sublinear_tf=True, norm="l2")

Xtr = tfidf.fit_transform(X_tr)
Xte = tfidf.transform(X_te)

# Linear SGDClassifier (hinge ≈ linear SVM; weight averaging to improve generalization, early stopping)
sgd = SGDClassifier(loss="hinge", alpha=1e-4, random_state=42, average=True, early_stopping=True, validation_fraction=0.1)
sgd.fit(Xtr, y_tr)

# Evaluate
yp = sgd.predict(Xte)
print("Accuracy:", accuracy_score(y_te, yp))
print("\nClassification Report:\n",
      classification_report(y_te, yp, target_names=["Negative","Positive"]))
cm = confusion_matrix(y_te, yp, labels=[0,1])
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for SGD Classifier")
plt.show()
