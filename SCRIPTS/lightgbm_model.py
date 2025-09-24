import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

df = pd.read_csv("/Users/athenavo/PycharmProjects/Project1/DATA/review_polarity_clean.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"].astype(str), df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, max_features=50000,
                        sublinear_tf=True, norm="l2")
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# LightGBM needs dense or CSR; CSR is fine:
clf = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
    colsample_bytree=0.9, random_state=42
)
clf.fit(Xtr, y_train)
pred = clf.predict(Xte)
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred, target_names=["Negative","Positive"]))

y_pred = clf.predict(Xte)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for LightGBM Classifier")
plt.show()
