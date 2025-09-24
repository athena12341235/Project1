import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import nltk
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import torch

# -----------------------------
# Download NLTK resources (only first run)
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("review_polarity_clean.csv")
print("Columns:", df.columns.tolist())
print(df.head())
print("\nDataset size:", len(df))
print("Class distribution:\n", df['label'].value_counts())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# -----------------------------
# Naive Bayes
# -----------------------------
nb_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('nb', MultinomialNB())
])
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("\nNaive Bayes Results")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# -----------------------------
# Logistic Regression
# -----------------------------
lr_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('lr', LogisticRegression(max_iter=1000))
])
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=["Negative", "Positive"]))

# -----------------------------
# SVM
# -----------------------------
svm_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('svm', LinearSVC())
])
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("\nSVM Results")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, target_names=["Negative", "Positive"]))

# -----------------------------
# BERT Dataset
# -----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_len
        )
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = ReviewDataset(X_train, y_train, tokenizer)
test_dataset = ReviewDataset(X_test, y_test, tokenizer)

# -----------------------------
# BERT Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# -----------------------------
# Training Setup
# -----------------------------
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",   # use eval_strategy (new in v4.46+)
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Train & Evaluate BERT
# -----------------------------
trainer.train()
bert_results = trainer.evaluate()
print("\nBERT Results")
print(bert_results)

# -----------------------------
# Save All Results to File
# -----------------------------
with open("results.txt", "w") as f:
    # NB
    f.write("Naive Bayes Results\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}\n")
    f.write(classification_report(y_test, y_pred_nb))
    f.write("\n\n")

    # LR
    f.write("Logistic Regression Results\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}\n")
    f.write(classification_report(y_test, y_pred_lr))
    f.write("\n\n")

    # SVM
    f.write("SVM Results\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}\n")
    f.write(classification_report(y_test, y_pred_svm))
    f.write("\n\n")

    # BERT
    f.write("BERT Results\n")
    for key, value in bert_results.items():
        f.write(f"{key}: {value}\n")
    f.write("\n")

print("✅ Results saved to results.txt")
