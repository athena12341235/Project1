"""
This script fine-tunes and evaluates a BERT (bert-base-uncased) model
for sentiment classification on the Movie Review Polarity dataset.
It expects a CSV file named 'review_polarity_clean.csv' with two columns:
    - clean_text: preprocessed movie review text
    - label: binary sentiment label (1 = positive, 0 = negative)

The script uses Hugging Face Transformers with the Trainer API.
The model is trained for multiple epochs, and performance is evaluated
using accuracy, precision, recall, and F1-score.

Input:
    - 'review_polarity_clean.csv' with two columns:
        * clean_text: preprocessed movie review text
        * label: binary sentiment label (1 = positive, 0 = negative)

Output:
    - Console printout of accuracy, precision, recall, F1
    - Model checkpoints in ./results
    - Logs in ./logs

Steps in the script:
1. Load dataset
2. Split into training/testing sets
3. Tokenize text with BERT tokenizer
4. Wrap into PyTorch Dataset objects
5. Load pre-trained BERT model with classification head
6. Configure Hugging Face Trainer
7. Train model
8. Evaluate model performance
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import Dataset
import torch

# 1. Load dataset from CSV (from DATA folder)
df = pd.read_csv(os.path.join("..", "DATA", "review_polarity_clean.csv"))
X = df["clean_text"]   # input text reviews
y = df["label"]        # binary sentiment labels

# 2. Split into train/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Load BERT tokenizer
# Tokenizer converts text into token IDs and attention masks
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 4. Define custom PyTorch Dataset for BERT
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        # Convert list of texts into BERT-compatible encodings
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,        # cut off reviews longer than max_len
            padding="max_length",   # pad reviews shorter than max_len
            max_length=max_len
        )
        self.labels = labels.tolist()  # store sentiment labels

    def __len__(self):
        return len(self.labels)  # number of samples

    def __getitem__(self, idx):
        # Retrieve encoded sample and add its label
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Wrap data into Dataset objects
train_dataset = ReviewDataset(X_train, y_train, tokenizer)
test_dataset  = ReviewDataset(X_test, y_test, tokenizer)

# 5. Load pre-trained BERT model with classification head
# num_labels=2 because this is binary classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 6. Define metric computation for evaluation
def compute_metrics(pred):
    labels = pred.label_ids                # true labels
    preds = pred.predictions.argmax(-1)    # predicted classes
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 7. Training configuration
training_args = TrainingArguments(
    output_dir="./results",          # save model checkpoints
    eval_strategy="epoch",           # evaluate after each epoch
    save_strategy="epoch",           # save checkpoint after each epoch
    per_device_train_batch_size=8,   # batch size per device (small for CPU/Mac)
    per_device_eval_batch_size=8,
    num_train_epochs=2,              # fine-tune for 2 epochs
    weight_decay=0.01,               # regularization
    logging_dir="./logs",            # log directory
    load_best_model_at_end=True      # automatically restore best checkpoint
)

# Data collator dynamically pads batches to max length
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Setup Hugging Face Trainer (handles training loop for us)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. Train the model
trainer.train()

# 9. Evaluate the model on the test set
results = trainer.evaluate()
print("\nBERT Evaluation Results:")
print(results)
