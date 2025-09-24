import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalMaxPooling1D,
                                     Concatenate, Dropout, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# 1) Load data
# -----------------------
df = pd.read_csv("/Users/athenavo/PycharmProjects/Project1/DATA/review_polarity_clean.csv")
texts = df["clean_text"].astype(str).tolist()
labels = df["label"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# -----------------------
# 2) Tokenize & pad
# -----------------------
MAX_VOCAB = 20000      # cap vocab for memory
MAX_LEN   = 256        # short sequences work well for TextCNN
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)  # fit on train only

Xtr = tokenizer.texts_to_sequences(X_train)
Xte = tokenizer.texts_to_sequences(X_test)

Xtr = pad_sequences(Xtr, maxlen=MAX_LEN, padding="post", truncating="post")
Xte = pad_sequences(Xte, maxlen=MAX_LEN, padding="post", truncating="post")

vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

# -----------------------
# 3) Build TextCNN
# -----------------------
EMB_DIM   = 100
FILTERS   = 128
KERNELS   = [3, 4, 5]
DROPOUT   = 0.5

inp = Input(shape=(MAX_LEN,), dtype="int32")
emb = Embedding(input_dim=vocab_size, output_dim=EMB_DIM, input_length=MAX_LEN)(inp)

convs = []
for k in KERNELS:
    c = Conv1D(filters=FILTERS, kernel_size=k, activation="relu")(emb)
    p = GlobalMaxPooling1D()(c)
    convs.append(p)

x = Concatenate()(convs)
x = Dropout(DROPOUT)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(DROPOUT)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# -----------------------
# 4) Train
# -----------------------
es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
history = model.fit(
    Xtr, y_train,
    validation_split=0.1,
    epochs=8,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

# -----------------------
# 5) Evaluate
# -----------------------
proba = model.predict(Xte, batch_size=256).ravel()
y_pred = (proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative","Positive"]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
