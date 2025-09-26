"""
This script preprocesses raw movie text data for sentiment analysis.
Converts to lowercase, removes punctuation and numbers, tokenizes, removes stopwords,
lemmatizes words.

Input:
    - Text files loaded in the dataset directory. Each file contains raw text for a single review.
        Positive reviews: ../DATA/review_polarity/txt_sentoken/neg/*.txt 
        Negative reviews: ../DATA/review_polarity/txt_sentoken/pos/*.txt 
Output: 
    - `review_polarity_clean.csv` containing two columns:
          - `clean_text`: preprocessed version of the review
          - `label`: sentiment label (0 = negative, 1 = positive)
"""

import os
import glob
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# One-time NLTK downloads
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("wordnet")

# ----- Paths -----
BASE_DIR = os.path.join("..", "DATA", "review_polarity", "txt_sentoken")
NEG_DIR  = os.path.join(BASE_DIR, "neg")
POS_DIR  = os.path.join(BASE_DIR, "pos")

# ----- Load raw reviews from folders -----
def read_folder(folder_path, label_value):
    rows = []
    for fp in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        rows.append({"text": text, "label": label_value})
    return rows

data = []
data += read_folder(NEG_DIR, 0)  # 0 = negative
data += read_folder(POS_DIR, 1)  # 1 = positive

df = pd.DataFrame(data)

# ----- Initialize tools -----
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove punctuation + numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords + lemmatize
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    # Join back into a single string
    return " ".join(tokens)

# ----- Apply cleaning -----
df["clean_text"] = df["text"].apply(preprocess)
out_df = df[["clean_text", "label"]]
out_path = os.path.join("..", "DATA", "review_polarity_clean.csv")
out_df.to_csv(out_path, index=False)
print(out_df.head())
