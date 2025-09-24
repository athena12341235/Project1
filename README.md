# Movie Review Sentiment Analysis: Evaluating Models


## 1. Software and Platform
- **Platform Used:** Mac was used to write and run scripts. The scripts can be run across platforms.
- **Software Used**: Python 3
 
- **Add-on Packages:**  
  - `pandas` - data loading and manipulation
  - `matplotlib` – plotting confusion matrices  
  - `seaborn` – visualizing confusion matrices
  - `lightgbm` (`lgb`) – gradient boosting framework for classification/regression  
  - `scikit-learn` (`sklearn`) library - machine learning models and metrics  
    - Subpackages, Classes, and Functions used:
        - `model_selection` → (`train_test_split`)
        - `feature_extraction.text` → (`TfidfVectorizer`)
        - `metrics` → (`accuracy_score`, `classification_report`, `confusion_matrix`)
        - `linear_model` → (`LogisticRegression`)
        - `svm` → (`LinearSVC`)
        - `naive_bayes` → (`MultinomialNB`)
  - `transformers` – deep learning models for NLP  
    - Classes used:
        - `BertTokenizer`
        - `BertForSequenceClassification`
        - `Trainer`
        - `TrainingArguments`
        - `DataCollatorWithPadding`  
  - `torch` – deep learning framework  
    - Subpackages and Classs used:
        - `torch.utils.data` → `dataset`

## 2. Documentation Map

## 3. Reproducing Our Results
 
 
