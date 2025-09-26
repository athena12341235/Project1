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
        - `linear_model` → (`LogisticRegression`, `SGDClassifier`)
        - `svm` → (`LinearSVC`)
        - `naive_bayes` → (`MultinomialNB`)
        - `decomposition` → 
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
The hierarchy of folders and files contained in this project are as follows:

```text
Project1
├── DATA
│   ├── review_polarity
│   │   └── txt_sentoken
│   │       ├── neg
│   │       │   ├── cv000_29416.txt
│   │       │   └── ... (more negative review files)
│   │       └── pos
│   │           ├── cv000_29590.txt
│   │           └── ... (more positive review files)
│   ├── README.md
│   ├── review_lengths_plots.png
│   ├── review_polarity_clean.csv
│   └── word_frequencies_plots.png
├── OUTPUT
│   ├── lightgbm_cm.png
│   ├── mlp_cm.pgn
│   ├── sgd_cm.png
│   └── svm_cm.png
├── SCRIPTS
│   ├── bert_model.py
│   ├── lightgbm_model.py
│   ├── logreg_model.py
│   ├── mlp_model.py
│   ├── naive_bayes_model.py
│   ├── preprocessing.py
│   ├── sgd_model.py
│   └── svm_model.py
├── LICENSE.md
└── README.md
```

## 3. Reproducing Our Results
  1. **Set up Python and install required add-on packages**
     - Clone this repository: https://github.com/athena12341235/Project1
     - Ensure you have Python 3 installed on your system.
     - See section 1 for packages needed.
  2. **Prepare the data**
     - If you wish to preprocess the raw data yourself, navigate to the `SCRIPTS` folder and run the `preprocessing.py` file, which will save the preprocessed data in a new `review_polarity_clean.csv` file within the `DATA` folder.
     - Otherwise, there is an existing `review_polarity_clean.csv` file in the `DATA` folder ready for use.
  4. **Run model scripts**
     - Navigate to the `SCRIPTS` folder.
     - Each script corresponds to a model (ex: Logistic Regression, SVM, Naive Bayes). Run each script in order of their postfix numbers. Note: The models are indepenent, so order does not matter, but we have numbered the files to mirror our process. 
  5. **Download and view outputs** 
     - Confusion matrix images (ex: `svm_cm.png`, `mlp_cm.png`) need to be saved manually after running the scripts.
