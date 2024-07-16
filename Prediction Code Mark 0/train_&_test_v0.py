
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import string

df = pd.read_excel('/content/drive/MyDrive/Project/Copy of dataset_v2.xlsx')

!pip install nltk

from nltk.corpus import stopwords

import nltk
nltk.download('wordnet')

dataset = pd.read_excel('/content/drive/MyDrive/Project/Copy of dataset_v2.xlsx')



import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import WordNetLemmatizer
import string

nlp_processor = spacy.load('en_core_web_sm')
lemmatizer_tool = WordNetLemmatizer()

def preprocess_text(input_text):
    filtered_words = []

    processed_text = nlp_processor(input_text)
    named_entities = {ent.text for ent in processed_text.ents}

    for token in processed_text:
        if token.text not in named_entities:
            filtered_words.append(token.text)
    clean_text = " ".join(filtered_words)

    # Further text preprocessing
    clean_text = clean_text.lower().strip()
    clean_text = clean_text.replace("</br>", " ")
    clean_text = clean_text.replace("-", " ")
    clean_text = "".join([char for char in clean_text if char not in string.punctuation and not char.isdigit()])
    clean_text = " ".join([word for word in clean_text.split() if word not in STOP_WORDS])
    clean_text = "".join([lemmatizer_tool.lemmatize(word) for word in clean_text])

    return clean_text

dataset['Text'] = dataset['Text'].apply(preprocess_text)
dataset.head()

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = dataset['Text'].tolist()
vectorizer = TfidfVectorizer(max_df=0.85, max_features=20000)
transformed_docs = vectorizer.fit_transform(corpus)
corpus_array = transformed_docs.toarray()
X, y = corpus_array, dataset['Category']

from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
X_training, X_validation, y_training, y_validation = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

print("Training data dimensions:", X_training.shape, y_training.shape)
print("Validation data dimensions:", X_validation.shape, y_validation.shape)

# Target variable encoding
df['Label'] = df['Category'].replace({'Trauma': 1, 'Non-Trauma': 0})
selected_data = df[["Text", "Label"]]
sample_data = selected_data.head(1000)

# Feature extraction using TF-IDF
corpus = sample_data['Text'].tolist()
vectorizer = TfidfVectorizer(max_df=0.85, max_features=20000)
transformed_docs = vectorizer.fit_transform(corpus)
corpus_array = transformed_docs.toarray()
X, y = corpus_array, sample_data['Label']

# Train-test split
RANDOM_STATE = 42
X_training, X_validation, y_training, y_validation = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

"""Decision Tree Classifier

"""

# Decision Tree Classifier
tree_classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree_classifier.fit(X_training, y_training)
tree_train_predictions = tree_classifier.predict(X_training)
tree_test_predictions = tree_classifier.predict(X_validation)
print("\nDecision Tree Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, tree_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, tree_test_predictions)))
print(classification_report(y_validation, tree_test_predictions, target_names=['Non-Trauma', 'Trauma']))

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=RANDOM_STATE)
rf_classifier.fit(X_training, y_training)
rf_train_predictions = rf_classifier.predict(X_training)
rf_test_predictions = rf_classifier.predict(X_validation)
print("\nRandom Forest Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, rf_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, rf_test_predictions)))
print(classification_report(y_validation, rf_test_predictions, target_names=['Non-Trauma', 'Trauma']))

# Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb_classifier.fit(X_training, y_training)
gb_train_predictions = gb_classifier.predict(X_training)
gb_test_predictions = gb_classifier.predict(X_validation)
print("\nGradient Boosting Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, gb_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, gb_test_predictions)))
print(classification_report(y_validation, gb_test_predictions, target_names=['Non-Trauma', 'Trauma']))

# Logistic Regression Classifier
lr_classifier = LogisticRegression(random_state=RANDOM_STATE)
lr_classifier.fit(X_training, y_training)
lr_train_predictions = lr_classifier.predict(X_training)
lr_test_predictions = lr_classifier.predict(X_validation)
print("\nLogistic Regression Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, lr_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, lr_test_predictions)))
print(classification_report(y_validation, lr_test_predictions, target_names=['Non-Trauma', 'Trauma']))

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(probability=True, random_state=RANDOM_STATE)
svm_classifier.fit(X_training, y_training)
svm_train_predictions = svm_classifier.predict(X_training)
svm_test_predictions = svm_classifier.predict(X_validation)
print("\nSupport Vector Machine (SVM) Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, svm_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, svm_test_predictions)))
print(classification_report(y_validation, svm_test_predictions, target_names=['Non-Trauma', 'Trauma']))

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_training, y_training)
knn_train_predictions = knn_classifier.predict(X_training)
knn_test_predictions = knn_classifier.predict(X_validation)
print("\nK-Nearest Neighbors (KNN) Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, knn_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, knn_test_predictions)))
print(classification_report(y_validation, knn_test_predictions, target_names=['Non-Trauma', 'Trauma']))

# Multi-layer Perceptron (MLP) Classifier
mlp_classifier = MLPClassifier(random_state=RANDOM_STATE)
mlp_classifier.fit(X_training, y_training)
mlp_train_predictions = mlp_classifier.predict(X_training)
mlp_test_predictions = mlp_classifier.predict(X_validation)
print("\nMulti-layer Perceptron (MLP) Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, mlp_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, mlp_test_predictions)))
print(classification_report(y_validation, mlp_test_predictions, target_names=['Non-Trauma', 'Trauma']))

# AdaBoost Classifier
adaboost_classifier = AdaBoostClassifier(random_state=RANDOM_STATE)
adaboost_classifier.fit(X_training, y_training)
adaboost_train_predictions = adaboost_classifier.predict(X_training)
adaboost_test_predictions = adaboost_classifier.predict(X_validation)
print("\nAdaBoost Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, adaboost_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, adaboost_test_predictions)))
print(classification_report(y_validation, adaboost_test_predictions, target_names=['Non-Trauma', 'Trauma']))

# XGBoost Classifier
xgboost_classifier = XGBClassifier(random_state=RANDOM_STATE)
xgboost_classifier.fit(X_training, y_training)
xgboost_train_predictions = xgboost_classifier.predict(X_training)
xgboost_test_predictions = xgboost_classifier.predict(X_validation)
print("\nXGBoost Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, xgboost_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, xgboost_test_predictions)))
print(classification_report(y_validation, xgboost_test_predictions, target_names=['Non-Trauma', 'Trauma']))

# LightGBM Classifier
lgbm_classifier = LGBMClassifier(random_state=RANDOM_STATE)
lgbm_classifier.fit(X_training, y_training)
lgbm_train_predictions = lgbm_classifier.predict(X_training)
lgbm_test_predictions = lgbm_classifier.predict(X_validation)
print("\nLightGBM Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, lgbm_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, lgbm_test_predictions)))
print(classification_report(y_validation, lgbm_test_predictions, target_names=['Non-Trauma', 'Trauma']))

classifiers = [
    ('DecisionTree', tree_classifier),
    ('NaiveBayes', GaussianNB()),  # Fixed the typo here
    ('RandomForest', rf_classifier),
    ('GradientBoosting', gb_classifier),
    ('LogisticRegression', lr_classifier),
    ('SVM', svm_classifier),
    ('KNN', knn_classifier),
    ('MLP', mlp_classifier),
    ('AdaBoost', adaboost_classifier),
    ('XGBoost', xgboost_classifier),
    ('LightGBM', lgbm_classifier),
]

vote_classifier = VotingClassifier(estimators=classifiers)
vote_classifier.fit(X_training, y_training)

# Convert sparse matrix to dense numpy array
X_validation = X_validation.toarray()

vote_train_predictions = vote_classifier.predict(X_training)
vote_test_predictions = vote_classifier.predict(X_validation)
print("\nVoting Classifier:")
print("Train Accuracy: {:.2f}%".format(100 * accuracy_score(y_training, vote_train_predictions)))
print("Test Accuracy: {:.2f}%".format(100 * accuracy_score(y_validation, vote_test_predictions)))
print(classification_report(y_validation, vote_test_predictions, target_names=['Non-Trauma', 'Trauma']))
