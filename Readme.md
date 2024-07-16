# Traumatic State Prediction Model

This project aims to predict whether a person is in a traumatic state based on text input using various machine learning classifiers.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Classifiers Used](#classifiers-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [How to Run](#how-to-run)

## Overview

This project utilizes text data to predict the traumatic state of an individual using different machine learning algorithms. The classifiers used range from basic models like Decision Tree and Logistic Regression to advanced ensemble methods like Random Forest, Gradient Boosting, XGBoost, and LightGBM. The final model is a Voting Classifier that combines the predictions of these individual classifiers.

## Installation

To run this project, you need to install the required libraries. Use the following command to install the necessary packages:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm nltk spacy seaborn matplotlib
```

Additionally, download the required NLTK and spaCy data:

```python
import nltk
nltk.download('wordnet')

import spacy
spacy.cli.download('en_core_web_sm')
```

## Dataset

The dataset used for this project is an Excel file named `dataset_v2.xlsx`. It contains text data categorized into two classes: 'Trauma' and 'Non-Trauma'.

## Preprocessing

Text data is preprocessed using the following steps:

- Named entity recognition to exclude named entities.
- Text normalization (lowercasing, removing punctuation and digits).
- Removal of stopwords.
- Lemmatization to reduce words to their base forms.

## Classifiers Used

The following classifiers are implemented in the project:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Logistic Regression Classifier
5. Support Vector Machine (SVM) Classifier
6. K-Nearest Neighbors (KNN) Classifier
7. Multi-layer Perceptron (MLP) Classifier
8. AdaBoost Classifier
9. XGBoost Classifier
10. LightGBM Classifier
11. Voting Classifier (combination of above classifiers)

## Evaluation Metrics

The performance of each classifier is evaluated using the following metrics:

- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

## Results

The results for each classifier are printed, including training and testing accuracies, along with detailed classification reports.

## How to Run

1. Clone the repository and navigate to the project directory.
2. Ensure the dataset `dataset_v2.xlsx` is in the correct path.
3. Install the required packages as mentioned in the [Installation](#installation) section.
4. Run the script to train and evaluate the models.

```bash
python script_name.py
```

Replace `script_name.py` with the name of your script file.

The script will preprocess the data, train multiple classifiers, evaluate their performance, and print the results.
