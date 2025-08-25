Aquí está el contenido estructurado para el archivo README.md:

# Titanic Survival Prediction
![Titanic Ship](oVUfCaYH7n+bP1Z2lNd4WrLrt73Xr9kFDn2eJ88en6Sffl+fn2i2+++eZecO3HZkcIQAj8H1xGOAO823bfAAAAAElFTkSuQmCC.jpg)

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preparation](#data-preparation)
   1. [Handling Missing Values](#handling-missing-values)
   2. [Feature Engineering](#feature-engineering)
   3. [Attribute Conversion (First Encoding)](#attribute-conversion-first-encoding)
   4. [Age Imputation](#age-imputation)
   5. [Scaling (First Time)](#scaling-first-time)
   6. [Feature Selection](#feature-selection)
   7. [PCA Application](#pca-application)
   8. [Second Encoding and Scaling (Saved)](#second-encoding-and-scaling-saved)
   9. [PCA Model Saving](#pca-model-saving)
4. [Modeling and Model Evaluation](#modeling-and-model-evaluation)
   1. [Cross-Validation with K-Fold](#cross-validation-with-k-fold)
   2. [Training and Evaluation with train_test_split](#training-and-evaluation-with-train_test_split)
   3. [New train-test split with original data](#new-train-test-split-with-original-data)

## Introduction

This project aims to predict whether a passenger on the Titanic survived the sinking or not, using machine learning techniques.

## Dataset

In this section, we load the `titanic_train.csv` dataset using the `pandas` library.

## Data Preparation

### Handling Missing Values

We start by handling the missing values present in the dataset. We fill the missing values in the `Embarked` and `Cabin` columns.

### Feature Engineering

We create new features that may be relevant for the model, such as `FamilySize` and `Title`.

### Attribute Conversion (First Encoding)

We encode all categorical columns using an ordinal encoder, so that we can work with them in the model.

### Age Imputation

We predict the missing values in the `Age` column using a linear regression model.

### Scaling (First Time)

We scale all features (excluding `Survived`, `PassengerId`, and `Name`) using a robust scaler.

### Feature Selection

We calculate the feature importance using a decision tree model and select the most relevant features.

### PCA Application

We apply Principal Component Analysis (PCA) to the selected features, retaining 99% of the variance.

### Second Encoding and Scaling (Saved)

We perform a second encoding of the relevant categorical columns and scale the final features. We save the encoder and scaler.

### PCA Model Saving

We save the PCA model after the second encoding and scaling.

## Modeling and Model Evaluation

### Cross-Validation with K-Fold

We evaluate the `RandomForestClassifier` model using cross-validation with `StratifiedKFold`.

### Training and Evaluation with train_test_split

We split the data into training and test sets, train the model on the training set, evaluate on the test set, and save the final model.

### New train-test split with original data

We perform a new split of the original data (excluding `Embarked` and `Cabin`) and apply the trained model.
