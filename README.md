
# Machine Learning Project: Multi-Model Implementation

## Overview

This project implements several machine learning models using scikit-learn to address different types of learning tasks including classification, ensemble methods, sequence labelling, dimensionality reduction, support vector machines, and Bayesian linear regression. Each model is carefully tuned and validated to achieve high performance on various datasets included in this repository.

## Project Structure

Below is an outline of the main components of this project:

- `activity_recognition_dataset/`: Contains datasets used for training models.
- `cw_lu21864.ipynb`: Jupyter notebook with detailed code, analysis, and visualizations.
- `cw_lu21864.pdf`: Comprehensive report documenting the methodology, results, and insights.
- `iris/`: Dataset used for initial testing and experimentation.
- `README.md`: This file, providing an overview and guide to the project.
- `SeoulBikeData.csv`: Dataset used for regression models.
- `wdbc.data`: Dataset for the Wisconsin Diagnostic Breast Cancer study.

## Models Implemented

### MLP Classifier for Activity Recognition

- **Dataset**: `activity_recognition_dataset`
- **Features**: Neural network MLP classifier enhanced with randomized search for hyperparameter tuning and cross-validation to mitigate overfitting.
- **Performance**: Achieved 89.98% accuracy on training data and 89.01% on testing data.

### Decision Tree Ensemble

- **Algorithm**: AdaBoost for ensemble learning.
- **Features**: Boosting ensemble method to improve prediction accuracy by focusing on mistakes of previous models.
- **Performance**: Accuracy ranges from 98.44% to 99.99% on the test set.

### Gaussian HMM for Sequence Labelling

- **Dataset**: Sequential data from the activity recognition dataset.
- **Features**: Models the sequential nature of data, capturing transitions and state dependencies.
- **Performance**: 90.50% accuracy on the test set.

### PCA for Dimensionality Reduction

- **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
- **Features**: PCA applied to reduce dimensions and focus on the most informative features.
- **Performance**: Explains 98.23% of the variance with the first principal component.

### SVM Classifier

- **Features**: Trained with optimized hyperparameters using grid search, tested on original and dimensionally reduced data.
- **Performance**: Improved accuracy from 93.22% to 94.07% on the test set with reduced dimensionality.

### Bayesian Linear Regression

- **Dataset**: `SeoulBikeData.csv`
- **Features**: Predicts bike rental counts with Bayesian priors tailored to capture underlying patterns in data.
- **Performance**: Detailed performance analysis available in the report.

## Installation and Setup

Ensure you have Python and Jupyter installed. You can install the required packages using:

```bash
pip install -r requirements.txt
```

## To open the Jupyter notebooks:

```bash
jupyter notebook cw_lu21864.ipynb
```

## Usage 
To replicate the findings and experiment with the models:

- Navigate to the respective notebook section.
- Run the cells sequentially to load data, train models, and visualize the outcomes.
