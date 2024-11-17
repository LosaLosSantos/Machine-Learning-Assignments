[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NhRt73hM)
# WUST Machine Learning - Laboratory # 4

**Linear Regression Model Selection and Hyperparameter Tuning Lab**

Term: Winter 2024/2025

Created by: [Daniel Kucharczyk](mailto:daniel.kucharczyk@pwr.edu.pl)

---

## Overview
In this lab, you will work with linear regression models to understand the impact of different regularization techniques and hyperparameter tuning. You'll use the Boston Housing dataset to predict house prices while learning how to prevent overfitting and optimize model performance.

## Objectives
- Understand the difference between simple linear regression and regularized versions (Ridge, Lasso, Elastic Net)
- Implement cross-validation techniques for linear regression
- Apply hyperparameter tuning to find optimal regularization parameters
- Evaluate model performance using appropriate regression metrics
- Visualize the impact of regularization on model coefficients

## Prerequisites
```python
# Required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
```

## Tasks

### Task 1: Data Preparation
```python
# TODO: Load and preprocess the Boston Housing dataset
def load_and_preprocess_data():
    """
    Load the Boston Housing dataset and perform necessary preprocessing steps.

    Your code should:
    1. Load the dataset
    2. Scale the features using StandardScaler
    3. Split into training and testing sets (80-20 split)

    Returns:
    - X_train, X_test, y_train, y_test
    """
    # Your code here
    pass

def visualize_data(X, y):
    """
    Create scatter plots to visualize relationships between features and target.

    Your code should:
    1. Create scatter plots for each feature vs. house price
    2. Add trend lines
    3. Add proper labels and titles
    """
    # Your code here
    pass
```

### Task 2: Implementing Different Linear Models
```python
# TODO: Implement various linear regression models
def train_linear_models(X_train, y_train):
    """
    Train different types of linear regression models.

    Implement:
    1. Simple Linear Regression
    2. Ridge Regression (L2 regularization)
    3. Lasso Regression (L1 regularization)
    4. Elastic Net Regression

    Returns:
    - Dictionary containing trained models
    """
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }

    # Your code here: Train each model

    return models
```

### Task 3: Cross-Validation Implementation
```python
# TODO: Implement k-fold cross-validation
def perform_cross_validation(model, X, y, k=5):
    """
    Perform k-fold cross-validation for regression models.

    Your code should:
    1. Implement k-fold cross-validation
    2. Calculate MSE and R² for each fold
    3. Return mean and std of performance metrics

    Parameters:
    - model: The regression model to evaluate
    - X: Feature matrix
    - y: Target vector
    - k: Number of folds

    Returns:
    - Dictionary with cross-validation metrics
    """
    # Your code here
    pass
```

### Task 4: Hyperparameter Tuning
```python
# TODO: Implement Grid Search for different models
def tune_model_parameters():
    """
    Perform grid search for hyperparameter tuning.

    Your code should:
    1. Define parameter grids for each model type:
       - Ridge: different alpha values
       - Lasso: different alpha values
       - Elastic Net: different alpha and l1_ratio values
    2. Implement GridSearchCV
    3. Return best parameters and scores

    Returns:
    - Dictionary with best parameters for each model
    """
    param_grids = {
        'ridge': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'lasso': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'elastic_net': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    }

    # Your code here
    pass
```

### Task 5: Model Evaluation and Visualization
```python
# TODO: Implement comprehensive model evaluation
def evaluate_models(models, X_test, y_test):
    """
    Evaluate models using multiple metrics.

    Your code should calculate:
    1. Mean Squared Error (MSE)
    2. Root Mean Squared Error (RMSE)
    3. Mean Absolute Error (MAE)
    4. R² Score

    Returns:
    - Dictionary with evaluation metrics for each model
    """
    # Your code here
    pass

def visualize_coefficients(models, feature_names):
    """
    Create visualizations comparing model coefficients.

    Your code should:
    1. Create bar plots of coefficients for each model
    2. Compare how different regularization techniques affect coefficients
    3. Add proper labels and legends
    """
    # Your code here
    pass
```

## Example Usage
```python
# Main execution flow
# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Visualize relationships in data
visualize_data(X_train, y_train)

# Train models
models = train_linear_models(X_train, y_train)

# Perform cross-validation
cv_results = {}
for name, model in models.items():
  cv_results[name] = perform_cross_validation(model, X_train, y_train)

# Tune hyperparameters
best_params = tune_model_parameters()

# Train final models with best parameters
final_models = {
  'ridge': Ridge(**best_params['ridge']),
  'lasso': Lasso(**best_params['lasso']),
  'elastic_net': ElasticNet(**best_params['elastic_net'])
}

# Evaluate and visualize results
final_metrics = evaluate_models(final_models, X_test, y_test)
visualize_coefficients(final_models, feature_names)
```

## Expected Deliverables
1. Completed implementation of all TODO sections
2. A report containing:
   - Data exploration visualizations
   - Cross-validation results for each model
   - Best hyperparameters found through Grid Search
   - Comparison of model performances
   - Visualization of coefficient values across different models
   - Analysis of how regularization affects model performance
4. Create learning curves to visualize model performance vs. training size
2. Implement feature selection based on Lasso coefficients
3. Implement your own custom cross-validation splitter

## Tips
- Pay attention to feature scaling - it's crucial for regularized models
- Experiment with different ranges of regularization parameters
- Look for features that become zero in Lasso regression
- Consider the trade-off between bias and variance
- Document your findings about which regularization technique works best and why


## Evaluation Criteria
- Data preprocessing and exploration (20%)
- Implementation of different linear models (20%)
- Cross-validation implementation (20%)
- Hyperparameter tuning implementation (20%)
- Visualization and analysis quality (20%)

## References
- Scikit-learn Linear Models: https://scikit-learn.org/stable/modules/linear_model.html
- Understanding Regularization: https://www.stat.cmu.edu/~ryantibs/statml/lectures/regularization.pdf

## Submission Guidelines
Include your report as a jupyter notebook document (notebooks).
Submit all files via the course management system by 13rd of November.


Good luck, and happy coding!
