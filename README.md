# Fuel Cell Performance Prediction

This repository contains two different implementations for predicting fuel cell performance using machine learning models. The project uses the Fuel_cell_performance_data-Full.csv dataset and implements both a custom machine learning pipeline and a PyCaret-based solution.

## Dataset Selection

- Dataset: Fuel_cell_performance_data-Full.csv
- Target Variable: Target4 (for roll numbers ending with 3 or 8)
- Features: F1 through F15

## Implementations

### 1. PyCaret Implementation (method1_pycaret.py)
This implementation uses PyCaret's automated machine learning capabilities for:
- Automated model comparison
- Built-in preprocessing
- Comprehensive model evaluation
- CatBoost model creation

Key features:
- Automated model selection
- Built-in visualization capabilities
- 70/30 train-test split
- Streamlined implementation

### 2. Custom Implementation (method2_sklearn.py)
This implementation uses scikit-learn to create a custom machine learning pipeline with the following models:
- Linear Regression
- Decision Tree
- Random Forest
- Support Vector Regression (SVR)

Key features:
- Manual data preprocessing using StandardScaler
- Custom metric calculations (MSE, RMSE, MAE, R² Score)
- 70/30 train-test split
- Explicit control over the machine learning pipeline

##  Install required packages
```bash
pip install scikit-learn pandas numpy
pip install pycaret[full]
```

## Results

Both methods provide comprehensive model performance metrics including:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

### Custom Implementation Results
The custom implementation provides results for four specific models with detailed metrics for each.

### PyCaret Implementation Results
PyCaret automatically compares multiple models and provides comprehensive performance metrics, with specific focus on the CatBoost model performance.
