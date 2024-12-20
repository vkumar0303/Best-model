import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def prepare_data():
    df = pd.read_csv('Fuel_cell_performance_data-Full.csv')
    X = df[[f'F{i}' for i in range(1, 16)]]
    y = df['Target1']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test

def evaluate_model(y_true, y_pred, model_name):
    return {
        'Model': model_name,
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2 Score': r2_score(y_true, y_pred)
    }

def run_models():
    X_train, X_test, y_train, y_test = prepare_data()
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR()
    }
    return pd.DataFrame([
        evaluate_model(y_test, model.fit(X_train, y_train).predict(X_test), name)
        for name, model in models.items()
    ])

results = run_models().round(4)
print("\nModel Performance Comparison:")
print(results.to_string(index=False))

best_model = results.loc[results['R2 Score'].idxmax()]
print(f"\nBest Model: {best_model['Model']}")
print(f"R2 Score of Best Model: {best_model['R2 Score']}")
