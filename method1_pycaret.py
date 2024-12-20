import pandas as pd
from pycaret.regression import *

data = pd.read_csv('Fuel_cell_performance_data-Full.csv')

s = setup(data=data, target='Target4', train_size=0.7)

print("\nComparing All Models:")
best_model = compare_models()

print(f"\nBest Model: {best_model}")
print(f"R2 Score of Best Model: {pull().iloc[0]['R2']}")
