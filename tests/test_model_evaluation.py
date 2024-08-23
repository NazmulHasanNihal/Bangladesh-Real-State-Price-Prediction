import pytest
import pandas as pd
from src.model_evaluation import evaluate_model, load_data, split_data
from src.utils import load_model

def test_evaluate_model():
    df = pd.DataFrame({
        'Floor_area': [1960, 1370, 2125],
        'Bedrooms': [3, 3, 3],
        'Bathrooms': [4, 3, 3],
        'Price_in_taka': [39000000, 12500000, 20000000]
    })
    X, y = split_data(df, 'Price_in_taka')
   
    model = load_model('workspaces/Bangladesh-Real-State-Price-Prediction/models/linear_regression.pkl')
    
    rmse, r2 = evaluate_model(model, X, y)
    
    assert rmse >= 0  
    assert 0 <= r2 <= 1  
