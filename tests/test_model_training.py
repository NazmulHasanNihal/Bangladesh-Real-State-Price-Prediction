import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.model_training import train_model

def test_train_model():
    df = pd.DataFrame({
        'Floor_area': [1960, 1370, 2125, 2687],
        'Bedrooms': [3, 3, 3, 3],
        'Bathrooms': [4, 3, 3, 3],
        'Price_in_taka': [39000000, 12500000, 20000000, 47500000]
    })
    X = df.drop('Price_in_taka', axis=1)
    y = df['Price_in_taka']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)
    
    model = train_model(RandomForestRegressor(random_state=42), X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    assert rmse >= 0  # RMSE should always be non-negative
    assert isinstance(model, RandomForestRegressor)
