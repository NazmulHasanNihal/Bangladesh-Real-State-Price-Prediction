# tests/test_feature_engineering.py

import pytest
import pandas as pd
from src.feature_engineering import create_features

def test_create_features():
    df = pd.DataFrame({
        'Price_in_taka': [39000000, 12500000],
        'Floor_area': [1960, 1370],
        'Bedrooms': [3, 3],
        'Bathrooms': [4, 3],
        'Floor_no': [3, 6]
    })
    df_with_features = create_features(df)
    assert 'Price_per_sqft' in df_with_features.columns
    assert 'Total_rooms' in df_with_features.columns
    assert 'Is_high_floor' in df_with_features.columns
    assert df_with_features['Price_per_sqft'].iloc[0] == 19897.96  # 39000000 / 1960
    assert df_with_features['Total_rooms'].iloc[1] == 6  # 3 + 3
    assert df_with_features['Is_high_floor'].iloc[1] == 1  # 6 > 4
