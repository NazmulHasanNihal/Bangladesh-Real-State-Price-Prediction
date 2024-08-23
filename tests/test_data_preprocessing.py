import pytest
import pandas as pd
from src.data_preprocessing import clean_price, clean_data, encode_categorical

def test_clean_price():
    assert clean_price('৳39,000,000') == 39000000.0
    assert clean_price('৳12,500,000') == 12500000.0
    assert clean_price('৳1,000,000') == 1000000.0

def test_clean_data():
    df = pd.DataFrame({
        'Bedrooms': [3, None, 2],
        'Bathrooms': [2, 3, None],
        'Price_in_taka': ['৳39,000,000', '৳12,500,000', '৳1,000,000']
    })
    df_cleaned = clean_data(df)
    assert df_cleaned['Bedrooms'].isnull().sum() == 0
    assert df_cleaned['Bathrooms'].isnull().sum() == 0
    assert df_cleaned['Price_in_taka'].dtype == float

def test_encode_categorical():
    df = pd.DataFrame({
        'City': ['dhaka', 'chattogram'],
        'Occupancy_status': ['vacant', 'occupied']
    })
    df_encoded = encode_categorical(df)
    assert 'City_chattogram' in df_encoded.columns
    assert df_encoded['Occupancy_status'].isin([0, 1]).all()
