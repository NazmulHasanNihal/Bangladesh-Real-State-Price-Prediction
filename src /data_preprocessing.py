import pandas as pd

def clean_price(price_str):
    """Convert price string to float after removing currency symbol and commas."""
    return float(price_str.replace('৳', '').replace(',', ''))

def preprocess_data(df):
    """Perform data cleaning and preprocessing."""
    df['Price_in_taka'] = df['Price_in_taka'].apply(clean_price)

    df['Bedrooms'].fillna(df['Bedrooms'].median(), inplace=True)
    df['Bathrooms'].fillna(df['Bathrooms'].median(), inplace=True)
    df['Floor_no'].fillna(df['Floor_no'].median(), inplace=True)
    df['Floor_area'].fillna(df['Floor_area'].median(), inplace=True)

    df.dropna(subset=['City', 'Location'], inplace=True)

    return df

def main():

    df = pd.read_csv('/workspaces/Bangladesh-Real-State-Price-Prediction/data/raw/house_price_bd.csv')

    df_cleaned = preprocess_data(df)

    df_cleaned.to_csv('/workspaces/Bangladesh-Real-State-Price-Prediction/data/processed/cleaned_data.csv', index=False)
    print("Cleaned data saved to 'workspaces/Bangladesh-Real-State-Price-Prediction/data/processed/cleaned_data.csv'")

if __name__ == "__main__":
    main()
