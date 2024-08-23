import pandas as pd

def create_features(df):
    """Create new features from existing data."""
    df['Price_per_sqft'] = df['Price_in_taka'] / df['Floor_area']

    df['Total_rooms'] = df['Bedrooms'] + df['Bathrooms']

    df['Is_high_floor'] = df['Floor_no'].apply(lambda x: 1 if x > 4 else 0)

    return df

def main():
    df = pd.read_csv('/workspaces/data/processed/cleaned_data.csv')

    df_with_features = create_features(df)

    df_with_features.to_csv('/workspaces/Bangladesh-Real-State-Price-Prediction/data/processed/data_with_features.csv', index=False)
    print("Data with new features saved to '/workspaces/Bangladesh-Real-State-Price-Prediction/data/processed/data_with_features.csv'")

if __name__ == "__main__":
    main()
