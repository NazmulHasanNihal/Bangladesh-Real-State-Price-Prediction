import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import save_model

def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def split_data(df, target_column):
    """Split the data into training and testing sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Train multiple machine learning models and return them."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"{name} training complete.")
    
    return models

def main():
    df = load_data('/workspaces/Bangladesh-Real-State-Price-Prediction/data/processed/data_with_features.csv')

    X_train, X_test, y_train, y_test = split_data(df, 'Price_in_taka')

    models = train_models(X_train, y_train)

    for name, model in models.items():
        save_model(model, f'/workspaces/Bangladesh-Real-State-Price-Prediction/models/{name.replace(" ", "_").lower()}.pkl')
        print(f"{name} saved successfully.")

if __name__ == "__main__":
    main()
