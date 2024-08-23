import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using RMSE and R² metrics."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def split_data(df, target_column):
    """Split the data into features and target."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def main():
    df = load_data('/workspaces/Bangladesh-Real-State-Price-Prediction/data/processed/data_with_features.csv')

    X, y = split_data(df, 'Price_in_taka')

    model_files = {
        'Random Forest': '/workspaces/Bangladesh-Real-State-Price-Prediction/models/random_forest.pkl',
        'Gradient Boosting': '/workspaces/Bangladesh-Real-State-Price-Prediction/models/gradient_boosting.pkl'
    }

    results = []

    for name, model_file in model_files.items():
        model = load_model(model_file)
        rmse, r2 = evaluate_model(model, X, y)
        print(f"{name} Evaluation:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}\n")
        results.append({
            'Model': name,
            'RMSE': rmse,
            'R²': r2
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('/workspaces/Bangladesh-Real-State-Price-Prediction/reports/tables/model_performance.csv', index=False)
    print("Model performance metrics saved to '/workspaces/Bangladesh-Real-State-Price-Prediction/reports/tables/model_performance.csv'")

if __name__ == "__main__":
    main()
