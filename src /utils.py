import pickle

def save_model(model, file_path):
    """Save the trained model to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """Load a model from a file."""
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model
