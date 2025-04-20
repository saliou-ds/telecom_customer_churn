import joblib

def load_model(filepath):
    """Load a trained model from a file."""
    return joblib.load(filepath)

def predict(model, X):
    """Make predictions using the trained model."""
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probabilities for the positive class (Churn = 1)
    threshold = 0.4
    y_pred_custom = (y_pred_proba >= threshold).astype(int) # Custom threshold for classification

    return y_pred_custom, y_pred_proba

if __name__ == "__main__":
    load_model(any)
    predict(any, any)