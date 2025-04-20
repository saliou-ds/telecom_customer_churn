from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = LogisticRegression(
    penalty='l2',          # Use L2 regularization (Ridge)
    random_state=221,      # For reproducibility
    max_iter=1000,          # Increase iterations if needed
    solver='lbfgs'        # Solver that supports L2 regularization
)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """Save the trained model to a file."""
    joblib.dump(model, filepath)
    
if __name__ == "__main__":
    train_model(any, any)
    save_model(any, any)