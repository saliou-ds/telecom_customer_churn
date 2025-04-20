from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (Churn = 1)
    threshold = 0.4
    y_pred_custom = (y_pred_proba >= threshold).astype(int) # Custom threshold for classification

    print(f"Custom threshold: {threshold}")
    print("Classification Report:\n", classification_report(y_test, y_pred_custom, target_names=['No Churn (0)', 'Churn (1)']))
    print(f"Confusion Matrix:\n{pd.DataFrame(confusion_matrix(y_test, y_pred_custom), index=['Actual:0', 'Actual:1'], columns=['Predicted:0', 'Predicted:1'])}")


    print(f"\nValidation ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}")

if __name__ == "__main__":
    evaluate_model(any, any, any) 