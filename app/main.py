import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import encode_categorical_features
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.prediction_service import load_model, predict

def main():
    # Load and preprocess data
    df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = preprocess_data(df)

    # Feature engineering
    customer_id = df['customerID']
    df_encoded = df.drop(columns=['customerID'])
    df_encoded = encode_categorical_features(df_encoded)

    # Split data
    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=221, stratify=y)

    smote = SMOTE(random_state=221)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train and evaluate model
    model = train_model(X_train_resampled, y_train_resampled)
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, 'models/churn_model.pkl')

    # Predict and export dashboard data
    y_pred, ypred_proba = predict(model, X_test)
    y_pred = pd.Series(y_pred, index=X_test.index)
    ypred_proba = pd.Series(ypred_proba, index=X_test.index)

    dashboard_x_test = df.loc[X_test.index]

    dashboard_df = pd.DataFrame({
        'Customer ID': dashboard_x_test['customerID'],
        'Time with Company (in Months)': dashboard_x_test['tenure'],
        'Contract Type': dashboard_x_test['Contract'],
        'Internet Service Provider': dashboard_x_test['InternetService'],
        'Total Charges': dashboard_x_test['TotalCharges'],
        'Predicted Churn': y_pred.map({1: 'Yes', 0: 'No'}),
        'Risk Level': ypred_proba.apply(
        lambda x: "3_High" if x >= 0.7 else "2_Medium" if x >= 0.4 else "1_Low"),
        'Predicted Probability': ypred_proba
    })
    print(dashboard_df.head())

    dashboard_df.to_csv('data/dashboard_data.csv', index=False)
    print("Dashboard data exported successfully.")

if __name__ == "__main__":
    main()