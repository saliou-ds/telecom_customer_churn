import pandas as pd

def load_data(filepath):

    column_types = {
	"customerID": pd.StringDtype(),
	"gender": "category",
	"SeniorCitizen": "category",
	"Partner": "category",
	"Dependents": "category",
	"tenure": "int64",
	"PhoneService": "category",
	"MultipleLines": "category",
	"InternetService": "category",
	"OnlineSecurity": "category",
	"OnlineBackup": "category",
	"DeviceProtection": "category",
	"TechSupport": "category",
	"StreamingTV": "category",
	"StreamingMovies": "category",
	"Contract": "category",
	"PaperlessBilling": "category",
	"PaymentMethod": "category",
	"MonthlyCharges": "float64",
	"TotalCharges": "float64",
	"Churn": "category"
}
    return pd.read_csv(filepath, dtype =column_types, na_values=[' '])

def preprocess_data(df):
    df = df.loc[df['tenure'] != 0]
    return df

if __name__ == "__main__":
    load_data(any)