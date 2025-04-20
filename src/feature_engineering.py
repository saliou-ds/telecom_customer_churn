import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_categorical_features(df):
    """One-hot encode categorical features."""

    df_target_var = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.drop(columns=['Churn'])

    categorical_data = df.select_dtypes(include=['category'])
    categorical_data = categorical_data.astype(str)

    encoder = OneHotEncoder(sparse_output=False, drop='first') #sparse_output false to have a numpy array as output & drop first to avoid multicollinearity
    
    
    df_encoded = encoder.fit_transform(categorical_data)

    # Get feature/variables names after encoding
    categorical_variables_names = encoder.get_feature_names_out(categorical_data.columns)

    # Convert to DataFrame for better visualization
    df_encoded = pd.DataFrame(df_encoded, columns=categorical_variables_names, index=categorical_data.index)

    # Concatenate the encoded DataFrame with the numerical variables from the original DataFrame
    numerical_data = df.select_dtypes(exclude=['category'])
    df_encoded = pd.concat([numerical_data, df_encoded, df_target_var], axis=1)

    return df_encoded

if __name__ == "__main__":
    encode_categorical_features(any)