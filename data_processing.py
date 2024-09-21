import pandas as pd

def load_data(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        return pd.read_csv(file)
    elif file_extension == 'xlsx':
        return pd.read_excel(file)
    elif file_extension == 'json':
        return pd.read_json(file)
    else:
        raise ValueError("Unsupported file format")

def clean_data(df):
    # Drop rows with missing values
    df_cleaned = df.dropna()

    # Identify numeric columns
    numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    df_cleaned = df_cleaned[numeric_columns]

    return df_cleaned
