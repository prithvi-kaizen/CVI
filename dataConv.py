import os
import numpy as np
import pandas as pd
from scipy.io import arff  # For ARFF files

def load_data(file_path, delimiter=None):

    # file ext.
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.arff':
        data, _ = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        # byte strings to regular strings if necessary.
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    elif ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.txt':
        # Use the provided delimiter if available; otherwise, assume whitespace.
        if delimiter is None:
            df = pd.read_csv(file_path, delim_whitespace=True)
        else:
            df = pd.read_csv(file_path, delimiter=delimiter)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"wrong file format: {ext}")

    #only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        raise ValueError("No numeric data found")
    #transpose data
    data_array = df_numeric.to_numpy().T
    return data_array

if __name__ == "__main__":

    file_path = "data.csv" #file_name
    try:
        data = load_data(file_path)
        print("Data shape (n_features, n_samples):", data.shape)
        print("Data:")
        print(data)
    except Exception as e:
        print("Error loading data:", e)
