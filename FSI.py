import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def load_data(file_path, delimiter=None):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.arff':
        data, _ = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    elif ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.txt':
        if delimiter is None:
            df = pd.read_csv(file_path, sep=r'\s+')
        else:
            df = pd.read_csv(file_path, delimiter=delimiter)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        raise ValueError("No numeric data found!")
    
    # missing value replaced by mean
    imputer = SimpleImputer(strategy='mean')
    df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
    
    return df_numeric.to_numpy().T

def compute_fsi(data, centers, membership, m=2):
    n_clusters = centers.shape[0]

    # Calculate overall mean of the dataset
    overall_mean = np.mean(data, axis=1)
    
    # First term: weighted sum of squared distances of each point to its cluster center
    term1 = 0.0
    for j in range(n_clusters):
        diff = data.T - centers[j]
        distances_sq = np.sum(diff**2, axis=1)
        term1 += np.sum(membership[j, :]**m * distances_sq)
    
    # Second term: sum of squared distances between each cluster center and the overall mean
    term2 = 0.0
    for j in range(n_clusters):
        diff = centers[j] - overall_mean
        term2 += np.sum(diff**2)
    
    fsi = term1 - term2
    return fsi

def apply_fcm(data, start_cluster, end_cluster):
    # data scaled
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.T).T  
    
    for n_clusters in range(start_cluster, end_cluster + 1):
        # apply FCM clustering
        centers, membership, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data_scaled, n_clusters, m=2, error=0.005, maxiter=1000
        )
        
        # fsi computing
        fsi_value = compute_fsi(data_scaled, centers, membership, m=2)
        print(f"Clusters: {n_clusters} | FPC: {fpc:.2f} | FSI: {fsi_value:.2f}")

if __name__ == "__main__":
    file_path = "s1.txt"  #file 
    start_cluster_num = 2
    end_cluster_num = 20

    try:
        data = load_data(file_path)
        print(f"Data shape (n_features, n_samples): {data.shape}")
        apply_fcm(data, start_cluster_num, end_cluster_num)
    except Exception as e:
        print(f"Error: {e}")