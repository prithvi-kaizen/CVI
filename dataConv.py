import os
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from sklearn.impute import SimpleImputer


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

    # handling categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols)

    # numeric data only
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        raise ValueError("No numeric data available after preprocessing.")
    
    # missing values replaced with mean
    imputer = SimpleImputer(strategy='mean')
    df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    # data return 
    return df_numeric.to_numpy().T


def apply_fcm(data, start_cluster, end_cluster):
    """FCM clustering & visualize"""
    # scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.T).T  # keeping (n_features, n_samples)

    for n_clusters in range(start_cluster, end_cluster + 1):
        # running fcm
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            # can change the values
            data_scaled, n_clusters, m=2, error=0.005, maxiter=1000
        )
        cluster_labels = np.argmax(u, axis=0)
        plot_data(data_scaled.T, cluster_labels, cntr, n_clusters, fpc)

def plot_data(data, labels, centers, n_clusters, fpc):
    """Visualize clusters using PCA if needed."""
    plt.figure(figsize=(10, 6))
    
    if data.shape[1] == 2:
        # 2D plotting
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', edgecolor='k')
    else:
        # high dim data 
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)  # (n_samples, 2)
        centers_pca = pca.transform(centers)  # (n_clusters, 2)
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='X', s=200, c='red', edgecolor='k')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
    
    plt.title(f'clusters = {n_clusters}, FPC: {fpc:.2f}')
    plt.colorbar(label='Cluster')
    plt.show()

if __name__ == "__main__":
    # change according to the need
    file_path = "google_review_ratings.csv"  # file path
    start_cluster_num = 2
    end_cluster_num = 5

    try:
        data = load_data(file_path)
        print(f"data shape (n_features, n_samples): {data.shape}")
        apply_fcm(data, start_cluster_num, end_cluster_num)
    except Exception as e:
        print(f"Error: {e}")