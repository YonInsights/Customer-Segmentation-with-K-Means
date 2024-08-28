import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from load_data import load_data
from preprocess_data import preprocess_data
from pca_analysis import standardize_data, apply_pca

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        print(f'Fit {k} clusters')

    plt.figure(figsize=(8, 6))
    plt.plot(iters, sse, marker='o')
    plt.xlabel('Cluster Centers')
    plt.ylabel('SSE')
    plt.title('SSE by Cluster Center Plot')
    plt.show()

def evaluate_clusters(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(data, labels)
    print(f'Silhouette Score for {n_clusters} clusters: {silhouette_avg}')
    return kmeans, silhouette_avg

if __name__ == "__main__":
    main_file_path = "D:/Osiri University/ML/Final Project 2/Customer-Segmentation-with-K-Means/data/DM_Sheet.csv"
    code_list_file_path = "D:/Osiri University/ML/Final Project 2/Customer-Segmentation-with-K-Means/data/CodelList.csv"
    
    main_data, code_list_data = load_data(main_file_path, code_list_file_path)
    
    print("Preprocessing Main Data:")
    main_data = preprocess_data(main_data)
    print(main_data.head())
    
    # Standardize the main data (excluding non-numeric columns)
    numeric_cols = main_data.select_dtypes(include=['number']).columns
    scaled_data = standardize_data(main_data[numeric_cols])
    
    # Apply PCA to reduce to 2 components for visualization
    principal_components, pca = apply_pca(scaled_data, n_components=2)
    
    # Find the optimal number of clusters
    find_optimal_clusters(principal_components, 10)
    
    # Evaluate clusters with an example number of clusters
    kmeans, silhouette_avg = evaluate_clusters(principal_components, 3)
    print(f'K-Means model with 3 clusters: {kmeans}')


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from load_data import load_data
from preprocess_data import preprocess_data
from pca_analysis import standardize_data, apply_pca

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        print(f'Fit {k} clusters')

    plt.figure(figsize=(8, 6))
    plt.plot(iters, sse, marker='o')
    plt.xlabel('Cluster Centers')
    plt.ylabel('SSE')
    plt.title('SSE by Cluster Center Plot')
    plt.show()

def evaluate_clusters(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(data, labels)
    print(f'Silhouette Score for {n_clusters} clusters: {silhouette_avg}')
    return kmeans, silhouette_avg, labels

def plot_clusters(data, labels, n_clusters):
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clusters Visualization')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main_file_path = "D:/Osiri University/ML/Final Project 2/Customer-Segmentation-with-K-Means/data/DM_Sheet.csv"
    code_list_file_path = "D:/Osiri University/ML/Final Project 2/Customer-Segmentation-with-K-Means/data/CodelList.csv"
    
    main_data, code_list_data = load_data(main_file_path, code_list_file_path)
    
    print("Preprocessing Main Data:")
    main_data = preprocess_data(main_data)
    print(main_data.head())
    
    # Standardize the main data (excluding non-numeric columns)
    numeric_cols = main_data.select_dtypes(include=['number']).columns
    scaled_data = standardize_data(main_data[numeric_cols])
    
    # Apply PCA to reduce to 2 components for visualization
    principal_components, pca = apply_pca(scaled_data, n_components=2)
    
    # Find the optimal number of clusters
    find_optimal_clusters(principal_components, 10)
    
    # Evaluate clusters with the chosen number of clusters (e.g., 3)
    kmeans, silhouette_avg, labels = evaluate_clusters(principal_components, 3)
    print(f'K-Means model with 3 clusters: {kmeans}')
    
    # Plot the clusters
    plot_clusters(principal_components, labels, 3)
    
    # Integrate cluster labels into the original data
    main_data['Cluster'] = labels
    print(main_data.head())
