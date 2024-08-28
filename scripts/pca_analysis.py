import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from load_data import load_data
from preprocess_data import preprocess_data

def standardize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return principal_components, pca

def plot_explained_variance(pca):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
    plt.title('Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.show()

def integrate_pca_results(data, principal_components):
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    integrated_data = pd.concat([data.reset_index(drop=True), pca_df], axis=1)
    return integrated_data

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
    
    # Plot the explained variance
    plot_explained_variance(pca)
    
    # Integrate PCA results into the DataFrame
    main_data_with_pca = integrate_pca_results(main_data, principal_components)
    print(main_data_with_pca.head())
