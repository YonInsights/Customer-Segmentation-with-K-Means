import pandas as pd

def preprocess_data(data):
    # Remove percentage signs and convert to float
    data = data.replace('%', '', regex=True).astype(float)
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    
    # Fill numeric columns with the mean
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Fill non-numeric columns with the mode, if mode exists
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        if not data[col].mode().empty:
            data[col] = data[col].fillna(data[col].mode().iloc[0])
        else:
            data[col] = data[col].fillna('Unknown')  # or any other placeholder
    
    return data

if __name__ == "__main__":
    main_file_path = "D:/Osiri University/ML/Final Project 2/Customer-Segmentation-with-K-Means/data/DM_Sheet.csv"
    code_list_file_path = "D:/Osiri University/ML/Final Project 2/Customer-Segmentation-with-K-Means/data/CodelList.csv"
    
    main_data = pd.read_csv(main_file_path)
    
    print("Preprocessing Main Data:")
    main_data = preprocess_data(main_data)
    print(main_data.head())
