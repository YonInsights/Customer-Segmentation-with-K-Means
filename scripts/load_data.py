import pandas as pd

def load_data(main_file_path, code_list_file_path):
    # Load the main dataset
    main_data = pd.read_csv(main_file_path)
    
    # Load the code list dataset
    code_list_data = pd.read_csv(code_list_file_path)
    
    return main_data, code_list_data

if __name__ == "__main__":
    main_file_path = "D:/Osiri University/ML/Final Project 2/Customer-Segmentation-with-K-Means/data/DM_Sheet.csv"
    code_list_file_path = "D:/Osiri University/ML/Final Project 2/Customer-Segmentation-with-K-Means/data/CodelList.csv"
    
    main_data, code_list_data = load_data(main_file_path, code_list_file_path)
    
    print("Main Data:")
    print(main_data.head())
    
    print("\nCode List Data:")
    print(code_list_data.head())
