import pandas as pd
def convert_xlsx_to_csv(xlsx_file_path: str, csv_file_path: str) -> None:
    """
    Convert an Excel file to CSV format.

    Parameters:
    xlsx_file_path (str): The path to the input Excel file.
    csv_file_path (str): The path where the output CSV file will be saved.
    """
    # Read the Excel file into a DataFrame
    df = pd.read_excel(xlsx_file_path)
    
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)       
    
    
convert_xlsx_to_csv('maroc_1988_2024.xlsx', 'maroc_1988_2024.csv')