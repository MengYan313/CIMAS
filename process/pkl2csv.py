import pandas as pd

file_name = 'telephony_core_service'

# Input and output file paths
pkl_file_path = '../files/' + file_name + '.pkl'  # Replace with your .pkl file path
excel_file_path = '../files/' + file_name + '.csv'  # Output .csv file path

# Read .pkl file
try:
    print("Loading .pkl file...")
    data = pd.read_pickle(pkl_file_path)  # Load data
    print("Successfully loaded .pkl file, data type:", type(data))
    print(data.head())
    print(data.shape)
    if "pros_name" in data.columns: # Output all contents of the `pros_name` column
        print("\n".join(map(str, data["pros_name"].tolist())))

    data.to_csv(excel_file_path, index=False)  # Save as csv
    print(f"Data has been successfully converted to .csv format, saved at: {excel_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# nohup python3 pkl2csv.py > pkl2csv.log 2>&1 &
# 2680846