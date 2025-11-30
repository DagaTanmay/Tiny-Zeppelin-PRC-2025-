import os
import pandas as pd

# paths
input_folder = r"C:\OpenSky\Parquet"
output_folder = r"C:\OpenSky\CSV"

# create folder if not existing
os.makedirs(output_folder, exist_ok=True)

# go through all prq data
for file in os.listdir(input_folder):
    if file.endswith(".parquet"):
        parquet_path = os.path.join(input_folder, file)
        csv_name = os.path.splitext(file)[0] + ".csv"
        csv_path = os.path.join(output_folder, csv_name)

        print(f"Verarbeite {file} ...")

        # read parquet
        df = pd.read_parquet(parquet_path)

        # save all csv
        df.to_csv(csv_path, index=False)

        print(f"Saved as {csv_path}")

print("All data saved as .csv")