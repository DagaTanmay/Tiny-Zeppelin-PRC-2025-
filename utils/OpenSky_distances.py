import os
import glob
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


INPUT_FOLDER = r"C:\OpenSky\CSV"
OUTPUT_FOLDER = r"C:\OpenSky\test_files"


# distance inbetween two coordinates (to calculate GTD)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # R_E in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in km


# check for output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# find all cdv data
all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
if not all_files:
    print("No .csv files found")
    exit()

# process every .csv file
for file in all_files:
    filename = os.path.basename(file)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"{name}_processed.csv")

    print(f"Processed file: {filename} ...")

    df = pd.read_csv(file)

    # sort for time by flight_id (already the case)
    df = df.sort_values(by=["flight_id", "timestamp"]).reset_index(drop=True)

    # add columns
    df["GTD"] = 0.0
    df["GTD_total"] = 0.0
    df["GCD_total"] = np.nan

    # distance per flight
    for flight_id, group in df.groupby("flight_id"):
        idx = group.index
        lats = group["latitude"].values
        lons = group["longitude"].values

        gtd = [0.0]
        for i in range(1, len(group)):
            dist = haversine(lats[i-1], lons[i-1], lats[i], lons[i])
            gtd.append(dist)

        gtd_total = np.cumsum(gtd)
        df.loc[idx, "GTD"] = gtd
        df.loc[idx, "GTD_total"] = gtd_total

        # GCD from first to last datapoint
        if len(group) > 1:
            gcd = haversine(lats[0], lons[0], lats[-1], lons[-1])
            df.loc[idx[-1], "GCD_total"] = gcd

    # Datei speichern
    df.to_csv(output_path, index=False)
    print(f"Done : {output_path}")

print("All data processed.")
