import pandas as pd
import os
import numpy as np

fuel_file = r"G:\MO-PLM\Transfer\Open Sky Data Challenge\Fuel Consumption\fuel_train.csv"
data_dir = r"./data"
output_dir = r"./data"
os.makedirs(output_dir, exist_ok=True)

TRAIN_DATA = True
metadata = pd.read_csv(fuel_file)
metadata = metadata[:100]

processed_count = 0
skipped_count = 0
flight_counter = {}

def fill_feature_dict(feature_dict, snippet_df):
    if snippet_df.empty:
        feature_dict["total_data_points"] = np.nan
        feature_dict["mean_altitude"] = np.nan
        feature_dict["min_altitude"] = np.nan
        feature_dict["max_altitude"] = np.nan
        feature_dict["mean_groundspeed"] = np.nan
        feature_dict["mean_roc"] = np.nan
        feature_dict["mean_longitude"] = np.nan
        feature_dict["mean_latitude"] = np.nan
        feature_dict["mean_track"] = np.nan
        feature_dict["mean_track_45"] = np.nan
        feature_dict["mean_track_90"] = np.nan
        feature_dict["airframe"] = np.nan
        feature_dict["GDT_total_start"] = np.nan
        feature_dict["CL"] = np.nan
        feature_dict["CR"] = np.nan
        feature_dict["DE"] = np.nan
        feature_dict["LVL"] = np.nan
        feature_dict["GND"] = np.nan
    else:
        feature_dict["total_data_points"] = extracted.shape[0]
        feature_dict["mean_altitude"] = extracted['altitude'].mean()
        feature_dict["min_altitude"] = extracted['altitude'].min()
        feature_dict["max_altitude"] = extracted['altitude'].min()
        feature_dict["mean_groundspeed"] = extracted['groundspeed'].mean()
        feature_dict["mean_roc"] = extracted['vertical_rate'].mean()
        feature_dict["mean_longitude"] =  extracted['longitude'].mean()
        feature_dict["mean_latitude"] =  extracted['latitude'].mean()
        feature_dict["mean_track"] = extracted['track'].mean()
        feature_dict["mean_track_45"] = 45 * int(extracted['track'].mean() / 45)
        feature_dict["mean_track_90"] = 90 * int(extracted['track'].mean() / 90)
        feature_dict["airframe"] =  extracted['typecode'].values[0]
        feature_dict["GDT_total_start"] =  extracted['GTD_total'].min()

        phase_counts = extracted["flight_phase"].value_counts()
        if "CL" in phase_counts.keys():
            meta_info_dict["CL"] = phase_counts["CL"] / extracted.shape[0]
        else:
            meta_info_dict["CL"] = 0
        if "CR" in phase_counts.keys():
            meta_info_dict["CR"] = phase_counts["CR"] / extracted.shape[0]
        else:
            meta_info_dict["CR"] = 0
        if "DE" in phase_counts.keys():
            meta_info_dict["DE"] = phase_counts["DE"] / extracted.shape[0]
        else:
            meta_info_dict["DE"] = 0
        if "LVL" in phase_counts.keys():
            meta_info_dict["LVL"] = phase_counts["LVL"] / extracted.shape[0]
        else:
            meta_info_dict["LVL"] = 0
        if "GND" in phase_counts.keys(): # GND is kind of taxi
            meta_info_dict["GND"] = phase_counts["GND"] / extracted.shape[0]
        else:
            meta_info_dict["GND"] = 0

    return feature_dict

# allowable tolerance (in seconds)
if __name__ == "__main__":
    TOLERANCE = pd.Timedelta(seconds=1)
    meta_info = []
    metadata['start'] = pd.to_datetime(metadata['start'], errors='coerce')
    metadata['end'] = pd.to_datetime(metadata['end'], errors='coerce')
    metadata['duration'] = (metadata['end'] - metadata['start']).dt.total_seconds()
    metadata["fuel_per_second"] = metadata['fuel_kg'] / metadata['duration']
    fuel_99th_p = metadata["fuel_per_second"].quantile(0.99)
    if TRAIN_DATA:
        start_rows = metadata.shape[0]
        metadata = metadata[metadata["fuel_per_second"] < fuel_99th_p]
        print(f"Excluding {start_rows - metadata.shape[0]} snippets "
              f"with fuel consumption more than {fuel_99th_p} kg per second")

    for _, row in metadata.iterrows():
        flight_number = str(row['flight_id']).strip()
        start_time = row['start']
        end_time = row['end']
        fuel_kg = row['fuel_kg']


        meta_info_dict = {
            "flight_id": flight_number,
            "start_time": start_time,
            "end_time": end_time,
            "total_time": row['duration'],
            "fuel_kg": fuel_kg
        }

        file_path = os.path.join(data_dir, f"{flight_number}_processed.csv")
        if not os.path.isfile(file_path):
            print(f"Skipping {flight_number} as file was not found")
            meta_info_dict["GTD_total_flight"] = np.nan
            meta_info_dict["GTD_total_start"] = fill_feature_dict(meta_info_dict, pd.DataFrame())
            skipped_count += 1
            if TRAIN_DATA:
                continue

        else:
            df = pd.read_csv(file_path)
            # time_col = df.columns[0]
            time_col = 'timestamp'
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            meta_info_dict["GTD_total_flight"] = df["GTD_total"].max()

            if df[time_col].isna().all():
                print(f"Skipping {flight_number} time column is all nans")
                meta_info_dict = fill_feature_dict(meta_info_dict, pd.DataFrame())
                skipped_count += 1
                if TRAIN_DATA:
                    continue
            else:
                # Keep only rows within [start_time - tol, end_time + tol]
                mask = (df[time_col] >= start_time - TOLERANCE) & (df[time_col] <= end_time + TOLERANCE)
                extracted = df.loc[mask].copy()

                # If nothing falls within that window, skip
                if extracted.empty:
                    print(f"Skipping {flight_number} snippet at {start_time} no data was found")
                    meta_info_dict = fill_feature_dict(meta_info_dict, pd.DataFrame())
                    skipped_count += 1
                    if TRAIN_DATA:
                        continue
                else:
                    # Increment suffix for same flight
                    flight_counter[flight_number] = flight_counter.get(flight_number, 0) + 1
                    suffix = flight_counter[flight_number]

                    output_path = os.path.join(output_dir, f"{flight_number}_{suffix}.parquet")
                    extracted.to_parquet(output_path, index=False)

                    fill_feature_dict(meta_info_dict, extracted)
                    # print(f"Processed {flight_number} snippet at {start_time}")
                    processed_count += 1

        meta_info.append(meta_info_dict)


    print(f"Processed {processed_count} flights, skipped {skipped_count}")
    meta_info_df = pd.DataFrame(meta_info)
    meta_info_df.to_csv("./labeled_data_distribution.csv", index=False)
