import pandas as pd
from pathlib import Path
import sys
sys.path.extend([str(Path(__file__).resolve().parent.parent / "utils")])
sys.path.extend([str(Path(__file__).resolve().parent.parent / "gradient_boost")])
from OpenSky_distances_gaps import process_file
from add_airframe_info import add_airframe_info
from features import get_features
from use_models import get_fuel_consumption
import numpy as np
import traceback
from tqdm import tqdm

from sklearn.metrics import root_mean_squared_error

input_data_file = "G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\Fuel Consumption\\fuel_rank_submission.csv"
flight_trajectories_folder = Path("G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\final_rank_processed_interpolation")

# input_data_file = Path("G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\rank1_interpolation_19features.csv")

time_col = "timestamp"
TOLERANCE = pd.Timedelta(seconds=1)
linear_parameter = 0.8972

models = ["CR_light", "CR_heavy", "CL_light", "CL_heavy", "DE_light", "DE_heavy"]

input_df = pd.read_csv(input_data_file)
input_df["used_min_fuel"] = False
input_df["out_of_trajectory"] = False
input_df["model_used"] = ""

# prob_idxs = [
# 51270, 121031, 38932, 12037, 3543, 20750, 114922, 7602, 6971, 89391, 91561, 46773, 1873, 47004
# ]
# prob_idxs = [
# 51270, 121031
# ]

# input_df = input_df[input_df['idx'].isin(prob_idxs)]

# input_df = input_df[input_df["flight_id"].isin(['prc788791392', 'prc778012785', 'prc775029188'])]
predicted_fuel_consumption = []
rmses = []

for flight_id, flight_input_df in tqdm(input_df.groupby("flight_id"), total=len(input_df)):
    # load right flight from flight_id, fill gaps and add GTD and flight phases as columns
    # trajectory_df_original = process_file(flight_trajectories_folder / f"{flight_id}.parquet", interpolation=True)
    trajectory_df_original = pd.read_csv(flight_trajectories_folder / f"{flight_id}_processed.csv")

    # TODO find better way to fill gaps during preprocessing
    no_na_trajectory_df = trajectory_df_original.dropna(subset=["altitude", "groundspeed", "vertical_rate", "GTD_total"])
    trajectory_df = add_airframe_info(no_na_trajectory_df, airframe_col="typecode")
    if trajectory_df.empty:
        trajectory_df = no_na_trajectory_df
        airframe_class = ""
    else:
        trajectory_df[time_col] = pd.to_datetime(trajectory_df[time_col])
        if not "MZFW" in trajectory_df.columns:
            airframe_class = ""
        elif trajectory_df['MZFW'].values[0]> 100:
            airframe_class = "heavy"
        else:
            airframe_class = "light"
        # airframe_class = trajectory_df["age_class"].values[0]
    # airframe_class = ""

    for i, row in flight_input_df.iterrows():
        # print(f"\nFlight {flight_id} at {row['start']} until {row['end']}")
        fuel_burned = 0

        # get snippet based on input start and end time
        start_time = pd.to_datetime(row["start"])
        end_time = pd.to_datetime(row["end"])
        mask = (trajectory_df[time_col] >= start_time - TOLERANCE) & (trajectory_df[time_col] <= end_time + TOLERANCE)
        snippet_df = trajectory_df[mask]

        # give fuel for time without trajectory
        if snippet_df.empty:
            out_of_trajectory_seconds = (end_time - start_time).total_seconds()
            idling_consumption = 0.25
            fuel_burned = fuel_burned + idling_consumption * out_of_trajectory_seconds
            input_df.loc[i, "out_of_trajectory"] = True

        else:
            before_trajectory_seconds = max(0, (snippet_df[time_col].min() - start_time).total_seconds())
            after_trajectory_seconds = max(0, (end_time - snippet_df[time_col].max()).total_seconds())
            out_of_trajectory_seconds = before_trajectory_seconds + after_trajectory_seconds

            if out_of_trajectory_seconds > 0:
                # TODO get different idling for different airframes (and conditions?)
                idling_consumption = 0.25
                fuel_burned = fuel_burned + idling_consumption * out_of_trajectory_seconds
                input_df.loc[i, "out_of_trajectory"] = True

            phases = snippet_df['flight_phase'].unique()
            phase_change_idxs = []
            if len(phases) > 1:
                phase_change = snippet_df['flight_phase'].ne(snippet_df['flight_phase'].shift().bfill()).astype(int)
                phase_change_idxs = phase_change[phase_change == 1].index.tolist()
            phase_change_idxs.append(snippet_df.index[-1]+1)
            prev_phase_idx = snippet_df.index[0]

            for phase_idx in phase_change_idxs:
                phase_df = snippet_df[(snippet_df.index >= prev_phase_idx) & (snippet_df.index < phase_idx)]
                prev_phase_idx = phase_idx
                phase = phase_df["flight_phase"].values[0]

                min_fuel = (phase_df.index.max() - phase_df.index.min()) * 0.09 #1st percentile of train data
                time_fuel = (phase_df.index.max() - phase_df.index.min()) * linear_parameter
                if phase in ["CL", "CR", "DE"]:
                    model_name = f"{phase}_{airframe_class}"
                elif phase == "LVL":
                    model_name = f"CR_{airframe_class}"
                else:
                    model_name = ""
                if model_name in models:
                    try:
                        input = get_features(model_name, phase_df, trajectory_df)
                        prediction = get_fuel_consumption(model_name, input)
                        if prediction < min_fuel:
                            input_df.loc[i, "used_min_fuel"] = True
                            fuel_from_input = time_fuel
                        else:
                            fuel_from_input = prediction
                            input_df.loc[i, "model_used"] = model_name
                        fuel_burned = fuel_burned + fuel_from_input
                    except Exception:
                        print(f"Could not estimate fuel burn because of:")
                        print(traceback.print_exc())
                        fuel_burned = fuel_burned + time_fuel
                        input_df.loc[i, "used_min_fuel"] = True
                else:
                    try:
                        input = get_features("UNKNOWN", phase_df, trajectory_df)
                        prediction = get_fuel_consumption("UNKNOWN", input)
                        if prediction < min_fuel:
                            input_df.loc[i, "used_min_fuel"] = True
                            fuel_from_input = time_fuel
                        else:
                            fuel_from_input = prediction
                            input_df.loc[i, "model_used"] = "UNKNOWN"
                        fuel_burned = fuel_burned + fuel_from_input
                    except Exception:
                        print(f"Could not estimate fuel burn because of:")
                        print(traceback.print_exc())
                        fuel_burned = fuel_burned + time_fuel
                        input_df.loc[i, "used_min_fuel"] = True


        predicted_fuel_consumption.append(fuel_burned)

        if fuel_burned == 0.0:
            print("Problem, no fuel consumption")

        # print(f"{fuel_burned:.2f} kg fuel predicted")
        if not np.isnan(row["fuel_kg"]):
            print(f"{row['fuel_kg']} fuel burned")
            rmse = root_mean_squared_error(np.array(row["fuel_kg"]).reshape(-1, 1), np.array(fuel_burned).reshape(-1, 1))
            rmses.append(rmse)
            print("The root mean squared error (RMSE) on validation set: {:.2f}".format(rmse))

input_df["predicted_fuel_kg"] = predicted_fuel_consumption
if len(rmses) > 0:
    print("The average RMSE on validation set: {:.2f}".format(np.mean(rmses)))

input_df.to_csv("single_phase_predictions_rank1.csv", index=False)