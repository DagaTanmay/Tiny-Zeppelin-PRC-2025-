import os
import glob
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from flightphase import fuzzylabels

INPUT_FOLDER = r"G:\MO-PLM\Transfer\Open Sky Data Challenge\flights_train"
OUTPUT_FOLDER = r"data\train_no_interpolation"

# ===============================================================
# Helper functions for filling missing values (NaN gaps)
# ===============================================================

def _fill_gap_int(a, b, gap_len):
    """
    Fill a gap of length `gap_len` between two integer values a (before) and b (after)
    linearly and proportionally with whole numbers. => for groundspeed and vertical_rate
    """
    a = int(round(a))
    b = int(round(b))
    if gap_len <= 0:
        return []

    # if both are the same, fill with same value
    if a == b:
        return [a] * gap_len

    inc = 1 if b > a else -1
    steps = abs(b - a)  # number of step levels between a and b

    # Distribute the gap length evenly across each step level
    base_run = gap_len // steps
    remainder = gap_len % steps

    filled = []
    for k in range(1, steps + 1):
        val = a + inc * k
        count = base_run + (1 if k <= remainder else 0)
        if count > 0:
            filled.extend([val] * count)

    # Safety correction
    if len(filled) > gap_len:
        filled = filled[:gap_len]
    elif len(filled) < gap_len:
        # pad with last value if slightly short
        if filled:
            filled.extend([filled[-1]] * (gap_len - len(filled)))
        else:
            filled = [a] * gap_len

    return filled


def _fill_gap_float(a, b, gap_len):
    """
    Fill a gap of length `gap_len` between two numbers a (before) and b (after)
    linearly with floating-point numbers (excluding endpoints). => for track
    """
    if gap_len <= 0:
        return []
    arr = np.linspace(a, b, gap_len + 2)[1:-1]
    return arr.tolist()


def fill_gaps_series(series: pd.Series, mode: str = "int", fill_edges: bool = True) -> pd.Series:

    values = series.values.copy()
    n = len(values)
    if n == 0:
        return series

 #Fill internal gaps (bounded on both sides)
    i = 0
    while i < n:
        if not pd.isna(values[i]):
            i += 1
            continue

        # found start of a gap
        start = i
        while i < n and pd.isna(values[i]):
            i += 1
        end = i  # first index after the gap

        left_idx = start - 1
        right_idx = end if end < n else None

        # Only fill if there are valid values on both sides
        if left_idx >= 0 and right_idx is not None and not pd.isna(values[left_idx]) and not pd.isna(values[right_idx]):
            a = values[left_idx]
            b = values[right_idx]
            gap_len = end - start

            if mode == "int":
                filled = _fill_gap_int(a, b, gap_len)
            else:
                filled = _fill_gap_float(a, b, gap_len)

            values[start:end] = filled


    #Fill leading/trailing edges
    if fill_edges:
        if np.all(pd.isna(values)):
            return pd.Series(values, index=series.index)

        # Fill leading NaNs with first valid value
        first_valid = next((idx for idx, v in enumerate(values) if not pd.isna(v)), None)
        if first_valid is not None and first_valid > 0:
            lead_val = values[first_valid]
            if mode == "int":
                lead_val = int(round(lead_val))
            values[:first_valid] = lead_val

        # Fill trailing NaNs with last valid value
        last_valid = next((n - 1 - idx for idx, v in enumerate(values[::-1]) if not pd.isna(v)), None)
        if last_valid is not None and last_valid < n - 1:
            trail_val = values[last_valid]
            if mode == "int":
                trail_val = int(round(trail_val))
            values[last_valid + 1:] = trail_val

    # Optional: cast back to int for "int" mode to ensure dtype consistency
    if mode == "int":
        out = pd.Series(values, index=series.index)
        return out.apply(lambda x: int(round(x)) if pd.notna(x) else np.nan)

    return pd.Series(values, index=series.index)



def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # distance in km

def process_file(file, interpolation=False):
    df = pd.read_parquet(file)
    df['int_ts'] = (np.array(df['timestamp']).astype(int) / 10E8).astype(int)
    df = df.groupby('int_ts').first()
    if interpolation:
        df = df.reindex(range(df.index.min(), df.index.max()))
        df['timestamp'] = df['timestamp'].interpolate(how='time')
        df.loc[:, 'flight_id'] = df['flight_id'].fillna(df['flight_id'].values[0])
        df.loc[:, 'typecode'] = df['typecode'].fillna(df['typecode'].values[0])
        interpolation_cols = ['latitude', 'longitude', 'altitude', 'groundspeed']
        df.loc[:, interpolation_cols] = df[interpolation_cols].interpolate()

    # Sort by flight_id and timestamp
    df = df.sort_values(by=["flight_id", "timestamp"]).reset_index(drop=True)

    df['flight_phase'] = 'NAN'


    # Fill gaps for each flight separately


    if "groundspeed" in df.columns or "track" in df.columns or "vertical_rate" in df.columns:
        for flight_id, group in df.groupby("flight_id"):
            idx = group.index

            if "groundspeed" in df.columns:
                df.loc[idx, "groundspeed"] = fill_gaps_series(df.loc[idx, "groundspeed"], mode="int", fill_edges=True)

            if "track" in df.columns:
                df.loc[idx, "track"] = fill_gaps_series(df.loc[idx, "track"], mode="float", fill_edges=True)

            if "vertical_rate" in df.columns:
                df.loc[idx, "vertical_rate"] = fill_gaps_series(df.loc[idx, "vertical_rate"], mode="int", fill_edges=True)


    # Add distance columns (GTD and GCD)

    df["GTD"] = 0.0
    df["GTD_total"] = 0.0
    df["GCD_total"] = np.nan

    for flight_id, group in df.groupby("flight_id"):
        idx = group.index
        lats = group["latitude"].values
        lons = group["longitude"].values

        # GTD per step
        gtd = [0.0]
        for i in range(1, len(group)):
            dist = haversine(lats[i-1], lons[i-1], lats[i], lons[i])
            gtd.append(dist)

        gtd_total = np.cumsum(gtd)
        df.loc[idx, "GTD"] = gtd
        df.loc[idx, "GTD_total"] = gtd_total

        # GCD from first to last point
        if len(group) > 1:
            gcd = haversine(lats[0], lons[0], lats[-1], lons[-1])
            df.loc[idx[-1], "GCD_total"] = gcd

        flight_phases = fuzzylabels(group.index, group['altitude'], group['groundspeed'], group['vertical_rate'])
        df.loc[idx, "flight_phase"] = flight_phases

        df.loc[(df['flight_phase'] == "LVL") & (df['altitude'] > 2000), "flight_phase"] = "CR"
    return df


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.parquet"))
    if not all_files:
        print("No .csv files found")
        exit()

    for file in all_files:
        filename = os.path.basename(file)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"{name}_processed.csv")

        print(f"Processing file: {filename} ...")
        # df = pd.read_csv(file)
        df = process_file(file)
        # Save
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    print("All data processed.")

