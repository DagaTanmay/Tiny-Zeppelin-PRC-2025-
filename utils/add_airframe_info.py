import pandas as pd

airframe_file = "G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\airframe distribution\\MZFW_airframes.csv"
airframe_df = pd.read_csv(airframe_file, sep=';')
airframe_df.set_index('type', inplace=True)


def add_airframe_info(df, airframe_col="airframe"):
    start_shape = df.shape[0]
    df = df[df[airframe_col].isin(airframe_df.index)].copy()
    shape_diff = start_shape - df.shape[0]
    if shape_diff > 0:
        print(f"Eliminated {start_shape - df.shape[0]} rows because airframe unknown")
        if df.empty:
            return df

    df[['MZFW', "age_class"]] = df[airframe_col].apply(
        lambda af: airframe_df.loc[af, ['MZFW max zero fuel weight [t]', 'age_class']])
    return df