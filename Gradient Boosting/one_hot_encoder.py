from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.extend([str(Path(__file__).parent.parent / "utils")])

from add_airframe_info import add_airframe_info
from sklearn.preprocessing import OneHotEncoder

from pickle import dump

input_file = Path("G:\\MO-PLM\\Transfer\\Open Sky Data Challenge\\train_no_interpolation_19features.csv")

input_df = pd.read_csv(input_file)
input_df.dropna(axis=0, inplace=True)
input_df = add_airframe_info(input_df)
# input_df['weight_class'] = 'light'
# input_df.loc[input_df['MZFW'] > 100, 'weight_class'] = 'heavy'
#
# for cl, df in input_df.groupby('weight_class'):
if True:
    df = input_df

    one_hot = OneHotEncoder(min_frequency=100, handle_unknown='infrequent_if_exist')
    one_hot.fit_transform(np.array(df['airframe']).reshape(-1, 1))

    # with open(f"one_hot_encoder_{cl}.pkl", "wb") as f:
    with open(f"one_hot_encoder.pkl", "wb") as f:
        dump(one_hot, f, protocol=5)