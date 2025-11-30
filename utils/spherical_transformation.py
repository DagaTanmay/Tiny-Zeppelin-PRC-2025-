import numpy as np
from pathlib import Path
import pandas as pd


def spherical_transformation(long, lat):

    x_coordinates = []
    y_coordinates = []
    z_coordinates = []
    radius = 1
    for phi, theta in zip(long, lat):
        theta = theta * np.pi / 180
        phi = phi * np.pi / 180
        x_coordinates.append(radius * np.sin(theta) * np.cos(phi))
        y_coordinates.append(radius * np.sin(theta) * np.sin(phi))
        z_coordinates.append(radius * np.cos(theta))


    return x_coordinates, y_coordinates, z_coordinates



if __name__ == '__main__':
    input_data_folder = Path("G:/MO-PLM/Transfer/Open Sky Data Challenge/workstation output/evaluation/251120")
    input_data_file = input_data_folder / "251120_Emy_all_snippets.csv"


    input_df = pd.read_csv(input_data_file)


    x,y,z = spherical_transformation(input_df['mean_longitude'], input_df['mean_latitude'])

    input_df = input_df.assign(x_coordinates = x, y_coordinates = y, z_coordinates = z)



