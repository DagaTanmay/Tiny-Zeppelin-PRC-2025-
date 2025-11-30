# Utils Folder

This folder contains a collection of Python utility scripts used for data preprocessing, transformation, and enhancement related to flight and OpenSky datasets. Each script performs a specific task.

## Files

- **`add_airframe_info.py`**  
  Adds or merges airframe-related metadata into existing datasets.

- **`flightphase.py`**  
  Determines or labels the flight phase based on aircraft state data.

- **`open_sky_file_splitter.py`**  
  Splits large OpenSky Network data files into smaller, manageable chunks.

- **`open_sky_distance`**  
  Utility to calculate distances between points in OpenSky trajectory data.

- **`open_sky_distance_gap`**  
  Identifies or computes gaps in OpenSky distance sequences.

- **`opensky_prq_to_csv.py`**  
  Converts OpenSky `.parquet` (PRQ) files into CSV format.

- **`spherical_transformation.py`**  
  Performs spherical coordinate transformations for geospatial data.
