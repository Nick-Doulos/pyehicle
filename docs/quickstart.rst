Quick Start
===========

This guide will help you get started with Pyehicle.

Data Format
-----------

Pyehicle expects trajectory data as pandas/polars DataFrames with columns:

+----------+------------------+---------------------------+
| Column   | Type             | Description               |
+==========+==================+===========================+
| lat      | float            | Latitude in WGS84         |
+----------+------------------+---------------------------+
| lon      | float            | Longitude in WGS84        |
+----------+------------------+---------------------------+
| time     | string/datetime  | Timestamp (configurable)  |
+----------+------------------+---------------------------+

Example CSV:

.. code-block:: csv

   lat,lon,time
   56.9496,24.1052,2023-01-01 10:00:00
   56.9497,24.1053,2023-01-01 10:00:05
   56.9498,24.1054,2023-01-01 10:00:10

Basic Usage
-----------

.. code-block:: python

   import pandas as pd
   import pyehicle as pye

   # Load GPS data
   df = pd.read_csv('trajectory.csv')

   # Preprocess
   compressed = pye.preprocessing.spatio_temporal_compress(df)
   matched = pye.preprocessing.leuven(compressed, lat_col='lat', lon_col='lon')
   filtered = pye.preprocessing.kalman_aeqd_filter(matched, lat_col='lat', lon_col='lon', time_col='time')

   # Load road network
   road_network, geometries, spatial_index = pye.utilities.load_road_network(
       pbf_file_path='map.osm.pbf',
       bbox=(24.0, 56.9, 24.2, 57.0)
   )

   # Reconstruct trajectory
   refined = pye.reconstructing.refine_trajectory(filtered, road_network, max_node_distance=10, time_col='time')
   final = pye.reconstructing.curve_interpolation(refined, road_network, lower_threshold=20, upper_threshold=80, max_node_distance=10, time_col='time')

   # Visualize
   pye.utilities.visualization.single(final, name='Trajectory', show_in_browser=True)

Pipeline Overview
-----------------

.. code-block:: text

   Raw GPS Data → Preprocessing → Reconstruction → Final Refined Trajectory
     (noisy)      (compress,      (combine,         (accurate, road-aligned)
                   map-match,      refine,
                   kalman)         curve_interp)
