Pyehicle Documentation
======================

**Pyehicle** is a production-ready Python library for GPS trajectory processing and map-matched reconstruction. It transforms noisy, fragmented GPS data into accurate, road-network-aligned trajectories using state-of-the-art algorithms including HMM map-matching, Kalman filtering, and intelligent trajectory reconstruction.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Installation
============

.. code-block:: bash

   pip install pyehicle

Quick Start
===========

.. code-block:: python

   import pandas as pd
   import pyehicle as pye

   # 1. Load GPS trajectory data
   df = pd.read_csv('trajectory.csv')

   # 2. Preprocessing: Compress, map-match, and filter
   compressed = pye.preprocessing.spatio_temporal_compress(df, spatial_radius_km=0.01, time_threshold_s=30)
   matched = pye.preprocessing.leuven(compressed, lat_col='lat', lon_col='lon', max_dist=50)
   filtered = pye.preprocessing.kalman_aeqd_filter(matched, lat_col='lat', lon_col='lon', time_col='time')

   # 3. Load road network
   road_network, geometries, spatial_index = pye.utilities.load_road_network(
       pbf_file_path='map.osm.pbf',
       bbox=(24.0, 56.9, 24.2, 57.0)
   )

   # 4. Reconstruction: Produce the final refined trajectory
   refined = pye.reconstructing.refine_trajectory(filtered, road_network, max_node_distance=10, time_col='time')
   final_trajectory = pye.reconstructing.curve_interpolation(refined, road_network, lower_threshold=20, upper_threshold=80, time_col='time')

   # 5. Visualize and save
   pye.utilities.visualization.single(final_trajectory, name='Reconstructed Trajectory', show_in_browser=True)

Key Features
============

* 🛣️ **Smart Trajectory Reconstruction**: Automatically combines topologically equivalent trajectories, detects and interpolates road curves, and enforces road network continuity at intersections
* 🗺️ **OpenStreetMap Integration**: Seamless integration with OSM data via iGraph for fast graph operations
* 🎯 **Metric-Accurate Processing**: All distance calculations use geodesic methods (WGS84) and AEQD projections
* 🔄 **Dual Map-Matching Engines**: Choose between Leuven's HMM-based matcher or Valhalla's Meili service
* 🧮 **Advanced Kalman Filtering**: 2D filter with velocity state and automatic parameter tuning
* 📊 **Research-Grade Evaluation**: Built-in RMF, Precision, Recall, and F1 metrics
* ⚡ **Performance Optimized**: Supports both Pandas and Polars DataFrames with Numba JIT compilation
* 🌐 **Fully Offline Capable**: Works completely offline with downloaded PBF files

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
