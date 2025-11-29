Examples
========

Basic Trajectory Processing
----------------------------

.. code-block:: python

   import pandas as pd
   import pyehicle as pye

   # Load and compress trajectory
   df = pd.read_csv('trajectory.csv')
   compressed = pye.preprocessing.spatio_temporal_compress(df, spatial_radius_km=0.01, time_threshold_s=30)
   print(f"Reduced from {len(df)} to {len(compressed)} points")

   # Map-match to road network
   matched = pye.preprocessing.leuven(compressed, lat_col='lat', lon_col='lon', max_dist=50)
   print(f"Matched {len(matched)} points to road network")

Complete Reconstruction Pipeline
---------------------------------

.. code-block:: python

   import pandas as pd
   import pyehicle as pye

   # Load data
   df = pd.read_csv('raw_trajectory.csv')

   # Preprocessing
   compressed = pye.preprocessing.spatio_temporal_compress(df, spatial_radius_km=0.01, time_threshold_s=30)
   matched = pye.preprocessing.leuven(compressed, lat_col='lat', lon_col='lon', max_dist=50)
   filtered = pye.preprocessing.kalman_aeqd_filter(matched, lat_col='lat', lon_col='lon', time_col='time')

   # Load road network
   road_network, geometries, spatial_index = pye.utilities.load_road_network(
       pbf_file_path='map.osm.pbf',
       bbox=(24.0, 56.9, 24.2, 57.0)
   )

   # Reconstruction
   refined = pye.reconstructing.refine_trajectory(filtered, road_network, max_node_distance=10, time_col='time')
   final_trajectory = pye.reconstructing.curve_interpolation(refined, road_network, lower_threshold=20, upper_threshold=80, time_col='time')

   # Evaluate
   ground_truth = pd.read_csv('ground_truth.csv')
   f1_score = pye.utilities.f1(final_trajectory, ground_truth)
   precision = pye.utilities.precision(final_trajectory, ground_truth)
   recall = pye.utilities.recall(final_trajectory, ground_truth)

   print(f"F1: {f1_score:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

   # Visualize
   pye.utilities.visualization.multiple(
       df_list=[matched, filtered, final_trajectory],
       names=['Map-Matched', 'Kalman Filtered', 'Reconstructed'],
       show_in_browser=True,
       cmap='tab10'
   )

Combining Topologically Equivalent Trajectories
------------------------------------------------

.. code-block:: python

   import pandas as pd
   import pyehicle as pye

   # Load multiple recordings of the same bus route
   route_66_run1 = pd.read_csv('bus_route_66_monday.csv')
   route_66_run2 = pd.read_csv('bus_route_66_tuesday.csv')
   route_66_run3 = pd.read_csv('bus_route_66_wednesday.csv')

   # Preprocess each recording independently
   processed_runs = []
   for i, run in enumerate([route_66_run1, route_66_run2, route_66_run3]):
       compressed = pye.preprocessing.spatio_temporal_compress(run, spatial_radius_km=0.01, time_threshold_s=30)
       matched = pye.preprocessing.leuven(compressed, lat_col='lat', lon_col='lon', max_dist=50)
       filtered = pye.preprocessing.kalman_aeqd_filter(matched, lat_col='lat', lon_col='lon', time_col='time')
       processed_runs.append(filtered)
       print(f"Run {i+1}: {len(run)} → {len(filtered)} points")

   # Combine all runs into a single canonical route
   canonical_route = pye.reconstructing.trajectory_combiner(
       processed_runs,
       lat_col='lat',
       lon_col='lon',
       time_col='time'
   )
   print(f"Combined into canonical route with {len(canonical_route)} points")

   # Load road network and refine the canonical route
   road_network, geometries, spatial_index = pye.utilities.load_road_network(
       pbf_file_path='map.osm.pbf',
       bbox=(24.0, 56.9, 24.2, 57.0)
   )

   refined = pye.reconstructing.refine_trajectory(canonical_route, road_network, max_node_distance=10, time_col='time')
   final_route = pye.reconstructing.curve_interpolation(refined, road_network, lower_threshold=20, upper_threshold=80, max_node_distance=10, time_col='time')

   # Visualize the final canonical route
   pye.utilities.visualization.single(final_route, name='Bus Route 66 - Canonical Trajectory', show_in_browser=True)
