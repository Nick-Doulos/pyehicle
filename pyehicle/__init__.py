"""
pyehicle - A Python library for GPS trajectory processing and reconstruction.

pyehicle provides a comprehensive suite of tools for processing, analyzing, and
reconstructing GPS trajectories using road network data.

Components
----------
- **preprocessing**: Trajectory preprocessing (compression, map-matching, filtering, segmentation)
- **reconstructing**: Trajectory reconstruction (combining, curve interpolation, refinement)
- **utilities**: Utility functions (road network management, evaluation metrics, visualization)

Quick Start
-----------
```python
import pyehicle as pye

# Load road network
road_network, geometries, spatial_index = pye.utilities.load_road_network(
    pbf_file_path='map.osm.pbf',
    bbox=(min_lon, min_lat, max_lon, max_lat),
    save_path='network.graphml'
)

# Preprocess trajectory
compressed = pye.preprocessing.spatio_temporal_compress(df, spatial_radius_km=0.01)
matched = pye.preprocessing.leuven(compressed)

# Reconstruct trajectory
combined = pye.reconstructing.trajectory_combiner([traj1, traj2])
interpolated = pye.reconstructing.curve_interpolation(combined, road_network)
refined = pye.reconstructing.refine_trajectory(interpolated, road_network)

# Evaluate results
f1_score = pye.utilities.f1(matched, ground_truth)

# Visualize
pye.utilities.visualization.multiple([ground_truth, matched, refined],
                                      names=['Ground Truth', 'Matched', 'Refined'])
```
"""

from pyehicle._version import __version__, __version_info__
from pyehicle import preprocessing, reconstructing, utilities

__all__ = [
    '__version__',
    '__version_info__',
    'preprocessing',
    'reconstructing',
    'utilities',
]
