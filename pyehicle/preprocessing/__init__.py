"""
Trajectory preprocessing module for pyehicle.

This module provides algorithms for preprocessing GPS trajectories, including:
- Compression: Reduce trajectory point count while preserving shape
- Map-matching: Align trajectories to road networks
- Filtering: Smooth trajectories using Kalman filtering
- Segmentation: Split trajectories into segments
- Interpolation: Add points between existing trajectory points
- Sampling rate: Calculate trajectory sampling rates
"""

__version__ = "0.0.1"
__author__ = "Nick Doulos"

# Compression
from pyehicle.preprocessing.compression import spatio_temporal_compress

# Map-matching
from pyehicle.preprocessing.map_matching import leuven, meili

# Filtering
from pyehicle.preprocessing.filtration import kalman_aeqd_filter

# Segmentation
from pyehicle.preprocessing.segmentation import by_time

# Interpolation
from pyehicle.preprocessing.interpolation import by_number_of_points, by_sampling_rate

# Sampling rate
from pyehicle.preprocessing.sampling_rate import get_sampling_rate

__all__ = [
    # Compression
    'spatio_temporal_compress',
    # Map-matching
    'leuven',
    'meili',
    # Filtering
    'kalman_aeqd_filter',
    # Segmentation
    'by_time',
    # Interpolation
    'by_number_of_points',
    'by_sampling_rate',
    # Sampling rate
    'get_sampling_rate',
]
