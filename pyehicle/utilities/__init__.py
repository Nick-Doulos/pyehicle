"""
Utilities module for the pyehicle library.

This module provides utility functions for road network management, trajectory
evaluation metrics, and visualization tools.
"""

from pyehicle.utilities.road_network import (
    load_road_network,
    build_igraph_graph,
    preprocess_road_segments,
    filter_road_network_by_bbox
)

# Import evaluation functions
try:
    from pyehicle.utilities.evaluation import (
        optimized_lengths,
        rmf,
        recall,
        precision,
        f1,
        length_index,
        clear_overpass_cache
    )
    _has_evaluation = True
except ImportError:
    _has_evaluation = False

# Import visualization submodule
from pyehicle.utilities import visualization

__all__ = [
    # Road network functions
    'load_road_network',
    'build_igraph_graph',
    'preprocess_road_segments',
    'filter_road_network_by_bbox',
    # Visualization module
    'visualization',
]

# Add evaluation functions if available
if _has_evaluation:
    __all__.extend([
        'optimized_lengths',
        'rmf',
        'recall',
        'precision',
        'f1',
        'length_index',
        'clear_overpass_cache',
    ])
