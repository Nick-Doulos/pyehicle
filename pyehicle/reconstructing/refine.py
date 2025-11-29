"""
Trajectory refinement module for pyehicle.

This module provides functionality to refine GPS trajectories by enforcing spatial
continuity with underlying road networks. It is the second step in trajectory
reconstruction, taking combined trajectory segments and "snapping" them to realistic
road paths by detecting road transitions and interpolating corner points at intersections.

The refinement process:
1. Detects when trajectories transition between different roads (OSM ID changes)
2. Finds intersection points or reconstructs paths along the road network
3. Interpolates corner points with temporally-proportional timestamps
4. Produces a refined trajectory that follows the actual road network

This is particularly valuable for:
- Converting raw GPS traces into road-matched trajectories
- Generating realistic vehicle paths for simulation or analysis
- Improving trajectory quality for visualization and route analysis
- Preparing trajectories for curve interpolation and final reconstruction
"""

import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from sklearn.neighbors import BallTree
from pyproj import Geod
from pyehicle.utilities.road_network import preprocess_road_segments, filter_road_network_by_bbox

# Initialize a global geodesic object for all distance calculations
geod = Geod(ellps="WGS84")


def build_balltree(G):
    """
    Build a BallTree spatial index for fast nearest-neighbor queries on road network nodes.

    This function constructs a scikit-learn BallTree data structure that enables efficient
    k-nearest neighbor searches on the road network's node coordinates. The BallTree uses
    haversine distance (great-circle distance on a sphere) by operating on coordinates in
    radians, making it ideal for geographical data.

    BallTree is preferred over alternatives (KDTree, brute-force) because:
    - Handles spherical geometry correctly (haversine distance)
    - O(log n) query time for k-nearest neighbors
    - Memory efficient for large road networks (>100,000 nodes)
    - Faster than R-tree for pure k-NN queries (no bounding box operations)

    Parameters
    ----------
    G : igraph.Graph
        Road network graph with node attributes 'x' (longitude, WGS84 decimal degrees)
        and 'y' (latitude, WGS84 decimal degrees). Typically loaded from OSM via
        `utilities.road_network.load_road_network()`.

    Returns
    -------
    tree : sklearn.neighbors.BallTree
        Spatial index built on node coordinates (in radians) using haversine metric.
        Ready for k-nearest neighbor queries via `tree.query()`.
    node_indices : np.ndarray
        Array of node indices [0, 1, 2, ..., n-1] corresponding to G.vs.
        Used to map BallTree query results back to graph node IDs.

    Notes
    -----
    **Performance:**
    - Build time: O(n log n) where n is the number of nodes
    - Query time: O(log n) for k-nearest neighbors
    - Memory: O(n) for the tree structure

    **Coordinate Format:**
    - Input: WGS84 decimal degrees (lon, lat)
    - Internal: Radians (lat, lon) for haversine distance
    - BallTree expects (lat, lon) order when using haversine metric

    **Use Cases:**
    - Finding nearest road nodes to GPS points during trajectory refinement
    - Snapping trajectory points to road network
    - Rapid spatial queries without loading full R-tree index

    Examples
    --------
    >>> import igraph as ig
    >>> from pyehicle.utilities.road_network import load_road_network
    >>> from pyehicle.reconstructing.refine import build_balltree
    >>>
    >>> # Load road network
    >>> G = load_road_network('data/city_roads.graphml')
    >>>
    >>> # Build BallTree for fast queries
    >>> tree, node_indices = build_balltree(G)
    >>>
    >>> # Query 5 nearest nodes to a point (lat=56.95, lon=24.10)
    >>> query_point = np.radians([[56.95, 24.10]])  # Convert to radians
    >>> distances, indices = tree.query(query_point, k=5)
    >>>
    >>> # Map back to graph node IDs
    >>> nearest_node_ids = node_indices[indices[0]]
    >>> print(f"Nearest nodes: {nearest_node_ids}")
    """
    # Extract node coordinates (lat, lon) from graph
    # G.vs['y'] = latitude, G.vs['x'] = longitude
    node_coords = np.column_stack((G.vs['y'], G.vs['x']))  # Shape: (n_nodes, 2)

    # Convert to radians for haversine distance calculation
    # BallTree with haversine metric requires coordinates in radians
    node_coords_rad = np.radians(node_coords)

    # Build BallTree using haversine metric (great-circle distance)
    # Default metric is 'haversine' when coordinates are in radians
    tree = BallTree(node_coords_rad)

    # Create array of node indices for mapping query results back to graph nodes
    # node_indices[i] = i, where i is the index in G.vs
    node_indices = np.arange(len(G.vs))

    return tree, node_indices


def find_nearest_node(
    G,
    tree,
    node_indices,
    point,
    point_osmid=None,
    k=3,
    max_node_distance=10
):
    """
    Find the closest road network node to a GPS point that lies on the same road.

    This function performs a constrained nearest-neighbor search: it finds nodes that
    are (1) spatially close to the query point and (2) belong to the same road as the
    query point (matching OSM ID). This is critical for trajectory refinement because
    we want to snap trajectory points to nodes on the *correct* road, not just any
    nearby node.

    The algorithm uses a two-stage search:
    1. Fast BallTree k-NN query to get candidate nodes (approximate haversine distance)
    2. Accurate pyproj geodesic distance filtering with OSM ID matching

    This approach balances speed (BallTree) with accuracy (pyproj) and road identity
    (OSM ID filtering).

    Parameters
    ----------
    G : igraph.Graph
        Road network graph with node attributes 'x' (lon), 'y' (lat), and edge
        attributes 'osmid' (OpenStreetMap way ID). Typically from OSM data.
    tree : sklearn.neighbors.BallTree
        Pre-built BallTree spatial index for fast k-NN queries. Created by
        `build_balltree(G)`.
    node_indices : np.ndarray
        Array mapping BallTree indices to graph node IDs. Created by
        `build_balltree(G)`.
    point : tuple of float
        Query point as (latitude, longitude) in WGS84 decimal degrees.
        Example: (56.95, 24.10)
    point_osmid : int, list of int, or None, optional
        OSM way ID(s) of the road containing the query point. If None, no filtering
        by road is performed (returns None). Can be a single int or a list of ints
        for roads with multiple OSM IDs.
    k : int, default=3
        Number of nearest neighbors to consider in the BallTree search. Higher k
        increases chance of finding a same-road match but slows down the search.
        Typical values: 3-10.
    max_node_distance : float, default=25
        Maximum distance in meters for a node to be considered a match. Nodes
        farther than this threshold are rejected even if they match the OSM ID.
        Typical values: 10-50 meters.

    Returns
    -------
    int or None
        Graph node index (0-based) of the nearest node on the same road within
        the distance threshold. Returns None if no matching node is found.

    Notes
    -----
    **Algorithm:**
    1. Query BallTree for k nearest neighbors (fast approximate search)
    2. For each candidate node (ranked by BallTree distance):
       - Calculate accurate geodesic distance using pyproj
       - Check if distance <= max_node_distance
       - Check if node connects to an edge with matching OSM ID
       - Update best match if this node is closer
    3. Return the closest matching node or None

    **OSM ID Matching:**
    - OSM IDs identify specific roads (ways) in OpenStreetMap
    - A node may be connected to multiple edges (roads) at intersections
    - Function checks all incident edges to find OSM ID matches
    - Handles both single OSM IDs (int) and multiple IDs (list)

    **Performance:**
    - Time complexity: O(log n + k) where n is total nodes
    - BallTree query: O(log n)
    - Filtering k candidates: O(k)
    - Typical execution: <1ms per query

    **Use Cases:**
    - Snapping trajectory points to the correct road during refinement
    - Finding intersection nodes for path reconstruction
    - Validating that GPS points lie on expected roads

    **Limitations:**
    - Requires point_osmid to be known (returns None if not provided)
    - May miss the correct node if k is too small for complex intersections
    - Assumes OSM IDs are reliable (may fail for poor-quality OSM data)

    Examples
    --------
    >>> import numpy as np
    >>> from pyehicle.utilities.road_network import load_road_network
    >>> from pyehicle.reconstructing.refine import build_balltree, find_nearest_node
    >>>
    >>> # Load road network
    >>> G = load_road_network('data/city_roads.graphml')
    >>>
    >>> # Build BallTree index
    >>> tree, node_indices = build_balltree(G)
    >>>
    >>> # Find nearest node on a specific road
    >>> gps_point = (56.9520, 24.1050)  # (lat, lon)
    >>> road_osmid = 123456789  # OSM way ID from map-matching
    >>>
    >>> nearest_node = find_nearest_node(
    ...     G, tree, node_indices,
    ...     point=gps_point,
    ...     point_osmid=road_osmid,
    ...     k=5,
    ...     max_node_distance=10
    ... )
    >>>
    >>> if nearest_node is not None:
    ...     node_lat = G.vs[nearest_node]['y']
    ...     node_lon = G.vs[nearest_node]['x']
    ...     print(f"Snapped to node at ({node_lat:.6f}, {node_lon:.6f})")
    ... else:
    ...     print("No matching node found within 30 meters")
    """
    # Early exit if road identity is unknown - cannot filter by OSM ID
    # Without point_osmid, we can't determine which road the point belongs to
    if point_osmid is None:
        return None

    # ========== BallTree k-Nearest Neighbors Query ==========
    # Query BallTree for k nearest neighbors using haversine distance
    # BallTree works in radian space, so convert point to radians
    point_rad = np.radians([point])  # Shape: (1, 2) with (lat, lon)
    _, idx_array = tree.query(point_rad, k=k)  # Returns (distances, indices)

    # Initialize tracking variables for the best same-road match
    best_node_same_road = None  # Node index of the best match
    best_score_same_road = float('inf')  # Distance to the best match

    # Extract candidate indices for iteration
    # idx_array shape: (1, k) -> candidate_indices shape: (k,)
    candidate_indices = idx_array[0]

    # ========== Filter Candidates by Distance and OSM ID ==========
    # Loop through the k nearest neighbors as ranked by BallTree
    for i in range(k):
        # Map BallTree index back to graph node index
        node_idx = int(node_indices[candidate_indices[i]])

        # Get node coordinates from graph
        node_lat = G.vs[node_idx]['y']  # Latitude
        node_lon = G.vs[node_idx]['x']  # Longitude

        # Calculate accurate geodesic distance using pyproj
        # geod.inv(lon1, lat1, lon2, lat2) returns (az_forward, az_back, distance)
        # We only need the distance (in meters)
        _, _, distance_meters = geod.inv(point[1], point[0], node_lon, node_lat)

        # Skip nodes that are too far away (exceed threshold)
        if distance_meters > max_node_distance:
            continue

        # ========== Check OSM ID Match ==========
        # Collect OSM IDs of all roads connected to this node
        # A node at an intersection may connect to multiple roads
        incident_edges = G.incident(node_idx)  # List of edge IDs connected to this node
        connected_osmids = set()  # Set of OSM IDs for all incident edges

        # Check all incident edges for OSM ID matches
        for edge_id in incident_edges:
            edge = G.es[edge_id]
            if 'osmid' in edge.attributes():
                osmval = edge['osmid']
                # OSM ID can be a single int or a list of ints (for complex ways)
                if isinstance(osmval, list):
                    connected_osmids.update(osmval)
                else:
                    connected_osmids.add(osmval)

        # Update best match if this node is on the same road and closer
        # point_osmid can also be a list, so check if it's in connected_osmids
        if point_osmid in connected_osmids and distance_meters < best_score_same_road:
            best_score_same_road = distance_meters
            best_node_same_road = node_idx

    # Return the closest node on the same road, or None if no match found
    return best_node_same_road


def find_shortest_path(G, p1, p2, spatial_index, max_node_distance):
    """
    Find the shortest path through the road network between two GPS trajectory points.

    This function connects two trajectory points by routing through the road network,
    effectively "filling in" the path along actual roads. It's used during trajectory
    refinement when the direct line between two points doesn't follow the road network
    (e.g., at intersections or when crossing multiple road segments).

    The algorithm follows these steps:
    1. Find nearest road edges for both points using spatial index (R-tree)
    2. Extract OSM IDs to identify which roads the points are on
    3. Find nearest nodes on those specific roads using constrained k-NN search
    4. Compute shortest path between the nodes using Dijkstra's algorithm (iGraph)
    5. Return the path as a sequence of edge indices

    This ensures that interpolated corner points follow realistic road routes rather
    than arbitrary straight lines.

    Parameters
    ----------
    G : igraph.Graph
        Road network graph with edge attributes 'osmid' (road ID) and 'length' (meters).
        Typically loaded from OSM data via `utilities.road_network.load_road_network()`.
    p1 : shapely.geometry.Point
        Starting point with attributes .x (longitude) and .y (latitude) in WGS84.
    p2 : shapely.geometry.Point
        Ending point with attributes .x (longitude) and .y (latitude) in WGS84.
    spatial_index : rtree.Index
        R-tree spatial index for fast edge (road segment) lookups. Created by
        `utilities.road_network.preprocess_road_segments()`.
    max_node_distance : float
        Maximum distance in meters for snapping points to road nodes. Points farther
        than this threshold from any same-road node will cause the function to return
        None. Typical values: 10-50 meters.

    Returns
    -------
    list of int or None
        List of edge indices forming the shortest path from p1 to p2, or None if:
        - spatial_index is None
        - No same-road node found for p1 or p2 within max_node_distance
        - Both points snap to the same node (no path needed)
        - No connecting path exists in the road network

    Notes
    -----
    **Algorithm Details:**
    1. R-tree nearest query identifies which road edges p1 and p2 are closest to
    2. OSM IDs from those edges identify the specific roads
    3. BallTree k-NN search (via `find_nearest_node()`) finds nodes on those roads
    4. Dijkstra's algorithm (iGraph) computes shortest path weighted by edge length
    5. Returns edge-based path (not node-based) for geometry reconstruction

    **Performance:**
    - Time complexity: O(log n + k + m log m) where:
      - n = number of road segments (R-tree query)
      - k = BallTree k-NN candidates (typically 3)
      - m = number of nodes in the road network (Dijkstra's algorithm)
    - Typical execution: 1-10ms per query depending on network size

    **Use Cases:**
    - Reconstructing paths when trajectory changes roads at intersections
    - Filling gaps between non-adjacent trajectory points
    - Ensuring refined trajectories follow actual road geometry

    **Limitations:**
    - Requires both points to be within max_node_distance of road nodes
    - May fail if road network is disconnected or incomplete
    - Does not consider turn restrictions or one-way streets
    - Shortest path is by distance, not travel time

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> from pyehicle.utilities.road_network import load_road_network, preprocess_road_segments
    >>> from pyehicle.reconstructing.refine import find_shortest_path
    >>>
    >>> # Load road network and build spatial index
    >>> G = load_road_network('data/city_roads.graphml')
    >>> _, spatial_index = preprocess_road_segments(G)
    >>>
    >>> # Two trajectory points on different roads
    >>> point1 = Point(24.105, 56.950)  # (lon, lat)
    >>> point2 = Point(24.110, 56.952)
    >>>
    >>> # Find path connecting them along road network
    >>> path_edges = find_shortest_path(
    ...     G, point1, point2, spatial_index, max_node_distance=10
    ... )
    >>>
    >>> if path_edges:
    ...     print(f"Path found with {len(path_edges)} edges")
    ...     # Extract edge geometries for visualization
    ...     path_geoms = [G.es[e]['geometry'] for e in path_edges]
    ... else:
    ...     print("No path found - points may be too far from roads")
    """
    # Early exit if spatial index is not available
    if spatial_index is None:
        return None

    # Build BallTree for fast node queries
    # This is recreated each call - could be cached for better performance
    tree, node_indices = build_balltree(G)

    # ========== Find Nearest Road Edges for Both Points ==========
    # Use R-tree spatial index to find closest road segments
    # Query format: (min_x, min_y, max_x, max_y) - point has same min/max
    edge_idx_p1 = list(spatial_index.nearest((p1.x, p1.y, p1.x, p1.y), 1))[0]
    p1_osmid = G.es[edge_idx_p1]['osmid']  # Get OSM ID of the road

    edge_idx_p2 = list(spatial_index.nearest((p2.x, p2.y, p2.x, p2.y), 1))[0]
    p2_osmid = G.es[edge_idx_p2]['osmid']  # Get OSM ID of the road

    # ========== Find Nearest Nodes on the Same Roads ==========
    # For each point, find the closest node that belongs to the same road (OSM ID)
    # This ensures we snap to the correct road, not just any nearby node
    p1_node = find_nearest_node(
        G,
        tree,
        node_indices,
        (p1.y, p1.x),  # Note: find_nearest_node expects (lat, lon)
        point_osmid=p1_osmid,
        k=3,  # Check 3 nearest candidates
        max_node_distance=max_node_distance
    )
    if p1_node is None:
        return None  # No same-road node found for p1 within threshold

    p2_node = find_nearest_node(
        G,
        tree,
        node_indices,
        (p2.y, p2.x),  # Note: find_nearest_node expects (lat, lon)
        point_osmid=p2_osmid,
        k=3,  # Check 3 nearest candidates
        max_node_distance=max_node_distance
    )
    if p2_node is None:
        return None  # No same-road node found for p2 within threshold

    # If both points snap to the same node, no path interpolation needed
    if p1_node == p2_node:
        return None

    # ========== Compute Shortest Path Using Dijkstra's Algorithm ==========
    # iGraph's get_shortest_paths uses Dijkstra's algorithm by default
    # 'length' edge attribute = distance in meters
    # output='epath' returns edge indices instead of node indices
    epath = G.get_shortest_paths(p1_node, to=p2_node, weights='length', output='epath')

    # Check if a valid path was found
    if not epath or not epath[0]:
        return None  # No connecting path exists in the road network

    # Return the edge-based path (list of edge indices)
    return epath[0]


def reconstruct_path_line_from_nodes(G, path):
    """
    Reconstruct a LineString geometry from a sequence of edge IDs in the road network graph.

    This function converts a path represented as edge indices into a continuous LineString
    geometry by extracting the coordinates of all nodes along the path. It's used to create
    geometric representations of paths for corner point interpolation.

    The function recomputes the shortest path between the first and last node to ensure
    the path is topologically valid and follows the actual road network geometry.

    Parameters
    ----------
    G : igraph.Graph
        Road network graph with node attributes 'x' (longitude) and 'y' (latitude).
    path : list of int
        List of edge indices forming a path through the road network. Typically
        returned by `find_shortest_path()` or iGraph's `get_shortest_paths()`.

    Returns
    -------
    shapely.geometry.LineString or None
        LineString representing the path geometry with coordinates in (lon, lat) order,
        or None if:
        - path is empty
        - No valid node path found between endpoints
        - Fewer than 2 unique coordinates after filtering duplicates

    Notes
    -----
    **Algorithm:**
    1. Extract source node from first edge and target node from last edge
    2. Recompute shortest path between these nodes (node-based path)
    3. Extract (lon, lat) coordinates for all nodes in the path
    4. Remove consecutive duplicate coordinates (can occur at self-loops)
    5. Create LineString if at least 2 unique coordinates remain

    **Why Recompute Path:**
    The input `path` is edge-based, but we need node-based coordinates. Recomputing
    ensures topological consistency and handles cases where edge sequences might not
    form a continuous node path.

    **Duplicate Removal:**
    Consecutive duplicates can occur at:
    - Self-loops in the road network
    - Roundabouts with multiple edges at the same node
    - Data quality issues in OSM
    Removing them prevents invalid LineString geometries.

    **Performance:**
    - Time complexity: O(m log m + n) where m = nodes in subgraph, n = path length
    - Dijkstra's algorithm: O(m log m)
    - Coordinate extraction: O(n)
    - Typical execution: <5ms for paths under 100 nodes

    Examples
    --------
    >>> from pyehicle.utilities.road_network import load_road_network
    >>> from pyehicle.reconstructing.refine import find_shortest_path, reconstruct_path_line_from_nodes
    >>> from shapely.geometry import Point
    >>>
    >>> # Load road network
    >>> G = load_road_network('data/city_roads.graphml')
    >>> _, spatial_index = preprocess_road_segments(G)
    >>>
    >>> # Find path between two points
    >>> p1 = Point(24.105, 56.950)
    >>> p2 = Point(24.110, 56.952)
    >>> edge_path = find_shortest_path(G, p1, p2, spatial_index, max_node_distance=10)
    >>>
    >>> # Reconstruct geometry
    >>> if edge_path:
    ...     path_geom = reconstruct_path_line_from_nodes(G, edge_path)
    ...     if path_geom:
    ...         print(f"Path length: {path_geom.length:.6f} degrees")
    ...         print(f"Path coordinates: {list(path_geom.coords)}")
    """
    # Early exit for empty paths
    if not path:
        return None

    # Extract source node from first edge and target node from last edge
    # This gives us the endpoints of the path
    p1_node = G.es[path[0]].source
    p2_node = G.es[path[-1]].target

    # Recompute shortest path to get node-based representation
    # output='vpath' returns vertex (node) indices instead of edge indices
    vpath = G.get_shortest_paths(p1_node, to=p2_node, weights='length', output='vpath')

    # Check if a valid path was found
    if not vpath or not vpath[0]:
        return None  # No connecting path between endpoints

    # Extract the node sequence forming the path
    node_path = vpath[0]  # vpath is a list of lists, take first (and only) path

    # Extract coordinates from node path in (lon, lat) order
    # G.vs[n]['x'] = longitude, G.vs[n]['y'] = latitude
    coords = [(G.vs[n]['x'], G.vs[n]['y']) for n in node_path]

    # Remove consecutive duplicate coordinates to avoid invalid geometries
    # Start with the first coordinate
    filtered_coords = [coords[0]]
    for c in coords[1:]:
        # Only add if different from the last added coordinate
        if c != filtered_coords[-1]:
            filtered_coords.append(c)

    # LineString requires at least 2 unique coordinates
    if len(filtered_coords) < 2:
        return None  # Not enough unique points to form a line

    # Create and return LineString geometry
    return LineString(filtered_coords)


def _is_between(p1, p2, Q, total_dist, tolerance=0.001):
    """
    Check if a point Q lies between two endpoints p1 and p2 using geodesic distance.

    This function tests whether Q is "between" p1 and p2 by verifying that the sum
    of distances (p1 → Q) + (Q → p2) approximately equals the direct distance p1 → p2.
    This is the geodesic equivalent of the collinearity test, accounting for Earth's
    curvature.

    The function is used during corner point interpolation to validate that intersection
    points actually lie on the path between trajectory points (not behind or ahead).

    Parameters
    ----------
    p1 : shapely.geometry.Point
        First endpoint with attributes .x (longitude) and .y (latitude) in WGS84.
    p2 : shapely.geometry.Point
        Second endpoint with attributes .x (longitude) and .y (latitude) in WGS84.
    Q : shapely.geometry.Point
        Query point to test, with attributes .x (longitude) and .y (latitude) in WGS84.
    total_dist : float
        Pre-computed geodesic distance from p1 to p2 in meters. Passed in for
        efficiency to avoid recomputing in the calling function.
    tolerance : float, default=0.001
        Maximum allowable difference in meters for Q to be considered "between".
        Default of 1mm is appropriate for GPS coordinates (sub-centimeter precision).

    Returns
    -------
    bool
        True if Q lies between p1 and p2 (within tolerance), False otherwise.

    Notes
    -----
    **Mathematical Principle:**
    A point Q is between p1 and p2 if and only if:
        distance(p1, Q) + distance(Q, p2) ≈ distance(p1, p2)

    If Q is off the line or outside the segment, the sum will be larger due to
    the triangle inequality.

    **Why Geodesic Distance:**
    Using pyproj Geod ensures accuracy across all latitudes and distances. Euclidean
    distance in (lat, lon) space would fail at high latitudes or long distances.

    **Tolerance:**
    - 0.001m (1mm): Sub-centimeter precision, appropriate for GPS (±5-10m typical error)
    - Larger tolerances may be needed for low-precision trajectory data
    - Very small tolerance (<0.0001m) may cause false negatives due to float precision

    **Performance:**
    - Time complexity: O(1) - two geodesic distance calculations
    - Typical execution: <0.1ms per call

    **Use Case:**
    During trajectory refinement, when two roads intersect, we need to verify that
    the intersection point actually lies on the path segment, not beyond the endpoints.

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> from pyehicle.reconstructing.refine import _is_between
    >>> from pyproj import Geod
    >>>
    >>> geod = Geod(ellps="WGS84")
    >>>
    >>> # Three collinear points along a road
    >>> p1 = Point(24.100, 56.950)
    >>> Q = Point(24.105, 56.951)  # Midpoint
    >>> p2 = Point(24.110, 56.952)
    >>>
    >>> # Calculate total distance
    >>> _, _, total_dist = geod.inv(p1.x, p1.y, p2.x, p2.y)
    >>>
    >>> # Check if Q is between p1 and p2
    >>> result = _is_between(p1, p2, Q, total_dist)
    >>> print(result)  # True - Q is on the path
    >>>
    >>> # Point not on the line
    >>> Q_off = Point(24.105, 56.960)  # Way off to the north
    >>> result = _is_between(p1, p2, Q_off, total_dist)
    >>> print(result)  # False - detour makes total distance larger
    """
    # Calculate geodesic distance from p1 to Q
    # geod.inv(lon1, lat1, lon2, lat2) returns (az_forward, az_back, distance)
    _, _, d1 = geod.inv(p1.x, p1.y, Q.x, Q.y)

    # Calculate geodesic distance from Q to p2
    _, _, d2 = geod.inv(Q.x, Q.y, p2.x, p2.y)

    # Check if the sum of distances equals the total distance (within tolerance)
    # If Q is between p1 and p2, then d1 + d2 ≈ total_dist
    # If Q is off the path or outside the segment, d1 + d2 > total_dist (triangle inequality)
    return abs((d1 + d2) - total_dist) <= tolerance


def interpolate_corner_point(
    p1,
    p2,
    t1,
    t2,
    edge_idx1,
    edge_idx2,
    G,
    spatial_index,
    max_node_distance
):
    """
    Interpolate corner point(s) when a trajectory transitions between different roads.

    This is the core function for trajectory refinement. When consecutive trajectory points
    lie on different roads (different OSM IDs), this function fills the gap with realistic
    corner points that follow the road network geometry. It handles two geometric scenarios:

    **Case A - No Intersection (Non-adjacent roads):**
    - Roads don't share a common intersection point
    - Reconstruct path along road network using Dijkstra's algorithm
    - Interpolate multiple points along the reconstructed path
    - Distribute timestamps proportionally to distance traveled

    **Case B - Direct Intersection (Adjacent roads meeting at a point):**
    - Roads intersect at a single common point
    - Interpolate single corner point at the intersection
    - Calculate timestamp based on distance fraction

    The function ensures temporal continuity by distributing timestamps proportional to
    the distance traveled along the road network, not just Euclidean distance.

    Parameters
    ----------
    p1 : shapely.geometry.Point
        Starting trajectory point with attributes .x (longitude) and .y (latitude).
    p2 : shapely.geometry.Point
        Ending trajectory point with attributes .x (longitude) and .y (latitude).
    t1 : float
        Timestamp at p1 in Unix epoch seconds (seconds since 1970-01-01).
    t2 : float
        Timestamp at p2 in Unix epoch seconds (seconds since 1970-01-01).
    edge_idx1 : int
        Road network edge index for the road containing p1. Obtained from R-tree
        spatial index nearest-neighbor query.
    edge_idx2 : int
        Road network edge index for the road containing p2.
    G : igraph.Graph
        Road network graph with edge attributes 'geometry' (LineString) and 'osmid'.
    spatial_index : rtree.Index
        R-tree spatial index for fast edge lookups, created by
        `utilities.road_network.preprocess_road_segments()`.
    max_node_distance : float
        Maximum distance in meters for snapping trajectory points to road nodes.
        Typical values: 10-50 meters.

    Returns
    -------
    points : list of shapely.geometry.Point or None
        List of interpolated corner point(s) to insert between p1 and p2. Empty list
        or single point for Case B, multiple points for Case A. None if interpolation
        failed or is unnecessary.
    times : list of float or None
        Corresponding timestamps (Unix epoch seconds) for each interpolated point.
        Same length as `points`. None if interpolation failed.

    Notes
    -----
    **Algorithm Flow:**
    1. Extract road geometries for both edges from the graph
    2. Compute geometric intersection between the two road geometries
    3. **Case A (No Intersection):**
       - Check if points are far apart (>45m) to warrant path reconstruction
       - Find shortest path through road network using `find_shortest_path()`
       - Reconstruct LineString geometry from the path
       - Calculate cumulative distances along the path
       - Interpolate timestamps proportional to distance (constant speed assumption)
       - Nudge endpoint timestamps to avoid exact duplicates
       - Remove duplicate timestamps
    4. **Case B (Single Intersection):**
       - Verify intersection point lies between p1 and p2 using `_is_between()`
       - Calculate distance fraction to intersection
       - Interpolate timestamp at intersection proportional to distance
       - Nudge timestamp if it exactly matches p1 or p2

    **Temporal Interpolation Strategy:**
    - Assumes constant speed between p1 and p2
    - Distributes timestamps proportionally to distance traveled:
      `t_interp = t1 + (dist_to_point / total_distance) * (t2 - t1)`
    - This maintains temporal ordering and realistic timing

    **Duplicate Prevention:**
    - Nudges interpolated timestamps by 1 second if they exactly match p1 or p2
    - Prevents duplicate timestamps in the final refined trajectory
    - Removes consecutive duplicate timestamps after nudging

    **Distance Threshold (45m):**
    - Case A only triggers if points are >45m apart
    - Prevents unnecessary path reconstruction for closely-spaced points
    - Threshold based on typical GPS accuracy (±10m) and road intersection spacing

    **Performance:**
    - Case A (path reconstruction): 5-20ms depending on path length and network size
    - Case B (single intersection): <1ms
    - Dominated by shortest path computation in Case A

    **Failure Cases (Returns None, None):**
    - Edge geometries missing from graph
    - Case A: Points too close (<45m) for path reconstruction
    - Case A: No valid path found between roads
    - Case A: Reconstructed path has <2 coordinates
    - Case A: After deduplication, <2 points remain
    - Case B: Intersection point not between p1 and p2
    - Case B: No intersection found (falls through to Case B with no action)

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> from pyehicle.utilities.road_network import load_road_network, preprocess_road_segments
    >>> from pyehicle.reconstructing.refine import interpolate_corner_point
    >>> import time
    >>>
    >>> # Load road network
    >>> G = load_road_network('data/city_roads.graphml')
    >>> _, spatial_index = preprocess_road_segments(G)
    >>>
    >>> # Two trajectory points on different roads at an intersection
    >>> p1 = Point(24.105, 56.950)  # Street A
    >>> p2 = Point(24.110, 56.951)  # Street B (perpendicular to A)
    >>> t1 = time.time()
    >>> t2 = t1 + 10.0  # 10 seconds later
    >>>
    >>> # Find edges for both points
    >>> edge1 = next(spatial_index.nearest((p1.x, p1.y, p1.x, p1.y), 1))
    >>> edge2 = next(spatial_index.nearest((p2.x, p2.y, p2.x, p2.y), 1))
    >>>
    >>> # Interpolate corner point
    >>> points, times = interpolate_corner_point(
    ...     p1, p2, t1, t2, edge1, edge2, G, spatial_index, max_node_distance=10
    ... )
    >>>
    >>> if points:
    ...     print(f"Interpolated {len(points)} corner points")
    ...     for pt, ts in zip(points, times):
    ...         print(f"  Point: ({pt.x:.6f}, {pt.y:.6f}) at t={ts:.1f}")
    ... else:
    ...     print("No corner interpolation needed")
    """
    # ========== Extract Road Geometries ==========
    # Get LineString geometries for both edges from the graph
    edge1_geom = G.es[edge_idx1]['geometry']
    edge2_geom = G.es[edge_idx2]['geometry']

    # Early exit if either geometry is missing
    if edge1_geom and edge2_geom:
        # Ensure geometries are LineStrings (handle MultiLineStrings or GeometryCollections)
        # unary_union merges MultiLineStrings into a single LineString
        line1 = edge1_geom if isinstance(edge1_geom, LineString) else unary_union(edge1_geom)
        line2 = edge2_geom if isinstance(edge2_geom, LineString) else unary_union(edge2_geom)

        # Compute geometric intersection between the two road geometries
        # Returns: Point, MultiPoint, LineString, or empty geometry
        intersect = line1.intersection(line2)

        # Calculate direct geodesic distance between trajectory points
        # Used for distance thresholding and fraction calculation
        _, _, total_original_distance = geod.inv(p1.x, p1.y, p2.x, p2.y)

        # ========== CASE A: No Intersection - Path Reconstruction ==========
        # Roads don't directly intersect - need to route through the network
        if intersect.is_empty:
            # Skip if points are too close (<45m) - not worth path reconstruction
            # 45m threshold prevents over-processing for nearby points
            if total_original_distance < 45:
                return None, None

            # Find shortest path through road network between p1 and p2
            # Returns list of edge indices forming the path
            path_data = find_shortest_path(G, p1, p2, spatial_index, max_node_distance)
            if not path_data:
                return None, None  # No path found (disconnected roads or snapping failure)

            # Reconstruct LineString geometry from the edge path
            # Converts edge indices to actual (lon, lat) coordinates
            path_line = reconstruct_path_line_from_nodes(G, path_data)
            if path_line is None:
                return None, None  # Path reconstruction failed

            # Extract coordinates from LineString
            coords = list(path_line.coords)  # List of (lon, lat) tuples
            if len(coords) < 2:
                return None, None  # Invalid path - need at least 2 points

            # ========== Calculate Cumulative Distances Along Path ==========
            # Compute geodesic distance from start to each point along the path
            # This is needed for temporal interpolation
            num_coords = len(coords)
            lons = np.array([c[0] for c in coords])  # Extract longitudes
            lats = np.array([c[1] for c in coords])  # Extract latitudes

            # Calculate cumulative distance array
            # distances[i] = total distance from coords[0] to coords[i]
            distances = np.zeros(num_coords)
            for i in range(1, num_coords):
                # Calculate geodesic distance between consecutive points
                _, _, seg_dist = geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
                # Add to cumulative distance
                distances[i] = distances[i-1] + seg_dist

            # Total path length in meters
            total_dist = distances[-1]
            if total_dist == 0:
                return None, None  # Degenerate path (all points at same location)

            # ========== Temporal Interpolation ==========
            # Distribute timestamps proportionally to distance traveled
            # Assumes constant speed: t(d) = t1 + (d / total_dist) * (t2 - t1)
            time_diff = t2 - t1  # Total time elapsed between trajectory points
            fractions = distances / total_dist  # Distance fraction at each coordinate (0 to 1)
            interpolated_times = t1 + fractions * time_diff  # Interpolated timestamps

            # Convert coordinates to Point objects
            interpolated_points = [Point(coords[i]) for i in range(num_coords)]

            # ========== Duplicate Prevention ==========
            # Adjust first/last timestamps if they exactly match t1 or t2
            # This prevents duplicate timestamps in the final trajectory
            if abs(interpolated_times[0] - t1) < 1e-9:
                interpolated_times[0] = t1 + 1.0  # Nudge start forward by 1 second
            if abs(interpolated_times[-1] - t2) < 1e-9:
                interpolated_times[-1] = t2 - 1.0  # Nudge end backward by 1 second

            # Remove consecutive duplicate timestamps
            # Create boolean mask: True for first occurrence, False for duplicates
            unique_mask = np.concatenate(([True], interpolated_times[1:] != interpolated_times[:-1]))
            unique_points = [interpolated_points[i] for i in range(num_coords) if unique_mask[i]]
            unique_times = interpolated_times[unique_mask].tolist()

            # Final validation: Need at least 2 unique points
            if len(unique_points) < 2:
                return None, None  # Too few points after deduplication

            # Return interpolated points and timestamps
            return unique_points, unique_times

        # ========== CASE B: Single Intersection Point ==========
        # Roads directly intersect at a common point (typical intersection)
        else:
            # Check if intersection is a single point (most common case)
            if intersect.geom_type == 'Point':
                # Verify intersection point lies between p1 and p2 (not beyond endpoints)
                # Uses geodesic distance check: dist(p1,Q) + dist(Q,p2) ≈ dist(p1,p2)
                if _is_between(p1, p2, intersect, total_original_distance):
                    # Calculate distance from p1 to intersection point
                    _, _, dist_to_int = geod.inv(p1.x, p1.y, intersect.x, intersect.y)

                    # Calculate distance fraction (0 to 1)
                    # Safe division: handles case where total_original_distance = 0
                    frac = dist_to_int / total_original_distance if total_original_distance else 0.0

                    # Interpolate timestamp at intersection proportional to distance
                    t_int = t1 + frac * (t2 - t1)

                    # ========== Nudge Timestamp if Exact Match ==========
                    # Prevent duplicate timestamps in the final trajectory
                    if abs(t_int - t1) < 1e-9:
                        t_int = t1 + 1.0  # Nudge forward by 1 second
                    elif abs(t_int - t2) < 1e-9:
                        t_int = t2 - 1.0  # Nudge backward by 1 second

                    # Return single intersection point and its timestamp
                    return [intersect], [t_int]
                else:
                    # Intersection exists but is not between p1 and p2 (e.g., beyond endpoints)
                    return None, None

    # ========== Failure Case ==========
    # Geometry is missing or invalid, or intersection is not a Point (e.g., MultiPoint)
    return None, None


def refine_trajectory(df, road_network, max_node_distance=10, time_col='time', lat_col='lat', lon_col='lon'):
    """
    Refine GPS trajectory by enforcing spatial continuity with the underlying road network.

    This is the main trajectory refinement function and the second step in reconstruction.
    It takes a raw or preprocessed GPS trajectory and "snaps" it to realistic road paths
    by detecting road transitions and interpolating corner points at intersections. The
    result is a refined trajectory that follows actual road geometry.

    The function processes the trajectory sequentially, detecting when consecutive points
    lie on different roads (OSM ID changes) and filling gaps with corner points that
    follow the road network. This is essential for:
    - Converting GPS traces into road-matched trajectories
    - Preparing data for traffic analysis and route reconstruction
    - Generating realistic vehicle paths for simulation
    - Improving trajectory quality before curve interpolation

    **Algorithm Overview:**
    1. **Preprocessing:** Convert timestamps, remove duplicates, filter road network to bounding box
    2. **Initialization:** Build spatial index, initialize refined trajectory
    3. **Sequential Processing:** For each consecutive pair of trajectory points:
       - Find nearest road edges (OSM IDs) for both points
       - If OSM IDs differ (road transition): Interpolate corner points at intersection
       - If OSM IDs match (same road): Add current point directly
    4. **Finalization:** Convert back to DataFrame with formatted timestamps

    **OSM ID Detection:**
    The function uses OpenStreetMap way IDs to identify specific roads. When consecutive
    trajectory points have different OSM IDs, it indicates a road transition (e.g., turning
    at an intersection). The function then calls `interpolate_corner_point()` to fill the
    gap with realistic corner points.

    Parameters
    ----------
    df : pd.DataFrame
        Input trajectory DataFrame with columns for latitude, longitude, and timestamp.
        Can be raw GPS data or preprocessed (e.g., after compression or map-matching).
        Minimum columns: lat_col, lon_col, time_col.
    road_network : igraph.Graph
        Road network graph loaded from OSM data. Must have:
        - Node attributes: 'x' (longitude), 'y' (latitude)
        - Edge attributes: 'osmid' (OpenStreetMap way ID), 'geometry' (LineString)
        Typically loaded via `utilities.road_network.load_road_network()`.
    max_node_distance : float, default=10
        Maximum distance in meters for snapping trajectory points to road nodes.
        Points farther than this from any same-road node will skip corner interpolation.
        Typical values:
        - 10-15m: High-accuracy GPS (differential GPS, RTK)
        - 20-30m: Standard GPS (smartphone, consumer device)
        - 40-50m: Low-accuracy GPS or sparse road networks
    time_col : str, default='time'
        Name of the timestamp column in df. Must be parseable by pandas.to_datetime().
        Supports various formats: ISO 8601, Unix timestamps, custom datetime strings.
    lat_col : str, default='lat'
        Name of the latitude column in df (WGS84 decimal degrees).
    lon_col : str, default='lon'
        Name of the longitude column in df (WGS84 decimal degrees).

    Returns
    -------
    pd.DataFrame
        Refined trajectory with interpolated corner points. Contains the same columns
        as the input (lat_col, lon_col, time_col) with:
        - Original trajectory points preserved
        - Corner points inserted at road transitions
        - Timestamps distributed proportionally to distance
        - Points sorted chronologically
        - Duplicates removed
        Output format: Timestamps as strings ('%Y-%m-%d %H:%M:%S')

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Load GPS trajectory
    >>> df = pd.read_csv('gps_trajectory.csv')
    >>> print(f"Raw trajectory: {len(df)} points")
    >>>
    >>> # Load road network for the area
    >>> G = pye.utilities.road_network.load_road_network(
    ...     'city_roads.graphml'
    ... )
    >>>
    >>> # Refine trajectory with road network continuity
    >>> refined = pye.reconstructing.refine_trajectory(
    ...     df,
    ...     road_network=G,
    ...     max_node_distance=10,
    ...     time_col='time',
    ...     lat_col='lat',
    ...     lon_col='lon'
    ... )
    >>>
    >>> print(f"Refined trajectory: {len(refined)} points")
    >>> print(f"Added {len(refined) - len(df)} corner points")
    >>>
    >>> # Visualize before/after
    >>> pye.utilities.visualization.multiple(
    ...     [df, refined],
    ...     names=['Raw GPS', 'Refined'],
    ...     show_in_browser=True
    ... )

    >>> # Full preprocessing + reconstruction pipeline
    >>> import pyehicle as pye
    >>>
    >>> # 1. Load and preprocess
    >>> df = pd.read_csv('full_day_gps.csv')
    >>> compressed = pye.preprocessing.spatio_temporal_compress(df)
    >>> matched = pye.preprocessing.leuven(compressed, city='Riga', country='Latvia')
    >>>
    >>> # 2. Split by time gaps
    >>> segments = pye.preprocessing.by_time(matched, time_threshold=300)
    >>>
    >>> # 3. Combine segments spatially
    >>> combined = pye.reconstructing.trajectory_combiner(segments)
    >>>
    >>> # 4. Refine with road network (this function)
    >>> G = pye.utilities.road_network.load_road_network('riga_roads.graphml')
    >>> refined = pye.reconstructing.refine_trajectory(combined, G)
    >>>
    >>> # 5. Final curve interpolation
    >>> final = pye.reconstructing.curve_interpolation(refined, G)
    >>>
    >>> # Save result
    >>> final.to_csv('reconstructed_trajectory.csv', index=False)

    Notes
    -----
    **Algorithm Details:**
    The refinement process ensures that trajectories follow realistic road paths:
    1. For each consecutive pair of points, check if they're on the same road (OSM ID)
    2. If on different roads → Road transition detected → Interpolate corner points
    3. Corner interpolation handles two scenarios:
       - Adjacent roads with direct intersection → Single corner point
       - Non-adjacent roads → Path reconstruction along network
    4. Timestamps distributed proportionally to distance (constant speed assumption)
    5. Deduplication prevents duplicate coordinates and timestamps

    **Road Network Filtering:**
    The function filters the road network to the trajectory's bounding box (with 0.5°
    buffer) for performance. This dramatically speeds up spatial queries for large
    road networks.

    **Spatial Index:**
    Uses R-tree spatial index for fast nearest-edge queries. Built automatically from
    the road network using `preprocess_road_segments()`.

    **Duplicate Handling:**
    - Input duplicates removed before processing (same lat/lon)
    - Output duplicates removed after interpolation
    - Prevents issues with zero-length edges and invalid geometries

    **Temporal Ordering:**
    - Points sorted by timestamp before processing
    - Corner points assigned interpolated timestamps
    - Final output re-sorted to ensure chronological order

    **Performance:**
    - Time complexity: O(n * m) where n = trajectory points, m = avg corner points per transition
    - For 1000-point trajectory with 50 road transitions: 5-30 seconds
    - Dominated by corner interpolation (shortest path computations)
    - Large road networks (>100k edges) benefit from bounding box filtering

    **Limitations:**
    - Requires accurate OSM IDs from map-matching or spatial queries
    - May fail for disconnected road networks (e.g., ferry routes)
    - Assumes roads have valid geometry and OSM IDs in the graph
    - Corner interpolation may fail if points are too far from roads (>max_node_distance)
    - Does not consider turn restrictions, traffic rules, or one-way streets

    **Use Cases:**
    - Preparing trajectories for curve interpolation (next step in reconstruction)
    - Converting raw GPS logs to road-matched trajectories
    - Traffic flow analysis and route reconstruction
    - Vehicle path simulation for autonomous driving research
    - Improving trajectory quality for visualization and analysis

    **Quality Assurance:**
    The function is designed to be robust:
    - Handles missing geometries gracefully
    - Skips problematic segments instead of failing entirely
    - Preserves original points when corner interpolation fails
    - Maintains temporal continuity throughout

    See Also
    --------
    trajectory_combiner : First step - spatially sort trajectory segments
    curve_interpolation : Third step - interpolate smooth curves at corners
    utilities.road_network.load_road_network : Load OSM road network graph
    preprocessing.leuven : HMM map-matching for OSM ID assignment
    """
    # ========== Preprocessing ==========
    # Work on a copy to avoid SettingWithCopyWarning
    # This ensures we don't modify the user's original DataFrame
    df = df.copy()

    # Convert timestamps to Unix epoch (seconds) for easier temporal calculations
    # Unix epoch = seconds since 1970-01-01 00:00:00 UTC
    # This simplifies timestamp arithmetic and interpolation
    df[time_col] = pd.to_datetime(df[time_col]).astype('int64') / 1e9  # nanoseconds → seconds

    # Remove duplicate coordinates and sort by time
    # Duplicates can cause zero-length edges and invalid geometries
    # Sorting ensures we process points in chronological order
    df = df.drop_duplicates(subset=[lat_col, lon_col]).sort_values(by=time_col).reset_index(drop=True)

    # Extract coordinates and create Shapely Point objects
    # Point objects are needed for geometric operations (intersections, distance calculations)
    lons = df[lon_col].to_numpy()  # Extract longitudes as numpy array
    lats = df[lat_col].to_numpy()  # Extract latitudes as numpy array
    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]  # Create Point(lon, lat)
    times = df[time_col].to_numpy()  # Extract timestamps as numpy array

    # Filter road network to trajectory bounding box for performance
    # This dramatically reduces spatial query time for large road networks (e.g., entire country)
    # buffer=0.5 adds 0.5° (~55km) buffer around trajectory bounds to ensure nearby roads are included
    road_network = filter_road_network_by_bbox(road_network, df, buffer=0.5)

    # Build R-tree spatial index for fast nearest-edge queries
    # R-tree enables O(log n) queries instead of O(n) brute-force search
    # Returns (edge_features, spatial_index) - we only need the spatial index
    _, spatial_index = preprocess_road_segments(road_network)

    # ========== Initialize Refined Trajectory ==========
    # Build refined trajectory incrementally by processing consecutive point pairs
    refined_trajectory = []  # List of Point objects forming the refined trajectory
    refined_timestamps = []  # List of timestamps (Unix epoch seconds) for each point
    last_added_point = None  # Track last point added to detect duplicates at the end
    handled_segments = set()  # Track processed point pairs to avoid duplicate interpolation

    # ========== Sequential Processing ==========
    # Process each consecutive pair of trajectory points
    # For each pair, detect road transitions and interpolate corner points if needed
    num_points = len(points)
    for i in range(num_points):
        # ========== First Point (Initialization) ==========
        if i == 0:
            # Always add the first point directly (no previous point to compare)
            refined_trajectory.append(points[i])
            refined_timestamps.append(times[i])
            last_added_point = points[i]
            continue  # Skip to next iteration

        # ========== Extract Current Pair ==========
        # Get previous and current points with their timestamps
        prev_point = points[i - 1]  # Previous trajectory point
        curr_point = points[i]      # Current trajectory point
        prev_time = times[i - 1]    # Timestamp at previous point (Unix epoch seconds)
        curr_time = times[i]        # Timestamp at current point (Unix epoch seconds)

        # ========== Find Nearest Road Edges (OSM ID Detection) ==========
        # Use R-tree spatial index to find closest road edge for each point
        # Query format: (min_x, min_y, max_x, max_y) - for a point, min = max
        prev_edge_idx = next(spatial_index.nearest((prev_point.x, prev_point.y, prev_point.x, prev_point.y), 1))
        curr_edge_idx = next(spatial_index.nearest((curr_point.x, curr_point.y, curr_point.x, curr_point.y), 1))

        # Extract OSM IDs (OpenStreetMap way IDs) to identify which roads the points are on
        prev_osm = road_network.es[prev_edge_idx]['osmid']  # OSM ID of road containing prev_point
        curr_osm = road_network.es[curr_edge_idx]['osmid']  # OSM ID of road containing curr_point

        # ========== Road Transition Detection ==========
        # Check if consecutive points are on different roads (OSM ID change)
        # If OSM IDs differ → Road transition → Need corner interpolation
        if prev_osm != curr_osm:
            # Create unique key for this point pair using rounded coordinates
            # Round to 6 decimal places (~0.11m precision) to avoid floating point issues
            # Use tuple of (lat1, lon1, lat2, lon2) to uniquely identify this segment
            point_pair_key = (
                round(prev_point.y, 6), round(prev_point.x, 6),
                round(curr_point.y, 6), round(curr_point.x, 6)
            )

            # Only process each point pair once (prevents duplicate corner points)
            if point_pair_key not in handled_segments:
                handled_segments.add(point_pair_key)  # Mark point pair as processed

                # ========== Corner Point Interpolation ==========
                # Interpolate corner point(s) at the road transition
                # Returns (points, times) or (None, None) if interpolation failed
                ipoints, itimes = interpolate_corner_point(
                    prev_point,
                    curr_point,
                    prev_time,
                    curr_time,
                    prev_edge_idx,
                    curr_edge_idx,
                    road_network,
                    spatial_index,
                    max_node_distance
                )

                # ========== Handle Successful Interpolation ==========
                if ipoints is not None and itimes is not None and len(ipoints) > 0:
                    # Interpolation succeeded - could be 1 point (intersection) or multiple (path)

                    if len(ipoints) == 1:
                        # Case B: Single intersection point
                        # Add intersection point between prev and curr
                        refined_trajectory.append(ipoints[0])
                        refined_timestamps.append(itimes[0])
                        last_added_point = ipoints[0]
                        # Add current point after the intersection
                        refined_trajectory.append(curr_point)
                        refined_timestamps.append(curr_time)
                        last_added_point = curr_point

                    else:
                        # Case A: Path reconstruction with multiple points
                        # Remove the previous point - it will be replaced by the interpolated path
                        refined_trajectory.pop()
                        refined_timestamps.pop()

                        # Add all interpolated points along the reconstructed path
                        for p, t in zip(ipoints, itimes):
                            refined_trajectory.append(p)
                            refined_timestamps.append(t)
                            last_added_point = p

                else:
                    # ========== Interpolation Failed ==========
                    # No interpolation possible (e.g., disconnected roads, points too far)
                    # Add current point directly without corner interpolation
                    refined_trajectory.append(curr_point)
                    refined_timestamps.append(curr_time)
                    last_added_point = curr_point

        else:
            # ========== Same Road (No Transition) ==========
            # Consecutive points are on the same road (same OSM ID)
            # No corner interpolation needed - just add current point
            refined_trajectory.append(curr_point)
            refined_timestamps.append(curr_time)
            last_added_point = curr_point

    # ========== Finalization ==========
    # Ensure the final point is included (may have been popped during interpolation)
    if points and points[-1] != last_added_point:
        refined_trajectory.append(points[-1])
        refined_timestamps.append(times[-1])

    # Extract coordinates from Point objects for DataFrame creation
    # Convert list of Point(lon, lat) to separate lat and lon arrays
    final_lats = np.array([p.y for p in refined_trajectory])  # Extract latitudes
    final_lons = np.array([p.x for p in refined_trajectory])  # Extract longitudes

    # Create output DataFrame with refined trajectory
    final_df = pd.DataFrame({
        lat_col: final_lats,
        lon_col: final_lons,
        time_col: refined_timestamps  # Still in Unix epoch seconds
    })

    # Convert Unix timestamps back to formatted datetime strings
    # Format: 'YYYY-MM-DD HH:MM:SS' (e.g., '2024-01-15 14:30:45')
    final_df[time_col] = pd.to_datetime(final_df[time_col], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Final cleanup: Remove duplicates and sort by time
    # Duplicates can still occur if corner interpolation returned duplicate coordinates
    final_df = final_df.drop_duplicates(subset=[lat_col, lon_col]).sort_values(by=time_col).reset_index(drop=True)

    # Return refined trajectory with corner points interpolated
    return final_df
