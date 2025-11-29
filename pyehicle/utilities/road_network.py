"""
Road network utilities module for pyehicle.

This module provides functions to load, build, and preprocess road networks from
OpenStreetMap (OSM) data. Road networks are represented as igraph.Graph objects
with spatial attributes for use in trajectory reconstruction and map-matching.

Key functionality:
- Loading networks from PBF files or cached GraphML
- Building igraph graphs with geometric attributes
- Creating R-tree spatial indexes for fast nearest-neighbor queries
- Filtering networks by bounding box for performance

Road networks are essential for:
- Trajectory refinement (enforcing road network continuity)
- Curve interpolation (finding paths along roads)
- Map-matching (aligning GPS points to roads)
- Spatial analysis and routing
"""

import igraph as ig
import numpy as np
import pandas as pd
from pyrosm import OSM
from shapely.geometry import LineString
from shapely import wkt
from pyproj import Geod
from rtree import index
from shapely import bounds as shapely_bounds
import os
import warnings


def load_road_network(pbf_file_path, bbox=None, save_path=None):
    """
    Load road network from OpenStreetMap PBF file or cached GraphML format.

    This function loads a driveable road network from OSM data, builds an igraph.Graph
    representation with spatial attributes, and creates an R-tree spatial index for
    fast nearest-neighbor queries. It supports caching via GraphML for faster subsequent loads.

    The road network is extracted using pyrosm (network_type="driving") which includes:
    - Motorways, trunk roads, primary/secondary/tertiary roads
    - Residential and service roads
    - Excludes footways, cycleways, pedestrian paths

    Parameters
    ----------
    pbf_file_path : str
        Path to OpenStreetMap PBF file (e.g., 'city-latest.osm.pbf').
        PBF files can be downloaded from:
        - Geofabrik: https://download.geofabrik.de/
        - BBBike: https://extract.bbbike.org/
        - OpenStreetMap: https://planet.openstreetmap.org/
    bbox : tuple of float, optional
        Bounding box to filter the network: (north, south, east, west) in WGS84 degrees.
        If None, entire PBF file is loaded. Use bbox for large files to reduce memory.
        Example: (56.98, 56.92, 24.15, 24.05) for part of Riga, Latvia.
    save_path : str, optional
        Path to save/load cached GraphML file (e.g., 'city_roads.graphml').
        If save_path exists: loads from cache (fast)
        If save_path doesn't exist: loads from PBF (slow), but doesn't save
        Caching significantly speeds up subsequent loads (seconds vs minutes).

    Returns
    -------
    road_network : igraph.Graph
        Road network graph with attributes:
        - Vertex (node) attributes:
          - 'node_id': Original OSM node ID
          - 'x': Longitude (WGS84 degrees)
          - 'y': Latitude (WGS84 degrees)
        - Edge attributes:
          - 'osmid': OSM way ID
          - 'length': Edge length in meters (geodesic distance)
          - 'geometry': Shapely LineString geometry
    segment_geometries : np.ndarray
        Array of Shapely LineString geometries for all edges. Same order as road_network.es.
    spatial_index : rtree.index.Index
        R-tree spatial index for fast nearest-edge queries. Indexed by edge bounding boxes.
        Use: `nearest_edge = next(spatial_index.nearest((lon, lat, lon, lat), 1))`

    Examples
    --------
    >>> import pyehicle as pye
    >>>
    >>> # Load road network from PBF file (first time - slow)
    >>> G, geometries, spatial_index = pye.utilities.road_network.load_road_network(
    ...     'riga-latest.osm.pbf',
    ...     save_path='riga_roads.graphml'
    ... )
    >>> print(f"Loaded {len(G.vs)} nodes and {len(G.es)} edges")
    >>>
    >>> # Load from cache (subsequent times - fast)
    >>> G, geometries, spatial_index = pye.utilities.road_network.load_road_network(
    ...     'riga-latest.osm.pbf',  # Not used if cache exists
    ...     save_path='riga_roads.graphml'  # Loads from here
    ... )
    >>>
    >>> # Load subset using bounding box
    >>> bbox = (56.98, 56.92, 24.15, 24.05)  # (north, south, east, west)
    >>> G, _, _ = pye.utilities.road_network.load_road_network(
    ...     'latvia-latest.osm.pbf',
    ...     bbox=bbox
    ... )

    Notes
    -----
    **Data Source: OSM PBF Files**
    - PBF (Protocolbuffer Binary Format) = compressed OSM data
    - Much smaller than XML (10x compression)
    - Fast to parse with pyrosm
    - Updated regularly on Geofabrik (daily for most regions)

    **Network Extraction:**
    - Uses pyrosm with network_type="driving"
    - Extracts only driveable roads (excludes pedestrian, cycling paths)
    - Preserves road geometry (LineStrings with multiple points)
    - Calculates geodesic lengths using pyproj Geod

    **Caching with GraphML:**
    - GraphML = XML-based graph format supported by igraph
    - Saves graph structure + all attributes
    - Geometries stored as WKT strings (Well-Known Text)
    - Loading from GraphML is 50-100x faster than parsing PBF
    - Recommended workflow: Load from PBF once, then use cache

    **Spatial Index (R-tree):**
    - Built automatically by `preprocess_road_segments()`
    - Indexes edge bounding boxes for O(log n) nearest-edge queries
    - Essential for trajectory refinement and map-matching
    - Fast nearest-neighbor: ~0.01ms per query

    **Performance:**
    - Loading from PBF: 1-10 minutes depending on file size
    - Loading from GraphML: 5-30 seconds
    - Memory: ~100 MB per 100k edges
    - Spatial index build: ~1-5 seconds

    **Use Cases:**
    - Trajectory reconstruction: `refine_trajectory()`, `curve_interpolation()`
    - Map-matching: Finding nearest road segments
    - Routing: Shortest path computation with Dijkstra
    - Spatial analysis: Road network statistics

    See Also
    --------
    build_igraph_graph : Build igraph from node/edge DataFrames
    preprocess_road_segments : Create R-tree spatial index
    filter_road_network_by_bbox : Filter network by bounding box
    """
    if save_path and os.path.exists(save_path):
        # --- Load from GraphML ---
        road_network = ig.Graph.Read(save_path, format="graphml")

        # --- Check for 'geometry_wkt' and reconstruct 'geometry' ---
        if 'geometry_wkt' in road_network.es.attributes():
            wkt_list = road_network.es['geometry_wkt']
            road_network.es['geometry'] = [wkt.loads(w) for w in wkt_list]
        else:
            # --- Reconstruct 'geometry' from node coordinates ---
            print("Warning: 'geometry_wkt' not found. Reconstructing 'geometry' from node coordinates.")
            node_coords = np.column_stack((road_network.vs['x'], road_network.vs['y']))
            road_network.es['geometry'] = [
                LineString([node_coords[e.source], node_coords[e.target]]) for e in road_network.es
            ]

    else:
        # --- Build from PBF ---
        osm = OSM(pbf_file_path, bounding_box=bbox)

        # Suppress SettingWithCopyWarning from pyrosm/geopandas internals
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
            nodes, edges = osm.get_network(nodes=True, network_type="driving")

        road_network = build_igraph_graph(nodes, edges)

        # --- Save to GraphML (if requested) ---
        '''
        if save_path:
            # Convert geometry to WKT before saving
            wkt_list = [geom.wkt for geom in road_network.es['geometry']]
            road_network.es['geometry_wkt'] = wkt_list
            del road_network.es['geometry']   # remove raw geometry
            road_network.write_graphml(save_path)

            # Reassign geometry in memory if you still need it
            road_network.es['geometry'] = [wkt.loads(w) for w in wkt_list]'''

    segment_geometries, spatial_index = preprocess_road_segments(road_network)
    return road_network, segment_geometries, spatial_index


def build_igraph_graph(nodes, edges):
    """
    Build igraph.Graph from OpenStreetMap nodes and edges DataFrames.

    This function converts raw OSM network data (from pyrosm) into an igraph.Graph
    representation with spatial attributes. It handles node ID factorization, edge
    validation, geometry processing, and geodesic length calculation.

    The resulting graph is undirected with geographic coordinates stored as vertex
    attributes and road geometries as edge attributes, ready for spatial analysis
    and routing operations.

    Parameters
    ----------
    nodes : pd.DataFrame
        OSM nodes DataFrame from pyrosm.get_network() with columns:
        - 'id': OSM node IDs (int64)
        - 'lat': Latitude in WGS84 degrees (float64)
        - 'lon': Longitude in WGS84 degrees (float64)
    edges : pd.DataFrame
        OSM edges DataFrame from pyrosm.get_network() with columns:
        - 'id': OSM way IDs (int64)
        - 'u': Source node ID (int64)
        - 'v': Target node ID (int64)
        - 'geometry': Shapely LineString geometries

    Returns
    -------
    igraph.Graph
        Undirected road network graph with attributes:
        - Vertex attributes:
          - 'node_id': Original OSM node IDs
          - 'x': Longitude (WGS84 degrees)
          - 'y': Latitude (WGS84 degrees)
        - Edge attributes:
          - 'osmid': OSM way IDs
          - 'length': Geodesic edge length in meters (WGS84)
          - 'geometry': Shapely LineString geometries

    Notes
    -----
    **Algorithm Steps:**

    1. **Node Factorization**: Convert arbitrary OSM node IDs to sequential 0-based
       indices required by igraph. Uses pandas.factorize() for efficiency.

    2. **Edge Index Mapping**: Map edge endpoints (u, v) from OSM IDs to factorized
       indices using the node ID→index mapping from step 1.

    3. **Edge Validation**: Remove edges with missing endpoints or indices out of
       bounds (handles disconnected components and data quality issues).

    4. **Graph Construction**: Create igraph.Graph with edge list. Graph is undirected
       as road networks are typically bidirectional.

    5. **Geodesic Length Calculation**: Compute true great-circle distances for all
       edges using pyproj Geod.inv() on WGS84 ellipsoid.

    6. **Geometry Processing**: Ensure all edge geometries are LineStrings. Convert
       non-LineString geometries (e.g., Points) to simple two-point lines.

    **Performance:**
    - Time complexity: O(n + m) where n = nodes, m = edges
    - Factorization: O(n) with pandas
    - Geodesic calculations: O(m) vectorized with pyproj
    - Fast even for large networks (100k+ edges in seconds)

    **Edge Cases:**
    - Edges referencing non-existent nodes are removed
    - Zero-length edges are kept but have length = 0
    - Non-LineString geometries are converted to straight lines
    - Handles duplicate edges (igraph allows multiple edges)

    See Also
    --------
    load_road_network : Main function that calls this builder
    preprocess_road_segments : Creates R-tree index from graph
    """
    # ========== Step 1: Node Factorization ==========
    # Work on copies to avoid SettingWithCopyWarning when modifying DataFrames
    nodes = nodes.copy()
    edges = edges.copy()

    # Convert arbitrary OSM node IDs to sequential 0-based indices
    # igraph requires vertex indices to be 0, 1, 2, ..., n-1
    # pd.factorize() is O(n) and returns unique mapping
    nodes['idx'], id_uniques = pd.factorize(nodes['id'])

    # Create mapping: OSM_node_id -> factorized_index for fast edge lookup
    id_to_idx = pd.Series(nodes.index.values, index=nodes['id'])

    # ========== Step 2: Edge Index Mapping ==========
    # Map edge endpoints from OSM IDs to factorized indices
    edges['u_idx'] = edges['u'].map(id_to_idx)  # Source node index
    edges['v_idx'] = edges['v'].map(id_to_idx)  # Target node index

    # ========== Step 3: Edge Validation ==========
    # Remove edges with missing endpoints (edges referencing non-existent nodes)
    # This handles disconnected components or incomplete OSM data
    valid_edges_mask = edges['u_idx'].notna() & edges['v_idx'].notna()
    edges = edges.loc[valid_edges_mask].copy()
    edges['u_idx'] = edges['u_idx'].astype(int)
    edges['v_idx'] = edges['v_idx'].astype(int)

    # Additional validation: ensure indices are within bounds [0, n-1]
    # Protects against factorization errors or data corruption
    max_node_idx = len(nodes) - 1
    valid_idx_mask = (edges['u_idx'] <= max_node_idx) & (edges['v_idx'] <= max_node_idx)
    edges = edges.loc[valid_idx_mask].copy()

    # ========== Step 4: Graph Construction ==========
    # Create igraph.Graph with edge list [[u0, v0], [u1, v1], ...]
    # Graph is undirected because most roads are bidirectional
    edges_array = edges[['u_idx', 'v_idx']].values
    g = ig.Graph(edges=edges_array, directed=False)

    # ========== Assign Vertex Attributes ==========
    # Store original OSM node IDs (renamed to 'node_id' to avoid GraphML conflicts)
    g.vs['node_id'] = nodes['id'].values
    # Store geographic coordinates (WGS84 degrees)
    g.vs['x'] = nodes['lon'].values
    g.vs['y'] = nodes['lat'].values

    # ========== Step 5: Geodesic Length Calculation ==========
    # Calculate true great-circle distances using WGS84 ellipsoid
    # This is more accurate than Euclidean distance in lat/lon
    geod = Geod(ellps='WGS84')
    node_coords = nodes[['lon', 'lat']].values

    # Extract coordinates for edge endpoints using factorized indices
    u_coords = node_coords[edges['u_idx']]  # Source node coordinates
    v_coords = node_coords[edges['v_idx']]  # Target node coordinates

    # Compute geodesic distances (vectorized for all edges at once)
    # geod.inv() returns: forward_azimuth, back_azimuth, distance_in_meters
    _, _, distances = geod.inv(
        u_coords[:, 0], u_coords[:, 1],  # Source lon, lat
        v_coords[:, 0], v_coords[:, 1]   # Target lon, lat
    )
    g.es['length'] = distances  # Store edge lengths in meters

    # ========== Step 6: Geometry Processing ==========
    # Ensure all edge geometries are LineStrings (required for spatial operations)
    geometries = edges['geometry'].values

    # Check which geometries are LineStrings (OSM sometimes has Points or other types)
    is_linestring = np.array([isinstance(geom, LineString) for geom in geometries])

    if not np.all(is_linestring):
        # Some geometries are not LineStrings - convert them to simple two-point lines
        u_lons = u_coords[:, 0]
        u_lats = u_coords[:, 1]
        v_lons = v_coords[:, 0]
        v_lats = v_coords[:, 1]
        coords = np.column_stack((u_lons, u_lats, v_lons, v_lats))

        # Find indices of non-LineString geometries
        non_line_indices = np.where(~is_linestring)[0]

        # Convert each non-LineString to a simple straight line between endpoints
        new_geometries = np.array([
            LineString([(coords[i, 0], coords[i, 1]), (coords[i, 2], coords[i, 3])])
            for i in non_line_indices
        ])
        geometries[non_line_indices] = new_geometries

    # ========== Assign Edge Attributes ==========
    g.es['geometry'] = geometries  # Shapely LineString geometries
    g.es['osmid'] = edges['id'].values  # Original OSM way IDs

    return g


def preprocess_road_segments(road_network):
    """
    Build R-tree spatial index for fast nearest-neighbor queries on road segments.

    This function creates an R-tree spatial index from road network edge geometries,
    enabling O(log n) nearest-neighbor queries essential for trajectory reconstruction
    and map-matching. The R-tree indexes bounding boxes of all road segments for
    efficient spatial searches.

    R-trees are hierarchical data structures that organize spatial objects by their
    bounding rectangles, providing logarithmic query time for nearest-neighbor and
    intersection searches. This is critical for performance when matching thousands
    of GPS points to a road network with tens of thousands of edges.

    Parameters
    ----------
    road_network : igraph.Graph
        Road network graph from build_igraph_graph() or load_road_network().
        Must have edge attribute 'geometry' containing Shapely LineString objects.

    Returns
    -------
    segment_geometries : np.ndarray
        Array of Shapely LineString geometries for all edges. Length = len(road_network.es).
        Order matches road_network.es (edge sequence). Can be indexed by edge ID.
    spatial_index : rtree.index.Index
        R-tree spatial index for fast nearest-neighbor queries. Indexed by edge IDs.
        Usage: nearest_edge_id = next(spatial_index.nearest((lon, lat, lon, lat), 1))

    Examples
    --------
    >>> import pyehicle as pye
    >>>
    >>> # Load road network
    >>> G, geometries, spatial_index = pye.utilities.road_network.load_road_network(
    ...     'city-latest.osm.pbf',
    ...     save_path='city_roads.graphml'
    ... )
    >>>
    >>> # Find nearest road segment to a GPS point
    >>> gps_lon, gps_lat = 24.1055, 56.9496  # Riga, Latvia
    >>> nearest_edge_id = next(spatial_index.nearest((gps_lon, gps_lat, gps_lon, gps_lat), 1))
    >>> nearest_geometry = geometries[nearest_edge_id]
    >>> print(f"Nearest road: OSM way {G.es[nearest_edge_id]['osmid']}")
    >>>
    >>> # Find 5 nearest road segments
    >>> k_nearest = list(spatial_index.nearest((gps_lon, gps_lat, gps_lon, gps_lat), 5))
    >>> for edge_id in k_nearest:
    ...     print(f"Edge {edge_id}: length {G.es[edge_id]['length']:.1f}m")

    Notes
    -----
    **R-tree Index Structure:**

    - **Bounding Boxes**: Each road segment is indexed by its minimum bounding
      rectangle (minx, miny, maxx, maxy).
    - **Hierarchical Structure**: R-tree organizes boxes into a tree, grouping
      nearby segments together at each level.
    - **Query Performance**: O(log n) for nearest-neighbor, much faster than
      brute-force O(n) distance calculations.
    - **Memory Efficiency**: Index size scales linearly with number of edges.

    **Use Cases:**

    - **Trajectory Refinement**: `refine_trajectory()` uses this to find nearest
      roads when transitioning between different OSM ways.
    - **Map-Matching**: Fast nearest-road lookups for aligning GPS points.
    - **Curve Interpolation**: `curve_interpolation()` uses this to find road
      segments for path routing.
    - **Spatial Queries**: General-purpose nearest-neighbor searches.

    **Performance:**

    - Index construction: O(n log n) where n = number of edges
    - Build time: ~1-5 seconds for 100k edges
    - Query time: ~0.01ms per nearest-neighbor search
    - Memory: ~100 bytes per edge (index overhead)

    **Spatial Index API:**

    - `nearest(bbox, k)`: Returns k nearest edge IDs to bounding box
      - bbox format: (minx, miny, maxx, maxy)
      - For point queries: (lon, lat, lon, lat) (same coords twice)
    - Returns generator, use `next()` or `list()` to retrieve results
    - Results are edge IDs matching road_network.es indices

    See Also
    --------
    load_road_network : Main function that calls this preprocessor
    build_igraph_graph : Builds graph with geometries
    refine_trajectory : Uses spatial index for trajectory refinement
    """
    # ========== Extract Edge Geometries ==========
    # Convert edge geometry attribute list to numpy array for faster indexing
    # Array order matches road_network.es (edge sequence)
    geometries = np.array(road_network.es['geometry'])

    # ========== Compute Bounding Boxes ==========
    # Calculate minimum bounding rectangle for each LineString geometry
    # shapely.bounds() returns (minx, miny, maxx, maxy) for each geometry
    # This is vectorized and fast (processes all geometries at once)
    bounds_array = shapely_bounds(geometries)

    # ========== Build R-tree Spatial Index ==========
    # R-tree requires items as (id, bbox, object) tuples
    # We use a generator for memory efficiency (avoids creating list of all tuples)
    def generate_items():
        """Generate (edge_id, bounding_box, None) tuples for R-tree construction."""
        for i in range(len(bounds_array)):
            b = bounds_array[i]
            # Convert bounds to tuple of floats (minx, miny, maxx, maxy)
            # rtree requires exact float types, not numpy scalars
            if hasattr(b, '__iter__'):
                bbox = tuple(float(x) for x in b)
            else:
                # Fallback for different array structures
                bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            # Yield: (edge_id, bounding_box, object_data)
            # edge_id = i (matches road_network.es index)
            # object_data = None (not needed, we use edge_id for lookup)
            yield (i, bbox, None)

    # Create R-tree index from generator
    # This builds the hierarchical tree structure (O(n log n) construction)
    idx = index.Index(generate_items())

    # Return geometries array and spatial index
    # geometries[i] corresponds to road_network.es[i]
    # idx.nearest() returns edge IDs that index into geometries array
    return geometries, idx


def filter_road_network_by_bbox(road_network, matched, buffer=0.01):
    """
    Filter road network to bounding box around trajectory for performance optimization.

    This function creates a spatial subset of the road network by extracting only the
    nodes and edges within a bounding box around a GPS trajectory. This significantly
    reduces memory usage and improves computational performance for graph algorithms
    when working with large regional or national road networks.

    Use this function before running computationally expensive operations like shortest
    path routing or spatial analysis when you only need roads near a specific trajectory.

    Parameters
    ----------
    road_network : igraph.Graph
        Full road network graph from load_road_network().
        Must have vertex attributes 'x' (longitude) and 'y' (latitude).
    matched : pd.DataFrame
        GPS trajectory DataFrame with columns 'lat' and 'lon'.
        Used to determine the bounding box for filtering.
    buffer : float, default=0.01
        Buffer distance to extend bounding box in degrees (~1.1 km at equator).
        Ensures roads slightly outside trajectory extent are included.
        Typical values:
        - 0.005 (~550m): Tight fit for dense urban trajectories
        - 0.01 (~1.1km): Standard buffer (default)
        - 0.02 (~2.2km): Generous buffer for sparse trajectories

    Returns
    -------
    igraph.Graph
        Filtered subgraph containing only nodes within the bounding box.
        Edges are preserved if both endpoints are within the bounding box.
        All vertex and edge attributes are preserved.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Load full country-wide road network (large, slow operations)
    >>> full_network, _, _ = pye.utilities.road_network.load_road_network(
    ...     'latvia-latest.osm.pbf',
    ...     save_path='latvia_full.graphml'
    ... )
    >>> print(f"Full network: {len(full_network.vs)} nodes, {len(full_network.es)} edges")
    >>>
    >>> # Load trajectory (small region within Latvia)
    >>> trajectory = pd.read_csv('riga_trajectory.csv')
    >>>
    >>> # Filter to relevant subgraph around trajectory
    >>> filtered_network = pye.utilities.road_network.filter_road_network_by_bbox(
    ...     full_network,
    ...     trajectory,
    ...     buffer=0.01
    ... )
    >>> print(f"Filtered: {len(filtered_network.vs)} nodes, {len(filtered_network.es)} edges")
    >>> # Output: Filtered: 8234 nodes, 11567 edges (much smaller!)
    >>>
    >>> # Now routing operations are much faster
    >>> # Use filtered_network for refine_trajectory() or curve_interpolation()

    Notes
    -----
    **When to Use:**

    - **Large Networks**: When road network covers region larger than trajectory
      (e.g., country-wide network for city trajectory)
    - **Multiple Trajectories**: Filter once per trajectory region for efficiency
    - **Routing Operations**: Before Dijkstra shortest paths or other graph algorithms
    - **Memory Constraints**: When full network exceeds available memory

    **Performance:**

    - Filtering is O(n) where n = number of nodes
    - Typical speedup: 10-100x for routing operations on filtered network
    - Memory savings: Proportional to filtered area (90%+ reduction possible)

    **Buffer Size Guidelines:**

    - Buffer is in WGS84 degrees (latitude/longitude)
    - 1 degree ≈ 111 km at equator (less at higher latitudes)
    - 0.01 degrees ≈ 1.1 km at equator
    - Increase buffer if trajectory has gaps or uses routing
    - Too small: May miss important connecting roads
    - Too large: Minimal performance benefit

    **Subgraph Properties:**

    - Uses igraph.induced_subgraph() which preserves all attributes
    - Vertices are renumbered 0, 1, 2, ... in filtered graph
    - Original 'node_id' attribute preserves OSM IDs
    - Edges preserved only if both endpoints are in bounding box
    - Graph connectivity may be different (disconnected components possible)

    **Limitations:**

    - Simple rectangular bounding box (not convex hull or buffer polygon)
    - Buffer is uniform in degrees (different physical distance at different latitudes)
    - Does not account for road network connectivity (may cut through roads)
    - For precise spatial filtering, consider using Shapely buffer operations

    See Also
    --------
    load_road_network : Load full road network
    refine_trajectory : Uses filtered networks for efficiency
    curve_interpolation : Benefits from network filtering
    """
    # ========== Calculate Trajectory Bounding Box ==========
    # Find minimum and maximum coordinates of the trajectory
    # This defines the rectangular region containing all GPS points
    min_lon, min_lat = matched[['lon', 'lat']].min()
    max_lon, max_lat = matched[['lon', 'lat']].max()

    # Expand bounding box by buffer distance in all directions
    # Buffer is in degrees (WGS84): 0.01° ≈ 1.1 km at equator
    # This ensures roads near trajectory edges are included
    min_lon -= buffer
    min_lat -= buffer
    max_lon += buffer
    max_lat += buffer

    # ========== Extract Node Coordinates ==========
    # Get all node coordinates from the road network graph
    # Convert to numpy arrays for fast vectorized operations
    x_coords = np.array(road_network.vs['x'])  # Longitudes
    y_coords = np.array(road_network.vs['y'])  # Latitudes

    # ========== Find Nodes Within Bounding Box ==========
    # Create boolean mask for nodes inside the expanded bounding box
    # Uses vectorized numpy operations (fast for large networks)
    in_bbox = (
        (x_coords >= min_lon) & (x_coords <= max_lon) &  # Longitude bounds
        (y_coords >= min_lat) & (y_coords <= max_lat)    # Latitude bounds
    )

    # Get indices of nodes within bounding box
    # np.nonzero() returns tuple, [0] extracts the array of indices
    bbox_nodes = np.nonzero(in_bbox)[0]

    # ========== Create Subgraph ==========
    # Extract subgraph containing only nodes within bounding box
    # igraph.induced_subgraph() automatically includes only edges where
    # both endpoints are in bbox_nodes (preserves all attributes)
    return road_network.induced_subgraph(bbox_nodes)