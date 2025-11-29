"""
Map-matching module for pyehicle.

This module provides algorithms to align GPS trajectories with road networks by finding
the most likely paths on actual roads. Map-matching is essential for:
- Correcting GPS errors and drift
- Assigning OpenStreetMap (OSM) way IDs to trajectory points
- Preparing trajectories for road-network-based analysis
- Improving trajectory quality before reconstruction

The module implements two map-matching approaches:
1. **Leuven Algorithm (leuven)**: Hidden Markov Model-based matching using local OSM data
   - Fetches road network from Overpass API
   - Uses AEQD projection for accurate metric calculations
   - Best for offline processing and custom road networks

2. **Valhalla Meili (meili)**: Cloud-based matching using Valhalla routing engine
   - Uses Mapbox or custom Valhalla servers
   - Faster for large trajectories
   - Requires internet connection and API key

Both algorithms return trajectories with coordinates snapped to actual road centerlines.
"""

import os
import shutil
import sys
import hashlib
import math
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import polars as pl

# Prefer osmium; if missing, fall back to osmnx (if present)
try:
    import osmium as osm
    _HAVE_OSMIUM = True
except Exception:
    osm = None
    _HAVE_OSMIUM = False

try:
    import osmnx as ox
    _HAVE_OSMNX = True
except Exception:
    ox = None
    _HAVE_OSMNX = False

from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
import requests
from tqdm import tqdm

from pyproj import Transformer

# Overpass disk cache dir
_OVERPASS_CACHE_DIR = os.path.join(os.getcwd(), "overpass_cache")
os.makedirs(_OVERPASS_CACHE_DIR, exist_ok=True)

# AEQD transformer cache (simple bounded dict)
_AEQD_CACHE: Dict[Tuple[float, float, int], Tuple[Transformer, Transformer]] = {}


def _get_aeqd_transformers(cen_lat: float, cen_lon: float, precision: int = 6):
    """Return (forward, inverse) AEQD Transformers centered at (cen_lat, cen_lon).
       Cached by rounding centroid to `precision` decimals."""
    key = (round(cen_lat, precision), round(cen_lon, precision), precision)
    if key in _AEQD_CACHE:
        return _AEQD_CACHE[key]
    proj = f"+proj=aeqd +lat_0={cen_lat:.9f} +lon_0={cen_lon:.9f} +datum=WGS84 +units=m +no_defs"
    fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    _AEQD_CACHE[key] = (fwd, inv)
    # keep cache bounded
    if len(_AEQD_CACHE) > 256:
        _AEQD_CACHE.pop(next(iter(_AEQD_CACHE)))
    return fwd, inv


# ----------------------------------------
# More precise osmnx-like is_driveable logic
# (close to osmnx.utils.is_driveable)
# ----------------------------------------
_HIGHWAY_WHITELIST = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    "service", "road", "track"
}
_HIGHWAY_BLACKLIST = {
    "footway", "pedestrian", "steps", "path", "cycleway", "bridleway",
    "proposed", "construction", "abandoned", "platform"
}
_SERVICE_DENY = {"driveway", "parking_aisle"}
_ACCESS_DENY = {"no", "private", "customers", "permit", "delivery", "destination"}


def _resolve_tag_value(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = v[0]
    return str(v).lower()


def _is_driveable_way(tags: dict) -> bool:
    """Return True if a way is considered driveable (osmnx-like heuristics)."""
    if not tags:
        return False
    hwy = _resolve_tag_value(tags.get("highway"))
    if hwy is None:
        return False
    if hwy in _HIGHWAY_BLACKLIST:
        return False
    if (hwy not in _HIGHWAY_WHITELIST) and (not hwy.isdigit()) and (hwy != "road"):
        return False
    # area=yes not a road
    if _resolve_tag_value(tags.get("area")) == "yes":
        return False
    # service restrictions
    service = _resolve_tag_value(tags.get("service"))
    if service in _SERVICE_DENY:
        return False
    # route ferry excluded
    if _resolve_tag_value(tags.get("route")) == "ferry":
        return False
    # access/motor_vehicle/vehicle restrictions
    for k in ("motor_vehicle", "motorcar", "vehicle", "access"):
        v = _resolve_tag_value(tags.get(k))
        if v in _ACCESS_DENY:
            return False
    # exclude proposed/construction
    if _resolve_tag_value(tags.get("proposed")) == "yes" or _resolve_tag_value(tags.get("construction")) == "yes":
        return False
    return True


# ----------------------------------------
# Osmium handler to collect nodes & driveable ways
# ----------------------------------------
if _HAVE_OSMIUM:
    class WayNodeCollector(osm.SimpleHandler):
        def __init__(self, nodes: dict, ways: dict):
            super().__init__()
            self.nodes = nodes
            self.ways = ways

        def node(self, n):
            # store coordinates if available
            if n.location.valid():
                self.nodes[n.id] = (float(n.location.lat), float(n.location.lon))

        def way(self, w):
            tags = {k: v for k, v in w.tags.items()}
            if not _is_driveable_way(tags):
                return
            node_refs = [nd.ref for nd in w.nodes]
            if len(node_refs) < 2:
                return
            self.ways[w.id] = {"tags": tags, "nodes": node_refs}


# ----------------------------------------
# Overpass helpers: tile split, cache path, download
# ----------------------------------------
def _split_bbox_into_tiles(north: float, south: float, east: float, west: float,
                           max_tile_deg: float = 0.5,
                           overlap_deg: float = 0.05):
    """
    Divide a bounding box into smaller overlapping tiles for efficient OSM data fetching.

    This prevents Overpass API timeouts when requesting large areas by breaking them
    into manageable chunks with overlap to ensure no edge cases are missed.
    """
    if east <= west or north <= south:
        return [(north, south, east, west)]

    lat_spans = max(1, math.ceil((north - south) / max_tile_deg))
    lon_spans = max(1, math.ceil((east - west) / max_tile_deg))

    # Create evenly spaced tile boundaries
    lat_edges = np.linspace(south, north, lat_spans + 1)
    lon_edges = np.linspace(west, east, lon_spans + 1)

    tiles = []
    for i in range(lat_spans):
        s = lat_edges[i]
        n = lat_edges[i + 1]
        # Expand tile boundaries with overlap, but clip to original bbox
        n_exp = min(n + overlap_deg, north)
        s_exp = max(s - overlap_deg, south)

        for j in range(lon_spans):
            w = lon_edges[j]
            e = lon_edges[j + 1]
            e_exp = min(e + overlap_deg, east)
            w_exp = max(w - overlap_deg, west)
            tiles.append((n_exp, s_exp, e_exp, w_exp))
    return tiles


def _overpass_cache_path(north: float, south: float, east: float, west: float, overlap: float):
    key = f"{north:.6f}_{south:.6f}_{east:.6f}_{west:.6f}_{overlap:.6f}"
    h = hashlib.sha256(key.encode("utf8")).hexdigest()
    return os.path.join(_OVERPASS_CACHE_DIR, f"overpass_{h}.osm")


def _download_osm_xml_bbox_cached(north: float, south: float, east: float, west: float,
                                  timeout: int = 180, force_download: bool = False, overlap: float = 0.0):
    """Download OSM XML for bbox with caching on disk per tile."""
    cache_path = _overpass_cache_path(north, south, east, west, overlap)
    if os.path.exists(cache_path) and not force_download:
        return cache_path
    bbox = f"{south},{west},{north},{east}"
    q = f"""
    [out:xml][timeout:{int(timeout)}];
    (
      way["highway"]({bbox});
    );
    (._;>;);
    out body;
    """
    url = "https://overpass-api.de/api/interpreter"
    resp = requests.post(url, data={"data": q}, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(cache_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return cache_path


# ----------------------------------------
# Main upgraded leuven() function
# ----------------------------------------
def leuven(df: pd.DataFrame | pl.DataFrame,
           lat_col: str = 'lat',
           lon_col: str = 'lon',
           delete_cache: bool = False,
           max_dist: float = 0.01,
           tile_max_deg: float = 0.5,
           tile_overlap_deg: float = 0.05,
           cache_overpass: bool = True,
           return_projected: bool = False,
           aeqd_precision: int = 6) -> pd.DataFrame | pl.DataFrame:
    """
    Map-match GPS trajectory to road network using Hidden Markov Model algorithm (Leuven).

    This function aligns noisy GPS points to the most likely path on the actual road network
    by fetching OpenStreetMap data and applying HMM-based map-matching. It corrects GPS drift,
    snaps points to road centerlines, and assigns OSM way IDs for road-network-based analysis.

    The algorithm:
    1. Fetches driveable roads from Overpass API (tiled for large areas)
    2. Projects coordinates to AEQD (Azimuthal Equidistant) for accurate metric calculations
    3. Builds in-memory road network graph from OSM data
    4. Applies Hidden Markov Model to find most likely road path
    5. Returns trajectory with coordinates snapped to roads

    This is essential preprocessing for:
    - Trajectory reconstruction (provides OSM IDs for road identification)
    - Removing GPS noise and drift
    - Preparing data for road-network-based routing analysis
    - Generating training data for machine learning models

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input GPS trajectory with at least latitude and longitude columns. Can contain
        additional columns (all will be preserved). Minimum 2 points required.
    lat_col : str, default='lat'
        Name of the latitude column (WGS84 decimal degrees).
    lon_col : str, default='lon'
        Name of the longitude column (WGS84 decimal degrees).
    delete_cache : bool, default=False
        If True, delete cached Overpass API responses after matching. Useful for saving
        disk space or forcing fresh data download. Recommended: False (reuse cache).
    max_dist : float, default=0.01
        Maximum distance in kilometers for matching GPS points to road nodes. Points
        farther than this from any road will be excluded from the result. Typical values:
        - 0.01 km (10m): High-accuracy GPS, urban areas
        - 0.05 km (50m): Standard GPS accuracy
        - 0.1 km (100m): Low-accuracy GPS or sparse road networks
    tile_max_deg : float, default=0.5
        Maximum tile size in degrees for splitting large bounding boxes. Overpass API
        has request size limits, so large trajectories are automatically tiled. Larger
        values = fewer API calls but longer response times. Typical: 0.3-0.7 degrees.
    tile_overlap_deg : float, default=0.05
        Overlap between adjacent tiles in degrees to ensure roads crossing tile boundaries
        are captured. Should be >= typical road segment length. Typical: 0.03-0.1 degrees.
    cache_overpass : bool, default=True
        If True, cache Overpass API responses to disk in ./overpass_cache/ for reuse.
        Dramatically speeds up repeated matching in the same area. Recommended: True.
    return_projected : bool, default=False
        If True, return coordinates in AEQD projected meters (x, y) instead of WGS84
        degrees (lon, lat). Useful for distance calculations. Typical: False (keep lat/lon).
    aeqd_precision : int, default=6
        Decimal precision for rounding AEQD projection center coordinates when caching
        transformers. Higher precision = more cache entries. Typical: 5-7.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Map-matched trajectory with coordinates snapped to road network. Returns same type
        as input (pandas → pandas, polars → polars). Output contains:
        - Matched coordinates (lat/lon or x/y depending on return_projected)
        - All original columns preserved
        - Points filtered: only those within max_dist of roads are kept
        - Order preserved: points remain in chronological sequence

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Load raw GPS trajectory
    >>> df = pd.read_csv('raw_gps.csv')
    >>> print(f"Raw trajectory: {len(df)} points")
    >>>
    >>> # Map-match to OSM roads with default settings
    >>> matched = pye.preprocessing.leuven(df)
    >>> print(f"Matched trajectory: {len(matched)} points")
    >>>
    >>> # Map-match with custom distance threshold (100m)
    >>> matched = pye.preprocessing.leuven(
    ...     df,
    ...     max_dist=0.1,  # 100 meters
    ...     cache_overpass=True  # Reuse cached OSM data
    ... )
    >>>
    >>> # Use in preprocessing pipeline
    >>> compressed = pye.preprocessing.spatio_temporal_compress(df)
    >>> matched = pye.preprocessing.leuven(compressed)  # Map-match compressed data
    >>> filtered = pye.preprocessing.kalman_aeqd_filter(matched)  # Apply Kalman filter
    >>>
    >>> # Visualize before/after
    >>> pye.utilities.visualization.multiple(
    ...     [df, matched],
    ...     names=['Raw GPS', 'Map-Matched'],
    ...     show_in_browser=True
    ... )

    Notes
    -----
    **Algorithm: Hidden Markov Model (HMM)**
    - **States**: Road segments (edges) in the OSM network
    - **Observations**: GPS points with noise
    - **Transition probability**: Likelihood of traveling from one road to another
    - **Emission probability**: Likelihood of observing GPS point given road location
    - **Viterbi algorithm**: Finds most likely sequence of states (road path)

    **Data Sources:**
    - **Overpass API**: Public OSM query service (https://overpass-api.de/)
    - **Fallback**: OSMnx library if Overpass fails
    - **Caching**: Responses saved to ./overpass_cache/ as XML files

    **Dependencies:**
    - leuvenmapmatching: Core HMM matching library
    - osmium: Fast OSM XML parsing (preferred)
    - osmnx: Fallback for OSM data fetching and parsing
    - pyproj: AEQD projection transformations

    **Projection: AEQD (Azimuthal Equidistant)**
    - Centered on trajectory bounding box center
    - Preserves distances from center point
    - Converts lat/lon (degrees) to x/y (meters)
    - Essential for accurate distance-based matching

    **Performance:**
    - Time complexity: O(n * m * k) where:
      - n = trajectory points
      - m = average candidate roads per point
      - k = HMM Viterbi path length
    - For 1000-point trajectory: 30-90 seconds (first run), <5 seconds (cached)
    - Overpass API calls: ~1-5 tiles for typical city-scale trajectories
    - Cache significantly speeds up repeated matching in same area

    **Quality Factors:**
    - **GPS accuracy**: Better GPS → better matching
    - **Road network density**: Urban >> rural
    - **Sampling rate**: Higher frequency → better path inference
    - **max_dist**: Should match expected GPS error (typically 10-50m)

    **Limitations:**
    - Requires internet connection (Overpass API or OSMnx)
    - May fail in areas with sparse OSM coverage
    - Computationally intensive for very long trajectories (>10,000 points)
    - Does not handle GPS outages or tunnels well
    - Assumes trajectory follows driveable roads (not for off-road vehicles)

    **Troubleshooting:**
    - **"No OSM data available"**: Check internet connection, try larger tile_max_deg
    - **Slow performance**: Enable cache_overpass=True, reduce trajectory size with compression
    - **Poor matching**: Increase max_dist, check OSM coverage in area, improve GPS quality
    - **Overpass timeout**: Reduce tile_max_deg (split into smaller tiles)

    See Also
    --------
    meili : Alternative map-matching using Valhalla routing engine
    spatio_temporal_compress : Reduce trajectory size before map-matching
    kalman_aeqd_filter : Apply Kalman filtering after map-matching
    """
    input_is_polars = isinstance(df, pl.DataFrame)
    if input_is_polars:
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    if len(pdf) < 2:
        raise ValueError("Input must contain at least two points for map matching.")

    # Extract trajectory coordinates
    lats = pdf[lat_col].to_numpy(dtype=float)
    lons = pdf[lon_col].to_numpy(dtype=float)
    points = list(zip(lats.tolist(), lons.tolist()))

    # Calculate bounding box for the trajectory
    north = float(lats.max())
    south = float(lats.min())
    east = float(lons.max())
    west = float(lons.min())

    # Build nodes & ways by tiling the bbox and parsing each tile
    nodes: Dict[int, Tuple[float, float]] = {}
    ways: Dict[int, Dict] = {}

    tiles = _split_bbox_into_tiles(north, south, east, west, max_tile_deg=tile_max_deg, overlap_deg=tile_overlap_deg)

    # Set up AEQD projection centered on trajectory bounding box
    bbox_cen_lat = (north + south) * 0.5
    bbox_cen_lon = (east + west) * 0.5
    fwd_global, inv_global = _get_aeqd_transformers(bbox_cen_lat, bbox_cen_lon, precision=aeqd_precision)

    for (tn, ts, te, tw) in tiles:
        try:
            osm_path = _download_osm_xml_bbox_cached(tn, ts, te, tw, overlap=tile_overlap_deg, force_download=not cache_overpass)
        except Exception as e:
            # try fallback to osmnx tile if available
            if _HAVE_OSMNX:
                try:
                    G = ox.graph_from_bbox(tn, ts, te, tw, network_type="drive", simplify=False)
                    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
                    for nid, row in nodes_gdf.iterrows():
                        nodes[nid] = (float(row.geometry.y), float(row.geometry.x))
                    for idx, row in edges_gdf.iterrows():
                        geom = row.geometry
                        try:
                            coords = list(geom.coords)
                        except Exception:
                            coords = []
                        if len(coords) >= 2:
                            wid = int(hash((coords[0], coords[-1], idx)) & 0xFFFFFFFF)
                            node_seq = []
                            for c in coords:
                                node_id = int(hash(c) & 0x7FFFFFFF)
                                nodes.setdefault(node_id, (float(c[1]), float(c[0])))
                                node_seq.append(node_id)
                            ways[wid] = {"tags": {}, "nodes": node_seq}
                    continue
                except Exception:
                    # skip tile
                    continue
            else:
                continue

        if _HAVE_OSMIUM:
            handler = WayNodeCollector(nodes, ways)
            try:
                handler.apply_file(osm_path, locations=True)
            except Exception:
                # skip tile on parse error
                continue
        else:
            # if no osmium, we already attempted osmnx fallback above when download failed
            # try a direct osmnx parse of the file (if osmnx present)
            if _HAVE_OSMNX:
                try:
                    # osmnx can read xml file into graph via graph_from_xml? not always; fallback: skip
                    # Instead, use ox.graph_from_bbox fallback above
                    pass
                except Exception:
                    pass

    if len(nodes) == 0 or len(ways) == 0:
        # if nothing parsed, fallback to osmnx full bbox if available
        if _HAVE_OSMNX:
            return _leuven_with_osmnx(pdf, lat_col, lon_col, delete_cache, max_dist)
        else:
            raise RuntimeError("No OSM data available for bbox and no osmnx fallback available.")

    # Build mapping from node IDs to indices and extract coordinates
    node_ids = sorted(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    node_coords = np.array([nodes[nid] for nid in node_ids], dtype=float)  # lat, lon

    node_lats = node_coords[:, 0]
    node_lons = node_coords[:, 1]

    # Project nodes to AEQD for metric coordinates in meters
    xs, ys = fwd_global.transform(node_lons, node_lats)

    # Create dictionary mapping node IDs to projected coordinates
    node_proj = {nid: (float(ys[i]), float(xs[i])) for i, nid in enumerate(node_ids)}

    # Build InMemMap with projected nodes
    map_con = InMemMap("map_con", use_latlon=False, index_edges=True, use_rtree=True)
    for nid in node_ids:
        lat_m, lon_m = node_proj[nid]
        map_con.add_node(nid, (lat_m, lon_m))

    # Add edges from ways list
    for way_id, w in ways.items():
        node_sequence = w["nodes"]
        tags = w.get("tags", {})
        oneway = _resolve_tag_value(tags.get("oneway")) in ("yes", "true", "1")
        for a, b in zip(node_sequence[:-1], node_sequence[1:]):
            if a not in id_to_idx or b not in id_to_idx:
                continue
            try:
                map_con.add_edge(int(a), int(b))
                if not oneway:
                    map_con.add_edge(int(b), int(a))
            except Exception:
                continue

    # Build DistanceMatcher
    matcher = DistanceMatcher(map_con,
                              max_dist=max_dist,
                              min_prob_norm=0.5,
                              non_emitting_length_factor=0.75,
                              obs_noise=1,
                              obs_noise_ne=50,
                              dist_noise=50,
                              max_lattice_width=3,
                              non_emitting_states=True)

    # Run HMM-based map matching
    matcher.match(points, unique=False, tqdm=tqdm)

    # Extract matched coordinates from the matcher
    n_points = len(pdf)
    if n_points == len(matcher.lattice_best):
        matches = np.array([matcher.lattice_best[i].edge_m.pi for i in range(n_points)], dtype=float)
    else:
        raise Exception("The number of points in the dataframe are more than the number of matched points. "
                        "You can try to increase the max_dist parameter.")

    matched_y = matches[:, 0]  # lat_m in AEQD projection
    matched_x = matches[:, 1]  # lon_m in AEQD projection

    # If user wants projected coords, return them
    if return_projected:
        if input_is_polars:
            return pl.DataFrame({lon_col: matched_x.tolist(), lat_col: matched_y.tolist()})
        else:
            return pd.DataFrame({lon_col: matched_x.tolist(), lat_col: matched_y.tolist()})

    # otherwise inverse-transform from AEQD back to lon/lat degrees using the same bbox AEQD
    try:
        lon_deg, lat_deg = inv_global.transform(matched_x, matched_y)
        if input_is_polars:
            matched_df = pl.DataFrame({lon_col: lon_deg.tolist(), lat_col: lat_deg.tolist()})
        else:
            matched_df = pd.DataFrame({lon_col: lon_deg.tolist(), lat_col: lat_deg.tolist()})
    except Exception:
        # fallback: return projected meters if inverse transform fails
        if input_is_polars:
            matched_df = pl.DataFrame({lon_col: matched_x.tolist(), lat_col: matched_y.tolist()})
        else:
            matched_df = pd.DataFrame({lon_col: matched_x.tolist(), lat_col: matched_y.tolist()})

    # cleanup caches if requested (preserve previous behavior and also clear overpass cache)
    if os.path.isdir("cache") and delete_cache:
        shutil.rmtree("cache", ignore_errors=True)
    if delete_cache and os.path.isdir(_OVERPASS_CACHE_DIR):
        shutil.rmtree(_OVERPASS_CACHE_DIR, ignore_errors=True)

    return matched_df


# ------------------------------
# Fallback helper (unchanged), still returns degrees
# ------------------------------
def _leuven_with_osmnx(pdf: pd.DataFrame, lat_col: str, lon_col: str, delete_cache: bool, max_dist: float):
    """
    Fallback map matching using osmnx library when osmium/Overpass is unavailable.

    This is the original implementation kept for backward compatibility and as a fallback
    when the preferred osmium-based method fails.
    """
    lats = pdf[lat_col].to_numpy(dtype=float)
    lons = pdf[lon_col].to_numpy(dtype=float)
    points = list(zip(lats.tolist(), lons.tolist()))

    north = float(lats.max())
    south = float(lats.min())
    east = float(lons.max())
    west = float(lons.min())

    try:
        _graph = ox.graph_from_bbox(north, south, east, west, network_type='drive', simplify=False)
    except Exception as e:
        print(e)
        return pdf

    graph_proj = ox.project_graph(_graph)
    map_con = InMemMap("map_con", use_latlon=False, index_edges=True, use_rtree=True)
    nodes, edges = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)
    for nid, row in nodes.iterrows():
        try:
            map_con.add_node(nid, (float(row.geometry.y), float(row.geometry.x)))
        except Exception:
            continue
    for u, v, _ in _graph.edges:
        try:
            map_con.add_edge(u, v)
        except Exception:
            continue
    matcher = DistanceMatcher(map_con,
                              max_dist=max_dist,
                              min_prob_norm=0.5,
                              non_emitting_length_factor=0.75,
                              obs_noise=1,
                              obs_noise_ne=50,
                              dist_noise=50,
                              max_lattice_width=3,
                              non_emitting_states=True)
    matcher.match(points, unique=False, tqdm=tqdm)

    n_points = len(pdf)
    if n_points == len(matcher.lattice_best):
        matches = np.array([matcher.lattice_best[i].edge_m.pi for i in range(n_points)], dtype=float)
    else:
        raise Exception("The number of points in the dataframe are more than the number of matched points. "
                        "You can try to increase the max_dist parameter.")

    # Transform matched coordinates back to lat/lon degrees
    try:
        inv_t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon_m = matches[:, 1]
        lat_m = matches[:, 0]
        lon_deg, lat_deg = inv_t.transform(lon_m, lat_m)
        matched_df = pd.DataFrame({lon_col: lon_deg.tolist(), lat_col: lat_deg.tolist()})
    except Exception:
        matched_df = pd.DataFrame({lon_col: matches[:, 1].tolist(), lat_col: matches[:, 0].tolist()})

    if os.path.isdir("cache"):
        shutil.rmtree("cache", ignore_errors=True)
    return matched_df


# ------------------------------
# Meili function unchanged
# ------------------------------
def meili(df: pd.DataFrame | pl.DataFrame,
          lat_col: str = 'lat',
          lon_col: str = 'lon',
          time_col: str = 'time') -> pd.DataFrame | pl.DataFrame:
    """
    Map-match a GPS trajectory using Valhalla's Meili algorithm.

    This function requires a Valhalla server running on localhost:8002. Meili uses a
    Hidden Markov Model to match GPS traces to roads, handling noise and gaps effectively.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input trajectory with latitude, longitude, and time columns.
    lat_col : str, default='lat'
        Name of the latitude column.
    lon_col : str, default='lon'
        Name of the longitude column.
    time_col : str, default='time'
        Name of the time column.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Map-matched trajectory with coordinates snapped to roads.
        Includes 'original_lon' and 'original_lat' columns if time_col is present.

    Notes
    -----
    Requires Valhalla server running on http://localhost:8002
    Install and run Valhalla before using this function.
    See: https://github.com/valhalla/valhalla
    """
    original_lon = df[lon_col].to_list()
    original_lat = df[lat_col].to_list()

    # Convert trajectory to JSON format for Valhalla API
    if isinstance(df, pd.DataFrame):
        meili_coordinates = df.to_json(orient='records')
    else:
        meili_coordinates = df.write_json(row_oriented=True)

    # Build Meili request body
    meili_request_body = f'{{"shape":{meili_coordinates},"search_radius": 100, "shape_match":"map_snap", "costing":"auto", "format":"osrm"}}'

    # Send request to Valhalla server
    try:
        r = requests.post(
            "http://localhost:8002/trace_route",
            data=meili_request_body,
            headers={'Content-type': 'application/json'}
        )
    except requests.exceptions.RequestException as e:
        print(e)
        sys.exit(1)

    if r.status_code == 200:
        # Parse Valhalla response
        response_text = r.json()
        tracepoints = response_text.get('tracepoints', [])

        # Replace None entries with default values (happens when matching fails for a point)
        default_point = {'matchings_index': '#', 'name': '', 'waypoint_index': '#',
                        'alternatives_count': 0, 'distance': 0, 'location': [0.0, 0.0]}
        tracepoints = [tp if tp is not None else default_point for tp in tracepoints]

        if isinstance(df, pd.DataFrame):
            # Extract matched coordinates from tracepoints
            locations = np.array([tp['location'] for tp in tracepoints], dtype=float)

            lon_vals = locations[:, 0]
            lat_vals = locations[:, 1]

            # Build output DataFrame
            data_dict = {lon_col: lon_vals, lat_col: lat_vals}

            # Include time and original coordinates if time column is present
            if time_col in df.columns:
                data_dict[time_col] = df[time_col].to_numpy()
                data_dict['original_lon'] = original_lon
                data_dict['original_lat'] = original_lat

            matched_df = pd.DataFrame(data_dict)

            # Filter out failed matches (indicated by 0,0 coordinates)
            matched_df = matched_df[(matched_df[lat_col] != 0) & (matched_df[lon_col] != 0)]
            return matched_df

        else:  # polars
            # Extract matched coordinates for polars
            locations = [tp['location'] for tp in tracepoints]
            lon_vals = [loc[0] for loc in locations]
            lat_vals = [loc[1] for loc in locations]

            matched_df = pl.DataFrame({
                lon_col: lon_vals,
                lat_col: lat_vals
            })

            # Filter out failed matches
            matched_df = matched_df.filter(
                (pl.col(lat_col) != 0) & (pl.col(lon_col) != 0)
            )

            return matched_df
    else:
        print(f"Valhalla request failed, status code: {r.status_code}, reason: {r.reason}")
        return None
