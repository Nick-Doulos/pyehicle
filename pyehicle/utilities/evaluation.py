import os
import shutil
import hashlib
import math
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree
from pyproj import Geod, Transformer

# Prefer osmium
try:
    import osmium as osm  # type: ignore
    _HAVE_OSMIUM = True
except Exception:
    osm = None
    _HAVE_OSMIUM = False

# Fallback to osmnx if needed
try:
    import osmnx as ox  # type: ignore
    _HAVE_OSMNX = True
except Exception:
    ox = None
    _HAVE_OSMNX = False

_GEOD = Geod(ellps="WGS84")

# Overpass cache directory
_OVERPASS_CACHE_DIR = os.path.join(os.getcwd(), "overpass_cache")
os.makedirs(_OVERPASS_CACHE_DIR, exist_ok=True)

# AEQD transformer cache
_transformer_cache: Dict[Tuple[float, float, int], Tuple[Transformer, Transformer]] = {}


def _get_aeqd_transformers(cen_lat: float, cen_lon: float, precision: int = 6):
    key = (round(cen_lat, precision), round(cen_lon, precision), precision)
    if key in _transformer_cache:
        return _transformer_cache[key]
    proj = f"+proj=aeqd +lat_0={cen_lat:.9f} +lon_0={cen_lon:.9f} +datum=WGS84 +units=m +no_defs"
    fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    _transformer_cache[key] = (fwd, inv)
    if len(_transformer_cache) > 256:
        _transformer_cache.pop(next(iter(_transformer_cache)))
    return fwd, inv


# Driveability checks (close to osmnx semantics)
_HIGHWAY_WHITELIST = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link", "service", "road"
}
_HIGHWAY_BLACKLIST = {"footway", "pedestrian", "steps", "path", "cycleway", "bridleway",
                      "proposed", "construction", "abandoned", "platform"}
_SERVICE_DENY = {"driveway", "parking_aisle", "alley"}
_ACCESS_DENY = {"no", "private", "customers", "permit", "delivery", "destination"}


def _resolve_tag_value(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        v = val[0]
    else:
        v = val
    return str(v).lower()


def _is_driveable_way(tags: Dict[str, str]) -> bool:
    if not tags:
        return False
    hwy = _resolve_tag_value(tags.get("highway"))
    if hwy is None:
        return False
    if hwy in _HIGHWAY_BLACKLIST:
        return False
    if (hwy not in _HIGHWAY_WHITELIST) and (not hwy.isdigit()) and (hwy != "road"):
        return False
    if _resolve_tag_value(tags.get("area")) == "yes":
        return False
    service = _resolve_tag_value(tags.get("service"))
    if service in _SERVICE_DENY:
        return False
    if _resolve_tag_value(tags.get("route")) == "ferry":
        return False
    for k in ("motor_vehicle", "motorcar", "vehicle", "access"):
        v = _resolve_tag_value(tags.get(k))
        if v in _ACCESS_DENY:
            return False
    if _resolve_tag_value(tags.get("proposed")) == "yes" or _resolve_tag_value(tags.get("construction")) == "yes":
        return False
    return True


if _HAVE_OSMIUM:
    class WayNodeCollector(osm.SimpleHandler):
        def __init__(self, nodes: Dict[int, Tuple[float, float]], ways: Dict[int, Dict]):
            super().__init__()
            self.nodes = nodes
            self.ways = ways

        def node(self, n):
            if n.location.valid():
                self.nodes[n.id] = (float(n.location.lat), float(n.location.lon))

        def way(self, w):
            # In newer osmium versions, TagList doesn't have .items()
            # Instead, iterate over tags directly where each tag has .k and .v attributes
            tags = {tag.k: tag.v for tag in w.tags}
            if not _is_driveable_way(tags):
                return
            node_refs = [nd.ref for nd in w.nodes]
            if len(node_refs) < 2:
                return
            self.ways[w.id] = {"tags": tags, "nodes": node_refs}


import requests  # local import allowed


def _bbox_hash(north: float, south: float, east: float, west: float, overlap: float):
    key = f"{north:.6f}_{south:.6f}_{east:.6f}_{west:.6f}_{overlap:.6f}"
    return hashlib.sha256(key.encode("utf8")).hexdigest()


def _overpass_cache_path(north: float, south: float, east: float, west: float, overlap: float):
    h = _bbox_hash(north, south, east, west, overlap)
    return os.path.join(_OVERPASS_CACHE_DIR, f"overpass_{h}.osm")


def _download_overpass_bbox(north: float, south: float, east: float, west: float,
                            timeout: int = 180, force_download: bool = False, overlap: float = 0.0) -> str:
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


def _split_bbox_into_tiles(north: float, south: float, east: float, west: float,
                           max_tile_deg: float = 0.5, overlap_deg: float = 0.05) -> List[Tuple[float, float, float, float]]:
    # Early exit for invalid or single-tile bbox
    if east <= west or north <= south:
        return [(north, south, east, west)]

    lat_spans = max(1, math.ceil((north - south) / max_tile_deg))
    lon_spans = max(1, math.ceil((east - west) / max_tile_deg))

    # Pre-compute tile edges for latitude and longitude
    lat_edges = np.linspace(south, north, lat_spans + 1)
    lon_edges = np.linspace(west, east, lon_spans + 1)

    tiles = []
    for i in range(lat_spans):
        s = lat_edges[i]
        n = lat_edges[i + 1]
        # Calculate expanded boundaries with overlap
        n_exp = min(n + overlap_deg, north)
        s_exp = max(s - overlap_deg, south)

        for j in range(lon_spans):
            w = lon_edges[j]
            e = lon_edges[j + 1]
            e_exp = min(e + overlap_deg, east)
            w_exp = max(w - overlap_deg, west)
            tiles.append((n_exp, s_exp, e_exp, w_exp))
    return tiles


def _build_segment_index(north: float, south: float, east: float, west: float,
                         tile_max_deg: float = 0.5,
                         tile_overlap_deg: float = 0.05,
                         cache_overpass: bool = True,
                         aeqd_precision: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Initialize lists to collect segment data
    seg_ids_list = []
    centroid_x_list = []
    centroid_y_list = []
    seg_points_list = []

    tiles = _split_bbox_into_tiles(north, south, east, west, max_tile_deg=tile_max_deg, overlap_deg=tile_overlap_deg)

    for (tn, ts, te, tw) in tiles:
        try:
            osm_path = _download_overpass_bbox(tn, ts, te, tw, overlap=tile_overlap_deg, force_download=not cache_overpass)
        except Exception:
            continue
        nodes = {}
        ways = {}
        if _HAVE_OSMIUM:
            handler = WayNodeCollector(nodes, ways)
            handler.apply_file(osm_path, locations=True)
        elif _HAVE_OSMNX:
            try:
                G = ox.graph_from_bbox(tn, ts, te, tw, network_type="drive", simplify=False)
                nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
                for nid, row in nodes_gdf.iterrows():
                    nodes[nid] = (float(row.geometry.y), float(row.geometry.x))
                for idx, row in edges_gdf.iterrows():
                    geom = row.geometry
                    try:
                        coords = list(geom.coords)
                        if len(coords) >= 2:
                            wid = int(hash((coords[0], coords[-1], idx)) & 0xFFFFFFFF)
                            node_seq = []
                            for c in coords:
                                node_id = int(hash(c) & 0x7FFFFFFF)
                                nodes.setdefault(node_id, (float(c[1]), float(c[0])))
                                node_seq.append(node_id)
                            ways[wid] = {"tags": {}, "nodes": node_seq}
                    except Exception:
                        continue
            except Exception:
                continue
        else:
            raise ImportError("Neither osmium nor osmnx available to build OSM segments.")
        if len(ways) == 0:
            continue

        # Calculate tile centroid and get AEQD transformer for this tile
        cen_lat = (tn + ts) * 0.5
        cen_lon = (te + tw) * 0.5
        fwd, _ = _get_aeqd_transformers(cen_lat, cen_lon, precision=aeqd_precision)

        for way_id, w in ways.items():
            node_seq = w["nodes"]
            # Process all consecutive node pairs in this way
            num_pairs = len(node_seq) - 1

            for i in range(num_pairs):
                a = node_seq[i]
                b = node_seq[i + 1]

                if a not in nodes or b not in nodes:
                    continue

                lat_a, lon_a = nodes[a]
                lat_b, lon_b = nodes[b]

                try:
                    xa, ya = fwd.transform(lon_a, lat_a)
                    xb, yb = fwd.transform(lon_b, lat_b)
                except Exception:
                    xa, ya = float(lon_a), float(lat_a)
                    xb, yb = float(lon_b), float(lat_b)

                # Calculate segment centroid
                cx = (xa + xb) * 0.5
                cy = (ya + yb) * 0.5

                seg_ids_list.append(int(way_id))
                centroid_x_list.append(float(cx))
                centroid_y_list.append(float(cy))
                seg_points_list.append((lat_a, lon_a, lat_b, lon_b))
    if len(seg_ids_list) == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.empty((0, 4), dtype=float))
    return (np.array(seg_ids_list, dtype=int),
            np.array(centroid_x_list, dtype=float),
            np.array(centroid_y_list, dtype=float),
            np.array(seg_points_list, dtype=float))


def __lengths(df_matched: pd.DataFrame | pl.DataFrame,
              df_true: pd.DataFrame | pl.DataFrame,
              lat_col: str = 'lat',
              lon_col: str = 'lon',
              delete_cache: bool = True,
              tile_max_deg: float = 0.5,
              tile_overlap_deg: float = 0.05,
              cache_overpass: bool = True,
              aeqd_precision: int = 6) -> Tuple[float, float, float]:
    geod = _GEOD

    # Prepare data based on input type
    if isinstance(df_matched, pd.DataFrame) and isinstance(df_true, pd.DataFrame):
        _df_matched = df_matched.copy().reset_index(drop=True)
        _df_true = df_true.copy().reset_index(drop=True)
        input_is_polars = False
        combined = pd.concat([_df_matched, _df_true], ignore_index=True)
    elif isinstance(df_matched, pl.DataFrame) and isinstance(df_true, pl.DataFrame):
        _df_matched = df_matched.clone()
        _df_true = df_true.clone()
        input_is_polars = True
        # Combine coordinates for bounding box calculation
        lons = _df_matched[lon_col].to_list() + _df_true[lon_col].to_list()
        lats = _df_matched[lat_col].to_list() + _df_true[lat_col].to_list()
        combined = pd.DataFrame({lon_col: lons, lat_col: lats})
    else:
        raise ValueError("df_matched and df_true must be the same type (both pandas or both polars).")

    # Early exit for empty data
    if len(combined) == 0:
        return 0.0, 0.0, 0.0

    # Calculate bounding box from all coordinates
    combined_lons = combined[lon_col].to_numpy(dtype=float)
    combined_lats = combined[lat_col].to_numpy(dtype=float)
    west, east = float(combined_lons.min()), float(combined_lons.max())
    south, north = float(combined_lats.min()), float(combined_lats.max())
    seg_ids, cent_x, cent_y, seg_points = _build_segment_index(north, south, east, west,
                                                               tile_max_deg=tile_max_deg,
                                                               tile_overlap_deg=tile_overlap_deg,
                                                               cache_overpass=cache_overpass,
                                                               aeqd_precision=aeqd_precision)
    # If no road segments found, fall back to simple geometric trajectory comparison
    if seg_ids.size == 0:
        def _sum_length_from_df(df):
            if isinstance(df, pl.DataFrame):
                lon_arr = np.array(df[lon_col].to_list(), dtype=float)
                lat_arr = np.array(df[lat_col].to_list(), dtype=float)
            else:
                lon_arr = df[lon_col].to_numpy(dtype=float)
                lat_arr = df[lat_col].to_numpy(dtype=float)

            if len(lon_arr) < 2:
                return 0.0

            # Calculate geodesic distances between consecutive points
            _, _, lens = geod.inv(lon_arr[:-1], lat_arr[:-1], lon_arr[1:], lat_arr[1:])
            return float(np.nansum(np.abs(lens)))

        matched_length = _sum_length_from_df(_df_matched)
        true_length = _sum_length_from_df(_df_true)

        if matched_length == 0.0:
            matched_length = true_length

        # Calculate common length by comparing trajectories directly
        # When no road data available, use simple coordinate matching
        common_length = 0.0

        # Extract coordinates for comparison
        if input_is_polars:
            m_lons = np.array(_df_matched.get_column(lon_col).to_list(), dtype=float)
            m_lats = np.array(_df_matched.get_column(lat_col).to_list(), dtype=float)
            t_lons = np.array(_df_true.get_column(lon_col).to_list(), dtype=float)
            t_lats = np.array(_df_true.get_column(lat_col).to_list(), dtype=float)
        else:
            m_lons = _df_matched[lon_col].to_numpy(dtype=float)
            m_lats = _df_matched[lat_col].to_numpy(dtype=float)
            t_lons = _df_true[lon_col].to_numpy(dtype=float)
            t_lats = _df_true[lat_col].to_numpy(dtype=float)

        # Iterate over segments in the shorter trajectory
        min_len = min(len(m_lons), len(t_lons))
        if min_len > 1:
            for i in range(min_len - 1):
                # Check if points are approximately the same (within 1 meter)
                # This handles rounding differences and small GPS variations
                dist1 = geod.inv(m_lons[i], m_lats[i], t_lons[i], t_lats[i])[2]
                dist2 = geod.inv(m_lons[i+1], m_lats[i+1], t_lons[i+1], t_lats[i+1])[2]

                # If both endpoints match within 1m, count the segment
                if abs(dist1) < 1.0 and abs(dist2) < 1.0:
                    _, _, seglen = geod.inv(m_lons[i], m_lats[i], m_lons[i+1], m_lats[i+1])
                    common_length += abs(seglen)

        if delete_cache:
            if os.path.isdir("cache"):
                shutil.rmtree("cache", ignore_errors=True)
            if os.path.isdir(_OVERPASS_CACHE_DIR):
                shutil.rmtree(_OVERPASS_CACHE_DIR, ignore_errors=True)

        return matched_length, true_length, common_length

    # Calculate global AEQD projection centered on the bounding box
    cen_lat = (north + south) * 0.5
    cen_lon = (east + west) * 0.5
    fwd_global, _ = _get_aeqd_transformers(cen_lat, cen_lon, precision=aeqd_precision)

    # Extract segment endpoint coordinates
    lat_a = seg_points[:, 0]
    lon_a = seg_points[:, 1]
    lat_b = seg_points[:, 2]
    lon_b = seg_points[:, 3]

    # Transform segment endpoints to AEQD projection
    xa, ya = fwd_global.transform(lon_a, lat_a)
    xb, yb = fwd_global.transform(lon_b, lat_b)

    # Calculate segment centroids in projected space
    centroid_x = (xa + xb) * 0.5
    centroid_y = (ya + yb) * 0.5

    # Build spatial index (KDTree) for fast segment lookup
    tree = cKDTree(np.column_stack((centroid_x, centroid_y)))

    # Extract coordinate arrays from input dataframes
    if input_is_polars:
        m_lons = np.array(_df_matched.get_column(lon_col).to_list(), dtype=float)
        m_lats = np.array(_df_matched.get_column(lat_col).to_list(), dtype=float)
        t_lons = np.array(_df_true.get_column(lon_col).to_list(), dtype=float)
        t_lats = np.array(_df_true.get_column(lat_col).to_list(), dtype=float)
    else:
        m_lons = _df_matched[lon_col].to_numpy(dtype=float)
        m_lats = _df_matched[lat_col].to_numpy(dtype=float)
        t_lons = _df_true[lon_col].to_numpy(dtype=float)
        t_lats = _df_true[lat_col].to_numpy(dtype=float)

    # Early exit for empty trajectories
    if len(m_lons) == 0 or len(t_lons) == 0:
        return 0.0, 0.0, 0.0

    # Transform trajectory points to AEQD projection
    mx, my = fwd_global.transform(m_lons, m_lats)
    tx, ty = fwd_global.transform(t_lons, t_lats)

    # Find nearest road segment for each trajectory point
    _, idx_matched = tree.query(np.column_stack((mx, my)))
    _, idx_true = tree.query(np.column_stack((tx, ty)))

    # Map trajectory points to their nearest road segment IDs
    matched_seg_ids = seg_ids[idx_matched]
    true_seg_ids = seg_ids[idx_true]

    # Calculate common length by finding consecutive matching segments
    # A segment is "common" if both trajectories are on the same road
    common_length = 0.0

    if not input_is_polars:
        dfm = _df_matched.reset_index(drop=True)
        dft = _df_true.reset_index(drop=True)
        dfm['road_id'] = matched_seg_ids
        dft['road_id'] = true_seg_ids

        # Extract arrays for iteration
        road_ids_m = dfm['road_id'].to_numpy(dtype=int)
        road_ids_t = dft['road_id'].to_numpy(dtype=int)
        lons_m = dfm[lon_col].to_numpy(dtype=float)
        lats_m = dfm[lat_col].to_numpy(dtype=float)

        # Use minimum length to avoid index errors when trajectories have different sizes
        npoints = min(len(dfm), len(dft))
        for i in range(npoints - 1):
            # Current segment goes from point i to point i+1
            # Check if both points are on matching roads in both trajectories
            cur_m = road_ids_m[i]
            cur_t = road_ids_t[i]
            next_m = road_ids_m[i + 1]
            next_t = road_ids_t[i + 1]

            # Segment is common if BOTH endpoints are on matching roads
            # This ensures we're counting the same road segment in both trajectories
            if (cur_m == cur_t) and (next_m == next_t):
                lon1 = lons_m[i]
                lat1 = lats_m[i]
                lon2 = lons_m[i + 1]
                lat2 = lats_m[i + 1]

                _, _, seglen = geod.inv(lon1, lat1, lon2, lat2)
                common_length += abs(seglen)
    else:
        dfm = _df_matched.with_column(pl.Series(name='road_id', values=matched_seg_ids.tolist()))
        dft = _df_true.with_column(pl.Series(name='road_id', values=true_seg_ids.tolist()))

        # Use minimum length to avoid index errors when trajectories have different sizes
        npoints = min(dfm.height, dft.height)
        for i in range(npoints - 1):
            # Current segment goes from point i to point i+1
            # Check if both points are on matching roads in both trajectories
            cur_m = int(dfm[i, 'road_id'])
            cur_t = int(dft[i, 'road_id'])
            next_m = int(dfm[i + 1, 'road_id'])
            next_t = int(dft[i + 1, 'road_id'])

            # Segment is common if BOTH endpoints are on matching roads
            # This ensures we're counting the same road segment in both trajectories
            if (cur_m == cur_t) and (next_m == next_t):
                lon1 = float(dfm[i, lon_col])
                lat1 = float(dfm[i, lat_col])
                lon2 = float(dfm[i + 1, lon_col])
                lat2 = float(dfm[i + 1, lat_col])

                _, _, seglen = geod.inv(lon1, lat1, lon2, lat2)
                common_length += abs(seglen)

    # Helper function to calculate total trajectory length
    def _sum_geodesic(lon_arr, lat_arr):
        if len(lon_arr) < 2:
            return 0.0
        _, _, lens = geod.inv(lon_arr[:-1], lat_arr[:-1], lon_arr[1:], lat_arr[1:])
        return float(np.nansum(np.abs(lens)))

    # Calculate total trajectory lengths
    if not input_is_polars:
        # Use already extracted arrays for pandas
        matched_length = _sum_geodesic(lons_m, lats_m)
        true_length = _sum_geodesic(t_lons, t_lats)
    else:
        matched_length = _sum_geodesic(
            np.array(dfm.get_column(lon_col).to_list(), dtype=float),
            np.array(dfm.get_column(lat_col).to_list(), dtype=float)
        )
        true_length = _sum_geodesic(
            np.array(dft.get_column(lon_col).to_list(), dtype=float),
            np.array(dft.get_column(lat_col).to_list(), dtype=float)
        )
    if matched_length == 0.0:
        matched_length = true_length
    if common_length == 0.0:
        if delete_cache:
            if os.path.isdir("cache"):
                shutil.rmtree("cache", ignore_errors=True)
            if os.path.isdir(_OVERPASS_CACHE_DIR):
                shutil.rmtree(_OVERPASS_CACHE_DIR, ignore_errors=True)
        return matched_length, true_length, 0.0
    if delete_cache:
        if os.path.isdir("cache"):
            shutil.rmtree("cache", ignore_errors=True)
        if os.path.isdir(_OVERPASS_CACHE_DIR):
            shutil.rmtree(_OVERPASS_CACHE_DIR, ignore_errors=True)
    return matched_length, true_length, common_length


# Public wrappers with tile_overlap_deg default 0.05
def optimized_lengths(df_matched: pd.DataFrame | pl.DataFrame,
                      df_true: pd.DataFrame | pl.DataFrame,
                      lat_col: str = 'lat',
                      lon_col: str = 'lon',
                      delete_cache: bool = True,
                      tile_max_deg: float = 0.5,
                      tile_overlap_deg: float = 0.05,
                      cache_overpass: bool = True,
                      aeqd_precision: int = 6) -> Tuple[float, float, float]:
    return __lengths(df_matched, df_true, lat_col, lon_col, delete_cache,
                     tile_max_deg, tile_overlap_deg, cache_overpass, aeqd_precision)


def rmf(df_matched: pd.DataFrame | pl.DataFrame,
        df_true: pd.DataFrame | pl.DataFrame,
        lat_col: str = 'lat',
        lon_col: str = 'lon',
        delete_cache: bool = True,
        tile_max_deg: float = 0.5,
        tile_overlap_deg: float = 0.05,
        cache_overpass: bool = True,
        aeqd_precision: int = 6) -> float:
    matched_length, true_length, common_length = __lengths(df_matched, df_true, lat_col, lon_col, delete_cache,
                                                           tile_max_deg, tile_overlap_deg, cache_overpass, aeqd_precision)
    if true_length == 0.0:
        return 0.0
    rmf_value = (matched_length + true_length - 2.0 * common_length) / true_length
    return float(abs(round(rmf_value, 10)))


def recall(df_matched: pd.DataFrame | pl.DataFrame,
           df_true: pd.DataFrame | pl.DataFrame,
           lat_col: str = 'lat',
           lon_col: str = 'lon',
           delete_cache: bool = True,
           tile_max_deg: float = 0.5,
           tile_overlap_deg: float = 0.05,
           cache_overpass: bool = True,
           aeqd_precision: int = 6) -> float:
    matched_length, true_length, common_length = __lengths(df_matched, df_true, lat_col, lon_col, delete_cache,
                                                           tile_max_deg, tile_overlap_deg, cache_overpass, aeqd_precision)
    if true_length == 0.0:
        return 0.0
    return float(round(common_length / true_length, 10))


def precision(df_matched: pd.DataFrame | pl.DataFrame,
              df_true: pd.DataFrame | pl.DataFrame,
              lat_col: str = 'lat',
              lon_col: str = 'lon',
              delete_cache: bool = True,
              tile_max_deg: float = 0.5,
              tile_overlap_deg: float = 0.05,
              cache_overpass: bool = True,
              aeqd_precision: int = 6) -> float:
    matched_length, true_length, common_length = __lengths(df_matched, df_true, lat_col, lon_col, delete_cache,
                                                           tile_max_deg, tile_overlap_deg, cache_overpass, aeqd_precision)
    if matched_length == 0.0:
        return 0.0
    return float(round(common_length / matched_length, 10))


def f1(df_matched: pd.DataFrame | pl.DataFrame,
       df_true: pd.DataFrame | pl.DataFrame,
       lat_col: str = 'lat',
       lon_col: str = 'lon',
       delete_cache: bool = True,
       tile_max_deg: float = 0.5,
       tile_overlap_deg: float = 0.05,
       cache_overpass: bool = True,
       aeqd_precision: int = 6) -> float:
    matched_length, true_length, common_length = __lengths(df_matched, df_true, lat_col, lon_col, delete_cache,
                                                           tile_max_deg, tile_overlap_deg, cache_overpass, aeqd_precision)
    if matched_length == 0.0 or true_length == 0.0:
        return 0.0
    prec = common_length / matched_length
    rec = common_length / true_length
    if prec == 0.0 or rec == 0.0:
        return 0.0
    return float(2.0 * (prec * rec) / (prec + rec))


def length_index(df_matched: pd.DataFrame | pl.DataFrame,
                 df_original: pd.DataFrame | pl.DataFrame,
                 lat_col: str = 'lat',
                 lon_col: str = 'lon') -> float:
    geod = _GEOD

    # Helper function to calculate trajectory length
    def _sum_len_df(df):
        if isinstance(df, pl.DataFrame):
            lon_arr = np.array(df[lon_col].to_list(), dtype=float)
            lat_arr = np.array(df[lat_col].to_list(), dtype=float)
        else:
            lon_arr = df[lon_col].to_numpy(dtype=float)
            lat_arr = df[lat_col].to_numpy(dtype=float)

        if len(lon_arr) < 2:
            return 0.0

        # Calculate geodesic distances between consecutive points
        _, _, lens = geod.inv(lon_arr[:-1], lat_arr[:-1], lon_arr[1:], lat_arr[1:])
        return float(np.nansum(np.abs(lens)))

    total_matched = _sum_len_df(df_matched)
    total_raw = _sum_len_df(df_original)

    return float(total_matched / total_raw) if total_raw != 0.0 else 0.0


def clear_overpass_cache():
    """Delete the on-disk Overpass cache directory (overpass_cache/)."""
    if os.path.isdir(_OVERPASS_CACHE_DIR):
        shutil.rmtree(_OVERPASS_CACHE_DIR, ignore_errors=True)
