"""
Kalman filtering module for pyehicle.

This module provides advanced GPS trajectory smoothing using Kalman filtering with
Rauch-Tung-Striebel (RTS) smoothing and Expectation-Maximization (EM) parameter learning.
The implementation uses Azimuthal Equidistant (AEQD) projection for accurate metric
calculations and supports:

- 2D constant-velocity motion model
- Automatic noise parameter learning via EM algorithm
- Outlier detection based on innovation statistics
- Large trajectory support via chunking and memory mapping
- Optional numba JIT compilation for performance
- Per-chunk or global AEQD projection strategies

Key Features:
- Kalman Filter: Forward pass estimating states from noisy observations
- RTS Smoother: Backward pass refining estimates using future information
- EM Algorithm: Learns optimal process (Q) and measurement (R) noise covariances
- Chunked Processing: Handles million-point trajectories via overlapping chunks
- AEQD Projection: Accurate metric calculations for all latitudes and distances
"""

from typing import Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import polars as pl
from pyproj import Transformer, Geod
from datetime import datetime, timezone

# ========== Optional Dependency Handling ==========
# Try importing optional libraries for enhanced performance and features
# If not available, fall back to numpy/basic functionality

# scipy.linalg: Faster and more stable matrix operations than numpy
try:
    import scipy.linalg as sla  # type: ignore
    _use_scipy = True
except Exception:
    sla = np.linalg  # Fallback to numpy's linear algebra
    _use_scipy = False

# numba: JIT compilation for 10-50x speedup on filter/smoother loops
try:
    import numba  # type: ignore
    _have_numba = True
except Exception:
    numba = None
    _have_numba = False

# tqdm: Progress bars for long-running operations
try:
    from tqdm import tqdm  # type: ignore
    _have_tqdm = True
except Exception:
    tqdm = None
    _have_tqdm = False

# ========== Global Geodesic Calculator ==========
# Single Geod instance for WGS84 ellipsoid calculations (reused for efficiency)
_geod = Geod(ellps="WGS84")

# ========== AEQD Projection Transformer Cache ==========
# Cache AEQD (Azimuthal Equidistant) projection transformers to avoid
# expensive re-creation. AEQD projects sphere to plane with accurate
# distances and directions from the center point.
#
# Key: (rounded_lat, rounded_lon, precision) -> (forward_transformer, inverse_transformer)
# Rounding reduces cache size while maintaining projection accuracy
_transformer_cache = {}

def _get_cached_aeqd_transformer(cen_lat: float, cen_lon: float, precision: int = 6):
    """
    Return cached AEQD Transformer pair for centroid rounded at given precision.

    AEQD (Azimuthal Equidistant) projection is critical for accurate metric
    calculations. It preserves true distances and directions from the center
    point, making it ideal for Kalman filtering in local regions.

    Caching dramatically improves performance by avoiding repeated transformer
    creation (which is expensive). Rounding to `precision` decimal places
    reduces cache size while maintaining millimeter-level accuracy.
    """
    # Create cache key from rounded coordinates
    key = (round(cen_lat, precision), round(cen_lon, precision), precision)

    # Check cache for existing transformer
    t = _transformer_cache.get(key)
    if t is not None:
        return t

    # Create new AEQD projection centered at (cen_lat, cen_lon)
    # +proj=aeqd: Azimuthal Equidistant projection
    # +lat_0, +lon_0: Projection center (true distances from this point)
    # +datum=WGS84: Standard GPS ellipsoid
    # +units=m: Output in meters (not degrees)
    proj = f"+proj=aeqd +lat_0={cen_lat:.9f} +lon_0={cen_lon:.9f} +datum=WGS84 +units=m +no_defs"

    # Create bidirectional transformers
    fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)  # WGS84 -> AEQD (lon, lat) -> (x, y)
    inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)  # AEQD -> WGS84 (x, y) -> (lon, lat)

    # Store in cache for future use
    _transformer_cache[key] = (fwd, inv)
    return fwd, inv

# ========== DataFrame Type Preservation Helpers ==========
# These helpers allow the filter to work with both pandas and polars DataFrames
# while maintaining type consistency (return same type as input)

def _to_pandas_preserve(df: Union[pd.DataFrame, pl.DataFrame]) -> Tuple[pd.DataFrame, bool]:
    """
    Convert input DataFrame to pandas and track original type.

    Returns: (pandas_df, was_polars_flag)
    """
    if isinstance(df, pl.DataFrame):
        return df.to_pandas(), True  # Convert polars to pandas, remember it was polars
    return df.copy(), False  # Already pandas, just copy

def _from_pandas_preserve(pdf: pd.DataFrame, was_polars: bool) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Convert pandas DataFrame back to original type if needed.

    If was_polars=True, converts back to polars. Otherwise returns pandas.
    """
    return pl.from_pandas(pdf) if was_polars else pdf

def _ensure_time_array(pdf: pd.DataFrame, time_col: Optional[str]) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Ensure a valid time array exists for Kalman filtering.

    If no time column is provided, creates synthetic timestamps with 1-second intervals.
    This allows the Kalman filter to work even on trajectories without explicit timestamps.

    The Kalman filter needs time differences (Δt) between points to model velocity.
    Without timestamps, we assume uniform 1-second spacing.

    Returns:
        times: np.ndarray of datetime64[ns] timestamps
        time_col_name: str name of the time column (original or synthetic)
        pdf: pd.DataFrame with time column added if it was missing
    """
    if (time_col is None) or (time_col not in pdf.columns):
        # ========== Create Synthetic Timestamps ==========
        # Generate timestamps: now, now+1s, now+2s, ..., now+n-1s
        # Start from current UTC time (rounded to seconds for cleaner values)
        start = np.datetime64(datetime.now(timezone.utc).replace(microsecond=0))
        times = start + np.arange(len(pdf), dtype="int64").astype("timedelta64[s]")

        # Add synthetic column to DataFrame
        synthetic = "_synthetic_time"
        pdf = pdf.copy()
        pdf[synthetic] = times

        return times, synthetic, pdf

    # ========== Use Existing Time Column ==========
    # Parse time column to numpy datetime64 array (nanosecond precision)
    times = pd.to_datetime(pdf[time_col]).to_numpy(dtype="datetime64[ns]")
    return times, time_col, pdf

# ========== Numba-Accelerated Algorithm Implementation ==========
# Numba JIT-compiles these functions to machine code for 10-50x speedup.
# The Kalman filter and RTS smoother involve many tight loops over small
# fixed-size matrices (4x4 state covariance, 2x2 observation covariance),
# which numba handles extremely well.
#
# Algorithm Overview:
# 1. Forward Kalman Filter: Estimates state at each time step using past observations
# 2. RTS Smoother: Backward pass refining estimates using future information
# 3. EM M-step: Learns optimal Q (process noise) and R (measurement noise) matrices
#
# State Vector: [x, vx, y, vy]  (position and velocity in AEQD coordinates)
# Observation: [x, y]  (GPS measurements)

def _make_numba_helpers():
    """
    Create numba JIT-compiled versions of the core filter/smoother algorithms.

    These functions implement the mathematical core of the Kalman filter:
    - Forward filter with prediction and update steps
    - Backward RTS smoother for optimal state estimation
    - EM M-step for learning noise covariance matrices

    Returns: (_compute_run_filter_smoother_numba, _compute_mstep_numba)
    """
    # Get reference to numba module
    nb = numba

    @nb.njit  # JIT compile for performance
    def _compute_run_filter_smoother_numba(n, dt_s, Z, Q_mat, R_mat):
        """
        Forward Kalman filter + backward RTS smoother (numba-compiled version).

        Implements:
        1. Forward Kalman Filter: Predict-update cycle for each time step
        2. Backward RTS Smoother: Refines estimates using future information

        Args:
            n: Number of trajectory points
            dt_s: Time differences (seconds) between consecutive points
            Z: Observations (n x 2) [x, y] in AEQD coordinates
            Q_mat: Process noise covariance (4 x 4)
            R_mat: Measurement noise covariance (2 x 2)

        Returns:
            x_pred: Predicted states (n x 4)
            P_pred: Predicted covariances (n x 4 x 4)
            x_filt: Filtered states (n x 4)
            P_filt: Filtered covariances (n x 4 x 4)
            x_smooth: Smoothed states (n x 4)  <- Best estimates!
            P_smooth: Smoothed covariances (n x 4 x 4)
            P_lag: Lag-one covariances (n-1 x 4 x 4) for EM algorithm
        """
        # ========== Initialize State and Covariance Arrays ==========
        x_pred = np.zeros((n, 4))    # Predicted states: [x, vx, y, vy]
        P_pred = np.zeros((n, 4, 4))  # Predicted covariance matrices
        x_filt = np.zeros((n, 4))     # Filtered states (after update)
        P_filt = np.zeros((n, 4, 4))  # Filtered covariance matrices

        # ========== Initialize First State ==========
        # Initial state: position from first observation, velocity = 0
        x_filt[0, 0] = Z[0, 0]  # x position
        x_filt[0, 2] = Z[0, 1]  # y position
        # Initial covariance: high uncertainty in velocity, measurement uncertainty in position
        P_filt[0] = np.eye(4)
        P_filt[0, 0, 0] = R_mat[0, 0]  # x position uncertainty (from GPS)
        P_filt[0, 2, 2] = R_mat[1, 1]  # y position uncertainty (from GPS)

        # ========== Observation Matrix H ==========
        # Maps state [x, vx, y, vy] to observation [x, y]
        # We only observe positions, not velocities
        H = np.array([[1.0, 0.0, 0.0, 0.0],  # Observe x (ignore vx, y, vy)
                      [0.0, 0.0, 1.0, 0.0]])  # Observe y (ignore x, vx, vy)

        # ========== FORWARD KALMAN FILTER ==========
        for k in range(1, n):
            # Get time step duration
            dt = dt_s[k - 1]

            # ========== State Transition Matrix F ==========
            # Constant-velocity motion model: x_new = x_old + vx*dt, vx_new = vx_old
            # Same for y and vy (independent 2D motion)
            F = np.array([[1.0, dt, 0.0, 0.0],   # x  = x  + vx*dt
                          [0.0, 1.0, 0.0, 0.0],   # vx = vx
                          [0.0, 0.0, 1.0, dt],    # y  = y  + vy*dt
                          [0.0, 0.0, 0.0, 1.0]])  # vy = vy

            # ========== PREDICTION STEP ==========
            # Predict next state using motion model
            x_pred[k] = F @ x_filt[k - 1]
            # Predict next covariance (uncertainty grows due to process noise Q)
            P_pred[k] = F @ P_filt[k - 1] @ F.T + Q_mat

            # ========== Innovation Covariance S ==========
            # Uncertainty in the predicted observation
            # S = H*P_pred*H' + R (observation space covariance)
            S = H @ P_pred[k] @ H.T + R_mat

            # ========== Invert S (2x2 matrix) ==========
            # Use closed-form inverse for 2x2 matrix (faster than general inverse)
            det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
            if det == 0:
                # Singular matrix - use pseudo-inverse (rare edge case)
                S_inv = np.linalg.pinv(S)
            else:
                # Standard 2x2 inverse: [[a,b],[c,d]]^-1 = [[d,-b],[-c,a]]/det
                S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det

            # ========== Kalman Gain K ==========
            # Optimal weighting between prediction and measurement
            # High K -> trust measurement more, Low K -> trust prediction more
            K = P_pred[k] @ H.T @ S_inv

            # ========== Innovation (measurement residual) ==========
            # Difference between actual observation and predicted observation
            y0 = Z[k, 0] - x_pred[k, 0]  # x residual
            y1 = Z[k, 1] - x_pred[k, 2]  # y residual
            y = np.array([y0, y1])

            # ========== UPDATE STEP ==========
            # Correct predicted state using innovation weighted by Kalman gain
            x_filt[k] = x_pred[k] + K @ y
            # Update covariance (uncertainty decreases after incorporating measurement)
            P_filt[k] = (np.eye(4) - K @ H) @ P_pred[k]

        # ========== BACKWARD RTS SMOOTHER ==========
        # Rauch-Tung-Striebel smoother: refines estimates using future information
        # Operates backward from last point to first
        x_smooth = np.zeros_like(x_filt)
        P_smooth = np.zeros_like(P_filt)
        P_lag = np.zeros((n - 1, 4, 4))  # Lag-one covariances for EM

        # Initialize smoothed estimates with filtered estimates at last time step
        x_smooth[n - 1] = x_filt[n - 1]
        P_smooth[n - 1] = P_filt[n - 1]

        # Backward pass
        for k in range(n - 2, -1, -1):  # From n-2 down to 0
            dt = dt_s[k]

            # State transition matrix (same as forward pass)
            F = np.array([[1.0, dt, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, dt],
                          [0.0, 0.0, 0.0, 1.0]])

            # ========== Smoother Gain C ==========
            # Weights future information to refine current estimate
            P_pred_k1 = P_pred[k + 1]
            invPpred = np.linalg.pinv(P_pred_k1)  # Pseudo-inverse (safe for singular matrices)
            Ck = P_filt[k] @ F.T @ invPpred

            # ========== Smooth State Estimate ==========
            # Combine filtered estimate with information from future
            x_smooth[k] = x_filt[k] + Ck @ (x_smooth[k + 1] - x_pred[k + 1])

            # ========== Smooth Covariance Estimate ==========
            P_smooth[k] = P_filt[k] + Ck @ (P_smooth[k + 1] - P_pred_k1) @ Ck.T

            # ========== Lag-One Covariance ==========
            # Needed for EM M-step to compute Q matrix
            P_lag[k] = Ck @ P_smooth[k + 1]

        return x_pred, P_pred, x_filt, P_filt, x_smooth, P_smooth, P_lag

    @nb.njit  # JIT compile for performance
    def _compute_mstep_numba(n, dt_s, x_smooth, P_smooth, P_lag, Z):
        """
        EM M-step: Compute optimal Q and R covariance matrices from smoothed estimates.

        The M-step computes maximum-likelihood estimates of the noise covariances:
        - Q (process noise): How much does the state vary beyond the motion model?
        - R (measurement noise): How noisy are the GPS observations?

        Uses the smoothed state estimates and covariances to compute expected
        sufficient statistics for Q and R estimation.

        Args:
            n: Number of trajectory points
            dt_s: Time differences between consecutive points
            x_smooth: Smoothed state estimates (n x 4)
            P_smooth: Smoothed covariances (n x 4 x 4)
            P_lag: Lag-one covariances (n-1 x 4 x 4)
            Z: Observations (n x 2)

        Returns:
            Q_new: Updated process noise covariance (4 x 4)
            R_new: Updated measurement noise covariance (2 x 2)
        """
        # ========== Compute Q Matrix (Process Noise Covariance) ==========
        # Q represents uncertainty in the motion model
        # Computed from state transition residuals: x[k+1] - F*x[k]
        sum_Q = np.zeros((4, 4))

        for k in range(n - 1):
            dt = dt_s[k]

            # State transition matrix (constant-velocity model)
            F = np.array([[1.0, dt, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, dt],
                          [0.0, 0.0, 0.0, 1.0]])

            # Reshape smoothed states to column vectors for matrix operations
            xk = x_smooth[k].reshape((4, 1))
            xk1 = x_smooth[k + 1].reshape((4, 1))

            # ========== Expected Sufficient Statistics ==========
            # Compute E[x*x'] accounting for uncertainty in state estimates
            # E[x*x'] = P (covariance) + x*x' (outer product of mean)
            E_xk_xkT = P_smooth[k] + xk @ xk.T
            E_xk1_xk1T = P_smooth[k + 1] + xk1 @ xk1.T
            E_xk_xk1T = P_lag[k] + xk @ xk1.T  # Cross-covariance between time steps

            # ========== Q Update Formula ==========
            # Q = E[(x[k+1] - F*x[k]) * (x[k+1] - F*x[k])']
            # Expanded form using expected sufficient statistics:
            term = E_xk1_xk1T - F @ E_xk_xk1T - E_xk_xk1T.T @ F.T + F @ E_xk_xkT @ F.T
            sum_Q += term

        # Average over all time steps
        Q_new = sum_Q / max(1, (n - 1))

        # ========== Compute R Matrix (Measurement Noise Covariance) ==========
        # R represents GPS measurement uncertainty
        # Computed from observation residuals: z[k] - H*x[k]
        sum_R = np.zeros((2, 2))

        # Observation matrix (observe positions only, not velocities)
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])

        for k in range(n):
            # Reshape observation and smoothed state to column vectors
            zk = Z[k].reshape((2, 1))
            xk = x_smooth[k].reshape((4, 1))

            # Expected sufficient statistic: E[x*x']
            E_xk_xkT = P_smooth[k] + xk @ xk.T

            # Predicted observation from smoothed state
            Hx = H @ xk

            # ========== R Update Formula ==========
            # R = E[(z[k] - H*x[k]) * (z[k] - H*x[k])']
            # Expanded form using expected sufficient statistics:
            term = zk @ zk.T - zk @ Hx.T - Hx @ zk.T + H @ E_xk_xkT @ H.T
            sum_R += term

        # Average over all time steps
        R_new = sum_R / max(1, n)

        return Q_new, R_new

    return _compute_run_filter_smoother_numba, _compute_mstep_numba

# prepare numba helpers only if numba available
_numba_helpers = None
if _have_numba:
    try:
        _numba_helpers = _make_numba_helpers()
    except Exception:
        _numba_helpers = None

# ----------------- core per-chunk routine (chooses numba or numpy path) -----------------

def _kalman_rts_em_chunk(
    xs: np.ndarray,
    ys: np.ndarray,
    times_ns: np.ndarray,
    Q_init: np.ndarray,
    R_init: np.ndarray,
    em_iters: int,
    em_tol: float,
    outlier_alpha: Optional[float],
    return_states: bool,
    use_numba: bool = False
):
    """
    Core Kalman filter and RTS smoother with EM parameter estimation for a trajectory chunk.

    This function implements a complete Kalman filtering pipeline:
    1. Forward Kalman filter to estimate states
    2. Backward RTS smoother for optimal state estimates
    3. EM algorithm to learn optimal noise parameters (Q and R matrices)
    4. Outlier detection based on innovation statistics
    """
    n = len(xs)
    if n == 0:
        return np.zeros((0,2)), (np.zeros((0,4)) if return_states else None), np.ones(0, dtype=bool), Q_init, R_init

    # Calculate time differences between consecutive points
    t_ints = times_ns.astype("datetime64[ns]").view("int64")
    dt_s = np.maximum(1.0, np.diff(t_ints) / 1e9)

    # Organize measurements as observation matrix Z
    Z = np.column_stack((xs, ys))

    H = np.array([[1.0,0.0,0.0,0.0],
                  [0.0,0.0,1.0,0.0]])

    x0 = np.array([xs[0], 0.0, ys[0], 0.0], dtype=float)
    P0 = np.eye(4)
    P0[0,0] = R_init[0,0]
    P0[2,2] = R_init[1,1]

    Q = Q_init.copy()
    R = R_init.copy()

    if use_numba and _numba_helpers is not None:
        run_nb, mstep_nb = _numba_helpers
        # EM loop using numba-compiled helper
        for em in range(em_iters + 1):
            x_pred, P_pred, x_filt, P_filt, x_smooth, P_smooth, P_lag = run_nb(n, dt_s, Z, Q, R)
            if em_iters == 0:
                break
            Q_new, R_new = mstep_nb(n, dt_s, x_smooth, P_smooth, P_lag, Z)
            # symmetrize & regularize
            Q_new = 0.5*(Q_new + Q_new.T) + np.eye(4)*1e-9
            R_new = 0.5*(R_new + R_new.T) + np.eye(2)*1e-9
            rel = (np.linalg.norm(Q_new - Q, ord='fro') / (np.linalg.norm(Q, ord='fro') + 1e-12)) + \
                  (np.linalg.norm(R_new - R, ord='fro') / (np.linalg.norm(R, ord='fro') + 1e-12))
            Q, R = Q_new, R_new
            if rel < em_tol:
                x_pred, P_pred, x_filt, P_filt, x_smooth, P_smooth, P_lag = run_nb(n, dt_s, Z, Q, R)
                break
    else:
        # fallback pure numpy EM (exact full-matrix)
        def run_np(Q_mat, R_mat):
            # forward filter
            x_pred = np.zeros((n,4)); P_pred = np.zeros((n,4,4))
            x_filt = np.zeros((n,4)); P_filt = np.zeros((n,4,4))
            x_filt[0] = x0; P_filt[0] = P0.copy()
            for k in range(1,n):
                dt = dt_s[k-1]
                F = np.array([[1.0, dt, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, dt],
                              [0.0, 0.0, 0.0, 1.0]])
                x_pred[k] = F @ x_filt[k-1]
                P_pred[k] = F @ P_filt[k-1] @ F.T + Q_mat
                S = H @ P_pred[k] @ H.T + R_mat
                try:
                    S_inv = sla.inv(S) if _use_scipy else np.linalg.inv(S)
                except Exception:
                    S_inv = np.linalg.pinv(S)
                K = P_pred[k] @ H.T @ S_inv
                y = Z[k] - (H @ x_pred[k])
                x_filt[k] = x_pred[k] + K @ y
                P_filt[k] = (np.eye(4) - K @ H) @ P_pred[k]
            # RTS smoother
            x_smooth = np.zeros_like(x_filt); P_smooth = np.zeros_like(P_filt); P_lag = np.zeros((n-1,4,4))
            x_smooth[-1] = x_filt[-1]; P_smooth[-1] = P_filt[-1]
            for k in range(n-2, -1, -1):
                dt = dt_s[k]
                F = np.array([[1.0, dt, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, dt],
                              [0.0, 0.0, 0.0, 1.0]])
                P_pred_k1 = P_pred[k+1]
                try:
                    invPpred = sla.inv(P_pred_k1) if _use_scipy else np.linalg.inv(P_pred_k1)
                except Exception:
                    invPpred = np.linalg.pinv(P_pred_k1)
                Ck = P_filt[k] @ F.T @ invPpred
                x_smooth[k] = x_filt[k] + Ck @ (x_smooth[k+1] - x_pred[k+1])
                P_smooth[k] = P_filt[k] + Ck @ (P_smooth[k+1] - P_pred_k1) @ Ck.T
                P_lag[k] = Ck @ P_smooth[k+1]
            return x_pred, P_pred, x_filt, P_filt, x_smooth, P_smooth, P_lag

        for em in range(em_iters + 1):
            x_pred, P_pred, x_filt, P_filt, x_smooth, P_smooth, P_lag = run_np(Q, R)
            if em_iters == 0:
                break
            # M-step compute full Q and R
            sum_Q = np.zeros((4,4))
            for k in range(n-1):
                dt = dt_s[k]
                F = np.array([[1.0, dt, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, dt],
                              [0.0, 0.0, 0.0, 1.0]])
                xk = x_smooth[k].reshape(4,1)
                xk1 = x_smooth[k+1].reshape(4,1)
                E_xk_xkT = P_smooth[k] + xk @ xk.T
                E_xk1_xk1T = P_smooth[k+1] + xk1 @ xk1.T
                E_xk_xk1T = P_lag[k] + xk @ xk1.T
                term = E_xk1_xk1T - F @ E_xk_xk1T - E_xk_xk1T.T @ F.T + F @ E_xk_xkT @ F.T
                sum_Q += term
            Q_new = sum_Q / max(1, (n-1))
            sum_R = np.zeros((2,2))
            for k in range(n):
                zk = Z[k].reshape(2,1)
                xk = x_smooth[k].reshape(4,1)
                E_xk_xkT = P_smooth[k] + xk @ xk.T
                Hx = H @ xk
                term = zk @ zk.T - zk @ Hx.T - Hx @ zk.T + H @ E_xk_xkT @ H.T
                sum_R += term
            R_new = sum_R / max(1, n)
            Q_new = 0.5*(Q_new + Q_new.T) + np.eye(4)*1e-9
            R_new = 0.5*(R_new + R_new.T) + np.eye(2)*1e-9
            rel = (np.linalg.norm(Q_new - Q, ord='fro') / (np.linalg.norm(Q, ord='fro') + 1e-12)) + \
                  (np.linalg.norm(R_new - R, ord='fro') / (np.linalg.norm(R, ord='fro') + 1e-12))
            Q, R = Q_new, R_new
            if rel < em_tol:
                x_pred, P_pred, x_filt, P_filt, x_smooth, P_smooth, P_lag = run_np(Q, R)
                break

    # ========== Outlier Detection (Optional) ==========
    # Detect outliers using Mahalanobis distance of innovation (prediction residual)
    # Innovation = observation - prediction (how surprising is this measurement?)
    # Mahalanobis distance accounts for correlation and variance in the innovation
    kept = np.ones(n, dtype=bool)  # Initially mark all points as kept

    if outlier_alpha is not None:
        # ========== Compute Chi-Squared Threshold ==========
        # Under the null hypothesis (no outliers), Mahalanobis distance follows chi-squared(df=2)
        # outlier_alpha = significance level (e.g., 0.01 = 1% false positive rate)
        try:
            import scipy.stats as sps  # type: ignore
            chi2_thresh = sps.chi2.ppf(1.0 - outlier_alpha, df=2)  # df=2 for 2D observations
        except Exception:
            # Fallback values for common alpha levels (from chi-squared table)
            chi2_thresh = 5.991 if outlier_alpha >= 0.05 else 9.21  # 5% or 1% threshold

        # ========== Test Each Observation ==========
        for k in range(1, n):
            # Innovation covariance (uncertainty in predicted observation)
            S = H @ P_pred[k] @ H.T + R

            # Invert S to compute Mahalanobis distance
            try:
                S_inv = sla.inv(S) if _use_scipy else np.linalg.inv(S)
            except Exception:
                S_inv = np.linalg.pinv(S)

            # ========== Mahalanobis Distance ==========
            # Measures how many "standard deviations" the observation is from prediction
            # resid = innovation (observation - prediction)
            resid = Z[k] - (H @ x_pred[k])
            # m2 = resid' * S^-1 * resid (squared Mahalanobis distance)
            m2 = float(resid @ S_inv @ resid)

            # Mark as outlier if distance exceeds threshold
            if m2 > chi2_thresh:
                kept[k] = False  # Outlier detected

    # ========== Extract Final Results ==========
    # Return smoothed positions (x, y) from full state vector [x, vx, y, vy]
    smoothed_positions = x_smooth[:, [0, 2]]  # Extract positions (columns 0 and 2)
    smoothed_states = x_smooth if return_states else None  # Full state if requested
    return smoothed_positions, smoothed_states, kept, Q, R

# ----------------- public API (sequential, per-chunk AEQD default, memmap optional) -----------------

def kalman_aeqd_filter(
    df: Union[pd.DataFrame, pl.DataFrame],
    lat_col: str = "lat",
    lon_col: str = "lon",
    time_col: Optional[str] = "time",
    process_noise_std_m_init: float = 1.0,
    measurement_noise_std_m_init: float = 10.0,
    em_iters: int = 3,
    em_tol: float = 1e-3,
    chunk_size: Optional[int] = None,
    overlap: int = 50,
    outlier_alpha: Optional[float] = 0.01,
    return_states: bool = False,
    per_chunk_aeqd: bool = True,
    use_mmap: bool = False,
    mmap_dir: Optional[str] = None,
    mmap_threshold: int = 200_000,
    return_QR: bool = False,
    use_numba: bool = False,
    transformer_cache_precision: int = 6,
    verbose: bool = False,
):
    """
    Apply Kalman filtering and RTS smoothing to a GPS trajectory with automatic noise parameter learning.

    This function smooths noisy GPS trajectories using a Kalman filter with a constant-velocity motion model.
    It uses an Azimuthal Equidistant (AEQD) projection for accurate metric calculations and can automatically
    learn optimal noise parameters through the EM algorithm. Optionally detects and flags outliers.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input trajectory with latitude, longitude, and optionally time columns.
    lat_col : str, default="lat"
        Name of the latitude column.
    lon_col : str, default="lon"
        Name of the longitude column.
    time_col : str or None, default="time"
        Name of the time column. If None, synthetic timestamps are created.
    process_noise_std_m_init : float, default=1.0
        Initial process noise standard deviation in meters (affects position).
    measurement_noise_std_m_init : float, default=10.0
        Initial measurement noise standard deviation in meters (GPS error).
    em_iters : int, default=3
        Number of EM iterations for learning noise parameters (0 = use initial values).
    em_tol : float, default=1e-3
        EM convergence tolerance (relative change in Q and R matrices).
    chunk_size : int or None, default=None
        Process trajectory in chunks of this size (useful for very long trajectories).
    overlap : int, default=50
        Number of overlapping points between chunks for smooth stitching.
    outlier_alpha : float or None, default=0.01
        Significance level for outlier detection (None = no outlier detection).
    return_states : bool, default=False
        If True, also return full state vectors (position + velocity).
    per_chunk_aeqd : bool, default=True
        If True, use a separate AEQD projection for each chunk.
    use_mmap : bool, default=False
        Use memory-mapped arrays for very large trajectories.
    mmap_dir : str or None, default=None
        Directory for memory-mapped files.
    mmap_threshold : int, default=200000
        Trajectory length above which to use memory mapping.
    return_QR : bool, default=False
        If True, also return learned Q and R covariance matrices.
    use_numba : bool, default=False
        Use numba JIT compilation for faster processing (if numba is available).
    transformer_cache_precision : int, default=6
        Number of decimal places for caching coordinate transformers.
    verbose : bool, default=False
        Print progress information during processing.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Smoothed trajectory with the same type as input. Contains smoothed lat/lon and a '_kept' column
        indicating which points passed outlier detection.
    np.ndarray (optional)
        If return_states=True, returns full state vectors (n x 4) with position and velocity.
    tuple (optional)
        If return_QR=True, returns (avg_Q, avg_R, qr_list) with learned covariance matrices.

    Notes
    -----
    The Kalman filter uses a 2D constant-velocity model:
    - State: [x, vx, y, vy] (position and velocity in AEQD projected coordinates)
    - Observation: [x, y] (GPS measurements)
    The EM algorithm learns optimal Q (process noise) and R (measurement noise) from the data.
    """
    # ========== Validate Parameters and Dependencies ==========
    # Check if numba is actually available and working
    use_numba = bool(use_numba and _have_numba)
    if use_numba and _numba_helpers is None:
        # numba present but helper compilation failed - fall back to numpy
        use_numba = False

    # ========== Convert Input to Pandas ==========
    # Process with pandas internally, convert back to original type at end
    pdf, was_polars = _to_pandas_preserve(df)
    n = len(pdf)

    # Handle empty trajectory
    if n == 0:
        if return_states:
            return _from_pandas_preserve(pdf, was_polars), np.zeros((0, 4))
        return _from_pandas_preserve(pdf, was_polars)

    # ========== Ensure Time Column Exists ==========
    # Create synthetic timestamps if none provided (needed for Δt calculation)
    times_ns, used_time_col, pdf = _ensure_time_array(pdf, time_col)

    # Extract coordinate arrays
    lats = pdf[lat_col].to_numpy(dtype=float)
    lons = pdf[lon_col].to_numpy(dtype=float)

    # ========== Memory Mapping Decision ==========
    # For very large trajectories, use memory-mapped arrays to reduce RAM usage
    use_mmap_local = bool(use_mmap) and (n >= int(mmap_threshold))
    tmp_files = []  # Track temporary files for cleanup

    # ========== Initialize Noise Covariance Matrices ==========
    # Q: Process noise (how much uncertainty in motion model)
    # R: Measurement noise (how noisy are GPS observations)
    q0 = process_noise_std_m_init ** 2  # Position variance
    r0 = measurement_noise_std_m_init ** 2  # Measurement variance
    q_vel = max(q0 * 0.01, 1e-6)  # Velocity variance (typically smaller than position)

    # Q matrix (4x4): diagonal for independent x and y motion
    # [q_x, q_vx, q_y, q_vy]
    Q_init = np.diag([q0, q_vel, q0, q_vel])

    # R matrix (2x2): diagonal for independent x and y measurements
    # [r_x, r_y]
    R_init = np.diag([r0, r0])

    # ========== Create Trajectory Chunks with Overlap ==========
    # Large trajectories are divided into overlapping chunks for:
    # 1. Memory efficiency (process sequentially)
    # 2. Per-chunk AEQD projection accuracy (each chunk gets own projection center)
    # 3. Parallelization potential (future enhancement)
    #
    # Chunks overlap to ensure smooth stitching at boundaries
    if (chunk_size is None) or (chunk_size >= n):
        # Process entire trajectory as single chunk
        chunks = [(0, n)]
    else:
        # Validate overlap parameter
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        # ========== Build Chunk List ==========
        # Each chunk: (chunk_start, chunk_end) with overlap regions
        # Example with chunk_size=1000, overlap=50:
        #   Chunk 0: [0, 1050]
        #   Chunk 1: [950, 2050]  (overlaps with chunk 0 by 100 points)
        #   Chunk 2: [1950, 3000] (overlaps with chunk 1 by 100 points)
        chunks = []
        start = 0
        while start < n:
            end = min(n, start + chunk_size)
            # Expand chunk boundaries by overlap (but stay within [0, n])
            cs = max(0, start - overlap)  # Chunk start (with left overlap)
            ce = min(n, end + overlap)    # Chunk end (with right overlap)
            chunks.append((cs, ce))
            start = end  # Move to next chunk (non-overlapping boundary)

    # prepare global outputs
    smoothed_lat_all = np.full(n, np.nan, dtype=float)
    smoothed_lon_all = np.full(n, np.nan, dtype=float)
    kept_mask_all = np.ones(n, dtype=bool)
    states_all = np.zeros((n,4), dtype=float) if return_states else None
    filled = np.zeros(n, dtype=bool)
    qr_list = []

    # Show progress information if requested
    chunk_iter = chunks
    if verbose:
        if _have_tqdm:
            chunk_iter = tqdm(chunks, desc="kalman chunks")
        else:
            print(f"Processing {len(chunks)} chunks...")

    # For global AEQD projection, build it once for all chunks
    if not per_chunk_aeqd:
        cen_lat_global = float(np.mean(lats))
        cen_lon_global = float(np.mean(lons))
        proj_fwd_global, proj_inv_global = _get_cached_aeqd_transformer(
            cen_lat_global, cen_lon_global, precision=transformer_cache_precision
        )
        xs_all, ys_all = proj_fwd_global.transform(lons, lats)

    # Process each chunk
    for cs, ce in chunk_iter:
        lats_chunk = lats[cs:ce]
        lons_chunk = lons[cs:ce]
        times_chunk = times_ns[cs:ce]

        if per_chunk_aeqd:
            # Use separate AEQD projection centered on this chunk's centroid
            cen_lat = float(np.mean(lats_chunk))
            cen_lon = float(np.mean(lons_chunk))
            fwd, inv = _get_cached_aeqd_transformer(cen_lat, cen_lon, precision=transformer_cache_precision)
            xs_chunk, ys_chunk = fwd.transform(lons_chunk, lats_chunk)
        else:
            # Use global AEQD projection for this chunk
            xs_chunk = xs_all[cs:ce]
            ys_chunk = ys_all[cs:ce]
            inv = proj_inv_global

        # call core per-chunk routine
        sm_pos, sm_states, kept, Q_est, R_est = _kalman_rts_em_chunk(
            xs_chunk, ys_chunk, times_chunk,
            Q_init=Q_init, R_init=R_init,
            em_iters=em_iters, em_tol=em_tol,
            outlier_alpha=outlier_alpha, return_states=return_states,
            use_numba=use_numba
        )

        qr_list.append((Q_est, R_est))

        # Transform smoothed coordinates back to lat/lon
        sm_xs = sm_pos[:, 0]
        sm_ys = sm_pos[:, 1]
        lon_sm_chunk, lat_sm_chunk = inv.transform(sm_xs, sm_ys)

        # Determine which part of the chunk to keep (trim overlap regions)
        chunk_len = ce - cs
        if len(chunks) == 1:
            local_keep_s, local_keep_e = 0, chunk_len
        else:
            left_trim = overlap if cs != 0 else 0
            right_trim = overlap if ce != n else 0
            local_keep_s = left_trim
            local_keep_e = chunk_len - right_trim

        # Stitch chunk results into global arrays
        global_start = cs + local_keep_s
        num_to_fill = local_keep_e - local_keep_s

        for local_idx in range(num_to_fill):
            li = local_keep_s + local_idx
            global_idx = global_start + local_idx

            if not filled[global_idx]:
                # First time filling this position
                smoothed_lat_all[global_idx] = lat_sm_chunk[li]
                smoothed_lon_all[global_idx] = lon_sm_chunk[li]
                if return_states:
                    states_all[global_idx] = sm_states[li]
                kept_mask_all[global_idx] = kept[li]
                filled[global_idx] = True
            else:
                # Already filled from another chunk - average the results
                smoothed_lat_all[global_idx] = 0.5 * (smoothed_lat_all[global_idx] + lat_sm_chunk[li])
                smoothed_lon_all[global_idx] = 0.5 * (smoothed_lon_all[global_idx] + lon_sm_chunk[li])
                if return_states:
                    states_all[global_idx] = 0.5 * (states_all[global_idx] + sm_states[li])
                kept_mask_all[global_idx] = kept_mask_all[global_idx] & kept[li]

    # Handle any unfilled positions (shouldn't happen but be safe)
    unfilled = np.where(~filled)[0]
    num_unfilled = len(unfilled)
    if num_unfilled > 0:
        smoothed_lat_all[unfilled] = lats[unfilled]
        smoothed_lon_all[unfilled] = lons[unfilled]
        if return_states:
            states_all[unfilled] = np.zeros((num_unfilled, 4))
        kept_mask_all[unfilled] = True

    # Build output DataFrame with smoothed coordinates
    out_pdf = pdf.copy()
    out_pdf[lat_col] = smoothed_lat_all
    out_pdf[lon_col] = smoothed_lon_all
    out_pdf["_kept"] = kept_mask_all

    if used_time_col == "_synthetic_time" and time_col is None:
        out_pdf = out_pdf.drop(columns=[used_time_col])

    out = _from_pandas_preserve(out_pdf, was_polars)

    if return_QR:
        # Compute average learned covariance matrices across all chunks
        num_qr = max(1, len(qr_list))
        avg_Q = sum(q for q, r in qr_list) / num_qr
        avg_R = sum(r for q, r in qr_list) / num_qr
        if return_states:
            return out, states_all, (avg_Q, avg_R, qr_list)
        return out, (avg_Q, avg_R, qr_list)

    if return_states:
        return out, states_all
    return out
