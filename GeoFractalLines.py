# -*- coding: utf-8 -*-

"""
***************************************************************************
*                                                                         *
*   GeoFractalLines v1.0.0                                               *
*   Box-Counting Fractal Analysis for Line Geometries                    *
*                                                                         *
***************************************************************************
"""

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterDefinition,
    QgsFeature,
    QgsGeometry,
    QgsVectorLayer,
    QgsFields,
    QgsField,
    QgsProject,
    QgsRectangle,
    QgsWkbTypes,
    QgsProcessingOutputVectorLayer,
    QgsProcessingOutputString
)
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
import pandas as pd
import os
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass, field, fields
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

__version__ = "1.0.0"

# ============================================================================
# CONSTANTS
# ============================================================================
EPS = 1e-15
MIN_ALPHA = 0.3
MAX_ALPHA = 3.5
MIN_LOCAL_R2 = 0.8
MIN_CURVATURE_RATIO = 0.7
SPLINE_SMOOTHING_FACTORS = [0.01, 0.05, 0.1, 0.2, 0.5]
SPLINE_MIN_KNOTS = 3
ADAPTIVE_SAMPLING_FACTOR = 0.8
SCALE_RANGE_LOW = 0.005
SCALE_RANGE_HIGH = 0.6
MAX_SCALES = 60
MIN_WINDOW = 6
WINDOW_STEP = 3
LOCAL_WINDOW_RATIO = 0.25
MIN_LOCAL_WINDOW = 5
LACUNARITY_MAX_SCALES = 8
MF_MAX_SCALES = 15
Q_FINE_POINTS = 200
Q_TOLERANCE = 0.5
MIN_VALID_COUNTS = 8
MIN_SURROGATE_COUNTS = 8
SURROGATE_STD_THRESHOLD = 0.001
BOOTSTRAP_MIN_COUNTS = 50
F_ALPHA_MIN = -0.1
CONCAVITY_THRESHOLD = 0.01
LOG_OFFSET = 1e-12
PROB_CLIP_MIN = 1e-15
GEOM_LENGTH_THRESHOLD = 5.0
CENTROID_DEFAULT = (0.0, 0.0)
PARALLEL_THRESHOLD = 50

KOCH_DIMENSION = 1.26186

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    base_path: str = ""
    bootstrap_iter: int = 500
    min_points_per_box: int = 5
    max_points: int = 200
    batch_size: int = 30
    q_values: np.ndarray = field(default_factory=lambda: np.arange(-5, 5.5, 0.5))
    min_scales_for_mf: int = 8
    surrogate_count: int = 100
    min_decades: float = 1.0
    isotropy_angles: int = 12
    tau_r2_min: float = 0.85
    isotropy_cv_max: float = 0.25
    surrogate_p_max: float = 0.05
    min_r2: float = 0.90
    min_mf_points: int = 50
    min_geometry_points: int = 20
    min_total_points: int = 20
    min_scales: int = 5
    random_seed: int = 42
    robust_box_counting: bool = True
    grid_offsets: int = 8
    adaptive_sampling: bool = True
    residual_check: bool = True
    use_parallel: bool = False
    max_workers: int = 4
    run_validation: bool = True

    def __post_init__(self):
        if not self.base_path:
            self.base_path = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(self.base_path):
            try:
                os.makedirs(self.base_path)
            except:
                self.base_path = os.path.join(os.path.expanduser("~"), "Downloads")
        np.random.seed(self.random_seed)
        self.timestamp = datetime.now().isoformat()

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass(slots=True)
class BaseMetrics:
    fid: int
    x: float
    y: float
    length_m: float
    d_fractal: float
    d_r2: float
    d_se: float
    d_ci95_low: float
    d_ci95_high: float
    d_ci95_range: float
    d_bootstrap_std: float
    d_p_value: float
    scale_decades: float
    n_scales: int
    n_points: int
    lacunarity: float
    surrogate_p_value: float
    significant_vs_csr: bool
    surrogate_d_mean: float
    surrogate_d_std: float
    isotropy_cv: float
    is_isotropic: bool
    scaling_quality: float

@dataclass(slots=True)
class MFMetrics:
    mf_dq_range: float = 0.0
    mf_dq_variance: float = 0.0
    mf_d0: float = 0.0
    mf_d1_direct: float = 0.0
    mf_d2: float = 0.0
    mf_alpha_range: float = 0.0
    mf_d0_spectrum: float = 0.0
    mf_asymmetry: float = 0.0
    mf_concave: bool = False
    mf_legendre_consistency: float = 0.0
    mf_tau_mean_r2: float = 0.0
    mf_is_reliable: bool = False
    has_mf_spectrum: bool = False

@dataclass(slots=True)
class FractalResult:
    base: BaseMetrics
    mf: MFMetrics = field(default_factory=MFMetrics)
    geom: Optional[QgsGeometry] = None

    def to_dict(self) -> dict:
        result = {}
        for f in fields(BaseMetrics):
            result[f.name] = getattr(self.base, f.name)
        for f in fields(MFMetrics):
            result[f.name] = getattr(self.mf, f.name)
        return result

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def results_to_dataframe(results_list: List[FractalResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results_list])

def safe_percentage(numerator: int, denominator: int) -> float:
    return 100 * numerator / max(denominator, 1)

def append_if_valid(result: Optional[FractalResult], results_list: List[FractalResult]) -> None:
    if result:
        results_list.append(result)

# ============================================================================
# BOX COUNTING
# ============================================================================
def _count_boxes_single(pts_array: np.ndarray, scale: float, min_xy: Tuple[float, float]) -> Tuple[int, Optional[np.ndarray]]:
    gx = np.floor((pts_array[:, 0] - min_xy[0]) / scale).astype(np.int32)
    gy = np.floor((pts_array[:, 1] - min_xy[1]) / scale).astype(np.int32)
    gx -= gx.min()
    gy -= gy.min()
    nx, ny = gx.max() + 1, gy.max() + 1
    if nx <= 0 or ny <= 0:
        return 0, None
    flat = np.ravel_multi_index((gx, gy), (nx, ny))
    return np.unique(flat).size, flat

def box_count(pts_array: np.ndarray, scale: float, min_xy: Tuple[float, float],
              robust: bool = True, offsets: int = 8, return_flat: bool = False) -> Tuple[float, Optional[np.ndarray]]:
    if scale <= 0 or len(pts_array) == 0:
        return 0, None
    if not robust:
        count, flat = _count_boxes_single(pts_array, scale, min_xy)
        return float(count), flat if return_flat else None
    counts = []
    all_flats = []
    offset_values = np.linspace(0, scale, offsets, endpoint=False)
    for ox in offset_values:
        for oy in offset_values:
            shifted_min = (min_xy[0] - ox, min_xy[1] - oy)
            count, flat = _count_boxes_single(pts_array, scale, shifted_min)
            if count > 0:
                counts.append(count)
                if return_flat:
                    all_flats.append(flat)
    if not counts:
        return 0, None
    mean_count = float(np.mean(counts))
    flat_result = all_flats[0] if return_flat and all_flats else None
    return mean_count, flat_result

# ============================================================================
# GEOMETRY PROCESSING
# ============================================================================
def geometry_to_array(geom: QgsGeometry, cfg: Config) -> Tuple[Optional[np.ndarray], Tuple[float, float]]:
    try:
        if not geom or geom.isEmpty() or geom.length() < GEOM_LENGTH_THRESHOLD:
            return None, (0, 0)
        if not geom.isGeosValid():
            geom = geom.makeValid()
        vertices = []
        for v in geom.vertices():
            vertices.append([v.x(), v.y()])
        target_points = max(cfg.min_geometry_points, min(cfg.max_points, int(geom.length() * ADAPTIVE_SAMPLING_FACTOR)))
        if len(vertices) < target_points:
            step = geom.length() / (target_points - len(vertices) + 1)
            for i in range(1, target_points - len(vertices) + 1):
                try:
                    pt = geom.interpolate(i * step)
                    if pt and not pt.isNull():
                        p = pt.asPoint()
                        if p:
                            vertices.append([p.x(), p.y()])
                except Exception:
                    continue
        pts_array = np.array(vertices, dtype=np.float64)
        _, idx = np.unique(pts_array, axis=0, return_index=True)
        pts_array = pts_array[np.sort(idx)]
        if len(pts_array) < cfg.min_geometry_points:
            return None, (0, 0)
        centroid = geom.centroid().asPoint()
        centroid_xy = (centroid.x(), centroid.y()) if centroid else CENTROID_DEFAULT
        return pts_array, centroid_xy
    except Exception:
        return None, (0, 0)

def check_isotropy(pts_array: np.ndarray, grid_size: float, cfg: Config) -> Tuple[float, bool]:
    try:
        if len(pts_array) < cfg.min_geometry_points:
            return 1.0, False
        centroid = np.mean(pts_array, axis=0)
        centered = pts_array - centroid
        angles = np.linspace(0, 180, cfg.isotropy_angles, endpoint=False)
        counts = []
        for angle in angles:
            theta = np.radians(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            rotated = centered @ rot_matrix.T
            min_xy_rot = (rotated[:, 0].min(), rotated[:, 1].min())
            count, _ = box_count(rotated, grid_size, min_xy_rot, robust=True, offsets=3)
            counts.append(count)
        mean_c = np.mean(counts)
        cv = float(np.std(counts) / mean_c) if mean_c > 0 else 1.0
        return cv, cv < cfg.isotropy_cv_max
    except Exception:
        return 1.0, False

# ============================================================================
# SCALING REGIME DETECTION
# ============================================================================
def _compute_scale_range(bbox: QgsRectangle) -> np.ndarray:
    L = max(bbox.width(), bbox.height())
    if L <= 0:
        return np.array([])
    return np.geomspace(L * SCALE_RANGE_LOW, L * SCALE_RANGE_HIGH, MAX_SCALES)

def _filter_valid_counts(pts_array: np.ndarray, scales: np.ndarray, min_xy: Tuple[float, float],
                         cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    all_counts = []
    for s in scales:
        count, _ = box_count(pts_array, s, min_xy, robust=cfg.robust_box_counting, offsets=cfg.grid_offsets)
        if cfg.min_points_per_box <= count <= len(pts_array) * 0.5:
            all_counts.append(count)
        else:
            all_counts.append(np.nan)
    valid_mask = ~np.isnan(all_counts)
    scales_v = scales[valid_mask]
    counts_v = np.array(all_counts)[valid_mask]
    return scales_v, counts_v

def _find_best_scaling_window(x: np.ndarray, y: np.ndarray, cfg: Config) -> Tuple[int, int]:
    best_aic = np.inf
    best_start, best_end = 0, len(x)
    window_min = max(MIN_WINDOW, len(x) // 4)
    for start in range(0, len(x) - window_min + 1, WINDOW_STEP):
        for end in range(start + window_min, len(x) + 1, WINDOW_STEP):
            decades = (x[end-1] - x[start]) / np.log(10)
            if decades < cfg.min_decades:
                continue
            try:
                xs, ys = x[start:end], y[start:end]
                n_seg = len(xs)
                slope, intercept, r_value, _, _ = stats.linregress(xs, ys)
                rss = np.sum((ys - (slope * xs + intercept))**2)
                if rss > 0 and n_seg > 2:
                    aic = n_seg * np.log(rss / n_seg) + 4
                    aic = aic - 2 * (end - start) / len(x)
                    if aic < best_aic:
                        best_aic = aic
                        best_start, best_end = start, end
            except Exception:
                continue
    return best_start, best_end

def _validate_residual_quality(x: np.ndarray, y: np.ndarray, best_start: int, best_end: int,
                                cfg: Config) -> Tuple[int, int]:
    if best_end - best_start < MIN_WINDOW:
        return best_start, best_end
    xs = x[best_start:best_end]
    ys = y[best_start:best_end]
    try:
        slope, _, _, _, _ = stats.linregress(xs, ys)
    except Exception:
        return best_start, best_end
    local_window = max(MIN_LOCAL_WINDOW, int(len(xs) * LOCAL_WINDOW_RATIO))
    local_r2 = []
    for i in range(local_window, len(xs) - local_window):
        try:
            _, _, local_r, _, _ = stats.linregress(
                xs[i-local_window:i+local_window],
                ys[i-local_window:i+local_window]
            )
            local_r2.append(local_r**2)
        except Exception:
            continue
    if local_r2 and min(local_r2) < MIN_LOCAL_R2:
        good_indices = np.where(np.array(local_r2) >= MIN_LOCAL_R2)[0]
        if len(good_indices) > MIN_WINDOW:
            best_start = best_start + good_indices[0]
            best_end = best_start + good_indices[-1] - good_indices[0] + 2 * local_window
    return best_start, best_end

def detect_scaling_regime(pts_array: np.ndarray, bbox: QgsRectangle, cfg: Config) -> Tuple[List[float], List[float]]:
    try:
        scales = _compute_scale_range(bbox)
        if len(scales) == 0:
            return [], []
        min_xy = (bbox.xMinimum(), bbox.yMinimum())
        scales_v, counts_v = _filter_valid_counts(pts_array, scales, min_xy, cfg)
        n_scales = len(scales_v)
        if n_scales < MIN_VALID_COUNTS:
            return [], []
        x = np.log(1.0 / scales_v)
        y = np.log(counts_v)
        best_start, best_end = _find_best_scaling_window(x, y, cfg)
        if cfg.residual_check:
            best_start, best_end = _validate_residual_quality(x, y, best_start, best_end, cfg)
        if best_end - best_start < MIN_WINDOW:
            return [], []
        return list(scales_v[best_start:best_end]), list(counts_v[best_start:best_end])
    except Exception:
        return [], []

# ============================================================================
# FRACTAL DIMENSION ANALYSIS
# ============================================================================
def bootstrap_fractal_dimension(x: np.ndarray, y: np.ndarray, cfg: Config) -> Dict[str, float]:
    n = len(x)
    boot_D = []
    for _ in range(cfg.bootstrap_iter):
        idx = np.random.randint(0, n, n)
        xx = x[idx]
        yy = y[idx]
        order = np.argsort(xx)
        try:
            bs, _, _, _, _ = stats.linregress(xx[order], yy[order])
            boot_D.append(bs)
        except Exception:
            continue
    try:
        slope, _, _, _, std_err = stats.linregress(x, y)
    except Exception:
        return {'ci_low': 0.0, 'ci_high': 0.0, 'ci_range': 0.0, 'boot_std': 0.0}
    if len(boot_D) > BOOTSTRAP_MIN_COUNTS:
        ci_low = float(np.percentile(boot_D, 2.5))
        ci_high = float(np.percentile(boot_D, 97.5))
        boot_std = float(np.std(boot_D))
        ci_range = float(ci_high - ci_low)
    else:
        ci_low = float(slope - 2 * std_err)
        ci_high = float(slope + 2 * std_err)
        boot_std = float(std_err)
        ci_range = float(4 * std_err)
    return {'ci_low': ci_low, 'ci_high': ci_high, 'ci_range': ci_range, 'boot_std': boot_std}

def test_against_csr(true_D: float, pts_array: np.ndarray, bbox: QgsRectangle,
                     regime_scales: List[float], cfg: Config) -> Tuple[float, bool, float, float]:
    n_scales = len(regime_scales)
    if n_scales < 5:
        return 1.0, False, 0.0, 0.0
    min_xy = (bbox.xMinimum(), bbox.yMinimum())
    max_xy = (bbox.xMaximum(), bbox.yMaximum())
    n_test_scales = min(LACUNARITY_MAX_SCALES, n_scales)
    indices = np.linspace(0, n_scales - 1, n_test_scales, dtype=int)
    test_scales = np.array(regime_scales)[indices]
    surrogate_Ds = []
    for _ in range(cfg.surrogate_count):
        surr = np.column_stack([
            np.random.uniform(min_xy[0], max_xy[0], len(pts_array)),
            np.random.uniform(min_xy[1], max_xy[1], len(pts_array))
        ])
        try:
            counts = []
            for s in test_scales:
                count, _ = box_count(surr, s, min_xy, robust=True, offsets=3)
                counts.append(count)
            if len(counts) >= 5:
                xs = np.log(1.0 / test_scales[:len(counts)])
                ys = np.log(np.array(counts) + LOG_OFFSET)
                s_D, _, _, _, _ = stats.linregress(xs, ys)
                surrogate_Ds.append(s_D)
        except Exception:
            continue
    if len(surrogate_Ds) < MIN_SURROGATE_COUNTS:
        return 1.0, False, 0.0, 0.0
    mean_surr = float(np.mean(surrogate_Ds))
    std_surr = float(np.std(surrogate_Ds))
    if std_surr > SURROGATE_STD_THRESHOLD:
        z_score = (true_D - mean_surr) / std_surr
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))
        significant = p_value < cfg.surrogate_p_max
    else:
        p_value = 1.0
        significant = False
    return p_value, significant, mean_surr, std_surr

def compute_lacunarity(pts_array: np.ndarray, bbox: QgsRectangle,
                       regime_scales: List[float], cfg: Config) -> float:
    min_xy = (bbox.xMinimum(), bbox.yMinimum())
    lac_vals = []
    for s in regime_scales[:LACUNARITY_MAX_SCALES]:
        count, flat = box_count(pts_array, s, min_xy, 
                               robust=cfg.robust_box_counting, 
                               offsets=cfg.grid_offsets, 
                               return_flat=True)
        if flat is not None and len(flat) > 0:
            _, cnts = np.unique(flat, return_counts=True)
            if len(cnts) > 1 and np.mean(cnts) > 0:
                lac_vals.append(float(np.var(cnts) / (np.mean(cnts)**2)))
    return float(np.mean(lac_vals)) if lac_vals else 0.0

# ============================================================================
# MULTIFRACTAL ANALYSIS
# ============================================================================
def _validate_mf_inputs(pts_array: np.ndarray, regime_scales: List[float], cfg: Config) -> bool:
    if len(pts_array) < cfg.min_mf_points:
        return False
    if len(regime_scales) < cfg.min_scales_for_mf:
        return False
    return True

def _select_mf_scales(regime_scales: List[float]) -> np.ndarray:
    n_total_scales = len(regime_scales)
    n_scales = min(MF_MAX_SCALES, n_total_scales)
    indices = np.linspace(0, n_total_scales - 1, n_scales, dtype=int)
    return np.array(regime_scales)[indices]

def _compute_probabilities(flat: np.ndarray, cfg: Config) -> Tuple[Optional[np.ndarray], float]:
    if flat is None:
        return None, 0.0
    _, counts = np.unique(flat, return_counts=True)
    total = np.sum(counts)
    if total < cfg.min_total_points:
        return None, 0.0
    probs = np.clip(counts.astype(np.float64) / total, PROB_CLIP_MIN, None)
    probs = probs / np.sum(probs)
    if len(probs) < 3:
        return None, 0.0
    entropy = float(-np.sum(probs * np.log(probs)))
    return probs, entropy

def _compute_Zq_for_scale(probs: np.ndarray, entropy: float, cfg: Config) -> np.ndarray:
    Zq_row = []
    for q in cfg.q_values:
        if abs(q) < EPS:
            Zq = float(len(probs))
        elif abs(q - 1.0) < EPS:
            Zq = np.exp(-entropy)
        else:
            powers = probs ** q
            powers = powers[np.isfinite(powers)]
            Zq = float(np.sum(powers)) if len(powers) > 0 else EPS
        Zq_row.append(max(Zq, EPS))
    return np.array(Zq_row)

def _compute_partition_function(pts_array: np.ndarray, bbox: QgsRectangle,
                                scales: np.ndarray, cfg: Config) -> Tuple[np.ndarray, List[float]]:
    min_xy = (bbox.xMinimum(), bbox.yMinimum())
    Zq_matrix = []
    entropy_vals = []
    for s in scales:
        count, flat = box_count(pts_array, s, min_xy,
                               robust=cfg.robust_box_counting,
                               offsets=cfg.grid_offsets,
                               return_flat=True)
        if flat is None:
            continue
        probs, entropy = _compute_probabilities(flat, cfg)
        if probs is None:
            continue
        Zq_row = _compute_Zq_for_scale(probs, entropy, cfg)
        Zq_matrix.append(Zq_row)
        entropy_vals.append(entropy)
    return np.array(Zq_matrix), entropy_vals

def _fit_tau_q(Zq_matrix: np.ndarray, scales: np.ndarray, entropy_vals: List[float],
                cfg: Config) -> Tuple[List[Dict], List[Dict], float]:
    valid_scales = scales[:len(Zq_matrix)]
    log_inv_scales = np.log(1.0 / valid_scales)
    D1_direct = 0.0
    if len(entropy_vals) >= 3:
        try:
            ent_slope, _, _, _, _ = stats.linregress(log_inv_scales[:len(entropy_vals)], entropy_vals)
            D1_direct = float(ent_slope)
        except Exception:
            pass
    tau_data = []
    Dq_data = []
    for q_idx, q in enumerate(cfg.q_values):
        log_Zq = np.log(Zq_matrix[:, q_idx])
        try:
            slope, _, r_value, _, std_err = stats.linregress(log_inv_scales, log_Zq)
            tau_q = -float(slope)
            if abs(q) < EPS:
                Dq = -tau_q
            elif abs(q - 1.0) < EPS:
                Dq = D1_direct if D1_direct > 0 else tau_q / (q - 1.0)
            else:
                Dq = tau_q / (q - 1.0)
            tau_data.append({
                'q': float(q), 'tau': tau_q,
                'r2': float(r_value**2), 'se': float(std_err)
            })
            Dq_data.append({
                'q': float(q), 'Dq': Dq,
                'r2': float(r_value**2)
            })
        except Exception:
            continue
    return tau_data, Dq_data, D1_direct

def _compute_legendre_spectrum(tau_data: List[Dict], cfg: Config) -> Dict[str, Any]:
    if len(tau_data) < 4:
        return {'alpha_range': 0.0, 'D0_spectrum': 0.0,
                'asym': 0.0, 'concave': False, 'is_reliable': False}
    q_arr = np.array([t['q'] for t in tau_data])
    tau_arr = np.array([t['tau'] for t in tau_data])
    try:
        best_spline = None
        best_score = -np.inf
        for s_factor in SPLINE_SMOOTHING_FACTORS:
            try:
                spline = UnivariateSpline(
                    q_arr, tau_arr,
                    s=s_factor * len(q_arr),
                    k=min(SPLINE_MIN_KNOTS, len(q_arr)-1)
                )
                alpha_test = spline.derivative()(q_arr)
                if len(alpha_test) > 1 and np.all(np.diff(alpha_test) >= 0):
                    residuals = tau_arr - spline(q_arr)
                    score = -np.mean(residuals**2)
                    if score > best_score:
                        best_score = score
                        best_spline = spline
            except Exception:
                continue
        if best_spline is None:
            best_spline = UnivariateSpline(q_arr, tau_arr, s=0.1 * len(q_arr), k=3)
        q_fine = np.linspace(q_arr.min(), q_arr.max(), Q_FINE_POINTS)
        alpha_fine = best_spline.derivative()(q_fine)
        f_alpha = q_fine * alpha_fine - best_spline(q_fine)
        valid = (alpha_fine > MIN_ALPHA) & (alpha_fine < MAX_ALPHA) & (f_alpha > F_ALPHA_MIN)
        a_v = alpha_fine[valid]
        f_v = f_alpha[valid]
        if len(a_v) <= 20:
            return {'alpha_range': 0.0, 'D0_spectrum': 0.0,
                    'asym': 0.0, 'concave': False, 'is_reliable': False}
        alpha_range = float(np.max(a_v) - np.min(a_v))
        D0_spectrum = float(np.max(f_v))
        f_max_i = np.argmax(f_v)
        a_max = a_v[f_max_i]
        a_l, a_r = a_v[:f_max_i], a_v[f_max_i+1:]
        asym = 0.0
        if len(a_l) and len(a_r):
            asym = float(((a_r[-1] - a_max) - (a_max - a_l[0])) /
                        (a_r[-1] - a_l[0] + EPS))
        si = np.argsort(a_v)
        a_s, f_s = a_v[si], f_v[si]
        d2 = np.diff(f_s, 2)
        da2 = np.diff(a_s)[:-1]**2 + EPS
        curv = d2 / da2
        concave = bool(np.mean(curv < CONCAVITY_THRESHOLD) > MIN_CURVATURE_RATIO)
        is_reliable = True
        return {
            'alpha_range': alpha_range,
            'D0_spectrum': D0_spectrum,
            'asym': asym,
            'concave': concave,
            'is_reliable': is_reliable
        }
    except Exception:
        return {'alpha_range': 0.0, 'D0_spectrum': 0.0,
                'asym': 0.0, 'concave': False, 'is_reliable': False}

def _create_mf_metrics(tau_data: List[Dict], Dq_data: List[Dict], D1_direct: float,
                       spectrum: Dict[str, Any], cfg: Config) -> Optional[MFMetrics]:
    if len(tau_data) < 4:
        return None
    q_arr = np.array([t['q'] for t in tau_data])
    Dq_arr = np.array([d['Dq'] for d in Dq_data])
    tau_mean_r2 = float(np.mean([t['r2'] for t in tau_data]))
    d0_idx = np.argmin(np.abs(q_arr))
    D0 = float(Dq_arr[d0_idx])
    d2_idx = np.argmin(np.abs(q_arr - 2))
    D2 = float(Dq_arr[d2_idx]) if any(abs(q_arr - 2) < Q_TOLERANCE) else 0.0
    return MFMetrics(
        mf_dq_range=round(float(np.max(Dq_arr) - np.min(Dq_arr)), 6),
        mf_dq_variance=round(float(np.var(Dq_arr)), 6),
        mf_d0=round(D0, 6),
        mf_d1_direct=round(D1_direct, 6),
        mf_d2=round(D2, 6),
        mf_alpha_range=round(spectrum['alpha_range'], 6),
        mf_d0_spectrum=round(spectrum['D0_spectrum'], 6),
        mf_asymmetry=round(spectrum['asym'], 6),
        mf_concave=spectrum['concave'],
        mf_legendre_consistency=0.0,
        mf_tau_mean_r2=round(tau_mean_r2, 6),
        mf_is_reliable=spectrum['is_reliable'],
        has_mf_spectrum=spectrum['is_reliable']
    )

def multifractal_analysis(pts_array: np.ndarray, bbox: QgsRectangle,
                         regime_scales: List[float], cfg: Config) -> Optional[MFMetrics]:
    try:
        if not _validate_mf_inputs(pts_array, regime_scales, cfg):
            return None
        scales = _select_mf_scales(regime_scales)
        Zq_matrix, entropy_vals = _compute_partition_function(pts_array, bbox, scales, cfg)
        if len(Zq_matrix) < cfg.min_scales_for_mf:
            return None
        tau_data, Dq_data, D1_direct = _fit_tau_q(Zq_matrix, scales, entropy_vals, cfg)
        if len(tau_data) < 4:
            return None
        spectrum = _compute_legendre_spectrum(tau_data, cfg)
        return _create_mf_metrics(tau_data, Dq_data, D1_direct, spectrum, cfg)
    except Exception:
        return None

# ============================================================================
# FEATURE ANALYSIS
# ============================================================================
def _compute_fractal_metrics(feature: QgsFeature, geom: QgsGeometry, bbox: QgsRectangle,
                             pts_array: np.ndarray, centroid_xy: Tuple[float, float],
                             regime_scales: List[float], regime_counts: List[float],
                             cfg: Config) -> Optional[FractalResult]:
    try:
        regime_scales_arr = np.asarray(regime_scales)
        regime_counts_arr = np.asarray(regime_counts)
        x = np.log(1.0 / regime_scales_arr)
        y = np.log(regime_counts_arr)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = float(r_value**2)
        if r2 < cfg.min_r2:
            return None
        
        scale_decades = float(abs(np.max(x) - np.min(x)) / np.log(10))
        if scale_decades < cfg.min_decades:
            return None
        
        n_scales = len(regime_scales)
        boot_results = bootstrap_fractal_dimension(x, y, cfg)
        surr_p, surr_sig, surr_mean, surr_std = test_against_csr(
            float(slope), pts_array, bbox, regime_scales, cfg
        )
        median_s = float(np.median(regime_scales))
        isotropy_cv, is_isotropic = check_isotropy(pts_array, median_s, cfg)
        mean_lac = compute_lacunarity(pts_array, bbox, regime_scales, cfg)
        
        scaling_quality = max(
            0.0,
            r2 * min(1.0, scale_decades / 2.0) * (1.0 - min(isotropy_cv, 1.0))
        )
        
        base = BaseMetrics(
            fid=feature.id(),
            x=round(centroid_xy[0], 6),
            y=round(centroid_xy[1], 6),
            length_m=round(geom.length(), 2),
            d_fractal=round(float(slope), 4),
            d_r2=round(r2, 4),
            d_se=round(float(std_err), 4),
            d_ci95_low=round(boot_results['ci_low'], 4),
            d_ci95_high=round(boot_results['ci_high'], 4),
            d_ci95_range=round(boot_results['ci_range'], 4),
            d_bootstrap_std=round(boot_results['boot_std'], 4),
            d_p_value=round(float(p_value), 6),
            scale_decades=round(scale_decades, 3),
            n_scales=n_scales,
            n_points=len(pts_array),
            lacunarity=round(mean_lac, 4),
            surrogate_p_value=round(surr_p, 6),
            significant_vs_csr=surr_sig,
            surrogate_d_mean=round(surr_mean, 4),
            surrogate_d_std=round(surr_std, 4),
            isotropy_cv=round(isotropy_cv, 4),
            is_isotropic=is_isotropic,
            scaling_quality=round(scaling_quality, 4)
        )
        return FractalResult(base=base, geom=QgsGeometry(geom))
    except Exception:
        return None

def analyze_single_feature(feature: QgsFeature, cfg: Config) -> Optional[FractalResult]:
    try:
        geom = feature.geometry()
        if not geom or geom.isEmpty() or geom.length() < GEOM_LENGTH_THRESHOLD:
            return None
        bbox = geom.boundingBox()
        pts_array, centroid_xy = geometry_to_array(geom, cfg)
        if pts_array is None:
            return None
        regime_scales, regime_counts = detect_scaling_regime(pts_array, bbox, cfg)
        n_regime = len(regime_scales)
        if n_regime < cfg.min_scales:
            return None
        result = _compute_fractal_metrics(feature, geom, bbox, pts_array, centroid_xy,
                                         regime_scales, regime_counts, cfg)
        if result is None:
            return None
        mf_metrics = multifractal_analysis(pts_array, bbox, regime_scales, cfg)
        if mf_metrics:
            result.mf = mf_metrics
        return result
    except Exception:
        return None

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================
def process_features_sequential(layer: QgsVectorLayer, cfg: Config) -> List[FractalResult]:
    features = list(layer.getFeatures())
    total = len(features)
    all_results = []
    logger.info("\nProcessing features (sequential)...\n")
    for i in range(0, total, cfg.batch_size):
        batch = features[i:i+cfg.batch_size]
        for feat in batch:
            result = analyze_single_feature(feat, cfg)
            append_if_valid(result, all_results)
        QCoreApplication.processEvents()
        progress = min(i + cfg.batch_size, total)
        logger.info(f"  [{progress}/{total}] Fractal: {len(all_results)}")
    logger.info(f"\nAnalysis Complete!")
    return all_results

def process_features_parallel_safe(layer: QgsVectorLayer, cfg: Config) -> List[FractalResult]:
    if not cfg.use_parallel:
        return process_features_sequential(layer, cfg)
    features = list(layer.getFeatures())
    total = len(features)
    all_results = []
    n_workers = min(cfg.max_workers, max(1, multiprocessing.cpu_count() - 2))
    logger.info(f"\nProcessing {total} features with {n_workers} workers (parallel)...\n")
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(analyze_single_feature, feat, cfg): i
                for i, feat in enumerate(features)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    append_if_valid(result, all_results)
                    completed += 1
                    if completed % max(1, total // 20) == 0:
                        QCoreApplication.processEvents()
                        logger.info(f"  [{completed}/{total}] Valid: {len(all_results)}")
                except Exception:
                    completed += 1
    except Exception:
        logger.error("Parallel processing error, falling back to sequential...")
        return process_features_sequential(layer, cfg)
    logger.info(f"\nParallel analysis complete: {len(all_results)} valid features")
    return all_results

def process_features(layer: QgsVectorLayer, cfg: Config) -> List[FractalResult]:
    if cfg.use_parallel and layer.featureCount() > PARALLEL_THRESHOLD:
        return process_features_parallel_safe(layer, cfg)
    return process_features_sequential(layer, cfg)

# ============================================================================
# OUTPUT FUNCTIONS - QGIS Layers
# ============================================================================
FRACTAL_LAYER_FIELDS = [
    ("D_fractal", QVariant.Double), ("D_R2", QVariant.Double),
    ("D_SE", QVariant.Double), ("D_CI95_low", QVariant.Double),
    ("D_CI95_high", QVariant.Double), ("D_CI95_range", QVariant.Double),
    ("D_bootstrap_std", QVariant.Double), ("D_p_value", QVariant.Double),
    ("Scale_decades", QVariant.Double), ("N_scales", QVariant.Int),
    ("N_points", QVariant.Int), ("Lacunarity", QVariant.Double),
    ("Surrogate_p_value", QVariant.Double), ("Significant_vs_CSR", QVariant.Int),
    ("Surrogate_D_mean", QVariant.Double), ("Surrogate_D_std", QVariant.Double),
    ("Isotropy_CV", QVariant.Double), ("Is_isotropic", QVariant.Int),
    ("Scaling_quality", QVariant.Double), ("Has_MF_spectrum", QVariant.Int)
]

MF_LAYER_FIELDS = [
    ("MF_Dq_range", QVariant.Double), ("MF_Dq_variance", QVariant.Double),
    ("MF_D0", QVariant.Double), ("MF_D1_direct", QVariant.Double),
    ("MF_D2", QVariant.Double), ("MF_alpha_range", QVariant.Double),
    ("MF_D0_spectrum", QVariant.Double), ("MF_asymmetry", QVariant.Double),
    ("MF_concave", QVariant.Int), ("MF_legendre_consistency", QVariant.Double),
    ("MF_tau_mean_r2", QVariant.Double), ("D_fractal", QVariant.Double),
    ("D_R2", QVariant.Double), ("Scaling_quality", QVariant.Double)
]

def create_memory_layer(name: str, fields_config: List[Tuple[str, QVariant.Type]],
                        results_list: List[FractalResult], crs: str,
                        extract_attrs_fn: callable) -> QgsVectorLayer:
    flds = QgsFields()
    for fname, ftype in fields_config:
        flds.append(QgsField(fname, ftype))
    layer = QgsVectorLayer(f"LineString?crs={crs}", name, "memory")
    prov = layer.dataProvider()
    prov.addAttributes(flds)
    layer.updateFields()
    feats = []
    for r in results_list:
        f = QgsFeature()
        if r.geom:
            f.setGeometry(r.geom)
        attrs = extract_attrs_fn(r)
        f.setAttributes(attrs)
        feats.append(f)
    prov.addFeatures(feats)
    layer.updateExtents()
    QgsProject.instance().addMapLayer(layer)
    logger.info(f"Layer '{name}' ({len(feats)} features)")
    return layer

def create_output_layers(all_results: List[FractalResult], crs: str) -> List[FractalResult]:
    mf_reliable = [r for r in all_results if r.mf.mf_is_reliable]
    n_results = len(all_results)
    n_mf = len(mf_reliable)
    n_sig = sum(1 for r in all_results if r.base.significant_vs_csr)
    n_iso = sum(1 for r in all_results if r.base.is_isotropic)
    logger.info(f"\nCreating output layers...")
    logger.info(f"  Geometric Fractal Scaling: {n_results}")
    logger.info(f"  Reliable Multifractal Spectra: {n_mf}")
    logger.info(f"  Significant vs CSR: {n_sig}")
    logger.info(f"  Isotropic: {n_iso}")
    
    create_memory_layer("Fractal_Scaling", FRACTAL_LAYER_FIELDS, all_results, crs,
                       lambda r: [r.base.d_fractal, r.base.d_r2, r.base.d_se,
                                 r.base.d_ci95_low, r.base.d_ci95_high, r.base.d_ci95_range,
                                 r.base.d_bootstrap_std, r.base.d_p_value,
                                 r.base.scale_decades, r.base.n_scales, r.base.n_points,
                                 r.base.lacunarity,
                                 r.base.surrogate_p_value, int(r.base.significant_vs_csr),
                                 r.base.surrogate_d_mean, r.base.surrogate_d_std,
                                 r.base.isotropy_cv, int(r.base.is_isotropic),
                                 r.base.scaling_quality, int(r.mf.has_mf_spectrum)])
    if mf_reliable:
        create_memory_layer("Multifractal_Spectra", MF_LAYER_FIELDS, mf_reliable, crs,
                           lambda r: [r.mf.mf_dq_range, r.mf.mf_dq_variance,
                                     r.mf.mf_d0, r.mf.mf_d1_direct, r.mf.mf_d2,
                                     r.mf.mf_alpha_range, r.mf.mf_d0_spectrum,
                                     r.mf.mf_asymmetry, int(r.mf.mf_concave),
                                     r.mf.mf_legendre_consistency, r.mf.mf_tau_mean_r2,
                                     r.base.d_fractal, r.base.d_r2, r.base.scaling_quality])
    return mf_reliable

# ============================================================================
# VALIDATION
# ============================================================================
def generate_koch_curve(iterations: int = 6, n_points: int = 1000) -> np.ndarray:
    t = np.linspace(0, 2*np.pi, 4)[:-1]
    points = np.column_stack([np.cos(t), np.sin(t)])
    
    def koch_iteration(pts):
        new_pts = []
        for i in range(len(pts)-1):
            p1 = pts[i]
            p2 = pts[i+1]
            new_pts.append(p1)
            new_pts.append(p1 + (p2 - p1)/3)
            mid = p1 + (p2 - p1)/2
            perp = np.array([-(p2 - p1)[1], (p2 - p1)[0]]) / (2*np.sqrt(3))
            new_pts.append(mid + perp)
            new_pts.append(p1 + 2*(p2 - p1)/3)
        new_pts.append(pts[-1])
        return np.array(new_pts)
    
    for _ in range(min(iterations, 6)):
        points = koch_iteration(points)
    
    cumdist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    cumdist = np.insert(cumdist, 0, 0)
    if cumdist[-1] > 0:
        dist_interp = np.linspace(0, cumdist[-1], n_points)
        x = np.interp(dist_interp, cumdist, points[:, 0])
        y = np.interp(dist_interp, cumdist, points[:, 1])
        return np.column_stack([x, y])
    return points[:n_points]

def validate_algorithm(cfg: Config) -> pd.DataFrame:
    logger.info("\nValidating with known fractals...")
    pts = generate_koch_curve()
    min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
    max_x, max_y = pts[:, 0].max(), pts[:, 1].max()
    margin = 0.01 * max(max_x - min_x, max_y - min_y)
    bbox = QgsRectangle(min_x - margin, min_y - margin, max_x + margin, max_y + margin)
    
    regime_scales, regime_counts = detect_scaling_regime(pts, bbox, cfg)
    
    if len(regime_scales) >= cfg.min_scales:
        x = np.log(1.0 / np.asarray(regime_scales))
        y = np.log(np.asarray(regime_counts))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        estimated_D = round(slope, 4)
        r2 = round(r_val**2, 4)
        error = round(100 * abs(slope - KOCH_DIMENSION) / KOCH_DIMENSION, 2)
    else:
        estimated_D = "Failed"
        r2 = "N/A"
        error = "N/A"
    
    return pd.DataFrame([{
        "Fractal": "Koch Curve",
        "Expected_D": KOCH_DIMENSION,
        "Estimated_D": estimated_D,
        "R2": r2,
        "Error_%": error
    }])

# ============================================================================
# EXPORT FUNCTIONS - FULL EXCEL REPORT
# ============================================================================
def export_summary(writer, df_all: pd.DataFrame, mf_reliable: List[FractalResult],
                   layer_name: str, crs: str, total_features: int, cfg: Config):
    n_results = len(df_all)
    n_mf = len(mf_reliable)
    n_sig = df_all['significant_vs_csr'].sum() if 'significant_vs_csr' in df_all.columns else 0
    n_iso = df_all['is_isotropic'].sum() if 'is_isotropic' in df_all.columns else 0
    
    summary_rows = [
        {"Section": "GENERAL INFORMATION", "Metric": "", "Value": ""},
        {"Section": "", "Metric": "Analysis Timestamp", "Value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
        {"Section": "", "Metric": "Software Version", "Value": __version__},
        {"Section": "", "Metric": "Layer Name", "Value": layer_name},
        {"Section": "", "Metric": "CRS", "Value": crs},
        {"Section": "", "Metric": "Total Features in Layer", "Value": total_features},
        {"Section": "", "Metric": "Valid Fractal Scaling", "Value": n_results},
        {"Section": "", "Metric": "Reliable Multifractal Spectra", "Value": n_mf},
        {"Section": "", "Metric": "Significant vs CSR", "Value": f"{n_sig} ({safe_percentage(n_sig, n_results):.1f}%)"},
        {"Section": "", "Metric": "Isotropic Geometries", "Value": f"{n_iso} ({safe_percentage(n_iso, n_results):.1f}%)"},
        {"Section": "", "Metric": "", "Value": ""},
        {"Section": "METHODS", "Metric": "", "Value": ""},
        {"Section": "", "Metric": "Robust Box Counting", "Value": "Yes" if cfg.robust_box_counting else "No"},
        {"Section": "", "Metric": "Grid Offsets", "Value": cfg.grid_offsets if cfg.robust_box_counting else "N/A"},
        {"Section": "", "Metric": "Adaptive Sampling", "Value": "Yes" if cfg.adaptive_sampling else "No"},
        {"Section": "", "Metric": "Residual Analysis", "Value": "Yes" if cfg.residual_check else "No"},
        {"Section": "", "Metric": "Parallel Processing", "Value": "Yes" if cfg.use_parallel else "No"},
        {"Section": "", "Metric": "q-Range", "Value": f"[{cfg.q_values[0]:.1f}, {cfg.q_values[-1]:.1f}]"},
        {"Section": "", "Metric": "Bootstrap Iterations", "Value": cfg.bootstrap_iter},
        {"Section": "", "Metric": "Surrogate Count", "Value": cfg.surrogate_count},
        {"Section": "", "Metric": "Random Seed", "Value": cfg.random_seed},
    ]
    
    summary_rows.append({"Section": "", "Metric": "", "Value": ""})
    summary_rows.append({"Section": "GEOMETRIC FRACTAL DIMENSION STATISTICS", "Metric": "", "Value": ""})
    
    for col in ["d_fractal", "d_r2", "d_se", "d_ci95_range", "scale_decades", "lacunarity", "scaling_quality"]:
        if col in df_all.columns:
            mean_val = df_all[col].mean()
            std_val = df_all[col].std()
            min_val = df_all[col].min()
            max_val = df_all[col].max()
            summary_rows.append({
                "Section": "", 
                "Metric": f"  {col}", 
                "Value": f"Mean={mean_val:.4f} +/- {std_val:.4f}  Range=[{min_val:.4f}, {max_val:.4f}]"
            })
    
    if mf_reliable:
        df_mf = results_to_dataframe(mf_reliable)
        summary_rows.append({"Section": "", "Metric": "", "Value": ""})
        summary_rows.append({"Section": "MULTIFRACTAL SPECTRA STATISTICS (Reliable Only)", "Metric": "", "Value": ""})
        
        for col in ["mf_dq_range", "mf_alpha_range", "mf_d0", "mf_d1_direct", "mf_d2"]:
            if col in df_mf.columns:
                mean_val = df_mf[col].mean()
                std_val = df_mf[col].std()
                summary_rows.append({
                    "Section": "", 
                    "Metric": f"  {col}", 
                    "Value": f"Mean={mean_val:.4f} +/- {std_val:.4f}"
                })
        
        n_concave = df_mf['mf_concave'].sum() if 'mf_concave' in df_mf.columns else 0
        summary_rows.append({
            "Section": "", 
            "Metric": "  Concave Spectra", 
            "Value": f"{n_concave} ({safe_percentage(n_concave, n_mf):.1f}%)"
        })
    
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="1_Executive_Summary", index=False)

def export_statistics(writer, df_all: pd.DataFrame, mf_reliable: List[FractalResult]):
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        stats_df = df_all[numeric_cols].describe(
            percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        ).round(4)
        stats_df.to_excel(writer, sheet_name="2_Full_Statistics")
        
        if len(numeric_cols) > 2:
            corr_matrix = df_all[numeric_cols].corr().round(3)
            corr_matrix.to_excel(writer, sheet_name="3_Correlation_Matrix")
        
        if 'd_fractal' in df_all.columns:
            df_dist = df_all.copy()
            df_dist['D_category'] = pd.cut(
                df_dist['d_fractal'],
                bins=[0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0],
                labels=["<1.0", "1.0-1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0", ">2.0"]
            )
            d_dist = df_dist['D_category'].value_counts().sort_index()
            d_dist.to_frame("Count").to_excel(writer, sheet_name="4_D_Distribution")
        
        extra_stats = []
        if 'significant_vs_csr' in df_all.columns:
            sig_count = df_all['significant_vs_csr'].sum()
            sig_pct = safe_percentage(sig_count, len(df_all))
            extra_stats.append({"Metric": "Significant vs CSR", "Value": f"{sig_count} ({sig_pct:.1f}%)"})
        
        if 'is_isotropic' in df_all.columns:
            iso_count = df_all['is_isotropic'].sum()
            iso_pct = safe_percentage(iso_count, len(df_all))
            extra_stats.append({"Metric": "Isotropic", "Value": f"{iso_count} ({iso_pct:.1f}%)"})
        
        if extra_stats:
            pd.DataFrame(extra_stats).to_excel(writer, sheet_name="4b_Extra_Stats", index=False)
    
    df_all.to_excel(writer, sheet_name="5_All_Features", index=False)
    
    if mf_reliable:
        df_mf = results_to_dataframe(mf_reliable)
        mf_cols = [c for c in df_mf.columns if c.startswith('mf_') or c in ['fid', 'd_fractal', 'd_r2', 'scaling_quality']]
        existing_cols = [c for c in mf_cols if c in df_mf.columns]
        df_mf[existing_cols].to_excel(writer, sheet_name="6_Multifractal_Features", index=False)
    else:
        pd.DataFrame({"Info": ["No reliable multifractal spectra found"]}).to_excel(
            writer, sheet_name="6_Multifractal_Features", index=False
        )

def export_validation(writer, cfg: Config):
    if not cfg.run_validation:
        pd.DataFrame({
            "Info": ["Validation skipped"]
        }).to_excel(writer, sheet_name="7_Validation", index=False)
        return
    
    try:
        val_df = validate_algorithm(cfg)
        val_df.to_excel(writer, sheet_name="7_Validation", index=False)
        logger.info("  Validation results exported")
    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        pd.DataFrame({"Info": [f"Validation error: {str(e)}"]}).to_excel(
            writer, sheet_name="7_Validation", index=False
        )

def export_excel_report(all_results: List[FractalResult], mf_reliable: List[FractalResult],
                        layer_name: str, crs: str, total_features: int, 
                        excel_path: str, cfg: Config) -> bool:
    logger.info("\nGenerating Excel Report...")
    df_all = results_to_dataframe(all_results)
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            export_summary(writer, df_all, mf_reliable, layer_name, crs, total_features, cfg)
            export_statistics(writer, df_all, mf_reliable)
            export_validation(writer, cfg)
        
        logger.info(f"Excel Report Saved: {excel_path}")
        return True
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        try:
            # Try to save minimal report
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                pd.DataFrame({"Error": [str(e)]}).to_excel(writer, sheet_name="Error", index=False)
                df_all.to_excel(writer, sheet_name="All_Features", index=False)
            logger.info(f"Minimal report saved: {excel_path}")
            return True
        except Exception as e2:
            logger.error(f"Complete export failure: {e2}")
            return False

def print_final_summary(all_results: List[FractalResult], mf_reliable: List[FractalResult],
                        excel_path: str, export_success: bool):
    df_all = results_to_dataframe(all_results)
    n_results = len(all_results)
    n_mf = len(mf_reliable)
    n_sig = df_all['significant_vs_csr'].sum() if 'significant_vs_csr' in df_all.columns else 0
    n_iso = df_all['is_isotropic'].sum() if 'is_isotropic' in df_all.columns else 0
    
    d_mean = df_all['d_fractal'].mean()
    d_std = df_all['d_fractal'].std()
    if pd.isna(d_std):
        d_std = 0.0
    
    logger.info("\n" + "=" * 60)
    logger.info("  ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\n  GEOMETRIC FRACTAL SCALING:")
    logger.info(f"    • Analyzed Features: {n_results}")
    logger.info(f"    • Mean D: {d_mean:.4f} +/- {d_std:.4f}")
    logger.info(f"    • D Range: [{df_all['d_fractal'].min():.4f}, {df_all['d_fractal'].max():.4f}]")
    logger.info(f"    • Mean R²: {df_all['d_r2'].mean():.4f}")
    logger.info(f"    • Mean Scaling Quality: {df_all['scaling_quality'].mean():.4f}")
    
    logger.info(f"\n  STATISTICAL VALIDATION:")
    logger.info(f"    • Significant vs CSR: {n_sig} ({safe_percentage(n_sig, n_results):.1f}%)")
    logger.info(f"    • Isotropic: {n_iso} ({safe_percentage(n_iso, n_results):.1f}%)")
    logger.info(f"    • Mean Lacunarity: {df_all['lacunarity'].mean():.4f}")
    
    if mf_reliable:
        df_mf = results_to_dataframe(mf_reliable)
        n_concave = df_mf['mf_concave'].sum() if 'mf_concave' in df_mf.columns else 0
        logger.info(f"\n  MULTIFRACTAL SPECTRA:")
        logger.info(f"    • Reliable Spectra: {n_mf} ({safe_percentage(n_mf, n_results):.1f}%)")
        logger.info(f"    • Mean Dq Range: {df_mf['mf_dq_range'].mean():.4f}")
        logger.info(f"    • Mean α Range: {df_mf['mf_alpha_range'].mean():.4f}")
        logger.info(f"    • Mean D1 (Information): {df_mf['mf_d1_direct'].mean():.4f}")
        logger.info(f"    • Concave Spectra: {n_concave} ({safe_percentage(n_concave, n_mf):.1f}%)")
    
    logger.info(f"\n  OUTPUTS:")
    logger.info(f"    • QGIS Layers: Fractal_Scaling, Multifractal_Spectra")
    logger.info(f"    • Excel Report: {excel_path if export_success else 'Export failed'}")
    logger.info("=" * 60)

# ============================================================================
# QGIS PROCESSING ALGORITHM
# ============================================================================
class GeoFractalLinesAlgorithm(QgsProcessingAlgorithm):
    def tr(self, string):
        return QCoreApplication.translate('GeoFractalLines', string)

    def createInstance(self):
        return GeoFractalLinesAlgorithm()

    def name(self):
        return 'geofractallines'

    def displayName(self):
        return self.tr('GeoFractalLines - Fractal Analysis')

    def group(self):
        return self.tr('GeoFractalLines')

    def groupId(self):
        return 'geofractallines'

    def shortHelpString(self):
        return self.tr("""GeoFractalLines v1.0.0

Box-Counting Fractal Analysis for Line Geometries.

OUTPUTS:
- Fractal_Scaling: Layer with fractal dimension results
- Multifractal_Spectra: Layer with multifractal analysis (if reliable)
- Excel Report: Full statistical report

METHODS:
- Box-counting with robust grid offsets
- Bootstrap confidence intervals
- Surrogate testing against Complete Spatial Randomness
- Lacunarity analysis
- Multifractal spectrum (q-range: -5 to 5)""")

    def initAlgorithm(self, config=None):
        # Input layer
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                'INPUT',
                self.tr('Input line layer'),
                [QgsProcessing.TypeVectorLine]
            )
        )
        
        # Output Excel file - USER CHOOSES WHERE TO SAVE
        self.addParameter(
            QgsProcessingParameterFileDestination(
                'OUTPUT_EXCEL',
                self.tr('Output Excel Report'),
                'Excel files (*.xlsx)'
            )
        )
        
        # Parameters
        self.addParameter(
            QgsProcessingParameterBoolean(
                'ROBUST_BOX_COUNTING',
                self.tr('Robust box counting (grid offsets)'),
                defaultValue=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                'GRID_OFFSETS',
                self.tr('Grid offsets (if robust)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                minValue=2,
                maxValue=20
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                'MIN_R2',
                self.tr('Minimum R² for scaling'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.90,
                minValue=0.50,
                maxValue=0.99
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                'MIN_DECADES',
                self.tr('Minimum scale decades'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.5,
                maxValue=3.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                'BOOTSTRAP_ITER',
                self.tr('Bootstrap iterations'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=500,
                minValue=100,
                maxValue=5000
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                'USE_PARALLEL',
                self.tr('Use parallel processing'),
                defaultValue=False
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                'MAX_WORKERS',
                self.tr('Max parallel workers'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=4,
                minValue=1,
                maxValue=16
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                'RUN_VALIDATION',
                self.tr('Run validation with Koch curve'),
                defaultValue=True
            )
        )
        
        # Outputs
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                'OUTPUT_FRACTAL',
                self.tr('Fractal Scaling Layer')
            )
        )
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                'OUTPUT_MULTIFRACTAL',
                self.tr('Multifractal Spectra Layer')
            )
        )
        self.addOutput(
            QgsProcessingOutputString(
                'OUTPUT_EXCEL_PATH',
                self.tr('Excel Report Path')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Get input layer
        layer = self.parameterAsVectorLayer(parameters, 'INPUT', context)
        if not layer:
            raise Exception("No valid input layer selected")
        
        # Get output Excel path - USER SELECTED
        excel_path = self.parameterAsFileOutput(parameters, 'OUTPUT_EXCEL', context)
        if not excel_path:
            # If user didn't select, use default
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_path = os.path.join(os.path.expanduser("~"), "Downloads", f"Fractal_Analysis_Report_{timestamp}.xlsx")
        elif not excel_path.endswith('.xlsx'):
            excel_path = excel_path + '.xlsx'
        
        # Get parameters
        robust_box = self.parameterAsBool(parameters, 'ROBUST_BOX_COUNTING', context)
        grid_offsets = self.parameterAsInt(parameters, 'GRID_OFFSETS', context)
        min_r2 = self.parameterAsDouble(parameters, 'MIN_R2', context)
        min_decades = self.parameterAsDouble(parameters, 'MIN_DECADES', context)
        bootstrap_iter = self.parameterAsInt(parameters, 'BOOTSTRAP_ITER', context)
        use_parallel = self.parameterAsBool(parameters, 'USE_PARALLEL', context)
        max_workers = self.parameterAsInt(parameters, 'MAX_WORKERS', context)
        run_validation = self.parameterAsBool(parameters, 'RUN_VALIDATION', context)
        
        # Setup config
        cfg = Config()
        cfg.base_path = os.path.dirname(excel_path)
        cfg.robust_box_counting = robust_box
        cfg.grid_offsets = grid_offsets
        cfg.min_r2 = min_r2
        cfg.min_decades = min_decades
        cfg.bootstrap_iter = bootstrap_iter
        cfg.use_parallel = use_parallel
        cfg.max_workers = max_workers
        cfg.run_validation = run_validation
        
        feedback.setProgressText("Starting GeoFractalLines analysis...")
        
        # Run analysis
        all_results = process_features(layer, cfg)
        
        if not all_results:
            feedback.reportError("No valid fractal scaling found!")
            return {}
        
        # Create output layers
        crs = layer.crs().authid()
        mf_reliable = create_output_layers(all_results, crs)
        
        # Export report
        export_success = export_excel_report(
            all_results, mf_reliable, layer.name(),
            crs, layer.featureCount(), excel_path, cfg
        )
        
        # Summary
        n_results = len(all_results)
        n_mf = len(mf_reliable)
        feedback.pushInfo(f"\nAnalysis Complete!")
        feedback.pushInfo(f"  Geometric Fractal Scaling: {n_results} features")
        feedback.pushInfo(f"  Reliable Multifractal Spectra: {n_mf} features")
        feedback.pushInfo(f"  Excel Report: {excel_path if export_success else 'Failed'}")
        
        return {
            'OUTPUT_FRACTAL': 'Fractal_Scaling',
            'OUTPUT_MULTIFRACTAL': 'Multifractal_Spectra' if mf_reliable else '',
            'OUTPUT_EXCEL_PATH': excel_path if export_success else ''
        }

# ============================================================================
# REGISTER ALGORITHM
# ============================================================================
def register_algorithm():
    """Register the algorithm with QGIS Processing"""
    from qgis.core import QgsApplication
    QgsApplication.processingRegistry().addProvider(GeoFractalLinesAlgorithm())

# Auto-register when script is loaded in QGIS
try:
    register_algorithm()
    logger.info("GeoFractalLines v1.0.0 registered successfully!")
    logger.info("Find it in: Processing Toolbox -> GeoFractalLines -> GeoFractalLines - Fractal Analysis")
except Exception as e:
    logger.error(f"Failed to register algorithm: {e}")

# ============================================================================
# MAIN - For testing in console
# ============================================================================
def main():
    """Run directly from QGIS Python console for testing"""
    try:
        layer = iface.activeLayer()
        if not layer:
            logger.error("Please select a line layer!")
            return
        if layer.geometryType() != QgsWkbTypes.LineGeometry:
            logger.error("Selected layer must be a line layer!")
            return
        
        # Run analysis with default config
        cfg = Config()
        cfg.use_parallel = False
        all_results = process_features(layer, cfg)
        
        if all_results:
            crs = layer.crs().authid()
            mf_reliable = create_output_layers(all_results, crs)
            
            # Generate report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_path = os.path.join(cfg.base_path, f"Fractal_Analysis_Report_{timestamp}.xlsx")
            export_success = export_excel_report(
                all_results, mf_reliable, layer.name(),
                crs, layer.featureCount(), excel_path, cfg
            )
            print_final_summary(all_results, mf_reliable, excel_path, export_success)
        else:
            logger.info("No valid results found.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()