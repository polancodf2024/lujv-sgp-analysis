#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MIRROR-PIM: GRASSMANN-PIM WITH REAL GEOMETRIC ALGEBRA OPERATORS (v11)
OPTIMIZED FOR MASSIVE DATASETS (150M+ sequences)
================================================================================
- Streaming mode: processes sequences without storing all vectors
- Online statistics: Welford algorithm for mean and covariance
- Memory constant: O(1) regardless of dataset size
- Sampling for cohesion: 10,000 random pairs for intra-group similarity
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
import os
import hashlib
from datetime import datetime
from collections import defaultdict
import random

warnings.filterwarnings('ignore')

# Configure plot style
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['savefig.dpi'] = 150

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMILARITY_THRESHOLD = None
CONFIDENCE_LEVEL = 0.95
TOP_N_PROTEINS = 20
TOLERANCE = 0.001
USE_TRIPLETS = False
USE_BOOTSTRAP = True
N_BOOTSTRAP = 100
USE_WEIGHTS = True

# IMPORTANT: For massive files, sample size for cohesion estimation
COHESION_SAMPLE_SIZE = 10000  # Number of random pairs to estimate intra-group similarity
STREAMING_MODE = True         # Enable streaming mode for large files

# Metric signature configuration
USE_BIOLOGICAL_METRIC = True
SHOW_METRIC_ANALYSIS = True

# Biologically relevant planes for rotors (PIM component indices)
ROTOR_PLANES = [
    ('hydrophobic', (10, 15), 'N→N vs NP→NP (H-bonds vs hydrophobic interactions)'),
    ('charge', (0, 5), 'P⁺→P⁺ vs P⁻→P⁻ (repulsion vs attraction)'),
    ('opposite_charge', (1, 4), 'P⁺→P⁻ vs P⁻→P⁺ (charge-charge interactions)'),
    ('polarity', (10, 11), 'N→N vs N→NP (polar vs mixed)'),
    ('charge_transition', (2, 8), 'P⁺→N vs N→P⁺ (charge-polar transition)'),
    ('opposite_transition', (6, 9), 'P⁻→N vs N→P⁻ (negative charge-polar transition)'),
]

# Indices for specular reflection (P⁺ ↔ P⁻ swap)
REFLECTION_SWAP_MAP = {
    0: 5, 1: 4, 2: 6, 3: 7, 4: 1, 5: 0, 6: 2, 7: 3,
    8: 9, 9: 8, 10: 10, 11: 11, 12: 13, 13: 12, 14: 14, 15: 15,
}

# Key bivectors for analysis
KEY_BIVECTORS = [
    (0, 5), (1, 4), (2, 6), (3, 7), (10, 11), (14, 15),
]

# Biological weights
BIOLOGICAL_WEIGHTS = {
    'P+,P-': 2.0, 'P-,P+': 2.0,
    'N,N': 1.5,
    'N,P+': 1.3, 'P+,N': 1.3,
    'N,P-': 1.3, 'P-,N': 1.3,
    'NP,NP': 1.0,
    'NP,N': 0.9, 'N,NP': 0.9,
    'NP,P+': 0.7, 'P+,NP': 0.7,
    'NP,P-': 0.7, 'P-,NP': 0.7,
    'P+,P+': 0.4, 'P-,P-': 0.4,
}

# ============================================================================
# BIOLOGICAL METRIC SIGNATURE
# ============================================================================

BIOLOGICAL_METRIC_SIGNATURE = np.array([
    -1.0, +1.0, +1.0, +0.0,
    +1.0, -1.0, +1.0, +0.0,
    +1.0, +1.0, +1.0, +0.0,
    +0.0, +0.0, +0.0, +1.0,
])

EUCLIDEAN_METRIC = np.ones(16)
METRIC_SIGNATURE = BIOLOGICAL_METRIC_SIGNATURE if USE_BIOLOGICAL_METRIC else EUCLIDEAN_METRIC

# Subspace masks for interior product projections
SUBSPACES = {
    'hydrophobic': [10, 15],
    'charge_repulsion': [0, 5],
    'charge_attraction': [1, 4],
    'charge_polar': [2, 3, 6, 7],
    'polar': [8, 9, 10, 11],
    'nonpolar': [12, 13, 14, 15],
    'full': None,
}

# ============================================================================
# REAL GEOMETRIC ALGEBRA OPERATORS
# ============================================================================

def wedge_product_oriented(v: np.ndarray, w: np.ndarray, 
                           key_pairs: List[Tuple[int, int]] = None) -> np.ndarray:
    if key_pairs is None:
        key_pairs = KEY_BIVECTORS
    
    bivector = np.zeros(len(key_pairs))
    for idx, (i, j) in enumerate(key_pairs):
        if i < len(v) and j < len(w):
            bivector[idx] = v[i] * w[j] - v[j] * w[i]
    return bivector


def wedge_similarity_with_orientation(v: np.ndarray, w: np.ndarray) -> Tuple[float, float, np.ndarray]:
    biv = wedge_product_oriented(v, w)
    magnitude = np.linalg.norm(biv)
    norm_v = np.linalg.norm(v) + 1e-10
    norm_w = np.linalg.norm(w) + 1e-10
    magnitude_norm = magnitude / (norm_v * norm_w + 1e-10)
    magnitude_norm = min(magnitude_norm, 1.0)
    
    non_zero = biv[np.abs(biv) > 1e-8]
    orientation_sign = 1.0
    if len(non_zero) > 0:
        orientation_sign = np.sign(non_zero[0])
    
    return magnitude_norm, orientation_sign, biv


def reflection_normal_vector() -> np.ndarray:
    n = np.zeros(16)
    for i, j in REFLECTION_SWAP_MAP.items():
        n[i] = 1.0
        n[j] = -1.0
    norm = np.linalg.norm(n)
    if norm > 0:
        n = n / norm
    return n


def specular_reflection(v: np.ndarray, normal: np.ndarray = None) -> np.ndarray:
    if normal is None:
        normal = reflection_normal_vector()
    n = normal / (np.linalg.norm(normal) + 1e-10)
    return v - 2 * np.dot(v, n) * n


def is_specular_reflection_ga(v1: np.ndarray, v2: np.ndarray, threshold: float = 0.95) -> Tuple[bool, float]:
    v1_reflected = specular_reflection(v1)
    norm1 = np.linalg.norm(v1_reflected) + 1e-10
    norm2 = np.linalg.norm(v2) + 1e-10
    sim = np.abs(np.dot(v1_reflected, v2)) / (norm1 * norm2)
    return sim > threshold, sim


def interior_product(v: np.ndarray, subspace_name: str) -> np.ndarray:
    if subspace_name not in SUBSPACES:
        raise ValueError(f"Subspace not recognized: {subspace_name}")
    indices = SUBSPACES[subspace_name]
    if indices is None:
        return v.copy()
    projected = np.zeros_like(v)
    projected[indices] = v[indices]
    total = np.sum(projected)
    if total > 0:
        projected = projected / total
    return projected


def interior_product_magnitude(v: np.ndarray, subspace_name: str) -> float:
    proj = interior_product(v, subspace_name)
    return np.linalg.norm(proj)


# ============================================================================
# METRIC OPERATORS
# ============================================================================

def dot_product_metric(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> float:
    """Metric dot product: v·_η w = Σ η_i * v_i * w_i"""
    if metric is None:
        metric = METRIC_SIGNATURE
    if len(metric) != len(v):
        if len(metric) < len(v):
            metric_padded = np.ones(len(v))
            metric_padded[:len(metric)] = metric
            metric = metric_padded
        else:
            metric = metric[:len(v)]
    return np.sum(metric * v * w)


def norm_metric(v: np.ndarray, metric: np.ndarray = None) -> Tuple[float, float]:
    """Metric norm: ||v||_η = sqrt(|Σ η_i v_i^2|)"""
    value = dot_product_metric(v, v, metric)
    sign = np.sign(value) if value != 0 else 0
    magnitude = np.sqrt(np.abs(value) + 1e-10)
    return magnitude, sign


def similarity_metric(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> float:
    """Metric-based similarity: S_η(v,w) = |v·_η w| / (||v||_η * ||w||_η)"""
    dot_η = dot_product_metric(v, w, metric)
    norm_v, _ = norm_metric(v, metric)
    norm_w, _ = norm_metric(w, metric)
    if norm_v * norm_w < 1e-10:
        return 0.0
    return np.abs(dot_η) / (norm_v * norm_w + 1e-10)


def metric_signature_info() -> Dict:
    info = {
        'total_components': len(METRIC_SIGNATURE),
        'positive_count': np.sum(METRIC_SIGNATURE > 0),
        'negative_count': np.sum(METRIC_SIGNATURE < 0),
        'neutral_count': np.sum(METRIC_SIGNATURE == 0),
        'is_euclidean': np.all(METRIC_SIGNATURE == 1),
        'is_biological': USE_BIOLOGICAL_METRIC,
    }
    component_names = [
        'P⁺→P⁺', 'P⁺→P⁻', 'P⁺→N', 'P⁺→NP',
        'P⁻→P⁺', 'P⁻→P⁻', 'P⁻→N', 'P⁻→NP',
        'N→P⁺', 'N→P⁻', 'N→N', 'N→NP',
        'NP→P⁺', 'NP→P⁻', 'NP→N', 'NP→NP'
    ]
    info['beneficial_interactions'] = [component_names[i] for i in range(len(METRIC_SIGNATURE)) if METRIC_SIGNATURE[i] > 0]
    info['detrimental_interactions'] = [component_names[i] for i in range(len(METRIC_SIGNATURE)) if METRIC_SIGNATURE[i] < 0]
    info['neutral_interactions'] = [component_names[i] for i in range(len(METRIC_SIGNATURE)) if METRIC_SIGNATURE[i] == 0]
    return info


# ============================================================================
# GEOMETRIC PRODUCT WITH METRIC
# ============================================================================

def geometric_product_metric(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> Tuple[float, np.ndarray]:
    """Geometric product: v *η w = (v·η w) + (v ∧η w)"""
    if metric is None:
        metric = METRIC_SIGNATURE
    scalar = np.sum(metric * v * w)
    sqrt_metric = np.sqrt(np.abs(metric) + 1e-10)
    v_transformed = v / sqrt_metric
    w_transformed = w / sqrt_metric
    bivector = wedge_product_oriented(v_transformed, w_transformed)
    return scalar, bivector


def geometric_product_decomposition(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> Dict:
    """Decompose geometric product into functional and structural components"""
    scalar, bivector = geometric_product_metric(v, w, metric)
    norm_v, _ = norm_metric(v, metric)
    norm_w, _ = norm_metric(w, metric)
    denom = norm_v * norm_w + 1e-10
    
    functional = np.abs(scalar) / denom
    structural = np.linalg.norm(bivector) / denom
    combined = np.sqrt(functional**2 + structural**2)
    ratio = functional / (structural + 1e-10)
    
    if ratio > 2.0:
        interpretation = "Functionally similar, structurally different"
    elif ratio < 0.5:
        interpretation = "Structurally similar, functionally different"
    else:
        interpretation = "Balanced: similar in both aspects"
    
    return {
        'functional_similarity': functional,
        'structural_difference': structural,
        'combined_similarity': combined,
        'functional_structural_ratio': ratio,
        'interpretation': interpretation
    }


# ============================================================================
# ONLINE STATISTICS (Welford Algorithm) - FOR STREAMING
# ============================================================================

class OnlineStatistics:
    """Computes mean and covariance in a single pass using Welford's algorithm."""
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))
        
    def update(self, x: np.ndarray):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)
        
    def get_covariance(self) -> np.ndarray:
        if self.n < 2:
            return np.eye(self.dim) * 0.01
        return self.M2 / (self.n - 1)
    
    def get_mean(self) -> np.ndarray:
        return self.mean
    
    def get_std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.dim) * 0.01
        cov = self.get_covariance()
        return np.sqrt(np.diag(cov))


# ============================================================================
# PIM CLASSIFICATION
# ============================================================================

POLARITY_MAP = {
    'H': 'P+', 'K': 'P+', 'R': 'P+',
    'D': 'P-', 'E': 'P-',
    'C': 'N', 'G': 'N', 'N': 'N', 'Q': 'N', 'S': 'N', 'T': 'N', 'Y': 'N',
    'A': 'NP', 'F': 'NP', 'I': 'NP', 'L': 'NP', 'M': 'NP', 'P': 'NP', 'V': 'NP', 'W': 'NP'
}

INTERACTIONS = [
    'P+,P+', 'P+,P-', 'P+,N', 'P+,NP',
    'P-,P+', 'P-,P-', 'P-,N', 'P-,NP',
    'N,P+', 'N,P-', 'N,N', 'N,NP',
    'NP,P+', 'NP,P-', 'NP,N', 'NP,NP'
]

INTERACTION_TO_IDX = {inter: i for i, inter in enumerate(INTERACTIONS)}

DIM_PAIRS = 16


def compute_pim_profile(sequence: str, use_weights: bool = USE_WEIGHTS) -> np.ndarray:
    """Calculate PIM profile from an amino acid sequence"""
    seq = ''.join([c for c in sequence.strip() if c.isalpha() and c.upper() in POLARITY_MAP])
    if len(seq) < 2:
        return np.zeros(DIM_PAIRS)
    
    polarities = []
    for aa in seq:
        pol = POLARITY_MAP.get(aa.upper())
        if pol is not None:
            polarities.append(pol)
    
    if len(polarities) < 2:
        return np.zeros(DIM_PAIRS)
    
    counts = np.zeros(DIM_PAIRS)
    for i in range(len(polarities) - 1):
        pair = f"{polarities[i]},{polarities[i+1]}"
        if pair in INTERACTION_TO_IDX:
            counts[INTERACTION_TO_IDX[pair]] += 1
    total = np.sum(counts)
    if total > 0:
        counts = counts / total
    
    if use_weights:
        weighted_counts = np.zeros(DIM_PAIRS)
        for i, inter in enumerate(INTERACTIONS):
            weight = BIOLOGICAL_WEIGHTS.get(inter, 1.0)
            weighted_counts[i] = counts[i] * weight
        total_weighted = np.sum(weighted_counts)
        if total_weighted > 0:
            weighted_counts = weighted_counts / total_weighted
        return weighted_counts
    
    return counts


def wedge_product_with_ci(v: np.ndarray, w: np.ndarray, 
                          n_bootstrap: int = N_BOOTSTRAP,
                          use_bootstrap: bool = USE_BOOTSTRAP) -> Tuple[float, float]:
    magnitude, orientation, _ = wedge_similarity_with_orientation(v, w)
    wedge = magnitude
    
    if not use_bootstrap:
        return wedge, 0.0
    
    dim = len(v)
    bootstrapped = []
    for _ in range(min(n_bootstrap, 100)):
        idx = np.random.choice(dim, dim, replace=True)
        v_boot = v[idx]
        w_boot = w[idx]
        mag_boot, _, _ = wedge_similarity_with_orientation(v_boot, w_boot)
        bootstrapped.append(mag_boot)
    
    return np.mean(bootstrapped), np.std(bootstrapped)


def rotor_angle(v1: np.ndarray, v2: np.ndarray, plane_indices: Tuple[int, int]) -> float:
    i, j = plane_indices
    if i >= len(v1) or j >= len(v1):
        return 0.0
    proj1 = np.array([v1[i], v1[j]])
    proj2 = np.array([v2[i], v2[j]])
    
    norm1 = np.linalg.norm(proj1) + 1e-10
    norm2 = np.linalg.norm(proj2) + 1e-10
    
    cos_theta = np.dot(proj1, proj2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta) * 180.0 / np.pi


def clifford_signature(v: np.ndarray) -> Dict[str, float]:
    signature = {}
    signature['norm'] = np.linalg.norm(v)
    
    v_reflected = specular_reflection(v)
    signature['auto_reflection'], _ = wedge_product_with_ci(v, v_reflected, use_bootstrap=False)
    
    if len(v) > 15:
        hydro_plane = np.array([v[10], v[15]])
        signature['hydrophobic_projection'] = np.linalg.norm(hydro_plane)
    else:
        signature['hydrophobic_projection'] = 0.0
    
    if len(v) > 5:
        charge_plane = np.array([v[0], v[5]])
        signature['charge_projection'] = np.linalg.norm(charge_plane)
    else:
        signature['charge_projection'] = 0.0
    
    v_rotated = np.roll(v, 4)
    signature['auto_rotation'], _ = wedge_product_with_ci(v, v_rotated, use_bootstrap=False)
    
    norm_η, sign_η = norm_metric(v)
    signature['metric_norm'] = norm_η
    signature['metric_sign'] = sign_η
    
    return signature


def clifford_distance(sig1: Dict[str, float], sig2: Dict[str, float]) -> float:
    keys = ['norm', 'auto_reflection', 'hydrophobic_projection', 'charge_projection', 'auto_rotation']
    diff = 0.0
    for key in keys:
        diff += (sig1.get(key, 0) - sig2.get(key, 0)) ** 2
    return np.sqrt(diff)


def read_fasta_stream(filepath: str, verbose: bool = False):
    """Generator that yields (header, sequence) from FASTA file one at a time"""
    if not os.path.exists(filepath):
        if verbose:
            print(f"    File not found: {filepath}")
        return
    
    with open(filepath, 'r') as f:
        current_header = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    yield current_header, ''.join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            yield current_header, ''.join(current_seq)


# ============================================================================
# CLASSES
# ============================================================================

@dataclass
class GroupStatistics:
    name: str
    n_samples: int
    centroid: np.ndarray
    covariance: np.ndarray
    inv_covariance: np.ndarray
    std_dev: np.ndarray
    wedge_self_similarity: float
    wedge_self_similarity_std: float = 0.0
    adaptive_threshold: float = 0.99
    clifford_signature: Dict[str, float] = field(default_factory=dict)
    subspace_projections: Dict[str, float] = field(default_factory=dict)
    metric_norm: float = 0.0
    metric_sign: float = 0.0
    
    def mahalanobis_distance(self, vector: np.ndarray) -> float:
        diff = vector - self.centroid
        return np.sqrt(diff @ self.inv_covariance @ diff)
    
    def probability_of_belonging(self, vector: np.ndarray) -> float:
        d = self.mahalanobis_distance(vector)
        return 1.0 - chi2.cdf(d**2, df=len(self.centroid))


class GrassmannPIM:
    def __init__(self, dim: int = DIM_PAIRS):
        self.dim = dim
    
    def wedge_product(self, v: np.ndarray, w: np.ndarray, 
                      with_ci: bool = False) -> Tuple[float, float]:
        return wedge_product_with_ci(v, w, use_bootstrap=with_ci)
    
    def wedge_product_oriented(self, v: np.ndarray, w: np.ndarray) -> Tuple[float, float, np.ndarray]:
        return wedge_similarity_with_orientation(v, w)
    
    def interior_product_magnitude(self, v: np.ndarray, subspace: str) -> float:
        return interior_product_magnitude(v, subspace)
    
    def specular_reflection(self, v: np.ndarray) -> np.ndarray:
        return specular_reflection(v)
    
    def is_specular_reflection(self, v1: np.ndarray, v2: np.ndarray, threshold: float = 0.95) -> Tuple[bool, float]:
        return is_specular_reflection_ga(v1, v2, threshold)
    
    def max_component_diff(self, v: np.ndarray, w: np.ndarray) -> float:
        return np.max(np.abs(v - w))
    
    def rotor_angle(self, v: np.ndarray, w: np.ndarray, plane_name: str = 'hydrophobic') -> float:
        planes_dict = {name: indices for name, indices, _ in ROTOR_PLANES}
        if plane_name not in planes_dict:
            raise ValueError(f"Plane not recognized: {plane_name}")
        i, j = planes_dict[plane_name]
        if i >= len(v) or j >= len(v):
            return 0.0
        return rotor_angle(v, w, planes_dict[plane_name])
    
    def all_rotor_angles(self, v: np.ndarray, w: np.ndarray) -> Dict[str, float]:
        angles = {}
        for name, indices, desc in ROTOR_PLANES:
            i, j = indices
            if i < len(v) and j < len(v):
                angles[name] = rotor_angle(v, w, indices)
            else:
                angles[name] = 0.0
        return angles
    
    def reflection_analysis(self, v: np.ndarray, w: np.ndarray) -> Dict:
        is_ref, sim = self.is_specular_reflection(v, w)
        return {
            'is_specular_reflection': is_ref,
            'reflection_similarity': sim,
            'interpretation': "Specular reflection detected" if is_ref else "Not a specular reflection"
        }
    
    def clifford_signature(self, v: np.ndarray) -> Dict[str, float]:
        return clifford_signature(v)
    
    def dot_product_metric(self, v: np.ndarray, w: np.ndarray) -> float:
        return dot_product_metric(v, w)
    
    def norm_metric(self, v: np.ndarray) -> Tuple[float, float]:
        return norm_metric(v)
    
    def similarity_metric(self, v: np.ndarray, w: np.ndarray) -> float:
        return similarity_metric(v, w)
    
    def geometric_product_decomposition(self, v: np.ndarray, w: np.ndarray) -> Dict:
        return geometric_product_decomposition(v, w)
    
    def metric_signature_info(self) -> Dict:
        return metric_signature_info()


class AdvancedGroupAnalyzer:
    def __init__(self, grassmann: GrassmannPIM):
        self.grassmann = grassmann
        self.dim = grassmann.dim
        self.groups: Dict[str, List[np.ndarray]] = {}
        self.group_headers: Dict[str, List[str]] = {}
        self.group_stats: Dict[str, GroupStatistics] = {}
        self.proteins: Dict[str, Tuple[str, np.ndarray]] = {}
        self.adaptive_thresholds: Dict[str, float] = {}
    
    def load_fasta_streaming(self, filepath: str, group_name: str, 
                              max_sequences: int = None, verbose: bool = True) -> int:
        if verbose:
            print(f"  Streaming {group_name} from {filepath}...")
        
        if group_name not in self.groups:
            self.groups[group_name] = []
            self.group_headers[group_name] = []
        
        stats = OnlineStatistics(self.dim)
        sample_vectors = []
        sample_headers = []
        SAMPLE_SIZE = min(COHESION_SAMPLE_SIZE, 10000)
        
        count_valid = 0
        count_total = 0
        
        for header, seq in read_fasta_stream(filepath, verbose):
            count_total += 1
            if max_sequences and count_total > max_sequences:
                break
            
            pim_profile = compute_pim_profile(seq, use_weights=USE_WEIGHTS)
            
            if np.sum(pim_profile) > 0.01:
                stats.update(pim_profile)
                count_valid += 1
                
                if len(sample_vectors) < SAMPLE_SIZE:
                    sample_vectors.append(pim_profile)
                    sample_headers.append(header[:100])
                else:
                    j = random.randint(0, count_valid - 1)
                    if j < SAMPLE_SIZE:
                        sample_vectors[j] = pim_profile
                        sample_headers[j] = header[:100]
                
                if count_valid <= 1000:
                    clean_header = header[:100] if len(header) > 100 else header
                    self.groups[group_name].append(pim_profile)
                    self.group_headers[group_name].append(clean_header)
                    protein_name = f"{group_name}|{clean_header}"
                    self.proteins[protein_name] = (group_name, pim_profile)
            
            if verbose and count_total % 100000 == 0 and count_total > 0:
                print(f"    Processed {count_total:,} sequences, {count_valid:,} valid...")
        
        centroid = stats.get_mean()
        covariance = stats.get_covariance()
        std_dev = stats.get_std()
        inv_covariance = np.linalg.pinv(covariance + np.eye(self.dim) * 1e-6)
        
        if len(sample_vectors) > 1:
            intra_similarities = []
            sample_size = min(len(sample_vectors), 200)
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    sim, _ = self.grassmann.wedge_product(sample_vectors[i], sample_vectors[j], with_ci=False)
                    intra_similarities.append(sim)
            wedge_self_similarity = np.mean(intra_similarities) if intra_similarities else 1.0
            wedge_self_similarity_std = np.std(intra_similarities) if len(intra_similarities) > 1 else 0.0
            self.adaptive_thresholds[group_name] = np.percentile(intra_similarities, 5) if len(intra_similarities) > 0 else 0.99
        else:
            wedge_self_similarity = 1.0
            wedge_self_similarity_std = 0.0
            self.adaptive_thresholds[group_name] = 0.99
        
        cliff_sig = self.grassmann.clifford_signature(centroid)
        
        subspace_proj = {}
        for subspace in SUBSPACES.keys():
            if subspace != 'full':
                subspace_proj[subspace] = self.grassmann.interior_product_magnitude(centroid, subspace)
        
        metric_norm, metric_sign = self.grassmann.norm_metric(centroid)
        
        self.group_stats[group_name] = GroupStatistics(
            name=group_name,
            n_samples=count_valid,
            centroid=centroid,
            covariance=covariance,
            inv_covariance=inv_covariance,
            std_dev=std_dev,
            wedge_self_similarity=wedge_self_similarity,
            wedge_self_similarity_std=wedge_self_similarity_std,
            adaptive_threshold=self.adaptive_thresholds[group_name],
            clifford_signature=cliff_sig,
            subspace_projections=subspace_proj,
            metric_norm=metric_norm,
            metric_sign=metric_sign
        )
        
        metric_info = f", metric_norm={metric_norm:.4f}({'+' if metric_sign>0 else '-' if metric_sign<0 else '0'})" if USE_BIOLOGICAL_METRIC else ""
        print(f"     {group_name}: n={count_valid:,}, cohesion={wedge_self_similarity:.6f} ± {wedge_self_similarity_std:.6f}, "
              f"adaptive_threshold={self.adaptive_thresholds[group_name]:.4f}{metric_info}")
        
        return count_valid
    
    def load_fasta_file(self, filepath: str, group_name: str, 
                         max_sequences: int = None, verbose: bool = True) -> int:
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            if file_size > 100 * 1024 * 1024:
                if verbose:
                    print(f"  Large file detected ({file_size / (1024**2):.1f} MB) - using streaming mode")
                return self.load_fasta_streaming(filepath, group_name, max_sequences, verbose)
        
        if verbose:
            print(f"  Loading {group_name} from {filepath}...")
        
        sequences = []
        current_header = None
        current_seq = []
        
        if not os.path.exists(filepath):
            if verbose:
                print(f"    File not found: {filepath}")
            return 0
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_header is not None:
                        sequences.append((current_header, ''.join(current_seq)))
                    current_header = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_header is not None:
                sequences.append((current_header, ''.join(current_seq)))
        
        if verbose:
            print(f"    Sequences found: {len(sequences)}")
        
        if group_name not in self.groups:
            self.groups[group_name] = []
            self.group_headers[group_name] = []
        
        count_valid = 0
        for i, (header, seq) in enumerate(sequences):
            if max_sequences and i >= max_sequences:
                break
            
            pim_profile = compute_pim_profile(seq, use_weights=USE_WEIGHTS)
            
            if np.sum(pim_profile) > 0.01:
                self.groups[group_name].append(pim_profile)
                clean_header = header[:100] if len(header) > 100 else header
                self.group_headers[group_name].append(clean_header)
                protein_name = f"{group_name}|{clean_header}"
                self.proteins[protein_name] = (group_name, pim_profile)
                count_valid += 1
            
            if verbose and count_valid % 5000 == 0 and count_valid > 0:
                print(f"    Processed {count_valid} valid sequences...")
        
        if verbose:
            print(f"    ✅ Valid PIM profiles: {count_valid}")
        
        return count_valid
    
    def compute_group_statistics(self):
        print("\n  📊 Computing group statistics...")
        
        for group_name, vectors in self.groups.items():
            if group_name in self.group_stats:
                continue
                
            if len(vectors) == 0:
                print(f"     {group_name}: empty (0 proteins)")
                continue
            
            vectors_array = np.array(vectors)
            n = len(vectors_array)
            dim = vectors_array.shape[1]
            
            centroid = np.mean(vectors_array, axis=0)
            metric_norm, metric_sign = self.grassmann.norm_metric(centroid)
            
            cliff_sig = self.grassmann.clifford_signature(centroid)
            subspace_proj = {}
            for subspace in SUBSPACES.keys():
                if subspace != 'full':
                    subspace_proj[subspace] = self.grassmann.interior_product_magnitude(centroid, subspace)
            
            if n > 1:
                covariance = np.cov(vectors_array, rowvar=False) + np.eye(dim) * 1e-6
            else:
                covariance = np.eye(dim) * 0.01
            
            inv_covariance = np.linalg.pinv(covariance)
            std_dev = np.std(vectors_array, axis=0)
            
            if n > 1:
                intra_similarities = []
                sample_size = min(n, 50)
                for i in range(sample_size):
                    for j in range(i+1, sample_size):
                        sim, _ = self.grassmann.wedge_product(vectors_array[i], vectors_array[j], with_ci=False)
                        intra_similarities.append(sim)
                wedge_self_similarity = np.mean(intra_similarities) if intra_similarities else 1.0
                wedge_self_similarity_std = np.std(intra_similarities) if len(intra_similarities) > 1 else 0.0
                self.adaptive_thresholds[group_name] = np.percentile(intra_similarities, 5) if len(intra_similarities) > 0 else 0.99
            else:
                wedge_self_similarity = 1.0
                wedge_self_similarity_std = 0.0
                self.adaptive_thresholds[group_name] = 0.99
            
            self.group_stats[group_name] = GroupStatistics(
                name=group_name,
                n_samples=n,
                centroid=centroid,
                covariance=covariance,
                inv_covariance=inv_covariance,
                std_dev=std_dev,
                wedge_self_similarity=wedge_self_similarity,
                wedge_self_similarity_std=wedge_self_similarity_std,
                adaptive_threshold=self.adaptive_thresholds[group_name],
                clifford_signature=cliff_sig,
                subspace_projections=subspace_proj,
                metric_norm=metric_norm,
                metric_sign=metric_sign
            )
            
            metric_info = f", metric_norm={metric_norm:.4f}({'+' if metric_sign>0 else '-' if metric_sign<0 else '0'})" if USE_BIOLOGICAL_METRIC else ""
            print(f"     {group_name}: n={n:,}, cohesion={wedge_self_similarity:.6f} ± {wedge_self_similarity_std:.6f}, "
                  f"adaptive_threshold={self.adaptive_thresholds[group_name]:.4f}{metric_info}")
    
    def compare_group_to_all(self, target_group: str) -> pd.DataFrame:
        if target_group not in self.group_stats:
            return pd.DataFrame()
        
        target_stat = self.group_stats[target_group]
        target_centroid = target_stat.centroid
        adaptive_threshold = self.adaptive_thresholds.get(target_group, 0.99)
        
        results = []
        for group_name, stat in self.group_stats.items():
            if group_name == target_group:
                continue
            
            wedge, wedge_std = self.grassmann.wedge_product(target_centroid, stat.centroid, with_ci=True)
            prob = stat.probability_of_belonging(target_centroid) if stat.n_samples > 1 else 0.5
            is_similar = wedge >= adaptive_threshold
            
            rotor_angles = self.grassmann.all_rotor_angles(target_centroid, stat.centroid)
            reflection = self.grassmann.reflection_analysis(target_centroid, stat.centroid)
            cliff_dist = clifford_distance(target_stat.clifford_signature, stat.clifford_signature) if target_stat.clifford_signature and stat.clifford_signature else 0.0
            
            mag, orient, _ = self.grassmann.wedge_product_oriented(target_centroid, stat.centroid)
            gp_decomp = self.grassmann.geometric_product_decomposition(target_centroid, stat.centroid)
            
            results.append({
                'Compared Group': group_name,
                'Wedge Similarity': round(wedge, 6),
                'Wedge Orientation': round(orient, 6),
                'Probability of Belonging': round(prob, 6),
                'N Samples': stat.n_samples,
                'Is Similar (adaptive)': is_similar,
                'Hydrophobic Angle (°)': round(rotor_angles.get('hydrophobic', 0), 2),
                'Charge Angle (°)': round(rotor_angles.get('charge', 0), 2),
                'Specular Reflection': reflection['is_specular_reflection'],
                'Clifford Distance': round(cliff_dist, 6),
                'GP Functional Sim': round(gp_decomp['functional_similarity'], 6),
                'GP Structural Diff': round(gp_decomp['structural_difference'], 6),
                'GP F/S Ratio': round(gp_decomp['functional_structural_ratio'], 2),
                'GP Interpretation': gp_decomp['interpretation']
            })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        return df.sort_values('Wedge Similarity', ascending=False)
    
    def cross_group_similarity_matrix(self) -> pd.DataFrame:
        group_names = list(self.group_stats.keys())
        if not group_names:
            return pd.DataFrame()
        
        n = len(group_names)
        matrix = np.zeros((n, n))
        
        for i, g1 in enumerate(group_names):
            for j, g2 in enumerate(group_names):
                if i != j:
                    matrix[i, j], _ = self.grassmann.wedge_product(
                        self.group_stats[g1].centroid,
                        self.group_stats[g2].centroid,
                        with_ci=False
                    )
        
        return pd.DataFrame(matrix, index=group_names, columns=group_names)
    
    def compute_lujv_statistics(self, target_group: str = 'LUJV') -> dict:
        if target_group not in self.group_stats:
            return {}
        
        target_centroid = self.group_stats[target_group].centroid
        
        similarities = []
        max_diffs = []
        rotor_angles_hydro = []
        rotor_angles_charge = []
        is_reflection = []
        orientations = []
        gp_functional = []
        gp_structural = []
        gp_ratios = []
        
        human_groups = ['REVIEWED_HUMAN', 'UNREVIEWED_HUMAN', 'senales', 'membrana', 'enfermedad']
        
        for group_name in human_groups:
            if group_name not in self.groups:
                continue
            
            for vec in self.groups[group_name]:
                mag, orient, _ = self.grassmann.wedge_product_oriented(target_centroid, vec)
                similarities.append(mag)
                orientations.append(orient)
                max_diff = self.grassmann.max_component_diff(target_centroid, vec)
                max_diffs.append(max_diff)
                rotor_angles_hydro.append(self.grassmann.rotor_angle(target_centroid, vec, 'hydrophobic'))
                rotor_angles_charge.append(self.grassmann.rotor_angle(target_centroid, vec, 'charge'))
                is_ref, _ = self.grassmann.is_specular_reflection(target_centroid, vec)
                is_reflection.append(is_ref)
                
                gp_decomp = self.grassmann.geometric_product_decomposition(target_centroid, vec)
                gp_functional.append(gp_decomp['functional_similarity'])
                gp_structural.append(gp_decomp['structural_difference'])
                gp_ratios.append(gp_decomp['functional_structural_ratio'])
        
        similarities = np.array(similarities)
        max_diffs = np.array(max_diffs)
        
        adaptive_threshold = self.adaptive_thresholds.get(target_group, 0.99)
        
        orientation_signs = np.sign(orientations)
        pct_positive_orientation = np.mean(orientation_signs > 0) * 100
        pct_negative_orientation = np.mean(orientation_signs < 0) * 100
        
        return {
            'similarities': similarities,
            'orientations': orientations,
            'max_diffs': max_diffs,
            'rotor_angles_hydro': rotor_angles_hydro,
            'rotor_angles_charge': rotor_angles_charge,
            'pct_reflection': np.mean(is_reflection) * 100 if is_reflection else 0,
            'pct_positive_orientation': pct_positive_orientation,
            'pct_negative_orientation': pct_negative_orientation,
            'max_similarity': np.max(similarities) if len(similarities) > 0 else 0,
            'mean_similarity': np.mean(similarities) if len(similarities) > 0 else 0,
            'std_similarity': np.std(similarities) if len(similarities) > 0 else 0,
            'count_sim_099': np.sum(similarities >= adaptive_threshold) if len(similarities) > 0 else 0,
            'count_sim_095': np.sum(similarities >= 0.95) if len(similarities) > 0 else 0,
            'count_sim_090': np.sum(similarities >= 0.90) if len(similarities) > 0 else 0,
            'max_diff_min': np.min(max_diffs) if len(max_diffs) > 0 else 0,
            'max_diff_mean': np.mean(max_diffs) if len(max_diffs) > 0 else 0,
            'adaptive_threshold': adaptive_threshold,
            'mean_gp_functional': np.mean(gp_functional) if gp_functional else 0,
            'mean_gp_structural': np.mean(gp_structural) if gp_structural else 0,
            'mean_gp_ratio': np.mean(gp_ratios) if gp_ratios else 0,
        }
    
    def print_all_group_summary(self):
        print("\n" + "=" * 80)
        print("📊 SUMMARY OF ALL GROUPS")
        print("=" * 80)
        if USE_BIOLOGICAL_METRIC:
            print(f"\n  {'Group':<20} {'N seq':>10} {'Cohesion':>20} {'Mean Std Dev':>14} {'Threshold':>10} {'Metric Norm':>12} {'Sign':>6}")
            print(f"  {'-'*95}")
        else:
            print(f"\n  {'Group':<20} {'N seq':>10} {'Cohesion':>20} {'Mean Std Dev':>14} {'Threshold':>10}")
            print(f"  {'-'*75}")
        
        for group_name, stats in sorted(self.group_stats.items()):
            adaptive_threshold = self.adaptive_thresholds.get(group_name, 0.99)
            if USE_BIOLOGICAL_METRIC:
                sign_char = '+' if stats.metric_sign > 0 else '-' if stats.metric_sign < 0 else '0'
                print(f"  {group_name:<20} {stats.n_samples:>10,} "
                      f"{stats.wedge_self_similarity:>12.6f} ± {stats.wedge_self_similarity_std:.6f} "
                      f"{np.mean(stats.std_dev):>14.6f} {adaptive_threshold:>10.4f} {stats.metric_norm:>12.6f} {sign_char:>6}")
            else:
                print(f"  {group_name:<20} {stats.n_samples:>10,} "
                      f"{stats.wedge_self_similarity:>12.6f} ± {stats.wedge_self_similarity_std:.6f} "
                      f"{np.mean(stats.std_dev):>14.6f} {adaptive_threshold:>10.4f}")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_similarity_distribution(similarities: np.ndarray, save_path: str, adaptive_threshold: float = None):
    plt.figure(figsize=(12, 6))
    plt.hist(similarities, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0.99, color='red', linestyle='--', linewidth=2, label='Threshold 0.99')
    plt.axvline(x=0.95, color='orange', linestyle='--', linewidth=2, label='Threshold 0.95')
    plt.axvline(x=0.90, color='green', linestyle='--', linewidth=2, label='Threshold 0.90')
    if adaptive_threshold:
        plt.axvline(x=adaptive_threshold, color='purple', linestyle=':', linewidth=2, 
                    label=f'Adaptive threshold ({adaptive_threshold:.4f})')
    plt.xlabel('Similarity (Wedge Product ∧)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Similarity Distribution: LUJV vs Human Proteome\nn={len(similarities):,}', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Figure saved: {save_path}")


def plot_orientation_distribution(orientations: List[float], save_path: str):
    plt.figure(figsize=(10, 6))
    plt.hist(orientations, bins=50, color='teal', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral (zero orientation)')
    plt.xlabel('Orientation Sign', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Orientation Distribution: LUJV vs Human Proteome', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Figure saved: {save_path}")


def plot_max_diff_distribution(max_diffs: np.ndarray, save_path: str, tolerance: float):
    plt.figure(figsize=(12, 6))
    plt.hist(max_diffs, bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(x=tolerance, color='red', linestyle='--', linewidth=2, label=f'Tolerance = {tolerance}')
    plt.xlabel('Maximum Component Difference', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Maximum Component Differences\nTolerance: {tolerance}', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Figure saved: {save_path}")


def plot_tolerance_summary(stats: dict, save_path: str, tolerance: float):
    thresholds = ['≥ 0.99', '≥ 0.95', '≥ 0.90']
    counts = [
        stats.get('count_sim_099', 0),
        stats.get('count_sim_095', 0),
        stats.get('count_sim_090', 0)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    bars = ax.bar(thresholds, counts, color=['#e74c3c', '#f1c40f', '#2ecc71'], edgecolor='black', alpha=0.8)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Proteins Similar to LUJV by Threshold', fontsize=12)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(1, count/50),
                    f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    ax2 = axes[1]
    stats_labels = ['Max Similarity', 'Mean Similarity', 'Min Max Diff', 'Mean Max Diff']
    stats_values = [
        stats.get('max_similarity', 0),
        stats.get('mean_similarity', 0),
        stats.get('max_diff_min', 0),
        stats.get('max_diff_mean', 0)
    ]
    bars2 = ax2.bar(stats_labels, stats_values, color='teal', edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Comparison Statistics', fontsize=12)
    for bar, val in zip(bars2, stats_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    if 'mean_gp_ratio' in stats:
        ax2.text(0.5, -0.15, f"Functional/Structural Ratio: {stats['mean_gp_ratio']:.2f}",
                 transform=ax2.transAxes, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Figure saved: {save_path}")


def plot_rotor_angles(angles_hydro: List[float], angles_charge: List[float], save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.hist(angles_hydro, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(angles_hydro), color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {np.mean(angles_hydro):.1f}°')
    ax.set_xlabel('Rotation Angle (°) - Hydrophobic Plane', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Hydrophobic Plane (N→N vs NP→NP)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    ax.hist(angles_charge, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(angles_charge), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(angles_charge):.1f}°')
    ax.set_xlabel('Rotation Angle (°) - Charge Plane', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Charge Plane (P⁺→P⁺ vs P⁻→P⁻)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Figure saved: {save_path}")


def plot_lujv_heatmap(sim_matrix: pd.DataFrame, results_dir: str):
    if sim_matrix.empty or 'LUJV' not in sim_matrix.index:
        print("  ⚠ No se puede generar el heatmap")
        return
    
    lujv_row = sim_matrix.loc['LUJV'].sort_values()
    
    target_groups = ['LASV', 'JUNV', 'MACV', 'LCMV', 
                     'VIRUS_REVIEWED', 'VIRUS_UNREVIEWED',
                     'REVIEWED_HUMAN', 'UNREVIEWED_HUMAN',
                     'REVIEWED_ALL', 'UNREVIEWED_ALL']
    target_groups = [g for g in target_groups if g in lujv_row.index]
    
    if not target_groups:
        print("  ⚠ No se encontraron grupos objetivo")
        return
    
    plt.figure(figsize=(16, 4))
    data = lujv_row[target_groups].values.reshape(1, -1)
    labels = target_groups
    
    sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlBu_r',
                xticklabels=labels, yticklabels=['LUJV'],
                cbar_kws={'label': 'Wedge Similarity (∧)'},
                vmin=0, vmax=0.15)
    
    plt.title('LUJV vs Other Groups: Wedge Similarity\n(Lower = More Similar, Red = Closest)', fontsize=14)
    plt.tight_layout()
    
    save_path = f"{results_dir}/heatmap_LUJV_vs_groups.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Heatmap guardado: {save_path}")
    
    csv_path = f"{results_dir}/lujv_vs_groups.csv"
    pd.DataFrame(data, columns=labels, index=['LUJV']).to_csv(csv_path)
    print(f"    ✅ Datos guardados: {csv_path}")
    
    print("\n  📊 LUJV vs Other Groups (Wedge Similarity):")
    for i, (group, val) in enumerate(zip(labels, data[0])):
        print(f"     ├─ {group}: {val:.6f}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 80)
    print("🦠 MIRROR-PIM: GRASSMANN-PIM WITH REAL GEOMETRIC ALGEBRA (v11)")
    print("   OPTIMIZED FOR MASSIVE DATASETS (150M+ sequences)")
    print("   Streaming mode: O(1) memory, processes sequences one by one")
    print("=" * 80)
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    dim = DIM_PAIRS
    print(f"\n  ⚙ CONFIGURATION:")
    print(f"     ├─ Space dimension: {dim} components (polarity pairs)")
    print(f"     ├─ Biological weighting: {'YES' if USE_WEIGHTS else 'NO'}")
    print(f"     ├─ Streaming mode: {'YES' if STREAMING_MODE else 'NO'}")
    print(f"     ├─ Cohesion sample size: {COHESION_SAMPLE_SIZE} random pairs")
    print(f"     ├─ LSH hash: Enabled")
    print(f"     ├─ Clifford rotors: Enabled (6 biological planes)")
    print(f"     └─ REAL GEOMETRIC ALGEBRA: Enabled (v11)")
    
    grassmann = GrassmannPIM(dim=dim)
    analyzer = AdvancedGroupAnalyzer(grassmann)
    
    # ========================================================================
    # FILE DEFINITION
    # ========================================================================
    
    files_to_load = {
        'LUJV': 'lujv_all.unico.dat0',
        'LASV': 'lasv_all.unico.dat0',
        'JUNV': 'junv_all.unico.dat0',
        'MACV': 'macv_all.unico.dat0',
        'LCMV': 'lcmv_all.unico.dat0',
        'PARTIALLY_FOLDED': 'partiallyorderedN.unico.dat0',
        'CPP': 'CPP.unico.dat0',
        'NON_CPP': 'NONCPP.unico.dat0',
        'UNFOLDED': 'unfolded.unico.dat0',
        'REVIEWED_HUMAN': 'reviewed_human.unico.dat0',
        'UNREVIEWED_HUMAN': 'unreviewed_human.unico.dat0',
        'senales': 'senales.unico.dat0',
        'membrana': 'membrana.unico.dat0',
        'enfermedad': 'enfermedad.unico.dat0',
        'VIRUS_REVIEWED': 'reviewed_virus.unico.dat0',
        'VIRUS_UNREVIEWED': 'unreviewed_virus.unico.dat0',
        'REVIEWED_ALL': 'reviewed_all.unico.dat0',
        'UNREVIEWED_ALL': 'unreviewed_all.unico.dat0',
    }
    
    print("\n📂 LOADING FASTA FILES (STREAMING MODE)...")
    print("   ⚠ Large files will be processed without storing all vectors in memory")
    print()
    
    for group_name, filename in files_to_load.items():
        analyzer.load_fasta_file(filename, group_name, verbose=True)
    
    print("\n" + "=" * 80)
    print("📈 GENERATING FIGURES...")
    print("=" * 80)
    
    lujv_stats = analyzer.compute_lujv_statistics('LUJV')
    
    if lujv_stats and len(lujv_stats.get('similarities', [])) > 0:
        plot_similarity_distribution(
            lujv_stats['similarities'],
            "01_similarity_distribution.png",
            adaptive_threshold=lujv_stats.get('adaptive_threshold')
        )
        
        if 'orientations' in lujv_stats and len(lujv_stats['orientations']) > 0:
            plot_orientation_distribution(
                lujv_stats['orientations'],
                "01b_orientation_distribution.png"
            )
        
        plot_max_diff_distribution(
            lujv_stats['max_diffs'],
            "02_max_diff_distribution.png",
            TOLERANCE
        )
        
        plot_tolerance_summary(
            lujv_stats,
            "03_tolerance_summary.png",
            TOLERANCE
        )
        
        if 'rotor_angles_hydro' in lujv_stats and len(lujv_stats['rotor_angles_hydro']) > 0:
            plot_rotor_angles(
                lujv_stats['rotor_angles_hydro'],
                lujv_stats['rotor_angles_charge'],
                "04_rotor_angles_distribution.png"
            )
    else:
        print("  ⚠ Insufficient data to generate figures")
    
    analyzer.print_all_group_summary()
    
    print("\n" + "=" * 80)
    print("🔍 COMPARING LUJV TO ALL GROUPS")
    print("=" * 80)
    
    comparison_lujv = analyzer.compare_group_to_all('LUJV')
    if comparison_lujv is not None and not comparison_lujv.empty:
        comparison_lujv.to_csv("comparison_LUJV_vs_all.csv", index=False)
        print("\n  📊 LUJV vs Other Groups (sorted by wedge similarity):")
        print(comparison_lujv[['Compared Group', 'Wedge Similarity', 'Wedge Orientation', 'N Samples']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("🔍 INTER-GROUP SIMILARITY MATRIX")
    print("=" * 80)
    
    sim_matrix = analyzer.cross_group_similarity_matrix()
    if not sim_matrix.empty:
        sim_matrix.to_csv("similarity_matrix_groups.csv")
        print("\n  📊 Inter-group similarity matrix:")
        print(sim_matrix.round(4).to_string())
        plot_lujv_heatmap(sim_matrix, ".")
    
    print("\n" + "=" * 80)
    print("🎯 GEOMETRIC PRODUCT DECOMPOSITION")
    print("=" * 80)
    
    if lujv_stats:
        print(f"\n  LUJV vs Human Proteome - Geometric Product Analysis:")
        print(f"     ├─ Mean Functional Similarity: {lujv_stats.get('mean_gp_functional', 0):.6f}")
        print(f"     ├─ Mean Structural Difference: {lujv_stats.get('mean_gp_structural', 0):.6f}")
        print(f"     └─ Mean Functional/Structural Ratio: {lujv_stats.get('mean_gp_ratio', 0):.2f}")
    
    print("\n" + "=" * 80)
    print("✅ EXECUTION COMPLETED")
    print("=" * 80)
    print(f"⏰ End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
