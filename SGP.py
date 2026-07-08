#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MIRROR-PIM: GRASSMANN-PIM WITH REAL GEOMETRIC ALGEBRA OPERATORS (v13.1)
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
import gc
import sys
import time

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SIMILARITY_THRESHOLD = None
CONFIDENCE_LEVEL = 0.95
TOP_N_PROTEINS = 20
TOLERANCE = 0.001
USE_TRIPLETS = False
USE_BOOTSTRAP = True
N_BOOTSTRAP = 100
USE_WEIGHTS = True
COHESION_SAMPLE_SIZE = 10000
USE_BIOLOGICAL_METRIC = True
SHOW_METRIC_ANALYSIS = True

# ============================================================================
# ⚙️ CONFIGURACIÓN DE PROCESAMIENTO POR LOTES
# ============================================================================

# Tamaño del lote en número de secuencias
BATCH_SIZE = 100000  # 100,000 secuencias por lote

# Tamaño máximo de muestra para almacenar en RAM (para comparaciones posteriores)
MAX_STORED_PROTEINS_PER_GROUP = 10000  # 10,000 por grupo (suficiente para estadísticas)

# Tamaño de muestra para cálculo de cohesión (se toma de la muestra almacenada)
COHESION_CALC_SAMPLE_SIZE = 500  # Aumentado de 200 a 500 para mejor representación

# Umbral para usar procesamiento por lotes (secuencias)
BATCH_THRESHOLD = 50000

# Intervalo de reporte de progreso (número de secuencias)
PROGRESS_REPORT_INTERVAL = 100000

# ============================================================================
# MAPEO DE NOMBRES DE GRUPOS PARA EXHIBICIÓN
# ============================================================================

GROUP_NAME_MAP = {
    'enfermedad': 'DISEASE',
    'membrana': 'MEMBRANE',
    'senales': 'SIGNALS',
}

def get_display_name(group_name: str) -> str:
    return GROUP_NAME_MAP.get(group_name, group_name)

# ============================================================================
# CONSTANTES BASE
# ============================================================================

DIM_PAIRS = 16
DIM_TRIPLETS = 64

ROTOR_PLANES = [
    ('hydrophobic', (10, 15), 'N→N vs NP→NP (H-bonds vs hydrophobic interactions)'),
    ('charge', (0, 5), 'P⁺→P⁺ vs P⁻→P⁻ (repulsion vs attraction)'),
    ('opposite_charge', (1, 4), 'P⁺→P⁻ vs P⁻→P⁺ (charge-charge interactions)'),
    ('polarity', (10, 11), 'N→N vs N→NP (polar vs mixed)'),
    ('charge_transition', (2, 8), 'P⁺→N vs N→P⁺ (charge-polar transition)'),
    ('opposite_transition', (6, 9), 'P⁻→N vs N→P⁻ (negative charge-polar transition)'),
]

REFLECTION_SWAP_MAP = {
    0: 5, 1: 4, 2: 6, 3: 7, 4: 1, 5: 0, 6: 2, 7: 3,
    8: 9, 9: 8, 10: 10, 11: 11, 12: 13, 13: 12, 14: 14, 15: 15,
}

KEY_BIVECTORS = [
    (0, 5), (1, 4), (2, 6), (3, 7), (10, 11), (14, 15),
]

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

BIOLOGICAL_METRIC_SIGNATURE = np.array([
    -1.0, +1.0, +1.0, +0.0,
    +1.0, -1.0, +1.0, +0.0,
    +1.0, +1.0, +1.0, +0.0,
    +0.0, +0.0, +0.0, +1.0,
])

EUCLIDEAN_METRIC = np.ones(16)
METRIC_SIGNATURE = BIOLOGICAL_METRIC_SIGNATURE if USE_BIOLOGICAL_METRIC else EUCLIDEAN_METRIC

SUBSPACES = {
    'hydrophobic': [10, 15],
    'charge_repulsion': [0, 5],
    'charge_attraction': [1, 4],
    'charge_polar': [2, 3, 6, 7],
    'polar': [8, 9, 10, 11],
    'nonpolar': [12, 13, 14, 15],
    'full': None,
}

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

# ============================================================================
# FUNCIONES BASE
# ============================================================================

def compute_pim_profile(sequence: str, use_weights: bool = USE_WEIGHTS) -> np.ndarray:
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

def pim_to_hash(pim_vector: np.ndarray, tolerance: float = TOLERANCE) -> str:
    discretized = np.round(pim_vector / tolerance) * tolerance
    vector_str = ','.join([f"{x:.6f}" for x in discretized])
    return hashlib.sha256(vector_str.encode()).hexdigest()[:32]

def compute_delta_pim(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return v1 - v2

# ============================================================================
# v10: COMMUTATOR, ANTICOMMUTATOR, AND METRIC OPERATORS
# ============================================================================

def commutator(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return wedge_product_oriented(v, w)

def commutator_norm(v: np.ndarray, w: np.ndarray) -> float:
    comm = commutator(v, w)
    mag = np.linalg.norm(comm)
    norm_v = np.linalg.norm(v) + 1e-10
    norm_w = np.linalg.norm(w) + 1e-10
    return mag / (norm_v * norm_w + 1e-10)

def anticommutator(v: np.ndarray, w: np.ndarray) -> float:
    return 2.0 * np.dot(v, w)

def anticommutator_similarity(v: np.ndarray, w: np.ndarray) -> float:
    anticomm = anticommutator(v, w)
    norm_v = np.linalg.norm(v) + 1e-10
    norm_w = np.linalg.norm(w) + 1e-10
    sim = np.abs(anticomm) / (2.0 * norm_v * norm_w + 1e-10)
    return min(sim, 1.0)

def dot_product_metric(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> float:
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
    value = dot_product_metric(v, v, metric)
    sign = np.sign(value) if value != 0 else 0
    magnitude = np.sqrt(np.abs(value) + 1e-10)
    return magnitude, sign

def similarity_metric(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> float:
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
# v11: GEOMETRIC PRODUCT WITH METRIC
# ============================================================================

def geometric_product_metric(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> Tuple[float, np.ndarray]:
    if metric is None:
        metric = METRIC_SIGNATURE
    scalar = np.sum(metric * v * w)
    sqrt_metric = np.sqrt(np.abs(metric) + 1e-10)
    v_transformed = v / sqrt_metric
    w_transformed = w / sqrt_metric
    bivector = wedge_product_oriented(v_transformed, w_transformed)
    return scalar, bivector

def geometric_product_decomposition(v: np.ndarray, w: np.ndarray, metric: np.ndarray = None) -> Dict:
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
# CLIFFORD SIGNATURE
# ============================================================================

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

# ============================================================================
# READ FASTA STREAM (sin límites)
# ============================================================================

def read_fasta_stream(filepath: str, verbose: bool = False):
    """Lee archivo FASTA sin límites, generando secuencia por secuencia"""
    if not os.path.exists(filepath):
        if verbose:
            print(f"    File not found: {filepath}")
        return
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
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
# CLASE: OnlineStatistics (sin límites)
# ============================================================================

class OnlineStatistics:
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
# CLASE: ProgressiveSampler (muestreo progresivo)
# ============================================================================

class ProgressiveSampler:
    """Mantiene una muestra representativa sin almacenar todo"""
    
    def __init__(self, max_samples: int = MAX_STORED_PROTEINS_PER_GROUP):
        self.max_samples = max_samples
        self.samples = []
        self.headers = []
        self.total_seen = 0
    
    def add(self, vector: np.ndarray, header: str):
        self.total_seen += 1
        
        if len(self.samples) < self.max_samples:
            self.samples.append(vector)
            self.headers.append(header)
        else:
            # Reemplazo aleatorio con probabilidad inversa
            j = random.randint(0, self.total_seen - 1)
            if j < self.max_samples:
                self.samples[j] = vector
                self.headers[j] = header
    
    def get_samples(self) -> List[np.ndarray]:
        return self.samples
    
    def get_headers(self) -> List[str]:
        return self.headers
    
    def size(self) -> int:
        return len(self.samples)

# ============================================================================
# CLASE: ProcessingTracker (monitoreo en tiempo real)
# ============================================================================

class ProcessingTracker:
    """Monitorea y registra estadísticas de procesamiento en tiempo real"""
    
    def __init__(self):
        self.total_sequences_processed = 0
        self.total_valid_pim = 0
        self.total_rejected = 0
        self.total_bytes_read = 0
        self.group_counts = {}
        self.group_valid = {}
        self.start_time = None
        self.last_report_time = None
        self.last_report_count = 0
        self.processing_rate = 0.0
        self.batch_count = 0
        self.total_batches = 0
        
    def update(self, group_name: str, is_valid: bool, bytes_read: int = 0):
        self.total_sequences_processed += 1
        self.total_bytes_read += bytes_read
        
        if is_valid:
            self.total_valid_pim += 1
        else:
            self.total_rejected += 1
            
        if group_name not in self.group_counts:
            self.group_counts[group_name] = 0
            self.group_valid[group_name] = 0
        
        self.group_counts[group_name] += 1
        if is_valid:
            self.group_valid[group_name] += 1
    
    def get_report(self) -> Dict:
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 1
        rate = self.total_sequences_processed / elapsed if elapsed > 0 else 0
        
        return {
            'total_sequences': self.total_sequences_processed,
            'valid_pim': self.total_valid_pim,
            'rejected': self.total_rejected,
            'valid_percentage': (self.total_valid_pim / self.total_sequences_processed * 100) 
                                if self.total_sequences_processed > 0 else 0,
            'group_counts': self.group_counts,
            'group_valid': self.group_valid,
            'total_bytes': self.total_bytes_read,
            'processing_rate': rate,
            'elapsed_seconds': elapsed,
            'batch_count': self.batch_count,
            'total_batches': self.total_batches
        }
    
    def print_progress(self, group_name: str = None, force: bool = False):
        """Imprime progreso en tiempo real"""
        if self.start_time is None:
            return
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.total_sequences_processed / elapsed if elapsed > 0 else 0
        
        # Solo reportar cada PROGRESS_REPORT_INTERVAL secuencias
        if not force and (self.total_sequences_processed - self.last_report_count) < PROGRESS_REPORT_INTERVAL:
            return
        
        self.last_report_count = self.total_sequences_processed
        
        # Estimar tiempo restante
        if rate > 0 and self.total_batches > 0:
            remaining_seqs = self.total_batches * BATCH_SIZE - self.total_sequences_processed
            eta_seconds = remaining_seqs / rate if rate > 0 else 0
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
        else:
            eta_str = "calculando..."
        
        group_info = f" [{group_name}]" if group_name else ""
        
        print(f"  📊 Progreso{group_info}: {self.total_sequences_processed:,} secuencias | "
              f"Válidas: {self.total_valid_pim:,} ({self.total_valid_pim/self.total_sequences_processed*100:.1f}%) | "
              f"Rate: {rate:,.0f} seq/s | ETA: {eta_str}")
        
        # Mostrar uso de memoria
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            print(f"  💾 Memoria: {mem_mb:.0f} MB | "
                  f"Almacenadas: {self.total_valid_pim} (muestra)")
        except ImportError:
            pass  # psutil no es crítico
    
    def print_summary(self):
        """Imprime resumen final"""
        print("\n" + "=" * 80)
        print("📊 RESUMEN GLOBAL DE PROCESAMIENTO")
        print("=" * 80)
        
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print(f"  Tiempo total: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"  Total secuencias leídas: {self.total_sequences_processed:,}")
        print(f"  Total PIM válidos: {self.total_valid_pim:,}")
        print(f"  Total rechazados: {self.total_rejected:,}")
        print(f"  Tasa de validez: {self.total_valid_pim/self.total_sequences_processed*100:.2f}%" 
              if self.total_sequences_processed > 0 else "0%")
        print(f"  Total bytes procesados: {self.total_bytes_read / (1024**3):.2f} GB")
        print(f"  Velocidad promedio: {self.total_sequences_processed/elapsed:,.0f} seq/s" 
              if elapsed > 0 else "N/A")
        
        print("\n  📊 DESGLOSE POR GRUPO:")
        print(f"  {'Grupo':<20} {'Total':>14} {'Válidos':>14} {'Rechazados':>14} {'% Válido':>10}")
        print(f"  {'-'*75}")
        
        for group in sorted(self.group_counts.keys()):
            total = self.group_counts[group]
            valid = self.group_valid.get(group, 0)
            rejected = total - valid
            pct = (valid / total * 100) if total > 0 else 0
            print(f"  {get_display_name(group):<20} {total:>14,} {valid:>14,} "
                  f"{rejected:>14,} {pct:>9.2f}%")

# ============================================================================
# CLASE: GrassmannPIM
# ============================================================================

class GrassmannPIM:
    def __init__(self, dim: int = DIM_PAIRS):
        self.dim = dim
    
    def wedge_product(self, v: np.ndarray, w: np.ndarray, with_ci: bool = False) -> Tuple[float, float]:
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
    
    def commutator_norm(self, v: np.ndarray, w: np.ndarray) -> float:
        return commutator_norm(v, w)
    
    def anticommutator_similarity(self, v: np.ndarray, w: np.ndarray) -> float:
        return anticommutator_similarity(v, w)

# ============================================================================
# CLASE: PIMHashIndex
# ============================================================================

class PIMHashIndex:
    def __init__(self, tolerance: float = TOLERANCE):
        self.tolerance = tolerance
        self.index: Dict[str, List[Tuple[str, str, np.ndarray]]] = defaultdict(list)
    
    def add_protein(self, protein_id: str, group: str, vector: np.ndarray):
        h = pim_to_hash(vector, tolerance=self.tolerance)
        self.index[h].append((protein_id, group, vector))
    
    def search(self, vector: np.ndarray) -> List[Tuple[str, str, np.ndarray]]:
        h = pim_to_hash(vector, tolerance=self.tolerance)
        return self.index.get(h, [])
    
    def build_from_samples(self, samples: Dict[str, List[Tuple[str, np.ndarray]]]):
        """Construye índice desde muestras almacenadas"""
        count = 0
        for group_name, sample_list in samples.items():
            for header, vector in sample_list:
                self.add_protein(header, group_name, vector)
                count += 1
        print(f"  ✅ Hash index built: {len(self.index)} unique buckets from {count} proteins")

# ============================================================================
# CLASE: GroupStatistics
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
    total_processed: int = 0
    sample_size: int = 0
    
    def mahalanobis_distance(self, vector: np.ndarray) -> float:
        if self.n_samples <= 1:
            return 1.0
        diff = vector - self.centroid
        return np.sqrt(diff @ self.inv_covariance @ diff)
    
    def probability_of_belonging(self, vector: np.ndarray) -> float:
        if self.n_samples <= 1:
            return 0.5
        d = self.mahalanobis_distance(vector)
        return 1.0 - chi2.cdf(d**2, df=len(self.centroid))

# ============================================================================
# CLASE: AdvancedGroupAnalyzer (MODIFICADA - SIN LÍMITES)
# ============================================================================

class AdvancedGroupAnalyzer:
    def __init__(self, grassmann: GrassmannPIM):
        self.grassmann = grassmann
        self.dim = grassmann.dim
        self.groups: Dict[str, List[np.ndarray]] = {}
        self.group_headers: Dict[str, List[str]] = {}
        self.group_stats: Dict[str, GroupStatistics] = {}
        self.proteins: Dict[str, Tuple[str, np.ndarray]] = {}
        self.adaptive_thresholds: Dict[str, float] = {}
        self.hash_index: Optional[PIMHashIndex] = None
        self.tracker = ProcessingTracker()
        self.start_time = None
        self.sample_size = MAX_STORED_PROTEINS_PER_GROUP
        
        # Almacenamiento de muestras para hash
        self.sample_data: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    
    def set_sample_size(self, size: int):
        """Configura el tamaño de la muestra a almacenar"""
        self.sample_size = size
        print(f"  ⚙️ Tamaño de muestra configurado a: {size:,} proteínas por grupo")
    
    def load_fasta_unlimited(self, filepath: str, group_name: str, verbose: bool = True) -> int:
        """
        Carga archivo FASTA SIN LÍMITES usando procesamiento por lotes
        y muestreo progresivo para mantener memoria controlada
        """
        if verbose:
            print(f"\n  📂 Procesando {get_display_name(group_name)} desde {filepath}...")
        
        if group_name not in self.groups:
            self.groups[group_name] = []
            self.group_headers[group_name] = []
            self.sample_data[group_name] = []
        
        if not os.path.exists(filepath):
            print(f"    ⚠️ Archivo no encontrado: {filepath}")
            return 0
        
        # Inicializar estadísticas online
        stats = OnlineStatistics(self.dim)
        sampler = ProgressiveSampler(self.sample_size)
        
        count_total = 0
        count_valid = 0
        batch_counter = 0
        
        # Leer archivo completo sin límites
        for header, seq in read_fasta_stream(filepath, verbose):
            count_total += 1
            
            # Calcular PIM
            pim_profile = compute_pim_profile(seq, use_weights=USE_WEIGHTS)
            is_valid = np.sum(pim_profile) > 0.01
            
            # Actualizar tracker
            self.tracker.update(group_name, is_valid, len(seq) + len(header))
            
            if is_valid:
                stats.update(pim_profile)
                count_valid += 1
                
                # Muestreo progresivo (solo almacena muestra)
                sampler.add(pim_profile, header[:100])
                
                # Almacenar una pequeña muestra para comparaciones posteriores
                if len(self.groups[group_name]) < self.sample_size:
                    self.groups[group_name].append(pim_profile)
                    self.group_headers[group_name].append(header[:100])
                    protein_name = f"{group_name}|{header[:100]}"
                    self.proteins[protein_name] = (group_name, pim_profile)
            
            # Reporte de progreso periódico
            if verbose and count_total % PROGRESS_REPORT_INTERVAL == 0:
                self.tracker.print_progress(group_name)
                
                # Garbage collection periódico
                if count_total % (PROGRESS_REPORT_INTERVAL * 10) == 0:
                    gc.collect()
        
        # Calcular estadísticas grupales
        centroid = stats.get_mean()
        covariance = stats.get_covariance()
        std_dev = stats.get_std()
        inv_covariance = np.linalg.pinv(covariance + np.eye(self.dim) * 1e-6)
        
        # Cohesión usando la muestra (aumentado de 200 a COHESION_CALC_SAMPLE_SIZE)
        sample_vectors = sampler.get_samples()
        if len(sample_vectors) > 1:
            intra_similarities = []
            sample_size_calc = min(len(sample_vectors), COHESION_CALC_SAMPLE_SIZE)
            for i in range(sample_size_calc):
                for j in range(i+1, sample_size_calc):
                    sim, _ = self.grassmann.wedge_product(sample_vectors[i], sample_vectors[j], with_ci=False)
                    intra_similarities.append(sim)
            wedge_self_similarity = np.mean(intra_similarities) if intra_similarities else 1.0
            wedge_self_similarity_std = np.std(intra_similarities) if len(intra_similarities) > 1 else 0.0
            self.adaptive_thresholds[group_name] = np.percentile(intra_similarities, 5) if len(intra_similarities) > 0 else 0.99
        else:
            wedge_self_similarity = 1.0
            wedge_self_similarity_std = 0.0
            self.adaptive_thresholds[group_name] = 0.99
        
        # Clifford signature
        cliff_sig = self.grassmann.clifford_signature(centroid)
        
        # Subspace projections
        subspace_proj = {}
        for subspace in SUBSPACES.keys():
            if subspace != 'full':
                subspace_proj[subspace] = self.grassmann.interior_product_magnitude(centroid, subspace)
        
        metric_norm, metric_sign = self.grassmann.norm_metric(centroid)
        
        # Guardar estadísticas
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
            metric_sign=metric_sign,
            total_processed=count_total,
            sample_size=len(self.groups[group_name])
        )
        
        # Guardar muestra para hash
        for vec, hdr in zip(sampler.get_samples(), sampler.get_headers()):
            self.sample_data[group_name].append((hdr, vec))
        
        # Resumen del grupo
        stored_count = len(self.groups[group_name])
        metric_info = f", metric_norm={metric_norm:.4f}({'+' if metric_sign>0 else '-' if metric_sign<0 else '0'})" if USE_BIOLOGICAL_METRIC else ""
        print(f"  ✅ {get_display_name(group_name)}: {count_valid:,} válidas de {count_total:,} totales | "
              f"Almacenadas: {stored_count:,} (muestra) | "
              f"Cohesión: {wedge_self_similarity:.6f} | "
              f"Umbral: {self.adaptive_thresholds[group_name]:.4f}{metric_info}")
        
        return count_valid
    
    def load_fasta_file(self, filepath: str, group_name: str, verbose: bool = True) -> int:
        """Wrapper para load_fasta_unlimited (sin límites)"""
        return self.load_fasta_unlimited(filepath, group_name, verbose)
    
    def compute_group_statistics(self):
        """Estadísticas ya calculadas durante la carga, solo imprime resumen"""
        print("\n  📊 Estadísticas grupales calculadas durante la carga")
        self.print_all_group_summary()
    
    def build_hash_index(self):
        """Construye índice hash a partir de las muestras almacenadas"""
        print("\n  🔨 Construyendo índice LSH hash...")
        self.hash_index = PIMHashIndex(tolerance=TOLERANCE)
        self.hash_index.build_from_samples(self.sample_data)
    
    def compare_group_to_all(self, target_group: str) -> pd.DataFrame:
        if target_group not in self.group_stats:
            print(f"  ⚠ Grupo objetivo '{target_group}' no encontrado")
            return pd.DataFrame()
        
        target_stat = self.group_stats[target_group]
        target_centroid = target_stat.centroid
        adaptive_threshold = self.adaptive_thresholds.get(target_group, 0.99)
        
        results = []
        for group_name, stat in self.group_stats.items():
            if group_name == target_group:
                continue
            
            wedge, wedge_std = self.grassmann.wedge_product(target_centroid, stat.centroid, with_ci=True)
            prob = stat.probability_of_belonging(target_centroid)
            is_similar = wedge >= adaptive_threshold
            
            rotor_angles = self.grassmann.all_rotor_angles(target_centroid, stat.centroid)
            reflection = self.grassmann.reflection_analysis(target_centroid, stat.centroid)
            cliff_dist = clifford_distance(target_stat.clifford_signature, stat.clifford_signature) if target_stat.clifford_signature and stat.clifford_signature else 0.0
            
            mag, orient, _ = self.grassmann.wedge_product_oriented(target_centroid, stat.centroid)
            
            comm_norm = self.grassmann.commutator_norm(target_centroid, stat.centroid)
            anticomm_sim = self.grassmann.anticommutator_similarity(target_centroid, stat.centroid)
            metric_sim = self.grassmann.similarity_metric(target_centroid, stat.centroid) if USE_BIOLOGICAL_METRIC else 0.0
            gp_decomp = self.grassmann.geometric_product_decomposition(target_centroid, stat.centroid)
            
            results.append({
                'Compared Group': get_display_name(group_name),
                'Wedge Similarity': round(wedge, 6),
                'Wedge Orientation': round(orient, 6),
                'Probability of Belonging': round(prob, 6),
                'N Samples': stat.n_samples,
                'Total Processed': stat.total_processed,
                'Is Similar (adaptive)': is_similar,
                'Hydrophobic Angle (°)': round(rotor_angles.get('hydrophobic', 0), 2),
                'Charge Angle (°)': round(rotor_angles.get('charge', 0), 2),
                'Specular Reflection': reflection['is_specular_reflection'],
                'Clifford Distance': round(cliff_dist, 6),
                'Commutator Norm': round(comm_norm, 6),
                'Anticommutator Sim': round(anticomm_sim, 6),
                'Metric Similarity': round(metric_sim, 6),
                'GP Functional Sim': round(gp_decomp['functional_similarity'], 6),
                'GP Structural Diff': round(gp_decomp['structural_difference'], 6),
                'GP Combined Sim': round(gp_decomp['combined_similarity'], 6),
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
    
    def classify_protein(self, protein_vector: np.ndarray, protein_name: str = ""):
        if not self.group_stats:
            return pd.DataFrame(), {'is_anomaly': True, 'reason': 'No groups loaded'}
        
        results = []
        for group_name, stat in self.group_stats.items():
            wedge, wedge_std = self.grassmann.wedge_product(protein_vector, stat.centroid, with_ci=True)
            prob = stat.probability_of_belonging(protein_vector)
            adaptive_threshold = self.adaptive_thresholds.get(group_name, 0.99)
            
            rotor_angles = self.grassmann.all_rotor_angles(protein_vector, stat.centroid)
            reflection = self.grassmann.reflection_analysis(protein_vector, stat.centroid)
            
            metric_sim = self.grassmann.similarity_metric(protein_vector, stat.centroid) if USE_BIOLOGICAL_METRIC else 0.0
            gp_decomp = self.grassmann.geometric_product_decomposition(protein_vector, stat.centroid)
            
            results.append({
                'Compared Group': get_display_name(group_name),
                'Wedge Similarity': round(wedge, 6),
                'Metric Similarity': round(metric_sim, 6),
                'GP Functional Sim': round(gp_decomp['functional_similarity'], 6),
                'GP Structural Diff': round(gp_decomp['structural_difference'], 6),
                'GP F/S Ratio': round(gp_decomp['functional_structural_ratio'], 2),
                'GP Interpretation': gp_decomp['interpretation'],
                'Probability of Belonging': round(prob, 6),
                'N Samples': stat.n_samples,
                'Total Processed': stat.total_processed,
                'Is Similar (adaptive)': wedge >= adaptive_threshold,
                'Hydrophobic Angle (°)': round(rotor_angles.get('hydrophobic', 0), 2),
                'Specular Reflection': reflection['is_specular_reflection'],
                'Best Match': False
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('Wedge Similarity', ascending=False)
        
        if len(df) > 0:
            df.iloc[0, df.columns.get_loc('Best Match')] = True
        
        best_sim = df.iloc[0]['Wedge Similarity']
        adaptive_threshold = 0.99
        
        is_anomaly = (best_sim < adaptive_threshold)
        
        anomaly_info = {
            'is_anomaly': is_anomaly,
            'reason': f"Best wedge={best_sim:.4f} (<{adaptive_threshold})" if is_anomaly else "Normal",
            'best_match_group': df.iloc[0]['Compared Group'],
            'best_match_similarity': best_sim
        }
        
        return df, anomaly_info
    
    # ============================================================
    # compute_lujv_statistics - PROCESA TODAS LAS SECUENCIAS
    # ============================================================
    def compute_lujv_statistics(self, target_group: str = 'LUJV') -> dict:
        """Calcula estadísticas de LUJV vs HUMANOS procesando TODAS las secuencias"""
        if target_group not in self.group_stats:
            print(f"  ⚠ Grupo objetivo '{target_group}' no encontrado")
            return {}
        
        target_centroid = self.group_stats[target_group].centroid
        
        print(f"\n  📊 Calculando estadísticas de {target_group} vs proteínas humanas...")
        print(f"  (Procesando TODAS las secuencias de REVIEWED_HUMAN y UNREVIEWED_HUMAN)")
        
        similarities = []
        max_diffs = []
        rotor_angles_hydro = []
        rotor_angles_charge = []
        is_reflection = []
        orientations = []
        commutator_norms = []
        anticommutator_sims = []
        metric_similarities = []
        gp_functional = []
        gp_structural = []
        gp_combined = []
        gp_ratios = []
        
        # SOLO REVIEWED_HUMAN + UNREVIEWED_HUMAN
        human_groups = ['REVIEWED_HUMAN', 'UNREVIEWED_HUMAN']
        
        for group_name in human_groups:
            # Verificar si existe el archivo
            filename = f"{group_name}.unico.dat0"
            if not os.path.exists(filename):
                print(f"  ⚠ Archivo no encontrado: {filename}")
                continue
            
            print(f"  📂 Procesando {group_name} desde {filename}...")
            count = 0
            
            # Leer TODAS las secuencias del archivo
            for header, seq in read_fasta_stream(filename, False):
                vec = compute_pim_profile(seq, use_weights=USE_WEIGHTS)
                if np.sum(vec) > 0.01:
                    # Calcular todas las métricas
                    mag, orient, _ = self.grassmann.wedge_product_oriented(target_centroid, vec)
                    similarities.append(mag)
                    orientations.append(orient)
                    max_diff = self.grassmann.max_component_diff(target_centroid, vec)
                    max_diffs.append(max_diff)
                    rotor_angles_hydro.append(self.grassmann.rotor_angle(target_centroid, vec, 'hydrophobic'))
                    rotor_angles_charge.append(self.grassmann.rotor_angle(target_centroid, vec, 'charge'))
                    is_ref, _ = self.grassmann.is_specular_reflection(target_centroid, vec)
                    is_reflection.append(is_ref)
                    commutator_norms.append(self.grassmann.commutator_norm(target_centroid, vec))
                    anticommutator_sims.append(self.grassmann.anticommutator_similarity(target_centroid, vec))
                    metric_similarities.append(self.grassmann.similarity_metric(target_centroid, vec))
                    
                    gp_decomp = self.grassmann.geometric_product_decomposition(target_centroid, vec)
                    gp_functional.append(gp_decomp['functional_similarity'])
                    gp_structural.append(gp_decomp['structural_difference'])
                    gp_combined.append(gp_decomp['combined_similarity'])
                    gp_ratios.append(gp_decomp['functional_structural_ratio'])
                    
                    count += 1
                    
                    # Reporte de progreso
                    if count % 100000 == 0:
                        print(f"     Procesadas {count:,} secuencias válidas de {group_name}...")
            
            print(f"     ✅ {group_name}: {count:,} secuencias válidas procesadas")
        
        if not similarities:
            print(f"  ⚠ No se procesaron secuencias válidas para {target_group}")
            return {}
        
        similarities = np.array(similarities)
        max_diffs = np.array(max_diffs)
        
        adaptive_threshold = self.adaptive_thresholds.get(target_group, 0.99)
        
        orientation_signs = np.sign(orientations)
        pct_positive_orientation = np.mean(orientation_signs > 0) * 100
        pct_negative_orientation = np.mean(orientation_signs < 0) * 100
        
        print(f"\n  ✅ Estadísticas completas:")
        print(f"     ├─ Total proteínas humanas analizadas: {len(similarities):,}")
        print(f"     ├─ Similitud máxima: {np.max(similarities):.6f}")
        print(f"     └─ Similitud media: {np.mean(similarities):.6f}")
        
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
            'mean_commutator_norm': np.mean(commutator_norms) if commutator_norms else 0,
            'mean_anticommutator_sim': np.mean(anticommutator_sims) if anticommutator_sims else 0,
            'mean_metric_similarity': np.mean(metric_similarities) if metric_similarities else 0,
            'mean_gp_functional': np.mean(gp_functional) if gp_functional else 0,
            'mean_gp_structural': np.mean(gp_structural) if gp_structural else 0,
            'mean_gp_combined': np.mean(gp_combined) if gp_combined else 0,
            'mean_gp_ratio': np.mean(gp_ratios) if gp_ratios else 0,
            'total_human_proteins': len(similarities)
        }
    
    def print_top_similar_proteins(self, target_group: str, 
                                    groups_to_search: List[str],
                                    n_top: int = TOP_N_PROTEINS):
        print(f"\n{'='*80}")
        print(f"🔍 TOP {n_top} PROTEÍNAS MÁS SIMILARES A {get_display_name(target_group)}")
        print(f"{'='*80}")
        
        if target_group not in self.group_stats:
            print(f"  ❌ Grupo {target_group} no encontrado")
            return pd.DataFrame()
        
        target_centroid = self.group_stats[target_group].centroid
        
        similarities = []
        for group_name in groups_to_search:
            if group_name not in self.groups:
                continue
            
            vectors = self.groups[group_name]
            headers = self.group_headers[group_name]
            
            for i, vec in enumerate(vectors):
                mag, orient, _ = self.grassmann.wedge_product_oriented(target_centroid, vec)
                rotor_hydro = self.grassmann.rotor_angle(target_centroid, vec, 'hydrophobic')
                is_ref, _ = self.grassmann.is_specular_reflection(target_centroid, vec)
                metric_sim = self.grassmann.similarity_metric(target_centroid, vec) if USE_BIOLOGICAL_METRIC else 0.0
                gp_decomp = self.grassmann.geometric_product_decomposition(target_centroid, vec)
                
                similarities.append({
                    'Rank': 0,
                    'Group': get_display_name(group_name),
                    'Protein ID': headers[i] if i < len(headers) else f"protein_{i}",
                    'Wedge Similarity': round(mag, 6),
                    'Orientation': round(orient, 6),
                    'Metric Similarity': round(metric_sim, 6),
                    'GP Functional': round(gp_decomp['functional_similarity'], 6),
                    'GP Structural': round(gp_decomp['structural_difference'], 6),
                    'GP Ratio': round(gp_decomp['functional_structural_ratio'], 2),
                    'GP Interp': gp_decomp['interpretation'],
                    'Hydro Angle (°)': round(rotor_hydro, 2),
                    'Specular': is_ref
                })
        
        similarities.sort(key=lambda x: x['Wedge Similarity'], reverse=True)
        
        for i, item in enumerate(similarities[:n_top]):
            item['Rank'] = i + 1
        
        df = pd.DataFrame(similarities[:n_top])
        
        if df.empty:
            print("  No similar proteins found")
            return df
        
        print(f"\n  {'Rank':<5} {'Group':<18} {'Wedge':>10} {'Orient':>7} {'GP Funct':>10} {'GP Struct':>10} {'GP Ratio':>9} {'Hydro':>8}")
        print(f"  {'-'*95}")
        
        for _, row in df.iterrows():
            protein_id = row['Protein ID']
            if len(protein_id) > 35:
                protein_id = protein_id[:32] + "..."
            orient_str = f"{row['Orientation']:+.3f}"
            print(f"  {row['Rank']:<5} {row['Group']:<18} {row['Wedge Similarity']:>10.6f} {orient_str:>7} "
                  f"{row['GP Functional']:>10.6f} {row['GP Structural']:>10.6f} "
                  f"{row['GP Ratio']:>9.2f} {row['Hydro Angle (°)']:>8.1f}")
        
        return df
    
    def print_all_group_summary(self):
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE TODOS LOS GRUPOS")
        print("=" * 80)
        print(f"\n  {'Grupo':<20} {'Procesados':>14} {'Válidos':>14} {'Almacenados':>14} {'Cohesión':>18} {'Umbral':>10}")
        print(f"  {'-'*95}")
        
        for group_name, stats in sorted(self.group_stats.items(), key=lambda x: x[1].total_processed, reverse=True):
            display_name = get_display_name(group_name)
            stored = len(self.groups.get(group_name, []))
            print(f"  {display_name:<20} {stats.total_processed:>14,} {stats.n_samples:>14,} "
                  f"{stored:>14,} {stats.wedge_self_similarity:>12.6f} "
                  f"{stats.adaptive_threshold:>10.4f}")
    
    def print_metric_info(self):
        print("\n" + "=" * 80)
        print("📐 METRIC SIGNATURE INFORMATION (Clifford Metric)")
        print("=" * 80)
        
        info = self.grassmann.metric_signature_info()
        
        print(f"\n  Metric type: {'Biological' if info['is_biological'] else 'Euclidean'}")
        print(f"  Total components: {info['total_components']}")
        print(f"  Positive (beneficial): {info['positive_count']}")
        print(f"  Negative (detrimental): {info['negative_count']}")
        print(f"  Neutral: {info['neutral_count']}")
        
        if info['beneficial_interactions']:
            print(f"\n  ✅ Beneficial interactions (+1):")
            for inter in info['beneficial_interactions'][:8]:
                print(f"     ├─ {inter}")
            if len(info['beneficial_interactions']) > 8:
                print(f"     └─ ... and {len(info['beneficial_interactions'])-8} more")
        
        if info['detrimental_interactions']:
            print(f"\n  ❌ Detrimental interactions (-1):")
            for inter in info['detrimental_interactions']:
                print(f"     ├─ {inter}")
        
        if info['neutral_interactions']:
            print(f"\n  ⚪ Neutral interactions (0):")
            for inter in info['neutral_interactions'][:5]:
                print(f"     ├─ {inter}")
            if len(info['neutral_interactions']) > 5:
                print(f"     └─ ... and {len(info['neutral_interactions'])-5} more")
    
    def print_processing_summary(self):
        """Imprime resumen completo del procesamiento"""
        self.tracker.print_summary()
        
        print("\n  📊 ALMACENAMIENTO POR GRUPO:")
        print(f"  {'Grupo':<20} {'Procesados':>14} {'Válidos':>14} {'Almacenados':>14} {'% Muestra':>12}")
        print(f"  {'-'*75}")
        for group_name in self.group_stats:
            stats = self.group_stats[group_name]
            stored = len(self.groups.get(group_name, []))
            pct = (stored / stats.n_samples * 100) if stats.n_samples > 0 else 0
            print(f"  {get_display_name(group_name):<20} {stats.total_processed:>14,} {stats.n_samples:>14,} "
                  f"{stored:>14,} {pct:>11.2f}%")

# ============================================================================
# FUNCIONES DE PLOTTING (CORREGIDAS)
# ============================================================================

def plot_similarity_distribution(similarities: np.ndarray, save_path: str, adaptive_threshold: float = None):
    if len(similarities) == 0:
        print(f"  ⚠ No similarities to plot for {save_path}")
        return
    
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
    plt.title('Similarity Distribution: LUJV vs Human Proteome', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Figure saved: {save_path}")

def plot_orientation_distribution(orientations: List[float], save_path: str):
    if len(orientations) == 0:
        print(f"  ⚠ No orientations to plot for {save_path}")
        return
    
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
    if len(max_diffs) == 0:
        print(f"  ⚠ No max_diffs to plot for {save_path}")
        return
    
    plt.figure(figsize=(12, 6))
    plt.hist(max_diffs, bins=50, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(x=tolerance, color='red', linestyle='--', linewidth=2, label=f'Tolerance = {tolerance}')
    plt.xlabel('Maximum Component Difference', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Maximum Component Differences', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Figure saved: {save_path}")

def plot_tolerance_summary(stats: dict, save_path: str, tolerance: float):
    if not stats or 'count_sim_099' not in stats:
        print(f"  ⚠ No statistics to plot for {save_path}")
        return
    
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
    
    if 'total_human_proteins' in stats:
        ax2.text(0.5, -0.25, f"Total human proteins analyzed: {stats['total_human_proteins']:,}",
                 transform=ax2.transAxes, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Figure saved: {save_path}")

def plot_rotor_angles(angles_hydro: List[float], angles_charge: List[float], save_path: str):
    if len(angles_hydro) == 0 or len(angles_charge) == 0:
        print(f"  ⚠ No rotor angles to plot for {save_path}")
        return
    
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

# ============================================================================
# FUNCIONES DE PLOTTING CORREGIDAS (Higher = More Similar)
# ============================================================================

def plot_lujv_vs_all_groups(sim_matrix: pd.DataFrame, results_dir: str):
    """
    Figura 8: LUJV vs ALL Groups - Wedge Similarity (bar chart)
    CORREGIDO: Higher = More Similar
    """
    if sim_matrix.empty or 'LUJV' not in sim_matrix.index:
        print("  ⚠ No se puede generar la figura LUJV vs ALL Groups")
        return
    
    lujv_row = sim_matrix.loc['LUJV']
    
    # CORREGIDO: Ordenar descendente (mayor similitud primero)
    # Higher = More Similar
    lujv_sorted = lujv_row.sort_values(ascending=False)
    
    # Excluir LUJV vs LUJV (diagonal = 0)
    lujv_sorted = lujv_sorted[lujv_sorted.index != 'LUJV']
    
    if len(lujv_sorted) == 0:
        print("  ⚠ No hay datos para LUJV vs ALL Groups")
        return
    
    plt.figure(figsize=(14, 8))
    
    colors = []
    display_labels = []
    for group in lujv_sorted.index:
        display_labels.append(get_display_name(group))
        if group in ['LASV', 'JUNV', 'MACV', 'LCMV']:
            colors.append('#e74c3c')
        elif group in ['CPP', 'NON_CPP', 'UNFOLDED', 'PARTIALLY_FOLDED']:
            colors.append('#3498db')
        elif group in ['REVIEWED_HUMAN', 'UNREVIEWED_HUMAN']:
            colors.append('#2ecc71')
        elif group in ['VIRUS_REVIEWED', 'VIRUS_UNREVIEWED']:
            colors.append('#f39c12')
        elif group in ['REVIEWED_ALL', 'UNREVIEWED_ALL']:
            colors.append('#9b59b6')
        elif group in ['senales', 'membrana', 'enfermedad']:
            colors.append('#1abc9c')
        else:
            colors.append('#95a5a6')
    
    bars = plt.barh(display_labels, lujv_sorted.values, color=colors, edgecolor='black', alpha=0.8)
    
    for bar, val in zip(bars, lujv_sorted.values):
        plt.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', ha='left', fontsize=9)
    
    # CORREGIDO: Higher = More Similar
    plt.xlabel('Wedge Similarity (∧) - Higher = More Similar', fontsize=12)
    plt.ylabel('Protein Group', fontsize=12)
    plt.title('LUJV vs All UniProt Groups: Wedge Similarity', fontsize=14)
    # Las líneas de umbral son orientativas, no cambian con la corrección
    plt.axvline(x=0.05, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label='Threshold 0.05')
    plt.axvline(x=0.10, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                label='Threshold 0.10')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    save_path = f"{results_dir}/08_lujv_vs_all_groups.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ New figure saved: {save_path}")
    
    # Guardar CSV con nombres de exhibición
    csv_path = f"{results_dir}/lujv_vs_all_groups.csv"
    df_out = pd.DataFrame({
        'Group': display_labels,
        'Wedge Similarity': lujv_sorted.values
    })
    df_out.to_csv(csv_path, index=False)
    print(f"    ✅ Data saved: {csv_path}")

def plot_lujv_heatmap(sim_matrix: pd.DataFrame, results_dir: str):
    """
    Heatmap: LUJV vs Other Groups
    CORREGIDO: Higher = More Similar
    """
    if sim_matrix.empty or 'LUJV' not in sim_matrix.index:
        print("  ⚠ No se puede generar el heatmap")
        return
    
    lujv_row = sim_matrix.loc['LUJV'].sort_values(ascending=False)
    
    target_groups = ['LASV', 'JUNV', 'MACV', 'LCMV', 
                     'VIRUS_REVIEWED', 'VIRUS_UNREVIEWED',
                     'REVIEWED_HUMAN', 'UNREVIEWED_HUMAN',
                     'REVIEWED_ALL', 'UNREVIEWED_ALL']
    target_groups = [g for g in target_groups if g in lujv_row.index]
    
    if not target_groups:
        print("  ⚠ No se encontraron grupos objetivo")
        return
    
    display_labels = [get_display_name(g) for g in target_groups]
    
    plt.figure(figsize=(16, 4))
    data = lujv_row[target_groups].values.reshape(1, -1)
    
    # CORREGIDO: Higher = More Similar
    # Usar vmax=0.20 para dar margen a valores como CPP=0.117
    sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlBu_r',
                xticklabels=display_labels, yticklabels=['LUJV'],
                cbar_kws={'label': 'Wedge Similarity (∧)'},
                vmin=0, vmax=0.20)
    
    # CORREGIDO: Higher = More Similar
    plt.title('LUJV vs Other Groups: Wedge Similarity\n(Higher = More Similar, Red = Closest)', fontsize=14)
    plt.tight_layout()
    
    save_path = f"{results_dir}/heatmap_LUJV_vs_groups.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✅ Heatmap guardado: {save_path}")
    
    csv_path = f"{results_dir}/lujv_vs_groups.csv"
    pd.DataFrame(data, columns=display_labels, index=['LUJV']).to_csv(csv_path)
    print(f"    ✅ Datos guardados: {csv_path}")
    
    print("\n  📊 LUJV vs Other Groups (Wedge Similarity):")
    for i, (group, val) in enumerate(zip(display_labels, data[0])):
        print(f"     ├─ {group}: {val:.6f}")

def print_classification_table(df: pd.DataFrame, anomaly_info: dict, protein_name: str):
    print(f"\n{'='*90}")
    print(f"🔍 CLASSIFICATION: {protein_name[:50]}")
    print(f"{'='*90}")
    
    if anomaly_info['is_anomaly']:
        print(f"\n  ⚠️ ANOMALY DETECTED: {anomaly_info['reason']}")
    else:
        print(f"\n  ✅ NORMAL PROTEIN")
        print(f"     └─ Best match: {anomaly_info['best_match_group']} "
              f"(similarity = {anomaly_info['best_match_similarity']:.6f})")
    
    if df.empty:
        print("  No classification data")
        return
    
    print(f"\n  {'Group':<18} {'Wedge':>10} {'GP Funct':>10} {'GP Struct':>10} {'GP Ratio':>9} {'Prob':>12} {'Similar?':>10}")
    print(f"  {'-'*90}")
    
    for _, row in df.head(10).iterrows():
        best_marker = "🏆" if row.get('Best Match', False) else "  "
        similar_mark = "✅" if row.get('Is Similar (adaptive)', False) else "❌"
        print(f"  {best_marker} {row['Compared Group']:<16} {row['Wedge Similarity']:>10.6f} "
              f"{row['GP Functional Sim']:>10.6f} {row['GP Structural Diff']:>10.6f} "
              f"{row['GP F/S Ratio']:>9.2f} {row['Probability of Belonging']:>12.6f} {similar_mark:>10}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 80)
    print("🦠 MIRROR-PIM: GRASSMANN-PIM WITH REAL GEOMETRIC ALGEBRA (v13.1)")
    print("   PROCESAMIENTO COMPLETO SIN LÍMITES")
    print("   Monitoreo en tiempo real + Gestión de memoria optimizada")
    print("   CORREGIDO: Visualización de figuras (Higher = More Similar)")
    print("=" * 80)
    print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n  ⚙️ CONFIGURACIÓN DE PROCESAMIENTO:")
    print(f"     ├─ Tamaño de lote: {BATCH_SIZE:,} secuencias")
    print(f"     ├─ Muestra por grupo: {MAX_STORED_PROTEINS_PER_GROUP:,} proteínas")
    print(f"     ├─ Muestra para cohesión: {COHESION_CALC_SAMPLE_SIZE} proteínas")
    print(f"     ├─ Intervalo de reporte: {PROGRESS_REPORT_INTERVAL:,} secuencias")
    print(f"     └─ Sin límites de secuencias por grupo")
    
    dim = DIM_PAIRS
    print(f"\n  ⚙ CONFIGURACIÓN DEL SISTEMA:")
    print(f"     ├─ Dimensión del espacio: {dim} componentes")
    print(f"     ├─ Ponderación biológica: {'SÍ' if USE_WEIGHTS else 'NO'}")
    print(f"     ├─ Modo streaming: SÍ (sin límites)")
    print(f"     ├─ Clifford rotors: 6 planos biológicos")
    print(f"     ├─ Producto Geométrico con Métrica: SÍ")
    print(f"     └─ Álgebra Geométrica Real: v13.1")
    
    grassmann = GrassmannPIM(dim=dim)
    analyzer = AdvancedGroupAnalyzer(grassmann)
    
    # Configurar tamaño de muestra
    analyzer.set_sample_size(MAX_STORED_PROTEINS_PER_GROUP)
    
    # Archivos a procesar (sin límites)
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
    
    print("\n📂 CARGANDO ARCHIVOS FASTA (SIN LÍMITES)...")
    print("=" * 80)
    
    analyzer.start_time = datetime.now()
    analyzer.tracker.start_time = analyzer.start_time
    
    total_valid = 0
    total_processed = 0
    
    for group_name, filename in files_to_load.items():
        count = analyzer.load_fasta_file(filename, group_name, verbose=True)
        total_valid += count
        # Actualizar tracker con totales del grupo
        if group_name in analyzer.tracker.group_counts:
            total_processed += analyzer.tracker.group_counts[group_name]
    
    # Imprimir resumen final del tracker
    analyzer.tracker.print_summary()
    
    # Mostrar resumen de almacenamiento
    analyzer.print_processing_summary()
    
    # Construir hash index
    analyzer.build_hash_index()
    
    # Crear directorio de resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results_mirrorpim_v13_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    if USE_BIOLOGICAL_METRIC:
        analyzer.print_metric_info()
    
    print("\n" + "=" * 80)
    print("📈 GENERANDO FIGURAS...")
    print("=" * 80)
    
    target_group = 'LUJV' if 'LUJV' in analyzer.group_stats else list(analyzer.group_stats.keys())[0]
    print(f"\n  Usando '{target_group}' como grupo de referencia")
    
    # Calcular estadísticas LUJV vs Humanos (procesando TODAS las secuencias)
    lujv_stats = analyzer.compute_lujv_statistics(target_group)
    
    if lujv_stats and len(lujv_stats.get('similarities', [])) > 0:
        total_human = len(lujv_stats['similarities'])
        print(f"\n  📊 Procesadas {total_human:,} proteínas humanas")
        
        plot_similarity_distribution(
            lujv_stats['similarities'],
            f"{results_dir}/01_similarity_distribution.png",
            adaptive_threshold=lujv_stats.get('adaptive_threshold')
        )
        
        if 'orientations' in lujv_stats and len(lujv_stats['orientations']) > 0:
            plot_orientation_distribution(
                lujv_stats['orientations'],
                f"{results_dir}/01b_orientation_distribution.png"
            )
        
        if 'max_diffs' in lujv_stats and len(lujv_stats['max_diffs']) > 0:
            plot_max_diff_distribution(
                lujv_stats['max_diffs'],
                f"{results_dir}/02_max_diff_distribution.png",
                TOLERANCE
            )
        
        if 'count_sim_099' in lujv_stats:
            plot_tolerance_summary(
                lujv_stats,
                f"{results_dir}/03_tolerance_summary.png",
                TOLERANCE
            )
        
        if 'rotor_angles_hydro' in lujv_stats and len(lujv_stats['rotor_angles_hydro']) > 0:
            plot_rotor_angles(
                lujv_stats['rotor_angles_hydro'],
                lujv_stats['rotor_angles_charge'],
                f"{results_dir}/04_rotor_angles_distribution.png"
            )
    else:
        print(f"  ⚠ Datos insuficientes para generar figuras")
    
    analyzer.print_all_group_summary()
    
    print("\n" + "=" * 80)
    print(f"🔍 COMPARANDO {target_group} CON TODOS LOS GRUPOS")
    print("=" * 80)
    
    comparison = analyzer.compare_group_to_all(target_group)
    if comparison is not None and not comparison.empty:
        filename = f"{results_dir}/comparison_{target_group}_vs_all.csv"
        comparison.to_csv(filename, index=False)
        print(f"\n  📊 {target_group} vs Otros Grupos:")
        print(comparison[['Compared Group', 'Wedge Similarity', 'Wedge Orientation', 
                         'Commutator Norm', 'Anticommutator Sim', 'N Samples']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("🔍 MATRIZ DE SIMILITUD INTER-GRUPOS")
    print("=" * 80)
    
    sim_matrix = analyzer.cross_group_similarity_matrix()
    if not sim_matrix.empty:
        sim_matrix.to_csv(f"{results_dir}/similarity_matrix_groups.csv")
        print("\n  📊 Matriz de similitud inter-grupos:")
        print(sim_matrix.round(4).to_string())
        
        plot_lujv_heatmap(sim_matrix, results_dir)
        plot_lujv_vs_all_groups(sim_matrix, results_dir)
    
    print("\n" + "=" * 80)
    print(f"🔍 TOP {TOP_N_PROTEINS} PROTEÍNAS MÁS SIMILARES A {target_group}")
    print("=" * 80)
    
    groups_to_search = ['REVIEWED_HUMAN', 'UNREVIEWED_HUMAN', 'REVIEWED_ALL', 'UNREVIEWED_ALL',
                        'senales', 'membrana', 'enfermedad', 'CPP', 'UNFOLDED', 'PARTIALLY_FOLDED']
    top_proteins_df = analyzer.print_top_similar_proteins(target_group, groups_to_search, TOP_N_PROTEINS)
    
    if not top_proteins_df.empty:
        top_proteins_df.to_csv(f"{results_dir}/top_similar_proteins_{target_group}.csv", index=False)
        print(f"\n  ✅ Archivo guardado: {results_dir}/top_similar_proteins_{target_group}.csv")
    
    print("\n" + "=" * 80)
    print("🔍 DETECCIÓN DE ANOMALÍAS")
    print("=" * 80)
    
    synthetic_protein = np.random.rand(analyzer.dim)
    synthetic_protein = synthetic_protein / np.sum(synthetic_protein)
    classification, anomaly = analyzer.classify_protein(synthetic_protein, "SYNTHETIC_PROTEIN_TEST")
    print_classification_table(classification, anomaly, "SYNTHETIC_PROTEIN_TEST")
    
    if target_group in analyzer.group_stats:
        target_centroid = analyzer.group_stats[target_group].centroid
        classification_target, anomaly_target = analyzer.classify_protein(target_centroid, f"{target_group}_CENTROID")
        print_classification_table(classification_target, anomaly_target, f"{target_group}_CENTROID")
        classification_target.to_csv(f"{results_dir}/classification_{target_group}.csv", index=False)
    
    print("\n" + "=" * 80)
    print("🎯 DESCOMPOSICIÓN DEL PRODUCTO GEOMÉTRICO")
    print("=" * 80)
    
    if lujv_stats:
        print(f"\n  {target_group} vs Proteoma Humano - Análisis de Producto Geométrico:")
        print(f"     ├─ Similitud Funcional Media: {lujv_stats.get('mean_gp_functional', 0):.6f}")
        print(f"     ├─ Diferencia Estructural Media: {lujv_stats.get('mean_gp_structural', 0):.6f}")
        print(f"     ├─ Similitud Combinada Media: {lujv_stats.get('mean_gp_combined', 0):.6f}")
        print(f"     ├─ Ratio Funcional/Estructural: {lujv_stats.get('mean_gp_ratio', 0):.2f}")
        print(f"     ├─ Norma del Conmutador Media: {lujv_stats.get('mean_commutator_norm', 0):.6f}")
        print(f"     └─ Similitud Métrica Media: {lujv_stats.get('mean_metric_similarity', 0):.6f}")
    
    print("\n" + "=" * 80)
    print("✅ EJECUCIÓN COMPLETADA")
    print("=" * 80)
    print(f"\n  📁 Resultados guardados en: {results_dir}/")
    
    print(f"\n  📊 ESTADÍSTICAS FINALES:")
    if lujv_stats:
        print(f"     ├─ Total proteínas humanas analizadas: {lujv_stats.get('total_human_proteins', 0):,}")
        print(f"     ├─ Similitud máxima con LUJV: {lujv_stats.get('max_similarity', 0):.6f}")
        print(f"     ├─ Similitud media con LUJV: {lujv_stats.get('mean_similarity', 0):.6f}")
        print(f"     ├─ Orientación positiva: {lujv_stats.get('pct_positive_orientation', 0):.1f}%")
        print(f"     ├─ Proteínas ≥ 0.99: {lujv_stats.get('count_sim_099', 0):,}")
        print(f"     ├─ Proteínas ≥ 0.95: {lujv_stats.get('count_sim_095', 0):,}")
        print(f"     └─ Proteínas ≥ 0.90: {lujv_stats.get('count_sim_090', 0):,}")
    
    print(f"\n  🎯 OPERADORES DE ÁLGEBRA GEOMÉTRICA IMPLEMENTADOS (v13.1):")
    print(f"     ├─ ✅ Producto exterior orientado (∧) - Preserva signo")
    print(f"     ├─ ✅ Reflexión especular real - v' = v - 2(v·n)n")
    print(f"     ├─ ✅ Producto interior (⌋) - Proyecciones a subespacios")
    print(f"     ├─ ✅ Conmutador [v,w] - No conmutatividad")
    print(f"     ├─ ✅ Anticonmutador {{v,w}} - Simetría métrica")
    print(f"     ├─ ✅ Producto punto métrico - Firma biológica (η)")
    print(f"     ├─ ✅ Producto Geométrico con Métrica - v *η w")
    print(f"     ├─ ✅ LSH hash: Búsqueda instantánea O(1)")
    print(f"     ├─ ✅ Ponderación biológica")
    print(f"     ├─ ✅ Umbral adaptativo")
    print(f"     ├─ ✅ Bootstrapping")
    print(f"     ├─ ✅ Análisis Δ-PIM")
    print(f"     ├─ ✅ Clifford rotors (6 planos biológicos)")
    print(f"     ├─ ✅ Clasificación y Detección de Anomalías")
    print(f"     ├─ ✅ PROCESAMIENTO SIN LÍMITES")
    print(f"     ├─ ✅ MONITOREO EN TIEMPO REAL")
    print(f"     ├─ ✅ GESTIÓN DE MEMORIA OPTIMIZADA")
    print(f"     └─ ✅ FIGURAS CORREGIDAS (Higher = More Similar)")
    
    print(f"\n⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Tiempo total de ejecución: {(datetime.now() - analyzer.start_time).total_seconds()/60:.1f} minutos")

if __name__ == "__main__":
    main()
