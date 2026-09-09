#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN: SGPMAIN6.2v.py vs BioPython
VERSIÓN CORREGIDA - EVALUANDO EL VIRUS LUJO
USANDO ARCHIVO todolujo.txt
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
import json
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 COMPARACIÓN: SGPMAIN6.2v.py vs BioPython")
print("   Análisis del virus LUJO")
print("   📌 USANDO FUNCIONES INTERNAS DE SGPMAIN6.2v.py")
print("   📌 ARCHIVO DE ENTRADA: todolujo.txt")
print("=" * 80)
print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# IMPORTAR BIOPYTHON
# ============================================================================

try:
    from Bio.SeqUtils import ProtParam
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
    print("✅ BioPython importado correctamente")
except ImportError:
    print("❌ BioPython no está instalado. Ejecuta: pip install biopython")
    sys.exit(1)

# ============================================================================
# IMPORTAR FUNCIONES DE SGPMAIN6.2v.py
# ============================================================================

SGPMAIN_PATH = "SGPMAIN6.2v.py"

import importlib.util

def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    sgp_module = import_from_file("sgpmain", SGPMAIN_PATH)
    print("✅ SGPMAIN6.2v.py importado correctamente")
    
    # Funciones principales
    compute_pim_profile = sgp_module.compute_pim_profile
    compute_trimer_profile = sgp_module.compute_trimer_profile
    shannon_entropy_true = sgp_module.shannon_entropy_true
    gini_coefficient_true = sgp_module.gini_coefficient_true
    structural_complexity_true = sgp_module.structural_complexity_true
    functional_modularity_true = sgp_module.functional_modularity_true
    morans_i_true = sgp_module.morans_i_true
    hellinger_distance_true = sgp_module.hellinger_distance_true
    spearman_correlation_true = sgp_module.spearman_correlation_true
    jensen_shannon_divergence_true = sgp_module.jensen_shannon_divergence_true
    wasserstein_distance_true = sgp_module.wasserstein_distance_true
    fractal_dimension_true = sgp_module.fractal_dimension_true
    radon_transform_true = sgp_module.radon_transform_true
    radon_similarity_true = sgp_module.radon_similarity_true
    
    # Grassmann
    grassmann_projection_distance_true = sgp_module.grassmann_projection_distance_true
    grassmann_fubini_study_true = sgp_module.grassmann_fubini_study_true
    grassmann_ricci_curvature_true = sgp_module.grassmann_ricci_curvature_true
    grassmann_geodesic_true = sgp_module.grassmann_geodesic_true
    grassmann_distance = sgp_module.grassmann_distance
    
    # Hodge
    hodge_dual_true_complete = sgp_module.hodge_dual_true_complete
    hodge_complementarity_true = sgp_module.hodge_complementarity_true
    
    # Clifford
    clifford_signature_true = sgp_module.clifford_signature_true
    clifford_distance_true = sgp_module.clifford_distance_true
    
    # Geometric product
    geometric_product_complete = sgp_module.geometric_product_complete
    geometric_product_full = sgp_module.geometric_product_full
    
    # Rotors
    rotor_angle_true = sgp_module.rotor_angle_true
    all_rotor_angles = sgp_module.all_rotor_angles_true
    general_rotor = sgp_module.general_rotor
    
    # Polarity Laplacian
    polarity_interaction_laplacian_true = sgp_module.polarity_interaction_laplacian_true
    
    # Karhunen-Loève
    karhunen_loeve_decomposition_true = sgp_module.karhunen_loeve_decomposition_true
    
    # Constantes
    POLARITY_MAP = sgp_module.POLARITY_MAP
    INTERACTIONS = sgp_module.INTERACTIONS
    INTERACTION_TO_IDX = sgp_module.INTERACTION_TO_IDX
    ROTOR_PLANES = sgp_module.ROTOR_PLANES
    BIOLOGICAL_WEIGHTS = sgp_module.BIOLOGICAL_WEIGHTS
    METRIC_SIGNATURE = sgp_module.METRIC_SIGNATURE
    
    print("✅ Todas las funciones de SGPMAIN6.2v.py disponibles")
        
except Exception as e:
    print(f"❌ Error importando SGPMAIN6.2v.py: {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN DEL VIRUS LUJO
# ============================================================================

LUJO_VIRUS = {
    'name': 'lujo',
    'display': 'LUJO',
    'pattern': r'Lujo|LUJO|lujo|LUJ|Luj|Arenavirus|lujo virus|Lujo virus'
}

# ============================================================================
# CONFIGURACIÓN DE GRUPOS
# ============================================================================

DATA_PATH = "/home/cpolanco/POLANCO/ARCHIVOMAESTRO"

GROUP_FILES = {
    'CPP': os.path.join(DATA_PATH, 'CPP.unico.dat0'),
    'NON_CPP': os.path.join(DATA_PATH, 'NONCPP.unico.dat0'),
    'UNFOLDED': os.path.join(DATA_PATH, 'unfolded.unico.dat0'),
    'PARTIALLY_FOLDED': os.path.join(DATA_PATH, 'partiallyorderedN.unico.dat0'),
    'REVIEWED_HUMAN': os.path.join(DATA_PATH, 'reviewed_human.unico.dat0'),
    'UNREVIEWED_HUMAN': os.path.join(DATA_PATH, 'unreviewed_human.unico.dat0'),
    'VIRUS_REVIEWED': os.path.join(DATA_PATH, 'reviewed_virus.unico.dat0'),
    'VIRUS_UNREVIEWED': os.path.join(DATA_PATH, 'unreviewed_virus.unico.dat0'),
    'REVIEWED_ALL': os.path.join(DATA_PATH, 'reviewed_all.unico.dat0'),
    'UNREVIEWED_ALL': os.path.join(DATA_PATH, 'unreviewed_all.unico.dat0'),
    'lujo': os.path.join(DATA_PATH, 'lujo.unico.dat0'),
    'lasv': os.path.join(DATA_PATH, 'lasv_all.unico.dat0'),
    'junv': os.path.join(DATA_PATH, 'junv_all.unico.dat0'),
    'macv': os.path.join(DATA_PATH, 'macv_all.unico.dat0'),
    'lcmv': os.path.join(DATA_PATH, 'lcmv_all.unico.dat0'),
    'nile1': os.path.join(DATA_PATH, 'nile1.unico.dat0'),
    'nile2': os.path.join(DATA_PATH, 'nile2.unico.dat0'),
    'enfermedad': os.path.join(DATA_PATH, 'enfermedad.unico.dat0'),
    'membrana': os.path.join(DATA_PATH, 'membrana.unico.dat0'),
    'senales': os.path.join(DATA_PATH, 'senales.unico.dat0'),
}

DISPLAY_NAMES = {
    'CPP': 'CPP',
    'NON_CPP': 'NON_CPP',
    'UNFOLDED': 'UNFOLDED',
    'PARTIALLY_FOLDED': 'PARTIALLY_FOLDED',
    'REVIEWED_HUMAN': 'REVIEWED_HUMAN',
    'UNREVIEWED_HUMAN': 'UNREVIEWED_HUMAN',
    'VIRUS_REVIEWED': 'VIRUS_REVIEWED',
    'VIRUS_UNREVIEWED': 'VIRUS_UNREVIEWED',
    'REVIEWED_ALL': 'REVIEWED_ALL',
    'UNREVIEWED_ALL': 'UNREVIEWED_ALL',
    'lujo': 'LUJO',
    'lasv': 'LASV',
    'junv': 'JUNV',
    'macv': 'MACV',
    'lcmv': 'LCMV',
    'nile1': 'NILE1',
    'nile2': 'NILE2',
    'enfermedad': 'DISEASE',
    'membrana': 'MEMBRANE',
    'senales': 'SIGNALS',
}

GROUPS_TO_ANALYZE = list(GROUP_FILES.keys())

# ============================================================================
# FUNCIONES DE LECTURA DE FASTA
# ============================================================================

def read_fasta_stream(filepath):
    """Lee archivo FASTA secuencialmente"""
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header is not None:
            yield header, ''.join(seq)

def get_filename(group_name):
    return GROUP_FILES.get(group_name, f"{group_name}.unico.dat0")

def get_display_name(group_name):
    return DISPLAY_NAMES.get(group_name, group_name)

def extract_lujo_sequences(input_file, max_seq=1000):
    """Extrae secuencias del virus LUJO del archivo"""
    if not os.path.exists(input_file):
        print(f"❌ Archivo no encontrado: {input_file}")
        return []
    
    print(f"\n📂 Leyendo archivo: {input_file}")
    
    lujo_sequences = []
    all_sequences = []
    
    for header, seq in read_fasta_stream(input_file):
        all_sequences.append((header, seq))
    
    print(f"  📊 Total de secuencias encontradas: {len(all_sequences)}")
    
    # 🔍 DEPURACIÓN: Mostrar las primeras 10 secuencias
    print("\n  🔍 DEPURACIÓN: Primeras 10 secuencias:")
    for i, (header, seq) in enumerate(all_sequences[:10]):
        print(f"     {i+1}: {header[:80]}...")
    
    # Buscar LUJO
    for header, seq in all_sequences[:max_seq]:
        # Buscar coincidencia con LUJO
        if re.search(LUJO_VIRUS['pattern'], header, re.IGNORECASE):
            lujo_sequences.append((header, seq))
            print(f"  🔍 DEBUG: Clasificado como LUJO: {header[:60]}...")
    
    print(f"\n  📊 Secuencias de LUJO encontradas: {len(lujo_sequences)}")
    
    if len(lujo_sequences) == 0:
        # Búsqueda alternativa más amplia
        print("\n  🔍 Búsqueda alternativa de LUJO...")
        keywords = ['lujo', 'LUJO', 'Luj', 'arenavirus', 'Arenavirus']
        for header, seq in all_sequences[:max_seq]:
            for keyword in keywords:
                if keyword in header:
                    lujo_sequences.append((header, seq))
                    print(f"  🔍 DEBUG: Clasificado como LUJO (keyword: {keyword}): {header[:60]}...")
                    break
    
    print(f"\n  📊 Total de secuencias de LUJO: {len(lujo_sequences)}")
    return lujo_sequences

# ============================================================================
# FUNCIONES PARA SGPMAIN6.2v
# ============================================================================

def compute_sgp_pim(sequence, use_weights=True):
    try:
        pim = compute_pim_profile(sequence, use_weights=use_weights)
        return pim
    except Exception as e:
        print(f"⚠️ Error en compute_sgp_pim: {e}")
        return np.zeros(16)

def compute_sgp_trimer(sequence):
    try:
        trimer = compute_trimer_profile(sequence)
        return trimer
    except Exception as e:
        print(f"⚠️ Error en compute_sgp_trimer: {e}")
        return np.zeros(64)

def compute_grassmann_similarity(v1, v2):
    try:
        dist = grassmann_distance(v1, v2)
        # Convertir distancia a similitud (0-1)
        return np.exp(-dist)
    except:
        return 0.0

def compute_sgp_enhanced_metrics(v1, v2):
    """Calcula todas las métricas avanzadas de SGPMAIN6.2v"""
    metrics = {}
    
    try:
        # Grassmann
        metrics['grassmann_projection'] = grassmann_projection_distance_true(v1, v2)
        metrics['fubini_study'] = grassmann_fubini_study_true(v1, v2)
        metrics['ricci_curvature'] = grassmann_ricci_curvature_true(v1, v2)
        metrics['grassmann_distance'] = grassmann_distance(v1, v2)
        
        # Hodge
        metrics['hodge_complementarity'] = hodge_complementarity_true(v1, v2)
        
        # Métricas de información
        metrics['jensen_shannon'] = jensen_shannon_divergence_true(v1, v2)
        metrics['hellinger'] = hellinger_distance_true(v1, v2)
        metrics['wasserstein'] = wasserstein_distance_true(v1, v2)
        metrics['spearman'] = spearman_correlation_true(v1, v2)
        
        # Fractal
        metrics['fractal_dim_v1'] = fractal_dimension_true(v1)
        metrics['fractal_dim_v2'] = fractal_dimension_true(v2)
        
        # Entropía
        metrics['entropy_v1'] = shannon_entropy_true(v1)
        metrics['entropy_v2'] = shannon_entropy_true(v2)
        
        # Gini
        metrics['gini_v1'] = gini_coefficient_true(v1)
        metrics['gini_v2'] = gini_coefficient_true(v2)
        
        # Morans I
        metrics['morans_i_v1'] = morans_i_true(v1)
        metrics['morans_i_v2'] = morans_i_true(v2)
        
        # Clifford
        sig1 = clifford_signature_true(v1)
        sig2 = clifford_signature_true(v2)
        metrics['clifford_distance'] = clifford_distance_true(sig1, sig2)
        
        # Rotor angles
        angles = all_rotor_angles(v1, v2)
        for name, angle in angles.items():
            metrics[f'rotor_angle_{name}'] = angle
        
    except Exception as e:
        print(f"⚠️ Error en enhanced metrics: {e}")
    
    return metrics

def get_lujo_representative_pim(lujo_sequences, max_seq=50):
    if not lujo_sequences:
        return None
    
    pims = []
    for header, seq in lujo_sequences[:max_seq]:
        seq_clean = ''.join([c for c in str(seq).strip() if c.isalpha()])
        if len(seq_clean) < 10:
            continue
        
        pim = compute_sgp_pim(seq_clean, use_weights=True)
        if np.sum(pim) > 0.01:
            pims.append(pim)
    
    if not pims:
        return None
    
    avg_pim = np.mean(pims, axis=0)
    total = np.sum(avg_pim)
    if total > 0:
        avg_pim = avg_pim / total
    
    return avg_pim

def run_sgp_for_lujo(input_file, groups_to_analyze, max_sequences=50):
    print(f"\n  🧬 Ejecutando SGPMAIN6.2v para LUJO...")
    
    lujo_sequences = extract_lujo_sequences(input_file, max_seq=1000)
    
    if not lujo_sequences:
        print(f"  ⚠️ No se encontraron secuencias para LUJO")
        return None
    
    lujo_pim = get_lujo_representative_pim(lujo_sequences, max_seq=max_sequences)
    
    if lujo_pim is None or np.sum(lujo_pim) < 0.01:
        print(f"  ❌ PIM inválido para LUJO")
        return None
    
    print(f"     ├─ Secuencias encontradas: {len(lujo_sequences)}")
    print(f"     ├─ PIM del virus: dimensión {len(lujo_pim)}, suma={np.sum(lujo_pim):.4f}")
    entropy_val = shannon_entropy_true(lujo_pim)
    print(f"     ├─ Entropía: {entropy_val:.4f}")
    print(f"     ├─ Gini: {gini_coefficient_true(lujo_pim):.4f}")
    print(f"     ├─ Fractal Dim: {fractal_dimension_true(lujo_pim):.4f}")
    
    results = {}
    enhanced_results = {}
    
    for group in groups_to_analyze:
        group_file = get_filename(group)
        if not os.path.exists(group_file):
            continue
        
        display = get_display_name(group)
        print(f"     ├─ Procesando {display}...")
        
        similarities = []
        enhanced_metrics_list = []
        count = 0
        
        for header, seq in read_fasta_stream(group_file):
            seq_clean = ''.join([c for c in str(seq).strip() if c.isalpha()])
            if len(seq_clean) < 10:
                continue
            
            pim_group = compute_sgp_pim(seq_clean, use_weights=True)
            if np.sum(pim_group) > 0.01:
                sim = compute_grassmann_similarity(lujo_pim, pim_group)
                similarities.append(sim)
                
                # Métricas avanzadas
                if count < 10:
                    enhanced = compute_sgp_enhanced_metrics(lujo_pim, pim_group)
                    enhanced_metrics_list.append(enhanced)
                
                count += 1
                if count >= max_sequences:
                    break
        
        if similarities:
            results[group] = np.mean(similarities)
            results[f'{group}_std'] = np.std(similarities)
            results[f'{group}_n'] = len(similarities)
            
            # Promediar métricas avanzadas
            if enhanced_metrics_list:
                avg_enhanced = {}
                for key in enhanced_metrics_list[0].keys():
                    values = [m.get(key, 0) for m in enhanced_metrics_list]
                    avg_enhanced[key] = np.mean(values)
                enhanced_results[group] = avg_enhanced
            
            print(f"        └─ Similitud media: {results[group]:.6f} ± {results[f'{group}_std']:.6f} (n={len(similarities)})")
        else:
            print(f"        └─ ⚠️ Sin PIMs válidos")
    
    return {
        'lujo_pim': lujo_pim,
        'similarities': results,
        'enhanced_metrics': enhanced_results,
        'lujo_sequences': len(lujo_sequences)
    }

# ============================================================================
# FUNCIONES PARA BIOPYTHON
# ============================================================================

def extract_biopython_features_enhanced(sequence):
    try:
        seq_str = ''.join([c for c in str(sequence).strip() if c.isalpha()])
        if len(seq_str) < 5:
            return None
        
        seq_obj = Seq(seq_str)
        analyzer = ProtParam.ProteinAnalysis(str(seq_obj))
        
        features = []
        
        # Amino acid composition (20 features)
        try:
            aa_counts = analyzer.get_amino_acids_percent()
            for aa in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
                features.append(aa_counts.get(aa, 0.0))
        except:
            features.extend([0.0] * 20)
        
        # Molecular weight
        try:
            mw = analyzer.molecular_weight()
            features.append(mw / (len(seq_str) + 1))
        except:
            features.append(0.0)
        
        # Isoelectric point
        try:
            features.append(analyzer.isoelectric_point())
        except:
            features.append(7.0)
        
        # Aromaticity
        try:
            features.append(analyzer.aromaticity())
        except:
            features.append(0.0)
        
        # Instability index
        try:
            features.append(analyzer.instability_index())
        except:
            features.append(50.0)
        
        # GRAVY
        try:
            features.append(analyzer.gravy())
        except:
            features.append(0.0)
        
        # Secondary structure fractions
        try:
            sec_struct = analyzer.secondary_structure_fraction()
            features.extend(sec_struct)
        except:
            features.extend([0.0, 0.0, 0.0])
        
        # Net charge
        try:
            charges = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}
            net_charge = sum(charges.get(aa, 0) for aa in seq_str)
            features.append(net_charge / (len(seq_str) + 1))
        except:
            features.append(0.0)
        
        # Log length
        features.append(np.log(len(seq_str) + 1))
        
        # Hydrophobicity
        hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        avg_hydro = sum(hydrophobicity.get(aa, 0) for aa in seq_str) / (len(seq_str) + 1)
        features.append(avg_hydro)
        
        # Flexibility
        flexibility = {
            'A': 0.0, 'R': 0.5, 'N': 0.3, 'D': 0.4, 'C': 0.1,
            'Q': 0.4, 'E': 0.5, 'G': 0.1, 'H': 0.3, 'I': 0.0,
            'L': 0.0, 'K': 0.5, 'M': 0.1, 'F': 0.0, 'P': 0.6,
            'S': 0.3, 'T': 0.2, 'W': 0.1, 'Y': 0.1, 'V': 0.0
        }
        avg_flex = sum(flexibility.get(aa, 0) for aa in seq_str) / (len(seq_str) + 1)
        features.append(avg_flex)
        
        # Pad to fixed length
        while len(features) < 35:
            features.append(0.0)
        
        return np.array(features)
        
    except Exception as e:
        return None

def normalize_features(features_list):
    if not features_list:
        return features_list
    
    features_array = np.array(features_list)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features_array)
    return normalized

def euclidean_similarity(v1, v2):
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    dist = np.linalg.norm(v1_norm - v2_norm)
    max_dist = np.sqrt(2)
    sim = 1 - (dist / max_dist)
    return max(0, min(1, sim))

def get_lujo_representative_features(lujo_sequences, max_seq=50):
    if not lujo_sequences:
        return None
    
    features_list = []
    for header, seq in lujo_sequences[:max_seq]:
        features = extract_biopython_features_enhanced(seq)
        if features is not None:
            features_list.append(features)
    
    if not features_list:
        return None
    
    avg_features = np.mean(features_list, axis=0)
    return avg_features

def run_biopython_for_lujo(input_file, groups_to_analyze, max_sequences=10):
    print(f"\n  🔬 Ejecutando BioPython para LUJO (35 características)...")
    
    lujo_sequences = extract_lujo_sequences(input_file, max_seq=1000)
    
    if not lujo_sequences:
        print(f"  ⚠️ No se encontraron secuencias para LUJO")
        return None
    
    lujo_features = get_lujo_representative_features(lujo_sequences, max_seq=50)
    
    if lujo_features is None:
        print(f"  ❌ Error extrayendo características de LUJO")
        return None
    
    print(f"     ├─ Secuencias encontradas: {len(lujo_sequences)}")
    print(f"     ├─ Dimensiones: {len(lujo_features)} características")
    
    results = {}
    
    for group in groups_to_analyze:
        group_file = get_filename(group)
        if not os.path.exists(group_file):
            continue
        
        print(f"     ├─ Procesando {get_display_name(group)}...")
        
        vectors = []
        count = 0
        for header, seq in read_fasta_stream(group_file):
            features = extract_biopython_features_enhanced(seq)
            if features is not None and len(features) > 0:
                vectors.append(features)
                count += 1
                if count >= max_sequences:
                    break
        
        if not vectors:
            print(f"        └─ ⚠️ Sin características válidas")
            continue
        
        # Normalizar
        all_vectors = [lujo_features] + vectors
        normalized_vectors = normalize_features(all_vectors)
        lujo_norm = normalized_vectors[0]
        vectors_norm = normalized_vectors[1:]
        
        similarities = []
        for vec in vectors_norm:
            sim = euclidean_similarity(lujo_norm, vec)
            similarities.append(sim)
        
        if similarities:
            results[group] = np.mean(similarities)
            results[f'{group}_std'] = np.std(similarities)
            results[f'{group}_n'] = len(similarities)
            print(f"        └─ Similitud media: {results[group]:.6f} ± {results[f'{group}_std']:.6f} (n={len(similarities)})")
        else:
            print(f"        └─ ⚠️ Sin similitudes válidas")
    
    return results

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def analyze_lujo_virus(input_file, groups_to_analyze, max_sequences=10):
    print("\n" + "=" * 80)
    print("🔬 ANALIZANDO: LUJO")
    print("=" * 80)
    
    if not os.path.exists(input_file):
        print(f"  ❌ Archivo no encontrado: {input_file}")
        return None
    
    print(f"  📌 Usando muestra de {max_sequences} secuencias por grupo")
    
    sgp_results = run_sgp_for_lujo(input_file, groups_to_analyze, max_sequences)
    biopython_results = run_biopython_for_lujo(input_file, groups_to_analyze, max_sequences)
    
    if biopython_results is None or len(biopython_results) == 0:
        print(f"\n❌ No se pudieron obtener resultados de BioPython para LUJO")
        return None
    
    # Tabla comparativa
    comparacion = []
    groups_compared = set()
    
    if sgp_results:
        groups_compared = set(sgp_results['similarities'].keys()) & set(biopython_results.keys())
    else:
        groups_compared = set(biopython_results.keys())
    
    print("\n" + "=" * 80)
    print("📋 TABLA COMPARATIVA: SGPMAIN6.2v vs BioPython (LUJO)")
    print("=" * 80)
    print(f"{'Grupo':<22} {'SGPMAIN':>12} {'BioPython':>12} {'Diferencia':>12} {'Interpretación':>15}")
    print("-" * 80)
    
    for grupo in sorted(groups_compared, 
                        key=lambda x: sgp_results['similarities'].get(x, 0) if sgp_results and x in sgp_results['similarities'] else 0, 
                        reverse=True):
        sgp_val = sgp_results['similarities'].get(grupo, 0.0) if sgp_results else 0.0
        biopy_val = biopython_results.get(grupo, 0.0)
        diff = abs(sgp_val - biopy_val) if sgp_val > 0 else 1.0
        
        if sgp_val == 0:
            interp = "⚠️ Sin SGP"
        elif diff < 0.01:
            interp = "✅ Excelente"
        elif diff < 0.03:
            interp = "✔️ Buena"
        elif diff < 0.05:
            interp = "⚠️ Moderada"
        else:
            interp = "❌ Diferente"
        
        display_name = get_display_name(grupo)
        print(f"{display_name:<22} {sgp_val:>12.6f} {biopy_val:>12.6f} "
              f"{diff:>12.6f} {interp:>15}")
        
        comparacion.append({
            'Virus': 'LUJO',
            'Grupo': display_name,
            'Grupo_original': grupo,
            'SGPMAIN': sgp_val,
            'BioPython': biopy_val,
            'Diferencia': diff,
            'Interpretación': interp
        })
    
    return {
        'virus': LUJO_VIRUS,
        'sgp': sgp_results,
        'biopython': biopython_results,
        'comparacion': comparacion
    }

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

# Cambiar el nombre del archivo de entrada a todolujo.txt
INPUT_FILE = "todolujo.txt"

print("\n📂 Verificando archivo de entrada...")
print(f"   Buscando: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Archivo no encontrado: {INPUT_FILE}")
    print("   Buscando archivos alternativos...")
    
    # Buscar archivos alternativos
    alternatives = ['todoebola.txt', 'ebola.txt', 'lujo.txt', 'todolujo.fasta', 'lujo.fasta']
    for alt in alternatives:
        if os.path.exists(alt):
            INPUT_FILE = alt
            print(f"✅ Usando archivo alternativo: {INPUT_FILE}")
            break
    else:
        print("❌ No se encontró ningún archivo de entrada")
        print("   Por favor, crea un archivo 'todolujo.txt' con las secuencias del virus LUJO")
        sys.exit(1)

print(f"✅ Archivo encontrado: {INPUT_FILE}")

# Verificar archivos .dat0
print("\n📂 Verificando archivos .dat0...")
found_files = 0
for group in GROUP_FILES:
    if os.path.exists(GROUP_FILES[group]):
        found_files += 1
print(f"  ✅ Archivos encontrados: {found_files} de {len(GROUP_FILES)}")

# Verificar secuencias de LUJO
lujo_sequences = extract_lujo_sequences(INPUT_FILE, max_seq=1000)

if not lujo_sequences:
    print("\n❌ No se encontraron secuencias del virus LUJO en el archivo")
    print("   🔍 Buscando todas las secuencias del archivo...")
    
    # Mostrar headers para depuración
    count = 0
    for header, seq in read_fasta_stream(INPUT_FILE):
        print(f"     └─ {header[:100]}")
        count += 1
        if count >= 5:
            break
    
    sys.exit(1)

print(f"\n  ✅ Se encontraron {len(lujo_sequences)} secuencias de LUJO")

# ============================================================================
# ANALIZAR LUJO
# ============================================================================

result = analyze_lujo_virus(INPUT_FILE, GROUPS_TO_ANALYZE, max_sequences=10)

if result and result['comparacion']:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"lujo_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar comparación
    df_comp = pd.DataFrame(result['comparacion'])
    df_comp.to_csv(f"{results_dir}/comparacion_sgp_biopython_lujo.csv", index=False)
    
    # Guardar resultados SGP
    if result['sgp']:
        sgp_data = {
            'lujo_pim': result['sgp']['lujo_pim'].tolist(),
            'similarities': result['sgp']['similarities'],
            'lujo_sequences': result['sgp']['lujo_sequences'],
            'enhanced_metrics': {}
        }
        # Convertir enhanced metrics a JSON serializable
        for group, metrics in result['sgp']['enhanced_metrics'].items():
            sgp_data['enhanced_metrics'][group] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                                    for k, v in metrics.items()}
        
        with open(f"{results_dir}/sgp_lujo_results.json", 'w') as f:
            json.dump(sgp_data, f, indent=2)
    
    # Guardar resultados BioPython
    if result['biopython']:
        with open(f"{results_dir}/biopython_lujo_results.json", 'w') as f:
            json.dump(result['biopython'], f, indent=2)
    
    print(f"\n  ✅ Resultados guardados en: {results_dir}/")

# ============================================================================
# RESULTADOS ADICIONALES
# ============================================================================

if result and result['sgp']:
    print("\n" + "=" * 80)
    print("📊 MÉTRICAS AVANZADAS DE SGPMAIN6.2v (LUJO)")
    print("=" * 80)
    
    # Mostrar métricas avanzadas para algunos grupos
    if result['sgp']['enhanced_metrics']:
        groups_with_metrics = list(result['sgp']['enhanced_metrics'].keys())[:5]
        if groups_with_metrics:
            print(f"\n  Métricas para {', '.join([get_display_name(g) for g in groups_with_metrics])}:")
            for group in groups_with_metrics:
                metrics = result['sgp']['enhanced_metrics'][group]
                print(f"\n  📍 {get_display_name(group)}:")
                print(f"     ├─ Grassmann Projection: {metrics.get('grassmann_projection', 0):.6f}")
                print(f"     ├─ Fubini-Study: {metrics.get('fubini_study', 0):.6f}")
                print(f"     ├─ Ricci Curvature: {metrics.get('ricci_curvature', 0):.6f}")
                print(f"     ├─ Jensen-Shannon: {metrics.get('jensen_shannon', 0):.6f}")
                print(f"     ├─ Hellinger: {metrics.get('hellinger', 0):.6f}")
                print(f"     ├─ Wasserstein: {metrics.get('wasserstein', 0):.6f}")
                print(f"     ├─ Spearman: {metrics.get('spearman', 0):.6f}")
                print(f"     └─ Clifford Distance: {metrics.get('clifford_distance', 0):.6f}")

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
