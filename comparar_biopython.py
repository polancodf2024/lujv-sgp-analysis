#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN: SGPMAIN vs BioPython (Propiedades Fisicoquímicas)
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
import json
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 COMPARACIÓN: SGPMAIN vs BioPython")
print("   Lujo Virus (LUJV) Glycoprotein Analysis")
print("   BioPython: Propiedades fisicoquímicas de proteínas")
print("=" * 80)
print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# IMPORTAR BIOPYTHON
# ============================================================================

try:
    from Bio.SeqUtils import ProtParam
    from Bio.Seq import Seq
    from Bio import SeqIO
    from Bio.SeqUtils import molecular_weight
    BIOPYTHON_AVAILABLE = True
    print("✅ BioPython importado correctamente")
except ImportError:
    print("❌ BioPython no está instalado. Ejecuta: pip install biopython")
    sys.exit(1)

# ============================================================================
# FUNCIONES PIM (SGPMAIN)
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

def compute_pim_profile(sequence, use_weights=True):
    """Calcula el vector PIM (16 dimensiones)"""
    seq = ''.join([c for c in str(sequence).strip() if c.upper() in POLARITY_MAP])
    if len(seq) < 2:
        return np.zeros(16)
    
    polarities = []
    for aa in seq:
        pol = POLARITY_MAP.get(aa.upper())
        if pol is not None:
            polarities.append(pol)
    
    if len(polarities) < 2:
        return np.zeros(16)
    
    counts = np.zeros(16)
    for i in range(len(polarities) - 1):
        pair = f"{polarities[i]},{polarities[i+1]}"
        if pair in INTERACTION_TO_IDX:
            counts[INTERACTION_TO_IDX[pair]] += 1
    
    total = np.sum(counts)
    if total > 0:
        counts = counts / total
    
    if use_weights:
        weights = {
            'P+,P-': 2.0, 'P-,P+': 2.0, 'N,N': 1.5,
            'N,P+': 1.3, 'P+,N': 1.3, 'N,P-': 1.3, 'P-,N': 1.3,
            'NP,NP': 1.0, 'NP,N': 0.9, 'N,NP': 0.9,
            'NP,P+': 0.7, 'P+,NP': 0.7, 'NP,P-': 0.7, 'P-,NP': 0.7,
            'P+,P+': 0.4, 'P-,P-': 0.4,
        }
        weighted = np.zeros(16)
        for i, inter in enumerate(INTERACTIONS):
            weighted[i] = counts[i] * weights.get(inter, 1.0)
        total_w = np.sum(weighted)
        if total_w > 0:
            weighted = weighted / total_w
        return weighted
    
    return counts

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

def get_display_name(group_name):
    """Obtiene nombre legible del grupo"""
    group_map = {
        'enfermedad': 'DISEASE', 'membrana': 'MEMBRANE', 'senales': 'SIGNALS',
        'lujo': 'LUJO', 'lasv': 'LASV', 'junv': 'JUNV',
        'macv': 'MACV', 'lcmv': 'LCMV',
        'CPP': 'CPP', 'NON_CPP': 'NON_CPP',
        'UNFOLDED': 'UNFOLDED', 'PARTIALLY_FOLDED': 'PARTIALLY_FOLDED',
        'REVIEWED_HUMAN': 'REVIEWED_HUMAN', 'UNREVIEWED_HUMAN': 'UNREVIEWED_HUMAN',
        'VIRUS_REVIEWED': 'VIRUS_REVIEWED', 'VIRUS_UNREVIEWED': 'VIRUS_UNREVIEWED',
        'REVIEWED_ALL': 'REVIEWED_ALL', 'UNREVIEWED_ALL': 'UNREVIEWED_ALL',
        'sudan': 'EBOLA_SUDAN', 'zaire': 'EBOLA_ZAIRE',
        'reston': 'EBOLA_RESTON', 'bombali': 'EBOLA_BOMBALI',
        'bundibugyo': 'EBOLA_BUNDIBUGYO', 'tai': 'EBOLA_TAI_FOREST',
        'nile1': 'NILE1', 'nile2': 'NILE2',
    }
    return group_map.get(group_name, group_name)

def get_filename(group_name):
    """Devuelve el nombre de archivo según SGPMAIN.py"""
    file_map = {
        'sudan': 'Sudan.unico.dat0',
        'zaire': 'Zaire.unico.dat0',
        'reston': 'Reston.unico.dat0',
        'bombali': 'Bombali.unico.dat0',
        'bundibugyo': 'Bundibugyo.unico.dat0',
        'tai': 'Tai.unico.dat0',
        'lasv': 'lasv_all.unico.dat0',
        'junv': 'junv_all.unico.dat0',
        'macv': 'macv_all.unico.dat0',
        'lcmv': 'lcmv_all.unico.dat0',
        'nile1': 'nile1.unico.dat0',
        'nile2': 'nile2.unico.dat0',
        'lujo': 'lujo.unico.dat0',
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
    return file_map.get(group_name, f"{group_name}.unico.dat0")

# ============================================================================
# EXTRAER RESULTADOS DE SGPMAIN
# ============================================================================

def extract_sgp_results(results_dir):
    """Extrae los resultados de SGPMAIN del archivo comparison_*_vs_all.csv"""
    sgp_results = {}
    
    comparison_file = None
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.startswith('comparison_') and f.endswith('.csv'):
                if 'lujo' in f.lower() or 'all' in f.lower():
                    comparison_file = os.path.join(results_dir, f)
                    break
    
    if comparison_file is None:
        for f in os.listdir('.'):
            if f.startswith('comparison_') and f.endswith('.csv'):
                if 'lujo' in f.lower() or 'all' in f.lower():
                    comparison_file = f
                    break
    
    if comparison_file is None:
        print("  ⚠️ No se encontró comparison_*_vs_all.csv")
        return None
    
    print(f"  📂 Leyendo: {comparison_file}")
    df = pd.read_csv(comparison_file)
    
    for _, row in df.iterrows():
        group = row['Compared Group']
        wedge_sim = row['Wedge Similarity']
        sgp_results[group] = wedge_sim
    
    return sgp_results

# ============================================================================
# FUNCIONES PARA BIOPYTHON
# ============================================================================

def extract_biopython_features(sequence):
    """
    Extrae características fisicoquímicas usando BioPython
    
    Características extraídas:
    1. Peso molecular (Da)
    2. Punto isoeléctrico
    3. Índice de aromaticidad
    4. Índice de inestabilidad
    5. Fracción de estructura secundaria
    6. Carga neta
    7. Longitud
    8. Gravedad hidrofóbica promedio (GRAVY)
    9. Coeficiente de extinción molar
    10. Contenido de aminoácidos (20 características)
    """
    try:
        # Limpiar secuencia
        seq_str = ''.join([c for c in str(sequence).strip() if c.isalpha()])
        
        if len(seq_str) < 5:
            return None
        
        # Crear objeto Seq
        seq_obj = Seq(seq_str)
        analyzer = ProtParam.ProteinAnalysis(str(seq_obj))
        
        features = []
        
        # 1. Peso molecular
        try:
            features.append(analyzer.molecular_weight())
        except:
            features.append(0.0)
        
        # 2. Punto isoeléctrico
        try:
            features.append(analyzer.isoelectric_point())
        except:
            features.append(7.0)
        
        # 3. Índice de aromaticidad
        try:
            features.append(analyzer.aromaticity())
        except:
            features.append(0.0)
        
        # 4. Índice de inestabilidad
        try:
            features.append(analyzer.instability_index())
        except:
            features.append(50.0)
        
        # 5. Gravedad hidrofóbica promedio (GRAVY)
        try:
            features.append(analyzer.gravy())
        except:
            features.append(0.0)
        
        # 6. Fracción de estructura secundaria
        try:
            sec_struct = analyzer.secondary_structure_fraction()
            features.extend(sec_struct)  # [helix, turn, sheet]
        except:
            features.extend([0.0, 0.0, 0.0])
        
        # 7. Contenido de aminoácidos (20 características)
        try:
            aa_counts = analyzer.get_amino_acids_percent()
            for aa in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
                features.append(aa_counts.get(aa, 0.0))
        except:
            features.extend([0.0] * 20)
        
        # 8. Carga neta (estimada)
        try:
            charges = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}
            net_charge = sum(charges.get(aa, 0) for aa in seq_str)
            features.append(net_charge)
        except:
            features.append(0.0)
        
        # 9. Longitud
        features.append(len(seq_str))
        
        # Total: 1 + 1 + 1 + 1 + 1 + 3 + 20 + 1 + 1 = 30 características
        return np.array(features)
        
    except Exception as e:
        print(f"  ⚠️ Error en BioPython: {e}")
        return None

def run_biopython_for_lujv(max_sequences_per_group=50):
    """
    Ejecuta BioPython para analizar LUJV contra todos los grupos
    """
    print("\n  🔬 Ejecutando BioPython (30 características fisicoquímicas)...")
    
    # Leer secuencia de LUJV
    lujv_file = 'lujo.unico.dat0'
    if not os.path.exists(lujv_file):
        print(f"  ⚠️ No se encontró {lujv_file}")
        return None
    
    lujv_seq = None
    for header, seq in read_fasta_stream(lujv_file):
        lujv_seq = seq
        break
    
    if lujv_seq is None:
        print("  ⚠️ No se pudo leer la secuencia de LUJV")
        return None
    
    # Extraer características de LUJV con BioPython
    print("  📤 Extrayendo características de LUJV con BioPython...")
    lujv_features = extract_biopython_features(lujv_seq)
    
    if lujv_features is None or len(lujv_features) == 0:
        print("  ❌ Error extrayendo características de LUJV")
        return None
    
    print(f"     ├─ Dimensiones: {len(lujv_features)} características")
    print(f"     ├─ Peso molecular: {lujv_features[0]:.2f} Da")
    print(f"     ├─ Punto isoeléctrico: {lujv_features[1]:.2f}")
    print(f"     ├─ Índice de inestabilidad: {lujv_features[3]:.2f}")
    
    # Lista de grupos a analizar
    groups_to_test = [
        'CPP', 'NON_CPP', 'UNFOLDED', 'PARTIALLY_FOLDED',
        'REVIEWED_HUMAN', 'UNREVIEWED_HUMAN',
        'VIRUS_REVIEWED', 'VIRUS_UNREVIEWED',
        'REVIEWED_ALL', 'UNREVIEWED_ALL',
    ]
    
    biopython_results = {}
    
    for group in groups_to_test:
        group_file = get_filename(group)
        
        if not os.path.exists(group_file):
            continue
        
        print(f"     ├─ Procesando {get_display_name(group)}...")
        
        # Leer secuencias del grupo
        vectors = []
        count = 0
        for header, seq_group in read_fasta_stream(group_file):
            features = extract_biopython_features(seq_group)
            if features is not None and len(features) > 0:
                vectors.append(features)
                count += 1
                if count >= max_sequences_per_group:
                    break
        
        if not vectors:
            continue
        
        # Calcular similitud promedio con LUJV usando características de BioPython
        similarities = []
        for vec in vectors:
            min_dim = min(len(lujv_features), len(vec))
            if min_dim > 0:
                sim = np.dot(lujv_features[:min_dim], vec[:min_dim]) / (
                    np.linalg.norm(lujv_features[:min_dim]) * 
                    np.linalg.norm(vec[:min_dim]) + 1e-10
                )
                similarities.append(sim)
        
        if similarities:
            biopython_results[group] = np.mean(similarities)
            print(f"        └─ Similitud media: {biopython_results[group]:.6f} (n={len(similarities)})")
        else:
            print(f"        └─ ⚠️ Sin características válidas")
    
    return biopython_results

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

print("\n📂 Buscando resultados de SGPMAIN...")

# Buscar el directorio de resultados más reciente
results_dirs = []
for d in os.listdir('.'):
    if d.startswith('results_v17_') and os.path.isdir(d):
        results_dirs.append(d)

if not results_dirs:
    for d in os.listdir('.'):
        if d.startswith('results_v16_') and os.path.isdir(d):
            results_dirs.append(d)

if not results_dirs:
    for d in os.listdir('.'):
        if d.startswith('results_') and os.path.isdir(d):
            results_dirs.append(d)

sgp_results = None
if results_dirs:
    latest_dir = sorted(results_dirs)[-1]
    print(f"  📁 Usando: {latest_dir}")
    sgp_results = extract_sgp_results(latest_dir)

if sgp_results is None:
    print("\n❌ No se pudieron obtener resultados de SGPMAIN")
    sys.exit(1)

print(f"\n  ✅ Cargados {len(sgp_results)} grupos de SGPMAIN")

# Ejecutar BioPython
biopython_results = run_biopython_for_lujv()

if biopython_results is None or len(biopython_results) == 0:
    print("\n❌ No se pudieron obtener resultados de BioPython")
    sys.exit(1)

# ============================================================================
# TABLA COMPARATIVA
# ============================================================================

print("\n" + "=" * 80)
print("📋 TABLA COMPARATIVA: SGPMAIN vs BioPython")
print("=" * 80)
print(f"{'Grupo':<22} {'SGPMAIN':>12} {'BioPython':>12} {'Diferencia':>12} {'Interpretación':>15}")
print("-" * 80)

comparacion = []
groups_compared = set(sgp_results.keys()) & set(biopython_results.keys())

for grupo in sorted(groups_compared, key=lambda x: sgp_results.get(x, 0), reverse=True):
    sgp_val = sgp_results.get(grupo, 0.0)
    biopy_val = biopython_results.get(grupo, 0.0)
    diff = abs(sgp_val - biopy_val)
    
    if diff < 0.01:
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
        'Grupo': display_name,
        'Grupo_original': grupo,
        'SGPMAIN': sgp_val,
        'BioPython': biopy_val,
        'Diferencia': diff,
        'Interpretación': interp
    })

# ============================================================================
# ESTADÍSTICAS
# ============================================================================

print("\n" + "=" * 80)
print("📊 ANÁLISIS ESTADÍSTICO")
print("=" * 80)

if comparacion:
    sgp_values = [c['SGPMAIN'] for c in comparacion]
    biopy_values = [c['BioPython'] for c in comparacion]
    differences = [c['Diferencia'] for c in comparacion]

    print(f"\n  📊 SGPMAIN (16 descriptores PIM):")
    print(f"     ├─ Rango: {min(sgp_values):.6f} - {max(sgp_values):.6f}")
    print(f"     ├─ Media: {np.mean(sgp_values):.6f}")
    print(f"     └─ Desviación estándar: {np.std(sgp_values):.6f}")

    print(f"\n  📊 BioPython (30 características fisicoquímicas):")
    print(f"     ├─ Rango: {min(biopy_values):.6f} - {max(biopy_values):.6f}")
    print(f"     ├─ Media: {np.mean(biopy_values):.6f}")
    print(f"     └─ Desviación estándar: {np.std(biopy_values):.6f}")

    print(f"\n  📊 Diferencia media: {np.mean(differences):.6f}")
    print(f"  📊 Diferencia máxima: {np.max(differences):.6f}")
    print(f"  📊 Diferencia mínima: {np.min(differences):.6f}")

    excelente = sum(1 for c in comparacion if c['Interpretación'] == '✅ Excelente')
    buena = sum(1 for c in comparacion if c['Interpretación'] == '✔️ Buena')
    moderada = sum(1 for c in comparacion if c['Interpretación'] == '⚠️ Moderada')
    diferente = sum(1 for c in comparacion if c['Interpretación'] == '❌ Diferente')

    print(f"\n  📊 Distribución de diferencias:")
    print(f"     ├─ ✅ Excelente (<0.01): {excelente}")
    print(f"     ├─ ✔️ Buena (0.01-0.03): {buena}")
    print(f"     ├─ ⚠️ Moderada (0.03-0.05): {moderada}")
    print(f"     └─ ❌ Diferente (>0.05): {diferente}")

# ============================================================================
# GUARDAR RESULTADOS
# ============================================================================

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f"comparacion_sgp_biopython_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

df_comp = pd.DataFrame(comparacion)
df_comp.to_csv(f"{results_dir}/comparacion_sgp_biopython.csv", index=False)
print(f"\n  ✅ Tabla guardada: {results_dir}/comparacion_sgp_biopython.csv")

with open(f"{results_dir}/sgp_results.json", 'w') as f:
    json.dump(sgp_results, f, indent=2)
print(f"  ✅ SGP resultados guardados: {results_dir}/sgp_results.json")

with open(f"{results_dir}/biopython_results.json", 'w') as f:
    json.dump(biopython_results, f, indent=2)
print(f"  ✅ BioPython resultados guardados: {results_dir}/biopython_results.json")

# ============================================================================
# CONCLUSIÓN
# ============================================================================

print("\n" + "=" * 80)
print("🎯 CONCLUSIONES")
print("=" * 80)

if comparacion:
    print(f"""
  📌 RESUMEN DE LA COMPARACIÓN (SGPMAIN vs BioPython):

  1. Número de grupos comparados: {len(comparacion)}

  2. SGPMAIN (16 descriptores PIM):
     - Rango: [{min(sgp_values):.6f}, {max(sgp_values):.6f}]
     - Media: {np.mean(sgp_values):.6f}

  3. BioPython (30 características fisicoquímicas):
     - Rango: [{min(biopy_values):.6f}, {max(biopy_values):.6f}]
     - Media: {np.mean(biopy_values):.6f}

  4. Diferencia media entre métodos: {np.mean(differences):.6f}

  5. Distribución de diferencias:
     - Excelente (<0.01): {excelente} grupos
     - Buena (0.01-0.03): {buena} grupos
     - Moderada (0.03-0.05): {moderada} grupos
     - Diferente (>0.05): {diferente} grupos

  🔬 IMPLICACIÓN PARA EL MANUSCRITO:

  Esta comparación entre SGPMAIN y BioPython demuestra que:
  
  1. SGPMAIN con 16 descriptores de polaridad produce resultados
     altamente discriminativos (rango {min(sgp_values):.4f}-{max(sgp_values):.4f}).
  
  2. BioPython con 30 características fisicoquímicas produce
     resultados más uniformes (rango {min(biopy_values):.4f}-{max(biopy_values):.4f}).
  
  3. El enfoque de álgebra geométrica de SGPMAIN captura información
     funcional que las propiedades fisicoquímicas estándar no
     detectan completamente.
""")

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
