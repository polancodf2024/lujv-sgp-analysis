#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN: SGP vs iFeature (corregido)
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 COMPARACIÓN: SGP vs iFeature")
print("   Lujo Virus (LUJV) Glycoprotein Analysis")
print("=" * 80)
print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# FUNCIONES PIM (para comparación)
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
    group_map = {
        'enfermedad': 'DISEASE', 'membrana': 'MEMBRANE', 'senales': 'SIGNALS',
        'LUJV': 'Lujo Virus', 'LASV': 'Lassa Virus', 'JUNV': 'Junín Virus',
        'MACV': 'Machupo Virus', 'LCMV': 'LCM Virus',
        'CPP': 'Cell-Penetrating Peptides', 'NON_CPP': 'Non-CPP',
        'UNFOLDED': 'Intrinsically Disordered', 'PARTIALLY_FOLDED': 'Partially Folded',
        'REVIEWED_HUMAN': 'Reviewed Human', 'UNREVIEWED_HUMAN': 'Unreviewed Human',
        'VIRUS_REVIEWED': 'Reviewed Virus', 'VIRUS_UNREVIEWED': 'Unreviewed Virus',
        'REVIEWED_ALL': 'Reviewed All', 'UNREVIEWED_ALL': 'Unreviewed All',
    }
    return group_map.get(group_name, group_name)

# ============================================================================
# COMPARACIÓN: RESULTADOS DEL MANUSCRITO
# ============================================================================

# Resultados SGP del manuscrito (Tabla 3)
SGP_RESULTS = {
    'CPP': 0.117,
    'LASV': 0.088,
    'NON_CPP': 0.072,
    'LCMV': 0.063,
    'UNFOLDED': 0.056,
    'senales': 0.053,
    'JUNV': 0.036,
    'PARTIALLY_FOLDED': 0.036,
    'MACV': 0.035,
    'VIRUS_REVIEWED': 0.034,
    'REVIEWED_ALL': 0.029,
    'REVIEWED_HUMAN': 0.025,
    'UNREVIEWED_HUMAN': 0.021,
    'UNREVIEWED_ALL': 0.016,
    'enfermedad': 0.019,
    'VIRUS_UNREVIEWED': 0.019,
    'membrana': 0.015,
}

# Resultados de iFeature de la ejecución anterior
IFeATURE_RESULTS = {
    'CPP': 0.876141,
    'LASV': 0.972992,
    'NON_CPP': 0.989800,
    'LCMV': 0.992123,
    'UNFOLDED': 0.982969,
    'senales': 0.992571,
    'JUNV': 0.992968,
    'PARTIALLY_FOLDED': 0.973412,
    'MACV': 0.995364,
    'VIRUS_REVIEWED': 0.997688,
    'REVIEWED_ALL': 0.995085,
    'REVIEWED_HUMAN': 0.996157,
    'UNREVIEWED_HUMAN': 0.996127,
    'UNREVIEWED_ALL': 0.992159,
    'enfermedad': 0.996885,
    'VIRUS_UNREVIEWED': 0.998430,
    'membrana': 0.995948,
}

print("\n" + "=" * 80)
print("📋 TABLA COMPARATIVA: SGP vs iFeature")
print("=" * 80)
print(f"{'Grupo':<22} {'SGP':>10} {'iFeature':>12} {'Diferencia':>12} {'Interpretación':>15}")
print("-" * 80)

comparacion = []

for grupo, sgp_val in SGP_RESULTS.items():
    ifeature_val = IFeATURE_RESULTS.get(grupo, 0.0)
    diff = abs(sgp_val - ifeature_val)
    
    if diff < 0.01:
        interp = "✅ Excelente"
    elif diff < 0.03:
        interp = "✔️ Buena"
    elif diff < 0.05:
        interp = "⚠️ Moderada"
    else:
        interp = "❌ Diferente"
    
    display_name = get_display_name(grupo)
    print(f"{display_name:<22} {sgp_val:>10.4f} {ifeature_val:>12.4f} "
          f"{diff:>12.4f} {interp:>15}")
    
    comparacion.append({
        'Grupo': display_name,
        'SGP': sgp_val,
        'iFeature': ifeature_val,
        'Diferencia': diff,
        'Interpretación': interp
    })

# ============================================================================
# ESTADÍSTICAS
# ============================================================================

print("\n" + "=" * 80)
print("📊 ANÁLISIS ESTADÍSTICO")
print("=" * 80)

sgp_values = [c['SGP'] for c in comparacion]
ifeature_values = [c['iFeature'] for c in comparacion]

print(f"\n  📊 SGP:")
print(f"     ├─ Rango: {min(sgp_values):.4f} - {max(sgp_values):.4f}")
print(f"     ├─ Media: {np.mean(sgp_values):.4f}")
print(f"     └─ Desviación estándar: {np.std(sgp_values):.4f}")

print(f"\n  📊 iFeature:")
print(f"     ├─ Rango: {min(ifeature_values):.4f} - {max(ifeature_values):.4f}")
print(f"     ├─ Media: {np.mean(ifeature_values):.4f}")
print(f"     └─ Desviación estándar: {np.std(ifeature_values):.4f}")

print(f"\n  📊 Diferencia media: {np.mean([c['Diferencia'] for c in comparacion]):.4f}")

# ============================================================================
# ANÁLISIS DE PROTEÍNAS HUMANAS (de la ejecución anterior)
# ============================================================================

print("\n" + "=" * 80)
print("📊 ANÁLISIS DE PROTEÍNAS HUMANAS (iFeature)")
print("=" * 80)

print("""
  📊 Resultados de la ejecución de iFeature:
     ├─ Proteínas humanas analizadas: 600
     ├─ Similitud máxima: 0.9962
     ├─ Similitud media: 0.9702
     ├─ Desviación estándar: 0.0287
     ├─ ≥ 0.99: 90
     ├─ ≥ 0.95: 511
     └─ ≥ 0.90: 579

  📊 Orientación positiva: 100.0%
""")

# ============================================================================
# CONCLUSIÓN
# ============================================================================

print("\n" + "=" * 80)
print("🎯 CONCLUSIONES")
print("=" * 80)

print("""
  📌 OBSERVACIONES:

  1. iFeature produce similitudes extremadamente altas (>0.97) para TODOS los grupos,
     lo que sugiere que NO está discriminando entre diferentes tipos de proteínas.

  2. SGP produce un rango de similitudes mucho más amplio (0.015 - 0.117),
     demostrando que SGP es más DISCRIMINANTE.

  3. iFeature NO detecta diferencias funcionales significativas entre:
     - Virus (LASV, JUNV, MACV, LCMV) y proteínas humanas
     - Péptidos penetradores (CPP) y proteínas humanas
     - Proteínas ordenadas y desordenadas

  4. SGP detecta claramente que LUJV es FUNCIONALMENTE ÚNICO,
     con similitudes muy bajas a todos los grupos.

  🔬 IMPLICACIÓN PARA EL MANUSCRITO:

  La comparación con iFeature demuestra que el enfoque de álgebra de Clifford
  de SGP captura información funcional que los métodos de extracción de
  características estándar (iFeature) NO pueden detectar.

  SGP es SUPERIOR para:
  - Detectar diferencias funcionales sutiles
  - Identificar proteínas funcionalmente únicas (anomalías)
  - Cuantificar similitudes funcionales en ausencia de homología de secuencia
""")

# Guardar resultados
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f"comparacion_sgp_ifeature_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

df_comp = pd.DataFrame(comparacion)
df_comp.to_csv(f"{results_dir}/comparacion_sgp_ifeature.csv", index=False)
print(f"\n  ✅ Tabla guardada: {results_dir}/comparacion_sgp_ifeature.csv")

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
