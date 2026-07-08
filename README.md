markdown

# 🦠 SGP.py - Specular Grassmann-PIM v13.1

**Optimized Geometric Algebra Framework for Massive Protein Sequence Analysis (150M+ sequences)**

---

## 📋 Overview

**SGP.py** (Specular Grassmann-PIM) is a high-performance computational framework that applies **Clifford (geometric) algebra** to protein sequence analysis. It transforms amino acid sequences into 16-dimensional polarity vectors using the **PIM 3.0v** (Polarity Index Method) and performs advanced comparisons using:

- **Grassmann wedge products** (functional similarity)
- **Real geometric algebra reflection** (specular/charge-swap detection)
- **Clifford rotors** (rotation angles in biologically relevant planes)
- **Geometric product decomposition** (functional vs structural components)
- **Biological metric signature** (η-weighted interactions)
- **Commutator and anticommutator** (non-commutativity and metric symmetry)

The framework is optimized for **massive datasets** (151M+ sequences) using **streaming mode** with **O(1) memory** consumption and processes in **<13 hours** on standard hardware.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Streaming Mode** | Processes sequences one by one without storing all vectors in memory |
| **Online Statistics** | Welford algorithm for mean and covariance in a single pass |
| **Real Geometric Algebra** | Full Clifford algebra implementation with specular reflection |
| **Biological Metric** | Weighted signature assigning beneficial (+1), detrimental (-1), or neutral (0) to polarity transitions |
| **Multi-Plane Rotor Analysis** | 6 biologically relevant planes (hydrophobic, charge, polar, etc.) |
| **Adaptive Thresholding** | 5th percentile of intra-group similarity for membership classification; 0.99 default for n=1 groups |
| **Anomaly Detection** | Zero-shot classification using Mahalanobis distance |
| **Visualization** | Automatic generation of histograms and heatmaps |
| **Lightweight** | Runs on standard laptop (Intel Core i5, 16 GB RAM) with ~500 MB peak memory |
| **LSH Instant Search** | Locality-Sensitive Hashing for O(1) similarity queries |
| **Progressive Reservoir Sampling** | Stores only 10,000 representative vectors per group |
| **Monitoreo en Tiempo Real** | ProcessingTracker with ETA, rate, and memory usage display |

---

## 🧬 Methodology

### 1. PIM 3.0v Polarity Classification

Amino acids are mapped to 4 polarity categories:

| Category | Symbol | Amino Acids |
|----------|--------|-------------|
| Positively Charged | **P⁺** | H, K, R |
| Negatively Charged | **P⁻** | D, E |
| Neutral Polar | **N** | C, G, N, Q, S, T, Y |
| Non-Polar (Hydrophobic) | **NP** | A, F, I, L, M, P, V, W |

### 2. 16-Component Transition Matrix

Each consecutive amino acid pair contributes to a 4×4 transition matrix (rows = "from", columns = "to"):

| From \ To | P⁺ | P⁻ | N | NP |
|-----------|----|----|---|----|
| P⁺ | 0 | 1 | 2 | 3 |
| P⁻ | 4 | 5 | 6 | 7 |
| N | 8 | 9 | 10 | 11 |
| NP | 12 | 13 | 14 | 15 |

### 3. Biological Weights

The weight function assigns functional importance to each interaction:

| Interaction | Weight | Biological Significance |
|-------------|--------|------------------------|
| P⁺→P⁻, P⁻→P⁺ | 2.0 | Charge attraction |
| N→N | 1.5 | Hydrogen bonds |
| P⁺→N, N→P⁺, P⁻→N, N→P⁻ | 1.3 | Polar interactions |
| NP→NP | 1.0 | Hydrophobic packing |
| P⁺→P⁺, P⁻→P⁻ | 0.4 | Charge repulsion |
| Others | 0.7 | Mixed interactions |

### 4. Geometric Algebra Operators

**Wedge Product (∧):** Measures functional similarity between two polarity vectors. **Higher values indicate greater similarity** (1.0 = identical, 0.0 = orthogonal).

v ∧ w = (v·w) / (||v|| ||w||)
text


**Specular Reflection:** Detects charge-swapped functional equivalence (P⁺ ↔ P⁻) using the reflection formula:

v' = v - 2(v·n)n
text


A wedge product > 0.95 between reflected and target vectors indicates specular reflection.

**Geometric Product:** Decomposes into scalar (functional) and bivector (structural) components:

v *η w = (v·η w) + (v ∧η w)
text


**Interpretation of F/S Ratio:**
- **> 2.0**: "Functionally similar, structurally different"
- **< 0.5**: "Structurally similar, functionally different"
- **Between**: "Balanced: similar in both aspects"

**Clifford Rotors:** Compute rotation angles in 6 biologically relevant planes:
- Hydrophobic (N→N vs NP→NP)
- Charge (P⁺→P⁺ vs P⁻→P⁻)
- Opposite Charge (P⁺→P⁻ vs P⁻→P⁺)
- Polarity (N→N vs N→NP)
- Charge Transition (P⁺→N vs N→P⁺)
- Opposite Transition (P⁻→N vs N→P⁻)

---

## 📁 File Structure

### Required Input Files

The program expects **FASTA files (.fasta or .dat0 extension)** in the same directory:

| File | Group | Description |
|------|-------|-------------|
| `lujv_all.unico.dat0` | LUJV | Lujo virus glycoprotein (target) |
| `lasv_all.unico.dat0` | LASV | Lassa virus (Old World pathogen) |
| `junv_all.unico.dat0` | JUNV | Junín virus (New World pathogen) |
| `macv_all.unico.dat0` | MACV | Machupo virus (New World pathogen) |
| `lcmv_all.unico.dat0` | LCMV | Lymphocytic choriomeningitis virus (non-pathogenic control) |
| `partiallyorderedN.unico.dat0` | PARTIALLY_FOLDED | Proteins with ordered and disordered regions |
| `CPP.unico.dat0` | CPP | Cell-penetrating peptides |
| `NONCPP.unico.dat0` | NON_CPP | Non-cell-penetrating peptides |
| `unfolded.unico.dat0` | UNFOLDED | Fully disordered proteins |
| `reviewed_human.unico.dat0` | REVIEWED_HUMAN | Manually annotated human proteins (Swiss-Prot) |
| `unreviewed_human.unico.dat0` | UNREVIEWED_HUMAN | Computationally annotated human proteins (TrEMBL) |
| `senales.unico.dat0` | SIGNALING | Human signaling proteins (KW-0950) |
| `membrana.unico.dat0` | MEMBRANE | Human membrane proteins (KW-1000) |
| `enfermedad.unico.dat0` | DISEASE | Human disease-associated proteins (KW-0225) |
| `reviewed_virus.unico.dat0` | VIRUS_REVIEWED | Manually annotated viral proteins |
| `unreviewed_virus.unico.dat0` | VIRUS_UNREVIEWED | Computationally annotated viral proteins |
| `reviewed_all.unico.dat0` | REVIEWED_ALL | Manually annotated proteins (all species) |
| `unreviewed_all.unico.dat0` | UNREVIEWED_ALL | Computationally annotated proteins (all species) |

### Output Files

The program generates:

| File | Description |
|------|-------------|
| `01_similarity_distribution.png` | Histogram of wedge similarities (LUJV vs human proteome) |
| `01b_orientation_distribution.png` | Histogram of orientation signs |
| `02_max_diff_distribution.png` | Maximum component difference distribution |
| `03_tolerance_summary.png` | Bar chart of similarity thresholds |
| `04_rotor_angles_distribution.png` | Rotor angle histograms (hydrophobic & charge planes) |
| `08_lujv_vs_all_groups.png` | Bar chart of LUJV vs all groups |
| `heatmap_LUJV_vs_groups.png` | Heatmap of LUJV vs all groups |
| `comparison_LUJV_vs_all.csv` | Detailed comparison table |
| `similarity_matrix_groups.csv` | Inter-group similarity matrix |
| `lujv_vs_groups.csv` | LUJV vs group data |
| `top_similar_proteins_LUJV.csv` | Top 20 proteins most similar to LUJV |

---

## 🚀 Installation

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy psutil

Clone Repository
bash

git clone https://github.com/polancodf2024/lujv-sgp-analysis.git
cd lujv-sgp-analysis

Download Test Files

Place the required FASTA files (listed above) in the same directory as SGP8.py.
💻 Usage
Basic Run
bash

python SGP8.py

Configuration (Inside SGP8.py)
Parameter	Default	Description
BATCH_SIZE	100,000	Number of sequences per batch
MAX_STORED_PROTEINS_PER_GROUP	10,000	Max vectors stored in RAM per group
COHESION_CALC_SAMPLE_SIZE	500	Sample size for cohesion calculation
PROGRESS_REPORT_INTERVAL	100,000	Frequency of progress reports
USE_BIOLOGICAL_METRIC	True	Use η-weighted metric signature
N_BOOTSTRAP	100	Bootstrap iterations for confidence intervals
TOLERANCE	0.001	Tolerance for maximum component difference
TOP_N_PROTEINS	20	Number of top similar proteins to report
Output Example
text

================================================================================
🦠 MIRROR-PIM: GRASSMANN-PIM WITH REAL GEOMETRIC ALGEBRA (v13.1)
   PROCESAMIENTO COMPLETO SIN LÍMITES
   Monitoreo en tiempo real + Gestión de memoria optimizada
================================================================================
⏰ Inicio: 2026-07-08 14:30:00

  ⚙️ CONFIGURACIÓN DE PROCESAMIENTO:
     ├─ Tamaño de lote: 100,000 secuencias
     ├─ Muestra por grupo: 10,000 proteínas
     ├─ Intervalo de reporte: 100,000 secuencias
     └─ Sin límites de secuencias por grupo

📂 CARGANDO ARCHIVOS FASTA (SIN LÍMITES)...
================================================================================

  📂 Procesando LUJV desde lujv_all.unico.dat0...
  ✅ LUJV: 1 válidas de 1 totales | Almacenadas: 1 (muestra) | Cohesión: 1.000000 | Umbral: 0.9900

  📊 Progreso [REVIEWED_HUMAN]: 100,000 secuencias | Válidas: 85,234 (85.2%) | Rate: 12,345 seq/s | ETA: 2.5h
  💾 Memoria: 245 MB | Almacenadas: 10,000 (muestra)
...

📊 Interpretation Guide
Wedge Similarity (∧)
Value	Interpretation
1.000	Identical polarity profiles
0.500 - 1.000	High functional similarity
0.100 - 0.500	Moderate functional similarity
< 0.100	Low functional similarity
< 0.050	Very low / orthogonal

Note: Higher values indicate greater similarity.
Geometric Product Decomposition
Component	Meaning	Biological Interpretation
Scalar (Functional)	Weighted dot product with η	Alignment of beneficial interactions
Bivector (Structural)	Oriented area	Sequential arrangement of polarity transitions
F/S Ratio	Functional/Structural	> 2.0 = "Functionally similar, structurally different"; < 0.5 = "Structurally similar, functionally different"
Orientation Sign
Sign	Meaning
+1	Same orientation (same order of bivector components)
-1	Opposite orientation (mirrored order of bivector components)
Adaptive Threshold

Defined as the 5th percentile of intra-group wedge products. A query protein is considered a member of the group if its wedge similarity ≥ adaptive threshold.

Special case: For groups with a single member (n = 1), the 5th percentile is mathematically undefined. The adaptive threshold is set to a conservative default value of 0.99, reflecting the expected self-similarity of a single vector with a 1% tolerance margin.
🔬 What SGP Does and Does Not Do
What SGP Does NOT Do

    Does NOT perform sequence alignment (BLAST, Smith-Waterman, etc.)

    Does NOT compare every protein against every other protein (would be O(N²))

    Does NOT store all protein vectors in memory (only 10,000 per group)

    Does NOT require multiple sequence alignments (MSA)

    Does NOT predict 3D structure

    Does NOT use GPU acceleration or distributed computing

What SGP Actually Does

    Reads sequences in streaming mode (line-by-line, O(1) memory)

    Computes a 16-component PIM vector for each sequence (simple O(L) operation)

    Updates online statistics for each group (Welford algorithm, single pass)

    Stores a representative sample of 10,000 vectors per group (reservoir sampling)

    Compares groups using centroids (only 18×17/2 = 153 comparisons)

    Provides instant similarity search via LSH (O(1) retrieval)

🧪 Validation: Synthetic Anomaly Detection

A synthetic random protein (sampled from uniform distribution) is correctly classified as an anomaly:

    Max wedge similarity: 0.145 (well below adaptive threshold)

    Probability of belonging: 0.000 for all groups

This demonstrates the framework's ability to detect "unnatural" polarity profiles without prior training.
🔬 Biological Applications

The framework has been validated on the Lujo virus (LUJV) glycoprotein and can be applied to:

    Pathogen characterization: Identifying unique polarity fingerprints of emerging viruses

    Vaccine safety screening: Detecting molecular mimicry with host proteins

    Synthetic protein quality control: Verifying intended polarity profiles

    Re-annotation of hypothetical proteins: Assigning functional classes without sequence homology

    Metagenomic screening: Detecting viral sequences in complex samples

📈 Performance
Metric	Value
Sequences processed	151,066,923
Processing time	< 13 hours
Processing rate	~3,200 sequences/sec
Peak memory usage	~500 MB
Largest file processed	149,234,636 sequences (~170 GB uncompressed)
Hardware	Intel Core i5, 16 GB RAM, 1 TB SSD
OS	Fedora 43 Linux (64-bit)
GPU required	No
Distributed computing	No
📚 Citation

If you use SGP.py in your research, please cite:

    Polanco, C., Uversky, V. N., Huberman, A., Rios Castro, M., Pimentel-Hernández, C., Martínez-Garcia, M., Hernández-Lemus, E., Martínez Memije, R., Becerra Luna, B., Solís Nájera, S. E., Calderón Dorantes, O. U., Vargas Alarcón, G., Díaz González, J. L., & Buhse, T. (2026). Specular Grassmann-PIM and Protein Intrinsic Disorder Predisposition of the Lujo virus glycoprotein: A geometric algebra framework for viral protein characterization. iMetaMed, (in press).

📜 License

Proprietary - All Rights Reserved

This software is the intellectual property of:

    Polarity Index Method (PIM) - Mexico, 2018

    PIM 3.0v - Mexico, 2023

    Specular Grassmann-PIM (SGP) - Mexico, 2026

All trademarks and copyright rights are reserved. Use of this software requires explicit permission from the corresponding author.
👥 Authors
Author	Affiliation
Carlos Polanco (Corresponding)	Instituto Nacional de Cardiología Ignacio Chávez, UNAM
Vladimir N. Uversky	University of South Florida
Alberto Huberman	Instituto Nacional de Ciencias Médicas y Nutrición
Martha Rios Castro	Instituto Nacional de Cardiología
Claudia Pimentel-Hernández	Instituto Nacional de Pediatría
Mireya Martínez-Garcia	UNAM
Juan Luciano Díaz González	UNAM
Raul Martínez Memije	Instituto Nacional de Cardiología
Brayans Becerra Luna	Instituto Nacional de Cardiología
Enrique Hernández-Lemus	Instituto Nacional de Medicina Genómica
Sergio Enrique Solís Nájera	UNAM
Osvaldo Uriel Calderón Dorantes	UNAM
Gilberto Vargas Alarcón	Instituto Nacional de Cardiología
Thomas Buhse (†)	Universidad Autónoma del Estado de Morelos
📧 Contact

Carlos Polanco
Email: polanco@unam.mx
Department of Scientific Innovation
Instituto Nacional de Cardiología Ignacio Chávez
Mexico City, 14080, Mexico

GitHub Repository: https://github.com/polancodf2024/lujv-sgp-analysis
