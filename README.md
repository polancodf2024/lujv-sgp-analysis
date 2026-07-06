# 🦠 SGP5.py - Specular Grassmann-PIM v11

**Optimized Geometric Algebra Framework for Massive Protein Sequence Analysis (150M+ sequences)**

---

## 📋 Overview

**SGP5.py** (Specular Grassmann-PIM) is a high-performance computational framework that applies **Clifford (geometric) algebra** to protein sequence analysis. It transforms amino acid sequences into 16-dimensional polarity vectors using the **PIM 3.0v** (Polarity Index Method) and performs advanced comparisons using:

- **Grassmann wedge products** (functional similarity)
- **Real geometric algebra reflection** (specular/charge-swap detection)
- **Clifford rotors** (rotation angles in biologically relevant planes)
- **Geometric product decomposition** (functional vs structural components)
- **Biological metric signature** (η-weighted interactions)

The framework is optimized for **massive datasets** (150M+ sequences) using **streaming mode** with **O(1) memory** consumption.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Streaming Mode** | Processes sequences one by one without storing all vectors in memory |
| **Online Statistics** | Welford algorithm for mean and covariance in a single pass |
| **Real Geometric Algebra** | Full Clifford algebra implementation with specular reflection |
| **Biological Metric** | Weighted signature assigning beneficial (+1), detrimental (-1), or neutral (0) to polarity transitions |
| **Multi-Plane Rotor Analysis** | 6 biologically relevant planes (hydrophobic, charge, polar, etc.) |
| **Adaptive Thresholding** | 5th percentile of intra-group similarity for membership classification |
| **Anomaly Detection** | Zero-shot classification using Mahalanobis distance |
| **Visualization** | Automatic generation of histograms and heatmaps |
| **Lightweight** | Runs on standard laptop (Intel Core i5, 16 GB RAM) |

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

### 3. Geometric Algebra Operators

**Wedge Product (∧):** Measures functional similarity between two polarity vectors:

v ∧ w = (v·w) / (||v|| ||w||)
text


**Specular Reflection:** Detects charge-swapped functional equivalence (P⁺ ↔ P⁻):

v' = v - 2(v·n)n
text


**Geometric Product:** Decomposes into scalar (functional) and bivector (structural) components:

v *η w = (v·η w) + (v ∧η w)
text


**Clifford Rotors:** Compute rotation angles in 6 biologically relevant planes.

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
| `heatmap_LUJV_vs_groups.png` | Heatmap of LUJV vs all groups |
| `comparison_LUJV_vs_all.csv` | Detailed comparison table |
| `similarity_matrix_groups.csv` | Inter-group similarity matrix |
| `lujv_vs_groups.csv` | LUJV vs group data |

---

## 🚀 Installation

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy

Clone Repository
bash

git clone https://github.com/yourusername/sgp3.git
cd sgp3

Download Test Files

Place the required FASTA files (listed above) in the same directory as SGP5.py.
💻 Usage
Basic Run
bash

python SGP5.py

Configuration (Inside SGP5.py)
Parameter	Default	Description
STREAMING_MODE	True	Enable O(1) memory streaming for large files
COHESION_SAMPLE_SIZE	10000	Number of random pairs for intra-group similarity
USE_BIOLOGICAL_METRIC	True	Use η-weighted metric signature
N_BOOTSTRAP	100	Bootstrap iterations for confidence intervals
TOLERANCE	0.001	Tolerance for maximum component difference
Output Example
text

================================================================================
🦠 MIRROR-PIM: GRASSMANN-PIM WITH REAL GEOMETRIC ALGEBRA (v11)
   OPTIMIZED FOR MASSIVE DATASETS (150M+ sequences)
   Streaming mode: O(1) memory, processes sequences one by one
================================================================================
⏰ Start: 2026-07-05 14:30:00

  ⚙ CONFIGURATION:
     ├─ Space dimension: 16 components (polarity pairs)
     ├─ Biological weighting: YES
     ├─ Streaming mode: YES
     ├─ Cohesion sample size: 10000 random pairs
     ├─ LSH hash: Enabled
     ├─ Clifford rotors: Enabled (6 biological planes)
     └─ REAL GEOMETRIC ALGEBRA: Enabled (v11)

📂 LOADING FASTA FILES (STREAMING MODE)...
   ⚠ Large files will be processed without storing all vectors in memory

  Streaming LUJV from lujv_all.unico.dat0...
    Processed 1,000,000 sequences, 856,234 valid...
     LUJV: n=1, cohesion=1.000000 ± 0.000000, adaptive_threshold=0.9900, metric_norm=0.3190(+)
...

📊 Interpretation Guide
Wedge Similarity (∧)
Value	Interpretation
1.000	Identical polarity profiles
0.500 - 1.000	High functional similarity
0.100 - 0.500	Moderate functional similarity
< 0.100	Low functional similarity
< 0.050	Very low / orthogonal
Geometric Product Decomposition
Component	Meaning	Biological Interpretation
Scalar (Functional)	Weighted dot product	Alignment of beneficial interactions
Bivector (Structural)	Oriented area	Sequential arrangement of polarity transitions
F/S Ratio	Functional/Structural	> 2.0 = functionally similar, structurally different; < 0.5 = structurally similar, functionally different
Orientation Sign
Sign	Meaning
+1	Same orientation (same order of bivector components)
-1	Opposite orientation (mirrored order of bivector components)
Adaptive Threshold

Defined as the 5th percentile of intra-group wedge products. A query protein is considered a member of the group if its wedge similarity ≥ adaptive threshold.
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
Processing time	< 60 seconds (streaming mode)
Memory usage	O(1) constant
Hardware	Intel Core i5, 16 GB RAM, 1 TB SSD
OS	Fedora 43 Linux (64-bit)
📚 Citation

If you use SGP5.py in your research, please cite:

    Polanco, C., Uversky, V. N., Huberman, A., et al. (2026). Specular Grassmann-PIM and Protein Intrinsic Disorder Predisposition of the Lujo virus glycoprotein: A geometric algebra framework for viral protein characterization. iMetaMed, (in press).

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
Department of Scientific Innovation, Instituto Nacional de Cardiología Ignacio Chávez
Mexico City, 14080, Mexico
