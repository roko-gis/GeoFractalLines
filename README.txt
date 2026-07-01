# GeoFractalLines

GeoFractalLines v1.0.0 is a QGIS-integrated Python framework for fractal and multifractal analysis of vector line geometries.  
It performs robust box-counting based scaling estimation, statistical validation, and multifractal spectrum reconstruction for geospatial datasets.

The software is designed for reproducible scientific analysis of spatial complexity in real-world GIS data.

---

# Usage

GeoFractalLines is integrated into QGIS as a Processing Toolbox tool with a graphical user interface (GUI) for parameter selection.

## How to use

1. Open:
   Processing Toolbox → GeoFractalLines → Fractal Analysis

2. Select input:
   - Input line vector layer

3. Configure parameters:
   - Fractal and multifractal settings are available in the GUI

4. Set output:
   - Choose Output Excel Report location and filename

5. Run analysis:
   - Click Run to start processing

---

## Output

The Excel report is automatically generated at the selected output location after processing is complete.

# Features

## Fractal Analysis
- Box-counting dimension for vector geometries
- Robust multi-offset grid sampling (reduced grid bias)
- Adaptive sampling based on geometry length
- Automatic scaling regime detection (AIC + R² + residual filtering)

## Statistical Validation
- Bootstrap confidence intervals (CI95)
- Surrogate testing against Complete Spatial Randomness (CSR)
- Isotropy analysis (directional stability check)
- Scaling quality metric

## Multifractal Analysis
- Partition function Z(q)
- Generalized dimensions Dq
- τ(q) estimation
- Information dimension D₁ from entropy scaling
- Legendre transform f(α) spectrum
- Spectrum reliability filtering

## GIS Integration
- Native QGIS processing (QgsVectorLayer, QgsGeometry)
- Memory layer output
- CRS-aware computation
- Batch and optional parallel processing

---

# Requirements

## Python dependencies
- numpy
- scipy
- pandas
- openpyxl

## QGIS environment
- QGIS 3.22+
- PyQt (bundled with QGIS)
- qgis.core

---

# Input Data

Supported input:
- QGIS LineString vector layers

Requirements:
- Valid geometries (auto-corrected when possible)
- Recommended: projected CRS (metric units)
- Minimum geometry length > 5 units

Preprocessing:
- Adaptive vertex sampling
- Duplicate point removal
- Optional interpolation for sparse geometries

---

# Output

## QGIS Layers

### 1_Fractal
Contains:
- fractal dimension (D)
- R² fit quality
- bootstrap confidence intervals
- lacunarity
- isotropy metrics
- CSR significance
- scaling quality index

### 2_Multifractal
Contains:
- Dq spectrum values
- τ(q) parameters
- α-range and spectrum shape metrics
- reliability flags

---

## Excel Report

Automatically generated file:

Fractal_Analysis_Report_YYYYMMDD_HHMMSS.xlsx

Includes:
- Executive summary
- Full descriptive statistics
- Correlation matrix
- D distribution
- All feature results
- Multifractal results
- Methodology description
- Validation against synthetic fractals

---

# Methodology

## Box-counting dimension

D = lim (ε → 0) [ log N(ε) / log(1/ε) ]

## Lacunarity

Λ = Var(N) / Mean(N)^2

## Multifractal formalism

Z_q(ε) = Σ p_i^q

τ(q) = -slope(log Z_q vs log(1/ε))

D_q = τ(q) / (q - 1)

## Information dimension

D₁ from entropy scaling:
D₁ = dH / d log(1/ε)

## Multifractal spectrum

f(α) obtained via Legendre transform of τ(q)


References
- Allain, C., & Cloitre, M. (1991). Characterizing the lacunarity of random and deterministic fractal sets. *Physical Review A*, 44(6), 3552–3558. DOI: https://doi.org/10.1103/PhysRevA.44.3552
- Halsey, T. C., Jensen, M. H., Kadanoff, L. P., Procaccia, I., & Shraiman, B. I. (1986). Fractal measures and their singularities. *Physical Review A*, 33(2), 1141–1151.DOI: https://doi.org/10.1103/PhysRevA.33.1141
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. W. H. Freeman.


# Validation

Validated using synthetic fractals:

- Koch curve (D ≈ 1.26186)
- Brownian motion
- Sierpinski-type structures
- Lévy C curve

Validation metrics:
- scaling accuracy
- R² fit quality
- relative error estimation

---

# Advantages over existing tools

Compared to FracLac, ImageJ plugins, and generic scripts:

- Direct vector geometry processing (no rasterization)
- Reduced grid bias via multi-offset sampling
- Automatic scaling regime detection
- Full statistical validation (bootstrap + CSR testing)
- Complete multifractal pipeline (not only fractal dimension)
- Reproducible scientific workflow
- QGIS-native integration with Excel export

---

# Applications

- Geomorphology (river networks, coastlines)
- Tectonics (fault systems)
- Urban morphology (road networks)
- Ecology (habitat fragmentation)
- Planetary science (Mars / Moon lineaments)
- Materials science (fracture patterns)

---

# Performance

- Supports batch processing of large vector layers
- Optional parallel execution
- Optimized NumPy-based computation

---

# Limitations

- Requires sufficient geometric density for scaling detection
- Not suitable for very short geometries (< 5 units)
- Recommended: projected coordinate systems only
- Multifractal analysis requires sufficient sampling density

---

# Version

GeoFractalLines v1.0.0

---

# Citation

If you use this software, please cite the corresponding reference provided in the source repository and the associated release.