# Laboratory of the Physics of Biological Systems (LPBS) - *C. elegans* Movement Analysis

## Project Overview

This project implements a comprehensive machine learning pipeline for analyzing movement patterns of *Caenorhabditis elegans* worms tracked on plates. The system can classify worms that have been administered drugs versus control worms and predict death proximity.

**📖 For detailed methodology, architecture decisions, and data leakage prevention strategies, see [METHODOLOGY.md](METHODOLOGY.md)**

## Project Architecture

```
LPBS/
├── 📁 data/                              # Raw experimental data
│   ├── 📁 Lifespan/                      # Lifespan experiments
│   │   ├── 📁 COMPANY DRUG/              # Drug treatment experiments
│   │   │   ├── 📁 COMPANY DRUG- (control)  # Control group
│   │   │   └── 📁 COMPANY DRUG+/           # Treatment group
│   │   └── 📁 TERBINAFINE/                # Terbinafine experiments
│   │       ├── 📁 TERBINAFINE- (control)   # Control group
│   │       └── 📁 TERBINAFINE+/            # Treatment group
│   ├── 📁 Optogenetics/                  # Optogenetics experiments
│   │   ├── 📁 ATR-/ & ATR+/              # ATR experiments (excluded)
│   └── 📁 Skeletonization code/          # Image processing utilities
│
├── 📁 preprocessed_data/                 # Processed trajectory data
│   ├── 📁 full/                          # Complete trajectory files
│   └── 📁 segments/                      # segment files
│
├── 📁 feature_data/                      # Extracted features
│   ├── 📄 full_features.csv              # Features from full trajectories
│   └── 📄 segments_features.csv          # Features from segments
│
├── 📁 EDA/                               # Exploratory Data Analysis
│   ├── 📄 correlation_analysis.py        # Feature correlation analysis
│   ├── 📄 eda_extracted_features.py      # Feature distribution analysis
│   ├── 📄 feature_comparison_analysis.py  # Drug vs control comparison
│   ├── 📄 temporal_segment_analysis.py   # Time series analysis
│   └── 📄 run_all_analyses.py            # Run all EDA scripts
│
├── 📄 preprocessing.py                   # Data preprocessing pipeline
├── 📄 feature_extraction.py             # Feature engineering
├── 📄 data_loader.py                     # Unified data loading interface
│
├── 📄 feature_segment_classification.py  # Feature-based ML models
├── 📄 ts_segment_classification.py      # Time series ML models (CNN/LSTM)
├── 📄 death_proximity_predictor.py      # Death classification (binary)
├── 📄 death_proximity_regressor.py      # Death regression (segments remaining)
│
├── 📄 Analysis & Utilities
├── 📄 feature_importance_analysis.py    # Feature selection analysis
├── 📄 first_vs_last_segment_analysis.py # Aging analysis
├── 📄 compare_proximity_thresholds.py   # Death prediction optimization
├── 📄 get_coordinate_bounds.py          # Data bounds calculation
```

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/lysandre-c/LPBS-Celegans.git
cd LPBS
pip install -r requirements.txt
```

### 2. Data Processing Pipeline

```bash
# 1. Preprocess raw data (splits into segments, normalizes)
python preprocessing.py

# 2. Extract features from preprocessed trajectories
python feature_extraction.py
```

### 3. Model Training & Evaluation

#### Feature-Based Classification
```bash
# Train feature-based models (Random Forest, Gradient Boosting, MLP)
python feature_segment_classification.py
```

#### Time Series Classification
```bash
# Train time series models (CNN, LSTM)
python ts_segment_classification.py
```

#### Death Proximity Prediction
```bash
# Classify segments as "close to death" or not (binary classification)
python death_proximity_predictor.py

# Predict number of segments remaining until death (regression)
python death_proximity_regressor.py
```

## Core Components

### Data Processing (`preprocessing.py`)

**Purpose**: Converts raw experimental data into analysis-ready format

**Key Features**:
- **Segment Creation**: Splits trajectories into segments (900 frames)
- **Data Cleaning**: Interpolates small gaps, removes large gaps
- **Normalization**: Coordinates to [0,1], angles to [-1,1]
- **Death Integration**: Uses lifespan data to exclude post-death segments
- **Quality Control**: Caps extreme speeds

**Output**: 
- `preprocessed_data/full/`: Complete worm trajectories
- `preprocessed_data/segments/`: Individual segments

### Feature Engineering (`feature_extraction.py`)

**Purpose**: Extracts 60+ biologically meaningful features from movement data

**Feature Categories**:

| Category | Features | Description |
|----------|----------|-------------|
| **Basic Movement** | Mean/std/max speed, distance, pause time | Fundamental locomotion metrics |
| **Turning Behavior** | Turning angles, frequency, meandering ratios | Path complexity and exploration |
| **Physics-Inspired** | Kinetic energy proxy, movement efficiency | Energy and momentum concepts |
| **Statistical** | Skewness, kurtosis, entropy | Distribution characteristics |
| **Frequency Domain** | FFT-based dominant frequencies | Periodic movement patterns |
| **Wavelet Analysis** | Multi-scale decomposition | Time-frequency features |
| **Behavioral States** | Activity levels, roaming vs dwelling | High-level behavior classification |
| **Complexity** | Fractal dimensions, frenetic scores | Movement pattern complexity |

**Output**: 
- `feature_data/segments_features.csv`: Features for each segment
- `feature_data/full_features.csv`: Features for complete trajectories

### Data Loading (`data_loader.py`)

**Purpose**: Unified interface for accessing processed data with proper CV splitting

**Key Features**:
- **File-based splitting**: Prevents data leakage in cross-validation
- **Custom labeling**: Death proximity, original labels, time-based labels
- **Flexible filtering**: First N segments, specific time windows
- **Multiple formats**: Features, time series, metadata

**Usage Example**:
```python
from data_loader import LPBSDataLoader

loader = LPBSDataLoader()
X, y, groups = loader.load_segment_features()  # worm-level groups inside
folds = loader.create_cv_splits(X, y, groups, n_splits=5)

for fold in folds:
    ...
```

## Machine Learning Models

### 1. Feature-Based Models (`feature_segment_classification.py`)

**Algorithms**:
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Sequential weak learner boosting  
- **Neural Network (MLP)**: Deep learning with regularization
- **K-Nearest Neighbors**: Instance-based learning

**Weighted Voting Strategies**:
- **Uniform**: Equal weight to all segments
- **Confidence**: Weight by prediction confidence
- **Late segments**: Higher weight to final segments
- **Last N segments**: Only consider final N segments

**Performance**: see comments at the end of the file

### 2. Time Series Models (`ts_segment_classification.py`)

**Deep Learning Architectures**:

#### 1D CNN
```python
Conv1D(128, kernel_size=3) → ReLU → 
Conv1D(256, kernel_size=3) → ReLU → 
GlobalAvgPool1D → Dense(2)
```

#### LSTM
```python
LSTM(128, num_layers=2) → Dense(2)
```

**Input Features**: 4D time series (x, y, speed, turning_angle)

### 3. Death Proximity Models

#### Binary Classification (`death_proximity_predictor.py`)

**Purpose**: Classify segments as "close to death" (last N segments) or "far from death"

**Model Performance**: See detailed results at the end of the file

#### Regression (`death_proximity_regressor.py`)

**Purpose**: Predict the continuous number of segments remaining until death

**Target Types**:
- `segments_from_end`: Number of segments remaining (interpretable)
- `life_percentage_remaining`: Percentage of life remaining (0-100%)
- `normalized_position`: Normalized position from death (0-1)

**Model Performance**: ~16% of life_percentage_remaining (MAE with cross-validation)

**Model Performance**:
see at the end of the file

## Analysis Tools

### Feature Importance Analysis (`feature_importance_analysis.py`)

**Methods**:
- **Random Forest Importance**: Built-in feature importance
- **Permutation Importance**: Model-agnostic feature ranking
- **Statistical Tests**: F-scores, mutual information
- **Combined Scoring**: Weighted ensemble of methods

**Output**: Ranked list of most discriminative features

### Aging Analysis (`first_vs_last_segment_analysis.py`)

**Purpose**: Identify features that change significantly with age

### Death Threshold Optimization (`compare_proximity_thresholds.py`)

**Purpose**: Compare performance of the predictive model depending on the definition of "close to death" (last X segments before death)

## 📈 Exploratory Data Analysis (EDA)

The `EDA/` directory contains comprehensive analysis scripts:

| Script | Purpose | Output |
|--------|---------|--------|
| `eda_extracted_features.py` | Feature distributions and statistics | Distribution plots, summary stats |
| `correlation_analysis.py` | Feature correlation matrices | Correlation heatmaps, clusters |
| `feature_comparison_analysis.py` | Drug vs control differences | Statistical comparisons, effect sizes |
| `temporal_segment_analysis.py` | Time series patterns | Temporal visualizations |
| `run_all_analyses.py` | Execute all EDA scripts | Comprehensive analysis suite |