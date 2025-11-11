# Laboratory of the Physics of Biological Systems (LPBS) - *C. elegans* Movement Analysis

## Project Overview

This project implements a comprehensive machine learning pipeline for analyzing movement patterns of *Caenorhabditis elegans* worms tracked on plates. The system can classify worms that have been administered drugs versus control worms and predict death proximity.

**📖 For detailed methodology, architecture decisions, data leakage prevention strategies, and main results see [METHODOLOGY.md](METHODOLOGY.md) or [REPORT.md](REPORT.md).**

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