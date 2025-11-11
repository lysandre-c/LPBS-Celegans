# LPBS Methodology & Architecture Documentation

## Table of Contents
1. [Data Flow Pipeline](#data-flow-pipeline)
2. [Preventing Data Leakage](#preventing-data-leakage)
3. [Cross-Validation Strategy](#cross-validation-strategy)
4. [Model Architectures](#model-architectures)
5. [Evaluation Methodology](#evaluation-methodology)
6. [Design Choices & Rationale](#design-choices--rationale)
7. [Validation & Verification](#validation--verification)

---

## Data Flow Pipeline

### Phase 1: Raw Data → Preprocessed Trajectories

```
Raw CSV Files (x, y, speed)
    │
    ├─ Frame Reset handling (recording artifacts)
    ├─ Turning angle calculation
    ├─ Speed Capping (remove tracking errors)
    ├─ Gap Cleaning (interpolate small gaps, remove large gaps)
    ├─ Segment creation (split into 900-frame windows)
    ├─ Death integration (exclude post-death segments)
    ├─ Coordinate Normalization (scale to [0,1])
    └─ Angle Normalization (scale to [-1,1])
    │
    ↓
Preprocessed Data
    ├── full/              (complete trajectories)
    └── segments/          (900-frame segments)
```

**Key Decisions**:

1. **Segment Length (900 frames)**: 
   - This segmentation matches the experimental design and makes it natural to isolate segments in this way
   - Allow to create more data to train on

2. **Gap Handling**:
   - Interpolate linearly gaps ≤6 frames
   - Remove gaps ≥7 frames

3. **Speed Capping (10.0 max)**:
   - Removes tracking artifacts (sudden jumps)
   - Prevents outliers from dominating statistics

4. **Post-Death Exclusion**:
   - Uses lifespan data to find death frame
   - Excludes all segments after death

### Phase 2: Preprocessed Data → Features

```
Preprocessed Segments (time series)
    │
    ├─ Basic Movement (mean/std/max speed, distance)
    ├─ Turning Behavior (angles, frequency, meandering)
    ├─ Physics-Inspired (kinetic energy, efficiency)
    ├─ Statistical (skewness, kurtosis, entropy)
    ├─ Frequency Domain (FFT dominant frequencies)
    ├─ Wavelet Analysis (multi-scale decomposition)
    ├─ Behavioral States (roaming, dwelling, activity levels)
    └─ Complexity (fractal dimension, frenetic score)
    │
    ↓
Feature Matrix (60+ features per segment)
```

### Phase 3: Features → Model Training & Evaluation

```
Feature Matrix + Labels + Metadata
    │
    ├─ File-Based Grouping (group segments by worm)
    ├─ Stratified K-Fold Split (maintain class balance, split at worm level)
    │
    ↓
Training Set (80%)          Test Set (20%)
    │                            │
    ├─ Model Training            │ (ISOLATED)
    ├─ Hyperparameter Tuning     │ (NEVER SEEN)
    ├─ Feature Selection         │ (NO LEAKAGE)
    │                            │
    └──────────────┬─────────────┘
                   │
                   ↓
            Final Predictions
                   │
                   ├─ Segment-Level Prediction
                   ↓
                   ├─ Worm-Level Prediction (aggregation by voting)
                   ↓
                   └─ Group-Level Prediction (aggregation by voting)
```

---

## Preventing Data Leakage

### 1. File-Based Cross-Validation Splits

**Problem**: If segments from the same worm appear in both training and test sets, the model can memorize worm-specific patterns rather than learning generalizable features.

**Solution**: Split at the **worm level** (file level), not segment level.

### 2. No Test Data in Preprocessing Parameters

**Problem**: If normalization/scaling parameters are calculated from the entire dataset, test data influences training.

**Note / Solution**: In this pipeline, normalization is done during preprocessing (before train/test split) because it's based on **physical bounds** (coordinate range, angle range), not data statistics. This is safe and doesn't leak information.

### 3. Group Prediction

For group-level classification, groups are formed after predictions.

Class balancing coefficients are calculated from training predictions only. Not test.

---

## Cross-Validation Strategy

### 5-Fold Stratified Cross-Validation (File-Based)

**Setup**:
```
Total: 104 worms
  ├─ Control: 52 worms
  └─ Treatment: 52 worms

Each Fold:
  ├─ Training: ~83 worms (80%)
  └─ Test: ~21 worms (20%)
```

**Stratification**: Ensures each fold maintains the same class distribution (50% control, 50% treatment).

**Reproducibility**: Fixed `random_state=42` ensures same splits across runs.

### Evaluation Metrics Across Folds

For each fold:
- Train model on training set
- Predict on test set
- Calculate metrics (accuracy, precision, recall, F1, AUC)

Final reported metrics:
- **Mean ± Std** across folds
- Shows both performance and consistency

---

## Model Architectures

### 1. Feature-Based Models

- Random Forest
- Gradient Boosting
- Multi-Layer Perceptron (MLP)

### 2. Time Series Models

#### 1D Convolutional Neural Network (CNN)
```
Input: (batch_size, time_steps, features)
    ↓
Conv1D(128, kernel_size=3) → ReLU → Dropout(0.3)
    ↓
Conv1D(256, kernel_size=3) → ReLU → Dropout(0.3)
    ↓
GlobalAveragePooling1D()
    ↓
Dense(2, activation='softmax')
```

#### Long Short-Term Memory (LSTM)
```
Input: (batch_size, time_steps, features)
    ↓
LSTM(128, return_sequences=True) → Dropout(0.3)
    ↓
LSTM(128, return_sequences=False) → Dropout(0.3)
    ↓
Dense(2, activation='softmax')
```

### 3. Death Proximity Models

#### Binary Classification (`death_proximity_predictor.py`)

Treats death proximity as a binary problem:
- **Class 0**: Far from death (not in last N segments)
- **Class 1**: Close to death (in last N segments)

Threshold N is tunable (tested: 1, 3, 5, 10, 15, 20, 25, 30 and 40 segments).


#### Regression (`death_proximity_regressor.py`)

Predicts continuous values:
- **Target 1**: `segments_from_end` (0 to max_segments)
- **Target 2**: `life_percentage_remaining` (0% to 100%)
- **Target 3**: `normalized_position` (0.0 to 1.0)

**Model Architectures**:
- Random Forest Regressor
- Gradient Boosting Regressor  
- Ridge/Lasso Regression
- MLP Regressor

**Why Multiple Target Types?**:
- Different scales may be easier to learn
- `segments_from_end`: Most interpretable
- `life_percentage_remaining`: Normalized across worms
- `normalized_position`: Normalized across worms

### Voting strategies

**Desgin choice**: To increase the amount of training data, we chose to train the models at the segment level. To make predictions at higher levels (such as worm level or group level), we aggregated the segment-level results using a voting approach. This strategy helps enhance the system's predictive power by mitigating the noise from individual predictions. We also tested several voting schemes to evaluate their effectiveness.

#### Strategies tested (segment → worm)

1. **Uniform**: All segments have equal weight.

2. **Confidence**: Each segment is weighted by its classification confidence.

3. **Early Sement**: Emphasize early behavior. Segments are weighted linearly with decreasing importance for later segments, from 1.5 (first segment) to 0.5 (last segment).

4. **Late Segments**: Emphasize late behavior. Segments are weighted linearly with increasing importance for later segments, from 0.5 (first segment) to 1.5 (last segment).

5. **Late Segments Confidence**: Each segment's weight is the product of a linearly increasing value (from 0.5 to 1.5, as in '4. Late Segments') and its confidence.

6. **Last N Segments**: Only the last N segments have weight 1; all earlier segments have weight 0.

7. **Last N Segments Confidence**: Only the last N segments contribute, and each is weighted by its confidence; all earlier segments have weight 0.

TODO: à essayer de faire un MLP qui trouve lui même les poids pour le voting

#### Strategy (worm → group)

For worm-to-group prediction predictions, we use a confidence voting strategy adjusted with class-balancing coefficients.

example:
```python
# Confidence voting
Worm 1: pred=0, confidence=0.9  →  contributes 0.9 to class 0
Worm 2: pred=0, confidence=0.7  →  contributes 0.7 to class 0
Worm 3: pred=1, confidence=0.6  →  contributes 0.6 to class 1
Worm 4: pred=0, confidence=0.8  →  contributes 0.8 to class 0
Worm 5: pred=1, confidence=0.5  →  contributes 0.5 to class 1

weighted_vote_0 = 0.9 + 0.7 + 0.8 = 2.4
weighted_vote_1 = 0.6 + 0.5 = 1.1

# Class-balancing coefficients
adjusted_vote_0 = weighted_vote_0 * class_balancing_coef_0
adjusted_vote_1 = weighted_vote_1 * class_balancing_coef_1

# Prediction
group_prediction = int(adjusted_vote_1 > adjusted_vote_0)
```


## Design Choices & Rationale

### Why Segment-Based Analysis?

**Advantages**:
1. More training samples (79 segments per worm on average)
2. Captures temporal dynamics (aging trajectory)
3. Enables fine-grained death proximity analysis
4. Allows weighting strategies (emphasize late segments)

**Challenges**:
1. Segments from same worm are not independent
2. Need careful CV splitting to avoid leakage
3. Aggregation needed for worm-level predictions

**Solution**: File-based CV + weighted voting strategies

### Why Both Feature-Based and Time Series Models?

**Feature-Based Models** (Random Forest, Gradient Boosting, MLP on features):
- **Pros**: Interpretable features, feature importance analysis, faster training
- **Cons**: Manual feature engineering, may miss temporal patterns

**Deep Learning Models** (CNN, LSTM on raw trajectories):
- **Pros**: Automatic feature learning, captures temporal dynamics, no feature engineering
- **Cons**: Black box, requires more data, harder to interpret

**Results**: Use both and compare. Feature-based outperformed deep learning models probably due to the high dimentionality of timeseries and not sufficient data.

### Why Fixed Preprocessing Parameters?

Coordinate normalization uses **data-independent bounds**:
```python
x_norm = (x - x_min) / (x_max - x_min)  # x_min, x_max from plate dimensions
angle_norm = angle / 180.0  # -180° to 180° → -1 to 1
```

**Why this is safe**: Bounds are based on physical constraints (plate size, angle range), not data statistics. No information leakage. (Data dependent normalization would require proper train-test split statistics)