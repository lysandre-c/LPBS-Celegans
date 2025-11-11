# Report

## Table of Contents
1. [Data Preprocessing](#data-preprocessing)
2. [Drugged / undrugged Classification](#2-drugged--undrugged-classification)
3. [Death Prediction](#3-death-prediction)
4. [Results and Models Analysis](#4-results-and-models-analysis)

---

## 1. Data Preprocessing

### Phase 1: Raw Data → Preprocessed Data (`preprocessing.py`)

- **Input:** Raw CSV Files (x, y, speed)
- **Processing Steps:**
  - Frame Reset handling
  - Turning angle calculation
  - Speed capping (removes outliers)
  - Gap cleaning (interpolate gaps with less than 7 consecutive missing values; remove large gaps)
  - Segment creation (split into 900-frame windows)
  - Exclude post-death segments
  - Coordinate normalization (scale to [0,1]) (used min and max values from `get_coordinate_bounds.py`)
  - Angle normalization (scale to [-1,1])
- **Output:** Preprocessed Data
  - `full/` folder: complete trajectories
  - `segments/` folder: 900-frame segments

### Phase 2: Preprocessed Data → Features (`feature_extraction.py`)

- **Input:** Preprocessed Data
- **Feature Extraction Steps:**
  - Basic Movement: mean, standard deviation, max speed, distance
  - Turning Behavior: angles, frequency, meandering
  - Physics-Inspired: kinetic energy, efficiency
  - Statistical: skewness, kurtosis, entropy
  - Frequency Domain: FFT dominant frequencies
  - Wavelet Analysis: multi-scale decomposition
  - Behavioral States: roaming, dwelling, activity levels
  - Complexity: fractal dimension, frenetic score
- **Output:** `feature_data` folder: Feature Matrix (60+ features per segment)

### Preventing Data Leakage (`data_loader.py`)

- Utility file used to load data in other scripts.
- File-based cross-validation splits:
  - Performance metrics are averaged over all folds for robust evaluation.
  - Problem: If segments from the same worm appear in both training and test sets, the model can memorize worm-specific patterns rather than learning generalizable features.
  - Solution: Split at the worm level (file level), not segment level. Prevent data leakage.


## 2. Drugged / undrugged Classification

### Feature-based Classification (`feature_segment_classification.py`)

- Uses top 10 features to predict if a segment corresponds to a worm that has been drugged or not.
- Implements several models:
  - Random Forest
  - Gradient Boosting
  - Multi-Layer Perceptron (MLP)
- Applies a voting strategy to aggregate the segment-level predictions to create worm-level predictions. Several strategies are implemented:
  - Uniform: All segments have equal weight.
  - Confidence: Each segment is weighted by its classification confidence.
  - Early Segment: Emphasize early behavior. Segments are weighted linearly with decreasing importance for later segments, from 1.5 (first segment) to 0.5 (last segment).
  - Late Segments: Emphasize late behavior. Segments are weighted linearly with increasing importance for later segments, from 0.5 (first segment) to 1.5 (last segment).
  - Late Segments Confidence: Each segment's weight is the product of a linearly increasing value (from 0.5 to 1.5, as in 'Late Segments') and its confidence.
  - Last N Segments: Only the last N segments have weight 1; all earlier segments have weight 0.
  - Last N Segments Confidence: Only the last N segments contribute, and each is weighted by its confidence; all earlier segments have weight 0.
- Applies the confidence voting strategy to aggregate the worm-level predictions to create group-level predictions.
  - Class balancing coefficients are also applied to the voting strategy. Class balancing coefficients are calculated from training predictions only, not test, to avoid data leakage.
  - It is applied as following:
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

Hyperparameters choice:
- top 10 features have been selected from the output of `feature_importance_analysis.py` (tried several values but 10 has shown to lead to the best results)
- The 'Last N Segments Confidence' strategy has shown to lead to the best results. The best N is 30 according to the output of `plot_last_x_segments_analysis.py`.
- The Gradient Boosting model has shown to lead to the best results (Ramdom Forest following closely).
- Only tried with group of 5 (because we have 102 worms, and 5 balances well the number of groups and their size)

Metrics:
- Accuracy
- F1 score

Results (according to `plot_last_x_segments_analysis.py`):
- At worm-level:
  - Best accuracy: 0.683
  - Best F1 score: 0.723
- At group-level:
  - Best accuracy: 0.800
  - Best F1 score: 0.771

### Timeseries-based Classification (`ts_segment_classification.py`)

Similar to [feature-based classification](#feature-based-classification-feature_segment_classificationpy)

- Uses angle and speed series values.
- Implements several models:
  - 1D Convutional Neural Network
  - LSTM Neural Network
- Applies a voting strategy to aggregate the segment-level predictions to create worm-level predictions. Several strategies are implemented [feature-based classification](#feature-based-classification-feature_segment_classificationpy))

Results:
- Close to random guess accuracy (0.5) at segment-level. Close to 0.55 at worm-level.


## 3. Death Prediction

***For this task, only feature-based models have been used.***

### Classification-based Death Prediction (`death_proximity_predictor.py`)

- Uses a binary classification approach to predict if a segment is "close to death"
- proximity_threshold parameter: defines how many segments from the end are considered "close to death" (e.g., last 20 segments)
- Uses 30 top aging features identified from `first_vs_last_segment_analysis.py`:
- Implements several models:
  - Random Forest
  - Gradient Boosting
  - Multi-Layer Perceptron (MLP)
- Uses SMOTE for class balancing (since death proximity samples are minority class)
- `compare_proximity_thresholds.py` tests on validation set several proximity_threshold values

Results (according to `compare_proximity_thresholds.py`):
- Gradient Boosting has shown the best results (close with Ramdom Forest)
- Threshold: 
  - 1 --> F1 score: 0.173
  - 3 --> F1 score: 0.208
  - 5 --> F1 score: 0.268
  - 10 --> F1 score: 0.437
  - 15 --> F1 score: 0.547
  - 20 --> F1 score: 0.611
  - 25 --> F1 score: 0.668
  - 30 --> F1 score: 0.706
  - 40 --> F1 score: 0.751

### Regression-based Death Prediction (`death_proximity_regressor.py`)

- Uses regression to predict continuous values instead of binary classification
- Predicts either:
  - Number of segments remaining until death (`segments_from_end`)
  - Percentage of life remaining (`life_percentage_remaining`)
- Uses the same top aging features as classification approach
- Implements several regression models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Ridge Regression
  - Lasso Regression
  - Multi-Layer Perceptron (MLP) Regressor
- No SMOTE needed (regression doesn't require class balancing)

Hyperparameters choice:
- Random Forest with max_depth=15 shows best performance

Metrics:
- MAE (Mean Absolute Error): average prediction error
- RMSE (Root Mean Squared Error): penalizes larger errors more
- R² Score: proportion of variance explained by the model

Results (from the code outputs):
- Random Forest predicting segments_from_end:
  - MAE: 17.854 ± 2.039 segments
  - RMSE: 22.482 ± 3.118 segments
  - R²: 0.343 ± 0.075
- Random Forest predicting life_percentage_remaining:
  - MAE: 16.226 ± 0.372 %
  - RMSE: 20.502 ± 0.266 %
  - R²: 0.508 ± 0.012
- The model can predict death proximity with moderate accuracy, with typical errors of 18 segments (on average we have 82 segments per worm) or 16% of remaining life. Then, using life_percentage_remaining as a target leads to better results.

## 4. Results and Models Analysis

### `plot_feature_importance_by_classification.py`

**Scientific Question:** *What behavioral features differ between drugged and undrugged worms, and which features are most important for distinguishing each group?*

This analysis trains a single classification model (Gradient Boosting) on all data to distinguish control vs treatment worms, then uses permutation importance to understand which features the model relies on most heavily for each group.

- **Top distinguishing features for Control (undrugged) worms:**
  - `median_meandering_ratio`: 0.2772
  - `min_meandering_ratio`: 0.1891
  - `turning_entropy`: 0.1884

- **Top distinguishing features for Treatment (drugged) worms:**
  - `median_meandering_ratio`: 0.2981 (7.5% more important)
  - `turning_entropy`: 0.2258 (**20% more important**)
  - `min_meandering_ratio`: 0.2049

### `plot_last_x_segments_analysis.py`

**Scientific Question:** *How many segments before death are most informative for predicting treatment status? Does focusing on end-of-life behavior improve classification accuracy?*

This analysis tests the `last_X_segments_confidence` weighting strategy with X ranging from 1 to 100, evaluating performance at both worm-level and group-level (groups of 5 worms).

**Performance Analysis:**
- Performance improves dramatically from X=1 to X=30
- But performance gets worse with more early segment (from X=30).

### `plot_feature_importance_life_stages.py`

**Scientific Question:** *Which features are most predictive of remaining life at different life stages?*

This analysis trains a single regression model (Random Forest) to predict `life_percentage_remaining` on all data, then uses permutation importance to analyze how feature contributions differ between life stages.

**Approach:**
- 2-stage analysis: Taking a look at the predictions Beginning of Life (0-50% lived) vs End of Life (50-100% lived)
- 4-stage analysis: Quartiles (Q1-Q4) for finer temporal resolution

**Beginning of Life (0-50% lived):**
- Most important features:
  - `mean_speed`: 0.4147
  - `std_roaming_score`: 0.3150
  - `speed_entropy`: 0.2804

**End of Life (50-100% lived):**
- Most important features:
  - `mean_speed`: **1.0435** (2.5× more important)
  - `mean_roaming_score`: 0.2946
  - `std_frenetic_score`: 0.2823