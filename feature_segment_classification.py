from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, silhouette_score, adjusted_rand_score, confusion_matrix

import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from data_loader import LPBSDataLoader


def calculate_segment_weights(weight_strategy, n_segments, segment_probs):
    """
    Common function to calculate segment weights based on strategy.
    
    Args:
        weight_strategy: Strategy name for weighting segments
        n_segments: Number of segments
        segment_probs: Predicted probabilities for segments
    
    Returns:
        numpy array of weights
    """
    if weight_strategy == 'uniform':
        weights = np.ones(n_segments)
    elif weight_strategy == 'confidence':
        weights = np.max(segment_probs, axis=1)
    elif weight_strategy == 'late_segments':
        weights = np.linspace(0.5, 1.5, n_segments)
    elif weight_strategy == 'early_segments':
        weights = np.linspace(1.5, 0.5, n_segments)
    elif re.match(r'^last_(\d+)_segments$', weight_strategy):
        match = re.match(r'^last_(\d+)_segments$', weight_strategy)
        X = int(match.group(1))
        weights = np.zeros(n_segments)    
        weights[-X:] = 1
    elif re.match(r'^last_(\d+)_segments_confidence$', weight_strategy):
        match = re.match(r'^last_(\d+)_segments_confidence$', weight_strategy)
        X = int(match.group(1))
        weights = np.zeros(n_segments)    
        weights[-X:] = 1 * np.max(segment_probs, axis=1)[-X:]
    elif weight_strategy == 'late_segments_confidence':
        weights = np.linspace(0.5, 1.5, n_segments) * np.max(segment_probs, axis=1)
    else:
        weights = np.ones(n_segments)
    
    return weights


def get_model(model_name: str, scaler = False) -> Pipeline:
    steps = []
    if scaler:
        steps.append(('scaler', StandardScaler()))

    # Knn and Kmeans led to bad results with all features
    if model_name == 'Limited Random Forest':
        steps.append(('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)))
    elif model_name == 'Random Forest':
        steps.append(('classifier', RandomForestClassifier(n_estimators=100, random_state=42)))
    elif model_name == 'Gradient Boosting':
        steps.append(('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42)))
    elif model_name == 'MLP':
        steps.append(('classifier', MLPClassifier(hidden_layer_sizes=(128, 64, 64), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=300, random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)))

    pipeline = Pipeline(steps)
    return pipeline


def feature_segment_classification(model: Pipeline, verbose=False, features=None):
    loader = LPBSDataLoader()
    X, y, groups = loader.load_segment_features()
    
    if features is not None:
        X = X[features]
    
    if verbose:
        print(f"Loaded data: {X.shape[0]:,} segments, {X.shape[1]} features")
        print(f"Class distribution: {y.value_counts()}")

    cv_splits = loader.create_cv_splits(X, y, groups, n_splits=5)
    auc_scores_train = []
    auc_scores_test = []
    f1_scores_train = []
    f1_scores_test = []
    acc_scores_train = []
    acc_scores_test = []

    for fold in tqdm(cv_splits):
        X_train = fold['X_train']
        X_test = fold['X_test']
        y_train = fold['y_train']
        y_test = fold['y_test']

        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        auc_train = roc_auc_score(y_train, y_pred_train)
        auc_test = roc_auc_score(y_test, y_pred_proba)
        auc_scores_train.append(auc_train)
        auc_scores_test.append(auc_test)

        f1_train = f1_score(y_train, y_pred_train, average='binary')
        f1_test = f1_score(y_test, y_pred_test, average='binary')
        f1_scores_train.append(f1_train)
        f1_scores_test.append(f1_test)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        acc_scores_train.append(acc_train)
        acc_scores_test.append(acc_test)

    if verbose:
        print(f"  Mean CV Test AUC: {np.mean(auc_scores_test):.3f} ± {np.std(auc_scores_test):.3f} (Train AUC: {np.mean(auc_scores_train):.3f} ± {np.std(auc_scores_train):.3f})")
        print(f"  Mean CV Test F1: {np.mean(f1_scores_test):.3f} ± {np.std(f1_scores_test):.3f} (Train F1: {np.mean(f1_scores_train):.3f} ± {np.std(f1_scores_train):.3f})")
        print(f"  Mean CV Test Acc: {np.mean(acc_scores_test):.3f} ± {np.std(acc_scores_test):.3f} Train Acc: {np.mean(acc_scores_train):.3f} ± {np.std(acc_scores_train):.3f}")

    return {
        "mean_auc_train": np.mean(auc_scores_train),
        "mean_auc_test": np.mean(auc_scores_test),
        "mean_train_f1": np.mean(f1_scores_train),
        "mean_test_f1": np.mean(f1_scores_test),
        "mean_train_acc": np.mean(acc_scores_train),
        "mean_test_acc": np.mean(acc_scores_test)
    }


def weighted_voting_classification(model: Pipeline, weight_strategy='confidence', verbose=False, features=None):
    """Weighted voting classifier with different weighting strategies."""
    loader = LPBSDataLoader()
    X, y, groups = loader.load_segment_features()
    
    if features is not None:
        X = X[features]
    
    if verbose:
        print(f"Loaded: {X.shape[0]:,} segments from {groups.nunique()} files")
        print(f"Weight strategy: {weight_strategy}")

    cv_splits = loader.create_cv_splits(X, y, groups, n_splits=5)
    
    file_predictions, file_true_labels = [], []
    vote_analysis = []
    
    for fold in tqdm(cv_splits):
        model.fit(fold['X_train'], fold['y_train'])
        
        for test_file in fold['test_files']:
            file_mask = fold['groups_test'] == test_file
            file_segments = fold['X_test'][file_mask]
            file_true_label = fold['y_test'][file_mask].iloc[0]
            
            segment_preds = model.predict(file_segments)
            segment_probs = model.predict_proba(file_segments)
            n_segments = len(segment_preds)
            
            weights = calculate_segment_weights(weight_strategy, n_segments, segment_probs)
            
            weighted_vote_0 = np.sum(weights[segment_preds == 0])
            weighted_vote_1 = np.sum(weights[segment_preds == 1])
            file_pred = int(weighted_vote_1 > weighted_vote_0)
            
            # Calculate confidence as normalized winning weight
            total_weight = weighted_vote_0 + weighted_vote_1
            confidence = max(weighted_vote_0, weighted_vote_1) / total_weight if total_weight > 0 else 0.5
            
            vote_analysis.append({
                'n_segments': n_segments,
                'weighted_pred': file_pred,
                'weighted_confidence': confidence,
                'avg_weight': weights.mean(),
                'weight_std': weights.std(),
                'true_label': file_true_label,
                'weighted_correct': file_pred == file_true_label,
            })
            
            file_predictions.append(file_pred)
            file_true_labels.append(file_true_label)
    
    file_predictions = np.array(file_predictions)
    file_true_labels = np.array(file_true_labels)
    
    accuracy = accuracy_score(file_true_labels, file_predictions)
    f1 = f1_score(file_true_labels, file_predictions, average='binary')
    cm = confusion_matrix(file_true_labels, file_predictions)
    
    vote_df = pd.DataFrame(vote_analysis)
    
    if verbose:
        print(f"Results: {len(file_predictions)} files, Acc: {accuracy:.3f}, F1: {f1:.3f}")
        print(f"\nWeighted Voting Analysis:")
        print(f"  Weighted accuracy: {vote_df['weighted_correct'].mean():.3f}")
        print(f"  Average confidence: {vote_df['weighted_confidence'].mean():.3f}")
        
    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "n_files": len(file_predictions),
        "vote_analysis": vote_df,
        "weight_strategy": weight_strategy
    }




def compare_weight_strategies(model: Pipeline, strategies=None, features=None, verbose=True):
    """Compare different weighted voting strategies."""
    results = {}
    
    if verbose:
        print("Comparing weighted voting strategies...")
    
    for strategy in strategies:
        if verbose:
            print(f"\n--- {strategy.replace('_', ' ').title()} Weighting ---")
        
        result = weighted_voting_classification(model, strategy, verbose=False, features=features)
        vote_df = result['vote_analysis']
        
        results[strategy] = {
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'avg_confidence': vote_df['weighted_confidence'].mean(),
        }
        
        if verbose:
            print(f"  Accuracy: {results[strategy]['accuracy']:.3f}")
            print(f"  F1: {results[strategy]['f1']:.3f}")
            print(f"  Avg Confidence: {results[strategy]['avg_confidence']:.3f}")
    
    if verbose:
        print(f"\nBest Strategy Ranking:")
        sorted_strategies = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for i, (strategy, metrics) in enumerate(sorted_strategies, 1):
            print(f"  {i}. {strategy.replace('_', ' ').title()}: {metrics['accuracy']:.3f} acc")
    
    return results



def group_prediction_cv(best_features, best_model_name, best_weight_strategy, group_size=5, n_splits=5, verbose=True):
    """
    Group prediction with proper CV to avoid data leakage.
    Groups worms of the same true class and uses confidence voting to predict group class.
    """
    loader = LPBSDataLoader()
    X, y, groups = loader.load_segment_features()
    X = X[best_features]
    
    if verbose:
        print(f"Group Prediction with {best_model_name} + {best_weight_strategy}")
        print(f"Using {len(best_features)} features, group size: {group_size}")
        print(f"Data: {X.shape[0]:,} segments from {groups.nunique()} files")
    
    cv_splits = loader.create_cv_splits(X, y, groups, n_splits=n_splits)
    all_group_results = []
    
    for fold_idx, fold in enumerate(tqdm(cv_splits, desc="CV Folds")):
        model = get_model(best_model_name, scaler=False)
        model.fit(fold['X_train'], fold['y_train'])
        
        # Get file-level predictions for test files
        file_predictions = {}
        for test_file in fold['test_files']:
            file_mask = fold['groups_test'] == test_file
            X_file = fold['X_test'][file_mask]
            true_label = fold['y_test'][file_mask].iloc[0]
            
            # Individual file prediction using weighted voting
            segment_preds = model.predict(X_file)
            segment_probs = model.predict_proba(X_file)
            n_segments = len(segment_preds)
            
            # Apply weighting strategy dynamically
            weights = calculate_segment_weights(best_weight_strategy, n_segments, segment_probs)
            
            weighted_vote_0 = np.sum(weights[segment_preds == 0])
            weighted_vote_1 = np.sum(weights[segment_preds == 1])
            file_pred = int(weighted_vote_1 > weighted_vote_0)
            
            # Calculate confidence as normalized winning weight
            total_weight = weighted_vote_0 + weighted_vote_1
            confidence = max(weighted_vote_0, weighted_vote_1) / total_weight if total_weight > 0 else 0.5
            
            file_predictions[test_file] = {
                'pred': file_pred, 
                'true': true_label, 
                'confidence': confidence
            }
        
        # Group files by true class
        control_files = [f for f, data in file_predictions.items() if data['true'] == 0]
        treatment_files = [f for f, data in file_predictions.items() if data['true'] == 1]
        
        # Create groups within each class
        for class_label, files in [(0, control_files), (1, treatment_files)]:
            for i in range(0, len(files), group_size):
                group_files = files[i:i + group_size]
                if len(group_files) == group_size:  # Only complete groups
                    # Get predictions and confidences for group members
                    group_preds = [file_predictions[f]['pred'] for f in group_files]
                    group_confidences = [file_predictions[f]['confidence'] for f in group_files]
                    
                    # Confidence-weighted voting for group prediction
                    group_preds = np.array(group_preds)
                    group_confidences = np.array(group_confidences)
                    
                    weighted_vote_0 = np.sum(group_confidences[group_preds == 0])
                    weighted_vote_1 = np.sum(group_confidences[group_preds == 1])
                    
                    group_prediction = int(weighted_vote_1 > weighted_vote_0)
                    
                    # Calculate group confidence
                    total_group_weight = weighted_vote_0 + weighted_vote_1
                    group_confidence = max(weighted_vote_0, weighted_vote_1) / total_group_weight if total_group_weight > 0 else 0.5
                    
                    all_group_results.append({
                        'fold': fold_idx,
                        'true_class': class_label,
                        'predicted_class': group_prediction,
                        'individual_preds': group_preds.tolist(),
                        'individual_confidences': group_confidences.tolist(),
                        'group_confidence': group_confidence,
                        'group_size': len(group_files)
                    })
    
    # Calculate results
    correct = sum(1 for r in all_group_results if r['predicted_class'] == r['true_class'])
    total = len(all_group_results)
    accuracy = correct / total if total > 0 else 0
    
    # Class-wise results
    control_groups = [r for r in all_group_results if r['true_class'] == 0]
    treatment_groups = [r for r in all_group_results if r['true_class'] == 1]
    
    control_acc = sum(1 for r in control_groups if r['predicted_class'] == r['true_class']) / len(control_groups) if control_groups else 0
    treatment_acc = sum(1 for r in treatment_groups if r['predicted_class'] == r['true_class']) / len(treatment_groups) if treatment_groups else 0
    
    if verbose:
        avg_group_confidence = np.mean([r['group_confidence'] for r in all_group_results])
        
        print(f"\n=== GROUP PREDICTION RESULTS ===")
        print(f"Total groups tested: {total}")
        print(f"Overall accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"Average group confidence: {avg_group_confidence:.3f}")
        print(f"Control groups: {len(control_groups)} (accuracy: {control_acc:.3f})")
        print(f"Treatment groups: {len(treatment_groups)} (accuracy: {treatment_acc:.3f})")
        
        # Show some example predictions
        print(f"\nExample group predictions:")
        for i, result in enumerate(all_group_results[:5]):
            class_name = "Control" if result['true_class'] == 0 else "Treatment"
            pred_name = "Control" if result['predicted_class'] == 0 else "Treatment"
            correct_mark = "✓" if result['predicted_class'] == result['true_class'] else "✗"
            conf_str = f"conf:{result['group_confidence']:.3f}"
            print(f"  Group {i+1}: True={class_name}, Pred={pred_name} {correct_mark} ({conf_str}, votes: {result['individual_preds']})")
    
    return {
        'accuracy': accuracy,
        'total_groups': total,
        'correct_groups': correct,
        'control_accuracy': control_acc,
        'treatment_accuracy': treatment_acc,
        'all_results': all_group_results
    }


if __name__ == "__main__":

    best_features = ['median_meandering_ratio', 'mean_meandering_ratio', 'min_meandering_ratio', 'wavelet_turning_level0',
                     'std_turning_angle', 'turning_entropy', 'wavelet_turning_level1', 'wavelet_turning_level2',
                     'speed_fractal_dim', 'wavelet_turning_level3']
    best_model_name = 'Gradient Boosting'
    best_weight_strategy = 'last_10_segments_confidence'
    
    print("===== Individual Prediction =====")
    model = get_model(best_model_name, scaler=False)
    results = weighted_voting_classification(model, best_weight_strategy, verbose=False, features=best_features)
    print("Accuracy:", results['accuracy'])
    print("F1:", results['f1'])
    print("Confusion Matrix:", results['confusion_matrix'])

    print("===== Group Prediction =====")
    model = get_model(best_model_name, scaler=False)
    results = group_prediction_cv(best_features, best_model_name, best_weight_strategy, group_size=5, n_splits=5, verbose=True)
    
# ===== Individual Prediction =====
# 100%|███████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:06<00:00,  1.32s/it]
# Accuracy: 0.6538461538461539
# F1: 0.7
# Confusion Matrix: [[26 26]
#  [10 42]]
# ===== Group Prediction =====
# Group Prediction with Gradient Boosting + last_10_segments_confidence
# Using 10 features, group size: 5
# Data: 8,197 segments from 104 files
# CV Folds: 100%|█████████████████████████████████████████████████████████████████████████████| 5/5 [00:06<00:00,  1.30s/it]
# 
# === GROUP PREDICTION RESULTS ===
# Total groups tested: 20
# Overall accuracy: 0.700 (14/20)
# Average group confidence: 0.764
# Control groups: 10 (accuracy: 0.400)
# Treatment groups: 10 (accuracy: 1.000)
# 
# Example group predictions:
#   Group 1: True=Control, Pred=Treatment ✗ (conf:0.603, votes: [1, 1, 1, 0, 0])
#   Group 2: True=Control, Pred=Treatment ✗ (conf:0.683, votes: [1, 0, 1, 0, 1])
#   Group 3: True=Treatment, Pred=Treatment ✓ (conf:0.844, votes: [0, 1, 1, 1, 1])
#   Group 4: True=Treatment, Pred=Treatment ✓ (conf:0.842, votes: [0, 1, 1, 1, 1])
#   Group 5: True=Control, Pred=Control ✓ (conf:0.598, votes: [0, 1, 1, 0, 0])