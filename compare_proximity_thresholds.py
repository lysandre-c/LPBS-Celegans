#!/usr/bin/env python3
"""
Compare Death Proximity Prediction with Different Thresholds

This script tests the death proximity prediction model with different
proximity thresholds (e.g., last 5, 10, 15 segments) to find the optimal
threshold for prediction accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from death_proximity_predictor import DeathProximityPredictor
from data_loader import LPBSDataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def compare_thresholds(thresholds, model_name):
    """
    Compare death proximity prediction performance across different thresholds.
    
    Args:
        thresholds: List of proximity thresholds to test
        
    Returns:
        DataFrame with comparison results
    """
    print("="*80)
    print("COMPARING DEATH PROXIMITY THRESHOLDS")
    print("="*80)
    print(f"Testing thresholds: {thresholds}")
    print()
    
    # Load data once
    df = pd.read_csv('feature_data/segments_features.csv')
    
    results = []
    
    for threshold in thresholds:
        print(f"\n{'='*50}")
        print(f"TESTING THRESHOLD: {threshold} segments")
        print(f"{'='*50}")
        
        # Initialize predictor
        predictor = DeathProximityPredictor(
            proximity_threshold=threshold,
            use_top_features=True
        )
        
        # Prepare data
        X, y, groups = predictor.prepare_data(df)
        
        # Perform CV evaluation like in main()
        loader = LPBSDataLoader()
        cv_splits = loader.create_cv_splits(X, y, groups, n_splits=5)
        
        fold_results = []
        optimal_thresholds = []
        
        for fold_idx, fold in enumerate(cv_splits):
            # Split training into train/val for threshold optimization
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                fold['X_train'], fold['y_train'], 
                test_size=0.2, random_state=42, stratify=fold['y_train']
            )
            
            # Create and train fold predictor
            fold_predictor = DeathProximityPredictor(threshold, use_top_features=True)
            fold_predictor.feature_names = predictor.feature_names
            
            fold_predictor.train_model(
                X_train_fold, y_train_fold, 
                model_name=model_name, use_smote=True, verbose=False
            )
            
            # Find optimal threshold on validation set
            optimal_threshold = fold_predictor.find_optimal_threshold(
                X_val_fold, y_val_fold
            )
            optimal_thresholds.append(optimal_threshold)
            
            # Evaluate on test set with optimal threshold
            results_fold = fold_predictor.evaluate_model(
                fold['X_test'], fold['y_test'], 
                optimal_threshold=optimal_threshold,
                verbose=False, plot_results=False
            )
            fold_results.append(results_fold)
        
        cv_auc_scores = [r['auc'] for r in fold_results]
        cv_acc_scores = [r['accuracy'] for r in fold_results]
        cv_f1_scores = [r['f1'] for r in fold_results]
        
        results.append({
            'threshold': threshold,
            'positive_samples': int(y.sum()),
            'negative_samples': int(len(y) - y.sum()),
            'positive_rate': float(y.mean()),
            'cv_auc_mean': np.mean(cv_auc_scores),
            'cv_auc_std': np.std(cv_auc_scores),
            'cv_accuracy_mean': np.mean(cv_acc_scores),
            'cv_accuracy_std': np.std(cv_acc_scores),
            'cv_f1_mean': np.mean(cv_f1_scores),
            'cv_f1_std': np.std(cv_f1_scores),
            'optimal_threshold_mean': np.mean(optimal_thresholds),
            'optimal_threshold_std': np.std(optimal_thresholds),
            'total_features': len(X.columns)
        })
        
        print(f"Positive samples: {y.sum()} ({y.mean():.3f})")
        print(f"Optimal decision threshold: {np.mean(optimal_thresholds):.3f} ± {np.std(optimal_thresholds):.3f}")
        print(f"CV AUC: {np.mean(cv_auc_scores):.3f} ± {np.std(cv_auc_scores):.3f}")
        print(f"CV Accuracy: {np.mean(cv_acc_scores):.3f} ± {np.std(cv_acc_scores):.3f}")
        print(f"CV F1: {np.mean(cv_f1_scores):.3f} ± {np.std(cv_f1_scores):.3f}")
    
    results_df = pd.DataFrame(results)
    return results_df

def plot_threshold_comparison(results_df):
    """
    Create visualizations comparing different thresholds.
    
    Args:
        results_df: DataFrame with comparison results
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cross-validation AUC
    axes[0, 0].errorbar(results_df['threshold'], results_df['cv_auc_mean'], 
                       yerr=results_df['cv_auc_std'], marker='o', capsize=5)
    axes[0, 0].set_xlabel('Proximity Threshold')
    axes[0, 0].set_ylabel('Cross-Validation AUC')
    axes[0, 0].set_title('CV AUC vs Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. CV Accuracy
    axes[0, 1].errorbar(results_df['threshold'], results_df['cv_accuracy_mean'], 
                       yerr=results_df['cv_accuracy_std'], marker='o', capsize=5, color='green')
    axes[0, 1].set_xlabel('Proximity Threshold')
    axes[0, 1].set_ylabel('CV Accuracy')
    axes[0, 1].set_title('CV Accuracy vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. CV F1 Score
    axes[1, 0].errorbar(results_df['threshold'], results_df['cv_f1_mean'], 
                       yerr=results_df['cv_f1_std'], marker='o', capsize=5, color='blue')
    axes[1, 0].set_xlabel('Proximity Threshold')
    axes[1, 0].set_ylabel('CV F1 Score')
    axes[1, 0].set_title('CV F1 vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. All metrics comparison
    axes[1, 1].plot(results_df['threshold'], results_df['cv_auc_mean'], 'o-', label='AUC', color='green')
    axes[1, 1].plot(results_df['threshold'], results_df['cv_accuracy_mean'], 'o-', label='Accuracy', color='blue')
    axes[1, 1].plot(results_df['threshold'], results_df['cv_f1_mean'], 'o-', label='F1', color='orange')
    axes[1, 1].set_xlabel('Proximity Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('All CV Metrics vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_threshold_summary(results_df):
    """Print a summary of threshold comparison results."""
    
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON SUMMARY")
    print("="*80)
    
    # Sort by CV AUC for ranking
    results_sorted = results_df.sort_values('threshold', ascending=False)
    
    print(f"{'Threshold':<10} {'Pos Rate':<10} {'Opt. Thresh.':<16} {'CV AUC':<18} {'CV Acc':<15} {'CV F1':<10}")
    print("-" * 100)
    
    for _, row in results_sorted.iterrows():
        opt_thresh_str = f"{row['optimal_threshold_mean']:.3f}±{row['optimal_threshold_std']:.3f}"
        cv_auc_str = f"{row['cv_auc_mean']:.3f}±{row['cv_auc_std']:.3f}"
        cv_acc_str = f"{row['cv_accuracy_mean']:.3f}±{row['cv_accuracy_std']:.3f}"
        cv_f1_str = f"{row['cv_f1_mean']:.3f}±{row['cv_f1_std']:.3f}"
        print(f"{int(row['threshold']):<10} {row['positive_rate']:<10.3f} "
              f"{opt_thresh_str:<16} {cv_auc_str:<18} "
              f"{cv_acc_str:<15} {cv_f1_str:<10}")
    
    # Find best thresholds
    best_cv_auc = results_sorted.iloc[0]
    best_f1 = results_df.loc[results_df['cv_f1_mean'].idxmax()]
    best_accuracy = results_df.loc[results_df['cv_accuracy_mean'].idxmax()]
    
    print(f"\nBest performing thresholds:")
    print(f"- Best CV AUC: {int(best_cv_auc['threshold'])} (AUC = {best_cv_auc['cv_auc_mean']:.3f}, Opt. Thresh. = {best_cv_auc['optimal_threshold_mean']:.3f})")
    print(f"- Best CV Accuracy: {int(best_accuracy['threshold'])} (Accuracy = {best_accuracy['cv_accuracy_mean']:.3f}, Opt. Thresh. = {best_accuracy['optimal_threshold_mean']:.3f})")
    print(f"- Best CV F1 Score: {int(best_f1['threshold'])} (F1 = {best_f1['cv_f1_mean']:.3f}, Opt. Thresh. = {best_f1['optimal_threshold_mean']:.3f})")
    
def main_comparison(model_name):
    """Main function to run threshold comparison."""
    
    print("Testing thresholds...")
    results_basic = compare_thresholds(thresholds=[1, 3, 5, 10, 15, 20, 25, 30, 40], model_name=model_name)
    
    plot_threshold_comparison(results_basic)    
    print_threshold_summary(results_basic)
    
    return results_basic

if __name__ == "__main__":
    results_basic = main_comparison(model_name='MLP')





# Some results:


# MLP 100-100-50
# ================================================================================
# THRESHOLD COMPARISON SUMMARY
# ================================================================================
# Threshold  Pos Rate   Opt. Thresh.     CV AUC             CV Acc          CV F1     
# -------------------------------------------------------------------------------------
# 1          0.025      0.088±0.132      0.794±0.018        0.892±0.026     0.082±0.024
# 3          0.050      0.386±0.359      0.792±0.020        0.896±0.024     0.163±0.034
# 5          0.075      0.090±0.101      0.783±0.013        0.817±0.021     0.275±0.008
# 10         0.137      0.048±0.025      0.765±0.021        0.754±0.038     0.407±0.022
# 15         0.198      0.140±0.047      0.764±0.028        0.738±0.012     0.488±0.052
# 20         0.261      0.164±0.139      0.745±0.020        0.712±0.020     0.548±0.032
# 25         0.322      0.196±0.146      0.748±0.023        0.702±0.017     0.611±0.016
# 30         0.385      0.154±0.129      0.749±0.021        0.696±0.018     0.661±0.015
# 40         0.511      0.106±0.100      0.722±0.021        0.659±0.012     0.710±0.006


# MLP 256-256-64
# ================================================================================
# THRESHOLD COMPARISON SUMMARY
# ================================================================================
# Threshold  Pos Rate   Opt. Thresh.     CV AUC             CV Acc          CV F1     
# -------------------------------------------------------------------------------------
# 1          0.025      0.204±0.197      0.798±0.015        0.929±0.019     0.096±0.049
# 3          0.050      0.196±0.150      0.802±0.013        0.890±0.016     0.151±0.031
# 5          0.075      0.302±0.324      0.786±0.013        0.853±0.030     0.241±0.050
# 10         0.137      0.074±0.051      0.776±0.024        0.770±0.032     0.410±0.030
# 15         0.198      0.110±0.084      0.760±0.016        0.741±0.016     0.487±0.030
# 20         0.261      0.128±0.046      0.757±0.021        0.727±0.021     0.565±0.019
# 25         0.322      0.206±0.172      0.741±0.018        0.689±0.027     0.599±0.019
# 30         0.385      0.210±0.224      0.745±0.022        0.691±0.028     0.650±0.020
# 40         0.511      0.102±0.073      0.715±0.014        0.672±0.021     0.711±0.018


# Random Forest
# ================================================================================
# THRESHOLD COMPARISON SUMMARY
# ================================================================================
# Threshold  Pos Rate   Opt. Thresh.     CV AUC             CV Acc          CV F1     
# -------------------------------------------------------------------------------------
# 1          0.025      0.724±0.075      0.827±0.007        0.959±0.018     0.151±0.055
# 3          0.050      0.570±0.094      0.823±0.013        0.833±0.057     0.197±0.032
# 5          0.075      0.504±0.143      0.826±0.011        0.794±0.050     0.299±0.041
# 10         0.137      0.552±0.039      0.835±0.022        0.789±0.024     0.464±0.035
# 15         0.198      0.504±0.045      0.827±0.022        0.756±0.012     0.542±0.022
# 20         0.261      0.464±0.051      0.823±0.018        0.747±0.018     0.621±0.018
# 25         0.322      0.444±0.050      0.823±0.013        0.748±0.016     0.679±0.018
# 30         0.385      0.400±0.009      0.823±0.015        0.748±0.020     0.716±0.016
# 40         0.511      0.406±0.020      0.810±0.018        0.740±0.023     0.758±0.017


# Gradient Boosting
# ================================================================================
# THRESHOLD COMPARISON SUMMARY
# ================================================================================
# Threshold  Pos Rate   Opt. Thresh.     CV AUC             CV Acc          CV F1     
# -------------------------------------------------------------------------------------
# 1          0.025      0.698±0.170      0.813±0.011        0.960±0.013     0.173±0.043
# 3          0.050      0.498±0.059      0.817±0.017        0.901±0.015     0.208±0.013
# 5          0.075      0.502±0.046      0.812±0.013        0.861±0.024     0.268±0.026
# 10         0.137      0.408±0.044      0.825±0.018        0.789±0.021     0.437±0.023
# 15         0.198      0.394±0.066      0.818±0.023        0.772±0.028     0.547±0.025
# 20         0.261      0.372±0.068      0.816±0.017        0.751±0.027     0.611±0.022
# 25         0.322      0.342±0.041      0.815±0.013        0.741±0.021     0.668±0.016
# 30         0.385      0.364±0.054      0.814±0.011        0.743±0.019     0.706±0.016
# 40         0.511      0.332±0.070      0.803±0.016        0.718±0.026     0.751±0.020