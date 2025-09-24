#!/usr/bin/env python3
"""
Compare Death Proximity Prediction with Different Thresholds

This script tests the death proximity prediction model with different
proximity thresholds (e.g., last 5, 10, 15 segments) to find the optimal
threshold for prediction accuracy.
"""

import pandas as pd
import matplotlib.pyplot as plt
from death_proximity_predictor import DeathProximityPredictor
import warnings
warnings.filterwarnings('ignore')

def compare_thresholds(thresholds=[1, 3, 5, 10, 15, 20, 25, 30], model_name='MLP'):
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
        
        # Train model (without plotting to avoid too many plots)
        cv_scores = predictor.train_model(X, y, groups, model_name=model_name, 
                                        use_smote=True, verbose=False)
        
        # Evaluate model (without plotting)
        eval_results = predictor.evaluate_model(X, y, plot_results=False)
        
        # Store results
        results.append({
            'threshold': threshold,
            'positive_samples': y.sum(),
            'negative_samples': len(y) - y.sum(),
            'positive_rate': y.mean(),
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_accuracy': eval_results['accuracy'],
            'test_auc': eval_results['auc'],
            'test_f1': eval_results['f1'],
            'total_features': len(X.columns)
        })
        
        print(f"Positive samples: {y.sum()} ({y.mean():.3f})")
        print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"Test Accuracy: {eval_results['accuracy']:.3f}")
        print(f"Test AUC: {eval_results['auc']:.3f}")
        print(f"Test F1: {eval_results['f1']:.3f}")
    
    results_df = pd.DataFrame(results)
    return results_df

def plot_threshold_comparison(results_df):
    """
    Create visualizations comparing different thresholds.
    
    Args:
        results_df: DataFrame with comparison results
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    
    # 1. Cross-validation AUC
    axes[0, 0].errorbar(results_df['threshold'], results_df['cv_auc_mean'], 
                       yerr=results_df['cv_auc_std'], marker='o', capsize=5)
    axes[0, 0].set_xlabel('Proximity Threshold')
    axes[0, 0].set_ylabel('Cross-Validation AUC')
    axes[0, 0].set_title('CV AUC vs Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Test AUC
    axes[0, 1].plot(results_df['threshold'], results_df['test_auc'], 'o-', color='green')
    axes[0, 1].set_xlabel('Proximity Threshold')
    axes[0, 1].set_ylabel('Test AUC')
    axes[0, 1].set_title('Test AUC vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Test Accuracy
    axes[0, 2].plot(results_df['threshold'], results_df['test_accuracy'], 'o-', color='blue')
    axes[0, 2].set_xlabel('Proximity Threshold')
    axes[0, 2].set_ylabel('Test Accuracy')
    axes[0, 2].set_title('Accuracy vs Threshold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. F1 Score
    axes[1, 0].plot(results_df['threshold'], results_df['test_f1'], 'o-', color='orange')
    axes[1, 0].set_xlabel('Proximity Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Class balance
    axes[1, 1].plot(results_df['threshold'], results_df['positive_rate'], 'o-', color='red')
    axes[1, 1].set_xlabel('Proximity Threshold')
    axes[1, 1].set_ylabel('Positive Class Rate')
    axes[1, 1].set_title('Class Balance vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Number of positive samples
    axes[1, 2].plot(results_df['threshold'], results_df['positive_samples'], 'o-', color='purple')
    axes[1, 2].set_xlabel('Proximity Threshold')
    axes[1, 2].set_ylabel('Number of Positive Samples')
    axes[1, 2].set_title('Positive Samples vs Threshold')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Performance summary (AUC vs F1)
    scatter = axes[2, 0].scatter(results_df['test_auc'], results_df['test_f1'], 
                               c=results_df['threshold'], cmap='viridis', s=100)
    axes[2, 0].set_xlabel('Test AUC')
    axes[2, 0].set_ylabel('F1 Score')
    axes[2, 0].set_title('AUC vs F1 (colored by threshold)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[2, 0])
    cbar.set_label('Proximity Threshold')
    
    # Add threshold labels to points
    for _, row in results_df.iterrows():
        axes[2, 0].annotate(f"{int(row['threshold'])}", 
                          (row['test_auc'], row['test_f1']), 
                          xytext=(5, 5), textcoords='offset points')
    
    # 8. Accuracy vs F1
    scatter2 = axes[2, 1].scatter(results_df['test_accuracy'], results_df['test_f1'], 
                                c=results_df['threshold'], cmap='plasma', s=100)
    axes[2, 1].set_xlabel('Test Accuracy')
    axes[2, 1].set_ylabel('F1 Score')
    axes[2, 1].set_title('Accuracy vs F1 (colored by threshold)')
    
    # Add colorbar
    cbar2 = plt.colorbar(scatter2, ax=axes[2, 1])
    cbar2.set_label('Proximity Threshold')
    
    # Add threshold labels to points
    for _, row in results_df.iterrows():
        axes[2, 1].annotate(f"{int(row['threshold'])}", 
                          (row['test_accuracy'], row['test_f1']), 
                          xytext=(5, 5), textcoords='offset points')
    
    # 9. Hide the last subplot
    axes[2, 2].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def print_threshold_summary(results_df):
    """Print a summary of threshold comparison results."""
    
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON SUMMARY")
    print("="*80)
    
    # Sort by CV AUC for ranking
    results_sorted = results_df.sort_values('cv_auc_mean', ascending=False)
    
    print(f"{'Threshold':<10} {'Pos Samples':<12} {'Pos Rate':<10} {'CV AUC':<15} {'Accuracy':<10} {'Test AUC':<10} {'F1 Score':<10}")
    print("-" * 90)
    
    for _, row in results_sorted.iterrows():
        cv_auc_str = f"{row['cv_auc_mean']:.3f}±{row['cv_auc_std']:.3f}"
        print(f"{int(row['threshold']):<10} {int(row['positive_samples']):<12} "
              f"{row['positive_rate']:<10.3f} {cv_auc_str:<15} "
              f"{row['test_accuracy']:<10.3f} {row['test_auc']:<10.3f} {row['test_f1']:<10.3f}")
    
    # Find best thresholds
    best_cv_auc = results_sorted.iloc[0]
    best_test_auc = results_df.loc[results_df['test_auc'].idxmax()]
    best_f1 = results_df.loc[results_df['test_f1'].idxmax()]
    best_accuracy = results_df.loc[results_df['test_accuracy'].idxmax()]
    
    print(f"\nBest performing thresholds:")
    print(f"- Best CV AUC: {int(best_cv_auc['threshold'])} (AUC = {best_cv_auc['cv_auc_mean']:.3f})")
    print(f"- Best Test AUC: {int(best_test_auc['threshold'])} (AUC = {best_test_auc['test_auc']:.3f})")
    print(f"- Best Accuracy: {int(best_accuracy['threshold'])} (Accuracy = {best_accuracy['test_accuracy']:.3f})")
    print(f"- Best F1 Score: {int(best_f1['threshold'])} (F1 = {best_f1['test_f1']:.3f})")
    
    # Recommendations
    print(f"\nRecommendations:")
    if best_cv_auc['threshold'] == best_test_auc['threshold']:
        print(f"- Threshold {int(best_cv_auc['threshold'])} performs best on both CV and test metrics")
    else:
        print(f"- Consider threshold {int(best_cv_auc['threshold'])} for most robust performance (best CV AUC)")
        print(f"- Consider threshold {int(best_test_auc['threshold'])} for highest discrimination (best test AUC)")
    
    # Class balance considerations
    balanced_thresholds = results_df[(results_df['positive_rate'] >= 0.05) & (results_df['positive_rate'] <= 0.15)]
    if len(balanced_thresholds) > 0:
        best_balanced = balanced_thresholds.loc[balanced_thresholds['cv_auc_mean'].idxmax()]
        print(f"- For balanced classes (5-15% positive): threshold {int(best_balanced['threshold'])} "
              f"(AUC = {best_balanced['cv_auc_mean']:.3f}, {best_balanced['positive_rate']:.1%} positive)")

def main_comparison(model_name='MLP'):
    """Main function to run threshold comparison."""
    
    # Test different thresholds
    print("Testing thresholds...")
    results_basic = compare_thresholds(thresholds=[1, 3, 5, 10, 15, 20, 25, 30, 40], model_name=model_name)
    
    # Create visualizations
    plot_threshold_comparison(results_basic)
    
    # Print summary
    print_threshold_summary(results_basic)
    
    return results_basic

if __name__ == "__main__":
    results_basic = main_comparison(model_name='GradientBoosting')
