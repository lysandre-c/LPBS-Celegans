#!/usr/bin/env python3
"""
Death Proximity Predictor for C. elegans

This module uses insights from the first vs last segment analysis to predict
when a worm is close to death. It leverages the most discriminative features
identified in the aging analysis to build a robust death proximity classifier.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                           confusion_matrix, classification_report, 
                           precision_recall_curve, roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class DeathProximityPredictor:
    """
    Predicts whether a worm segment is close to death based on movement features.
    
    Uses insights from aging analysis to focus on the most discriminative features:
    - Activity patterns (high/mixed/low activity fractions)
    - Speed metrics (mean, std, max speed)
    - Pausing behavior (time_paused, fraction_paused)
    - Movement quality (roaming scores, movement efficiency)
    - Movement variability (jerk, entropy measures)
    """
    
    def __init__(self, proximity_threshold=5, use_top_features=True):
        """
        Initialize the death proximity predictor.
        
        Args:
            proximity_threshold: Number of segments from end to consider "close to death"
            use_top_features: Whether to use only the most discriminative features
        """
        self.proximity_threshold = proximity_threshold
        self.use_top_features = use_top_features
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
        # Top discriminative features based on aging analysis
        self.top_aging_features = [
            # Activity patterns (strongest predictors)
            'high_activity_fraction', 'mixed_activity_fraction', 'low_activity_fraction',
            
            # Speed metrics (major decreases with age)
            'mean_speed', 'std_speed', 'max_speed', 'speed_entropy',
            
            # Roaming and exploration (major decreases)
            'mean_roaming_score', 'std_roaming_score', 'fraction_roaming',
            
            # Movement quality (deteriorates with age)
            'movement_efficiency', 'fraction_efficient_movement',
            
            # Pausing behavior (increases with age)
            'time_paused', 'fraction_paused',
            
            # Movement dynamics (decreases with age)
            'mean_jerk', 'max_jerk', 'kinetic_energy_proxy',
            
            # Meandering patterns
            'mean_meandering_ratio', 'std_meandering_ratio',
            
            # Wavelet features (speed patterns)
            'wavelet_speed_level0', 'wavelet_speed_level1', 'wavelet_speed_level2', 'wavelet_speed_level3',
            
            # Frenetic activity
            'mean_frenetic_score', 'std_frenetic_score',
            
            # Additional speed characteristics
            'speed_persistence', 'activity_level', 'speed_skewness', 'speed_kurtosis'
        ]
    
    def prepare_data(self, df):
        """
        Prepare data for death proximity prediction.
        
        Args:
            df: DataFrame with segment features
            
        Returns:
            X, y, groups: Features, labels, and worm groups
        """
        # Extract segment index if needed
        if 'segment_index' not in df.columns or df['segment_index'].isna().all():
            df['segment_index'] = df['filename'].str.extract(r'segment(\d+(?:\.\d+)?)', expand=False).astype(float)
        
        # Calculate position from end for each worm
        max_segments = df.groupby('original_file')['segment_index'].max()
        df['segments_from_end'] = df.apply(
            lambda row: max_segments[row['original_file']] - row['segment_index'], axis=1
        )
        
        # Create death proximity labels
        df['close_to_death'] = (df['segments_from_end'] <= self.proximity_threshold).astype(int)
        
        # Select features
        if self.use_top_features:
            features = [f for f in self.top_aging_features if f in df.columns]
        else:
            metadata_cols = ['label', 'filename', 'relative_path', 'file', 'worm_id', 'segment_number', 
                           'segment_index', 'original_file', 'max_segment_index', 'segments_from_end', 'close_to_death']
            features = [col for col in df.columns if col not in metadata_cols and df[col].dtype in ['float64', 'int64']]
        
        self.feature_names = features
        
        # Print summary
        pos_count = df['close_to_death'].sum()
        print(f"Data prepared: {len(df)} segments, {len(features)} features")
        print(f"  Close to death: {pos_count} ({df['close_to_death'].mean():.3f})")
        print(f"  Not close: {len(df) - pos_count}")
        
        # Prepare feature matrix and handle missing values
        X = df[features].fillna(df[features].median())
        y = df['close_to_death']
        groups = df['original_file']
        
        return X, y, groups
    
    
    def train_model(self, X, y, model_name='RandomForest', use_smote=True, verbose=True):
        """
        Train the death proximity prediction model.
        
        Args:
            X: Feature matrix
            y: Labels
            model_name: Type of model to use
            use_smote: Whether to use SMOTE for class balancing
            verbose: Print training progress
        """
        if verbose:
            print(f"Training {model_name}...")
        
        # Update feature names
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else list(range(X.shape[1]))
        
        # Create model pipeline
        model_configs = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42, class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=1000
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(256, 256, 64), random_state=42, max_iter=500
            )
        }
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model_configs[model_name])
        ])
        
        # Convert to numpy arrays
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        # Apply SMOTE if requested
        if use_smote:
            if verbose:
                print(f"  Before SMOTE: {len(X)} samples, positive rate: {y_array.mean():.3f}")
            smote = SMOTE(random_state=42)
            X_array, y_array = smote.fit_resample(X_array, y_array)
            if verbose:
                print(f"  After SMOTE: {len(X_array)} samples, positive rate: {y_array.mean():.3f}")
        
        # Train model
        self.model = model
        self.model.fit(X_array, y_array)
        
        # Get feature importance if available
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
    
    def find_optimal_threshold(self, X_val, y_val):
        """
        Find optimal probability threshold to maximize a given metric on validation set.
        
        Args:
            X_val: Validation feature matrix
            y_val: Validation true labels
            
        Returns:
            float: Optimal threshold value
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                
            score = f1_score(y_val, y_pred_thresh)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def evaluate_model(self, X, y, optimal_threshold=None, plot_results=False, verbose=True):
        """
        Evaluate the trained model.
        
        Args:
            X: Feature matrix
            y: True labels
            optimal_threshold: Custom decision threshold (if None, uses 0.5)
            plot_results: Whether to create evaluation plots
            verbose: Whether to print detailed results
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Get probabilities
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Apply threshold
        if optimal_threshold is None:
            optimal_threshold = 0.5
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba),
            'f1': f1_score(y, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'threshold': optimal_threshold
        }
        
        if verbose:
            print("="*60)
            print("MODEL EVALUATION RESULTS")
            print("="*60)
            print(f"Decision Threshold: {optimal_threshold:.3f}")
            print(f"Accuracy: {results['accuracy']:.3f}")
            print(f"AUC-ROC: {results['auc']:.3f}")
            print(f"F1-Score: {results['f1']:.3f}")
            print()
            print("Classification Report:")
            print(classification_report(y, y_pred, target_names=['Not Close to Death', 'Close to Death']))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y, y_pred))
        
        if plot_results:
            self._plot_evaluation_results(y, y_pred, y_pred_proba)
        
        return results
    
    def _plot_evaluation_results(self, y_true, y_pred, y_pred_proba):
        """Create evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        axes[0, 1].plot(recall, precision)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # Feature Importance (if available)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 15 Feature Importances')
        
        plt.tight_layout()
        plt.show()
    
    def predict_death_risk(self, X, return_probabilities=True):
        """
        Predict death risk for new segments.
        
        Args:
            X: Feature matrix for new segments
            return_probabilities: Whether to return probabilities or binary predictions
            
        Returns:
            Death risk predictions or probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Handle missing features
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0  # Default value for missing features
        
        # Ensure correct feature order
        X = X[self.feature_names]
        
        if return_probabilities:
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)

def analyze_death_risk_by_segment_position(df, predictor):
    """
    Analyze how death risk changes by segment position.
    
    Args:
        df: DataFrame with segments and predictions
        predictor: Trained DeathProximityPredictor
    
    Returns:
        pandas.DataFrame: Aggregated statistics of death risk by life stage
        bins with columns ['life_stage','mean','std','count'].
    """
    # Work with a copy to avoid modifying the original
    df_analysis = df.copy()
    
    # Extract segment index
    if df_analysis['segment_index'].isna().all():
        df_analysis['segment_index'] = df_analysis['filename'].str.extract(r'segment(\d+(?:\.\d+)?)', expand=False).astype(float)
    
    # Calculate relative position (percentage through life)
    worm_stats = df_analysis.groupby('original_file')['segment_index'].max().reset_index()
    worm_stats.columns = ['original_file', 'max_segment_index']
    df_analysis = df_analysis.merge(worm_stats, on='original_file', how='left')
    df_analysis['life_percentage'] = (df_analysis['segment_index'] / df_analysis['max_segment_index']) * 100
    
    # Get predictions - prepare features without modifying the dataframe
    metadata_cols = ['label', 'filename', 'relative_path', 'file', 'worm_id', 'segment_number', 
                    'segment_index', 'original_file']
    available_features = [f for f in predictor.feature_names if f in df_analysis.columns]
    X = df_analysis[available_features].fillna(df_analysis[available_features].median())
    
    death_probabilities = predictor.predict_death_risk(X)
    df_analysis['death_risk'] = death_probabilities
    
    # Analyze by life percentage bins
    df_analysis['life_stage'] = pd.cut(df_analysis['life_percentage'], 
                             bins=[0, 20, 40, 60, 80, 100], 
                             labels=['Early (0-20%)', 'Young (20-40%)', 'Mid (40-60%)', 'Mature (60-80%)', 'Late (80-100%)'])
    
    stage_analysis = df_analysis.groupby('life_stage')['death_risk'].agg(['mean', 'std', 'count']).reset_index()
    
    print("="*60)
    print("DEATH RISK BY LIFE STAGE")
    print("="*60)
    for _, row in stage_analysis.iterrows():
        print(f"{row['life_stage']}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})")
    
    # Plot death risk progression
    plt.figure(figsize=(12, 8))
    
    # Box plot by life stage
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df_analysis, x='life_stage', y='death_risk')
    plt.xticks(rotation=45)
    plt.title('Death Risk by Life Stage')
    plt.ylabel('Death Risk Probability')
    
    # Scatter plot of death risk vs life percentage
    plt.subplot(2, 2, 2)
    plt.scatter(df_analysis['life_percentage'], df_analysis['death_risk'], alpha=0.5)
    plt.xlabel('Life Percentage')
    plt.ylabel('Death Risk Probability')
    plt.title('Death Risk vs Life Progression')
    
    # Average death risk by 10% bins
    plt.subplot(2, 2, 3)
    df_analysis['life_bin'] = (df_analysis['life_percentage'] // 10) * 10
    bin_means = df_analysis.groupby('life_bin')['death_risk'].mean()
    plt.plot(bin_means.index, bin_means.values, 'o-')
    plt.xlabel('Life Percentage (10% bins)')
    plt.ylabel('Average Death Risk')
    plt.title('Death Risk Progression')
    
    # Distribution of death risk scores
    plt.subplot(2, 2, 4)
    plt.hist(df_analysis['death_risk'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Death Risk Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Death Risk Scores')
    
    plt.tight_layout()
    plt.show()
    
    return stage_analysis

def main(proximity_threshold, model_name):
    """
    Main function to demonstrate death proximity prediction.
    
    Args:
        proximity_threshold: Number of segments from end to consider "close to death"
    """
    from data_loader import LPBSDataLoader
    
    print("="*80)
    print("C. ELEGANS DEATH PROXIMITY PREDICTION")
    print("="*80)
    print(f"Proximity threshold: {proximity_threshold} segments from end")
    print()
    
    # Load data using LPBSDataLoader
    print("Loading segment features data...")
    df = pd.read_csv('feature_data/segments_features.csv')
    
    # Initialize predictor
    predictor = DeathProximityPredictor(
        proximity_threshold=proximity_threshold,
        use_top_features=True
    )
    
    # Prepare data
    X, y, groups = predictor.prepare_data(df)
    
    # Create CV splits using LPBSDataLoader's file-based splitting
    loader = LPBSDataLoader()
    cv_splits = loader.create_cv_splits(X, y, groups, n_splits=5)
    
    # Evaluate across all CV folds
    print(f"\nEvaluating across {len(cv_splits)} CV folds...")
    print("="*80)
    
    fold_results = []
    
    for fold_idx, fold in enumerate(cv_splits):
        print(f"\nFold {fold_idx + 1}/{len(cv_splits)}:")
        print(f"  Train: {len(fold['X_train'])} segments from {len(fold['train_files'])} files")
        print(f"  Test:  {len(fold['X_test'])} segments from {len(fold['test_files'])} files")
        
        # Create and train fold predictor
        fold_predictor = DeathProximityPredictor(proximity_threshold, use_top_features=True)
        fold_predictor.feature_names = predictor.feature_names
        
        fold_predictor.train_model(
            fold['X_train'], fold['y_train'], 
            model_name=model_name, use_smote=True, verbose=(fold_idx == 0)
        )
        
        # Evaluate on test set
        results = fold_predictor.evaluate_model(fold['X_test'], fold['y_test'], verbose=False)
        fold_results.append(results)
        
        print(f"    Accuracy: {results['accuracy']:.3f}, AUC: {results['auc']:.3f}, F1: {results['f1']:.3f}")
    
    # Calculate average metrics across folds
    metrics = {
        'accuracy': [r['accuracy'] for r in fold_results],
        'auc': [r['auc'] for r in fold_results],
        'f1': [r['f1'] for r in fold_results]
    }
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}
    
    # Print summary
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS (Test Set Performance)")
    print("="*80)
    print(f"Accuracy:  {avg_metrics['accuracy']:.3f} ± {std_metrics['accuracy']:.3f}")
    print(f"AUC-ROC:   {avg_metrics['auc']:.3f} ± {std_metrics['auc']:.3f}")
    print(f"F1-Score:  {avg_metrics['f1']:.3f} ± {std_metrics['f1']:.3f}")

    return {
        'fold_results': fold_results,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics
    }

if __name__ == "__main__":
    #main(proximity_threshold=20, model_name='RandomForest')
    main(proximity_threshold=20, model_name='MLP')