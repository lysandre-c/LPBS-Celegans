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
import joblib
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
        print("Preparing death proximity prediction data...")
        
        # Extract segment index from filename if needed
        if df['segment_index'].isna().all():
            df['segment_index'] = df['filename'].str.extract(r'segment(\d+(?:\.\d+)?)', expand=False).astype(float)
        
        # Calculate position from end for each worm
        worm_stats = df.groupby('original_file')['segment_index'].max().reset_index()
        worm_stats.columns = ['original_file', 'max_segment_index']
        df = df.merge(worm_stats, on='original_file', how='left')
        df['segments_from_end'] = df['max_segment_index'] - df['segment_index']
        
        # Create death proximity labels
        df['close_to_death'] = (df['segments_from_end'] <= self.proximity_threshold).astype(int)
        
        print(f"Segments close to death (≤{self.proximity_threshold} segments from end): {df['close_to_death'].sum()}")
        print(f"Segments not close to death: {len(df) - df['close_to_death'].sum()}")
        print(f"Class balance: {df['close_to_death'].mean():.3f} positive rate")
        
        # Select features
        if self.use_top_features:
            available_features = [f for f in self.top_aging_features if f in df.columns]
            print(f"Using {len(available_features)} top aging-related features")
        else:
            metadata_cols = ['label', 'filename', 'relative_path', 'file', 'worm_id', 'segment_number', 
                           'segment_index', 'original_file', 'max_segment_index', 'segments_from_end', 'close_to_death']
            available_features = [col for col in df.columns if col not in metadata_cols and df[col].dtype in ['float64', 'int64']]
            print(f"Using all {len(available_features)} available features")
        
        self.feature_names = available_features
        
        # Prepare feature matrix
        X = df[available_features].copy()
        y = df['close_to_death'].copy()
        groups = df['original_file'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y, groups
    
    
    def train_model(self, X, y, groups, model_name='RandomForest', use_smote=True, 
                   cv_folds=5, verbose=True):
        """
        Train the death proximity prediction model.
        
        Args:
            X: Feature matrix
            y: Labels
            groups: Worm groups for proper cross-validation
            model_name: Type of model to use
            use_smote: Whether to use SMOTE for class balancing
            cv_folds: Number of cross-validation folds
            verbose: Print training progress
        """
        print(f"Training {model_name} model for death proximity prediction...")
        
        # Update feature names
        self.feature_names = X.columns.tolist()
        
        if verbose:
            print(f"Total features: {len(self.feature_names)}")
        
        # Create model pipeline
        if model_name == 'RandomForest':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
        elif model_name == 'GradientBoosting':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ))
            ])
        elif model_name == 'LogisticRegression':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                ))
            ])
        elif model_name == 'MLP':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(100, 100, 50),
                    random_state=42,
                    max_iter=500
                ))
            ])
        
        # Apply SMOTE if requested
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            if verbose:
                print(f"After SMOTE: {len(X_resampled)} samples, positive rate: {y_resampled.mean():.3f}")
        else:
            X_resampled, y_resampled = X, y
        
        # Train model
        self.model = model
        self.model.fit(X_resampled, y_resampled)
        
        # Get feature importance if available
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Cross-validation evaluation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
        
        if verbose:
            print(f"Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return cv_scores
    
    def evaluate_model(self, X, y, plot_results=True):
        """
        Evaluate the trained model.
        
        Args:
            X: Feature matrix
            y: True labels
            plot_results: Whether to create evaluation plots
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        f1 = f1_score(y, y_pred)
        
        print("="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC-ROC: {auc:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print()
        
        print("Classification Report:")
        print(classification_report(y, y_pred, target_names=['Not Close to Death', 'Close to Death']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        
        if plot_results:
            self._plot_evaluation_results(y, y_pred, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
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
    
    def save_model(self, filepath='death_proximity_model.joblib'):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'proximity_threshold': self.proximity_threshold,
            'top_aging_features': self.top_aging_features
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath='death_proximity_model.joblib'):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.proximity_threshold = model_data['proximity_threshold']
        self.top_aging_features = model_data['top_aging_features']
        
        print(f"Model loaded from: {filepath}")

def analyze_death_risk_by_segment_position(df, predictor):
    """
    Analyze how death risk changes by segment position.
    
    Args:
        df: DataFrame with segments and predictions
        predictor: Trained DeathProximityPredictor
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
    print("="*80)
    print("C. ELEGANS DEATH PROXIMITY PREDICTION")
    print("="*80)
    print(f"Proximity threshold: {proximity_threshold} segments from end")
    print()
    
    # Load data
    print("Loading segment features data...")
    df = pd.read_csv('feature_data/segments_features.csv')
    
    # Initialize predictor
    predictor = DeathProximityPredictor(
        proximity_threshold=proximity_threshold,
        use_top_features=True
    )
    
    # Prepare data
    X, y, groups = predictor.prepare_data(df)
    
    # Train model
    print("\nTraining death proximity prediction model...")
    cv_scores = predictor.train_model(X, y, groups, model_name=model_name, use_smote=True)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    results = predictor.evaluate_model(X, y, plot_results=True)
    
    # Save model with threshold in filename
    model_filename = f'death_proximity_model_threshold_{proximity_threshold}.joblib'
    predictor.save_model(model_filename)
    
    # Analyze death risk by segment position
    print("\nAnalyzing death risk by segment position...")
    stage_analysis = analyze_death_risk_by_segment_position(df, predictor)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Files generated:")
    print(f"- {model_filename}")
    
    return predictor, results

if __name__ == "__main__":
    #main(proximity_threshold=20, model_name='RandomForest') #Accuracy: 0.833 AUC-ROC: 0.918 F1-Score: 0.739
    main(proximity_threshold=20, model_name='MLP')

# with RandomForest
#================================================================================
#THRESHOLD COMPARISON SUMMARY
#================================================================================
#Threshold  Pos Rate   CV AUC          Accuracy   Test AUC   F1 Score  
#------------------------------------------------------------------------------------------
#1          0.025      0.834±0.017     0.868      0.961      0.256    
#3          0.050      0.829±0.012     0.813      0.941      0.332   
#5          0.075      0.832±0.024     0.813      0.930      0.422   
#10         0.137      0.836±0.023     0.830      0.931      0.601     
#15         0.198      0.830±0.029     0.836      0.926      0.690     
#20         0.261      0.822±0.031     0.833      0.918      0.739    
#25         0.322      0.823±0.027     0.836      0.918      0.777      
#30         0.385      0.819±0.023     0.836      0.916      0.801     
#40         0.511      0.804±0.018     0.823      0.910      0.822


# with MLP
#================================================================================
#THRESHOLD COMPARISON SUMMARY
#================================================================================
#Threshold  Pos Rate   CV AUC          Accuracy   Test AUC   F1 Score  
#------------------------------------------------------------------------------------------
#1          0.025      0.814±0.018     0.995      0.999      0.898     
#3          0.050      0.793±0.019     0.984      0.997      0.857     
#5          0.075      0.787±0.023     0.980      0.998      0.882     
#10         0.137      0.771±0.031     0.933      0.989      0.802     
#15         0.198      0.768±0.028     0.943      0.988      0.869     
#20         0.261      0.740±0.026     0.963      0.993      0.930     
#30         0.385      0.732±0.030     0.938      0.986      0.922     
#25         0.322      0.731±0.016     0.938      0.985      0.909     
#40         0.511      0.700±0.030     0.952      0.990      0.953


# with GradientBoosting
#================================================================================
#Threshold  Pos Samples  Pos Rate   CV AUC          Accuracy   Test AUC   F1 Score  
#------------------------------------------------------------------------------------------
#1          206          0.025      0.815±0.022     0.999      1.000      0.980     
#3          412          0.050      0.815±0.021     0.996      1.000      0.964     
#5          615          0.075      0.811±0.018     0.986      0.998      0.913  
#10         1126         0.137      0.817±0.023     0.970      0.995      0.899     
#15         1627         0.198      0.809±0.026     0.958      0.989      0.898    
#20         2136         0.261      0.802±0.024     0.951      0.988      0.908    
#25         2643         0.322      0.810±0.022     0.943      0.986      0.913     
#30         3159         0.385      0.803±0.024     0.939      0.986      0.921     
#40         4186         0.511      0.787±0.019     0.945      0.989      0.945