"""This module uses regression to predict the number of segments remaining until death."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import LPBSDataLoader


class DeathProximityRegressor:
    """
    Predicts the number of segments remaining until death using regression.
    """
    
    def __init__(self, use_top_features=True, target_type='segments_from_end'):
        """
        Initialize the death proximity regressor.
        
        Args:
            use_top_features: Whether to use only the most discriminative features
            target_type: Type of target to predict:
                - 'segments_from_end': Number of segments remaining (default)
                - 'life_percentage_remaining': Percentage of life remaining (0-100)
        """
        self.use_top_features = use_top_features
        self.target_type = target_type
        self.model = None
        self.feature_names = None
        
        # Top discriminative features for aging
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
        Prepare data for death proximity regression.
        
        Args:
            df: DataFrame with segment features
            
        Returns:
            X, y, groups: Features, targets (continuous), and worm groups
        """
        # Extract segment index if needed
        if 'segment_index' not in df.columns or df['segment_index'].isna().all():
            df['segment_index'] = df['filename'].str.extract(r'segment(\d+(?:\.\d+)?)', expand=False).astype(float)
        
        # Calculate position from end for each worm
        max_segments = df.groupby('original_file')['segment_index'].max()
        df['segments_from_end'] = df.apply(
            lambda row: max_segments[row['original_file']] - row['segment_index'], axis=1
        )
        
        # Calculate life percentage for alternative targets
        df['max_segment_index'] = df['original_file'].map(max_segments)
        df['life_percentage'] = (df['segment_index'] / df['max_segment_index']) * 100
        df['life_percentage_remaining'] = 100 - df['life_percentage']
        
        # Select target based on target_type
        if self.target_type == 'segments_from_end':
            target_col = 'segments_from_end'
        elif self.target_type == 'life_percentage_remaining':
            target_col = 'life_percentage_remaining'
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
        
        # Select features
        if self.use_top_features:
            features = [f for f in self.top_aging_features if f in df.columns]
        else:
            metadata_cols = ['label', 'filename', 'relative_path', 'file', 'worm_id', 'segment_number', 
                           'segment_index', 'original_file', 'max_segment_index', 'segments_from_end',
                           'life_percentage', 'life_percentage_remaining']
            features = [col for col in df.columns if col not in metadata_cols and df[col].dtype in ['float64', 'int64']]
        
        self.feature_names = features
        
        # Print summary
        print(f"Data prepared: {len(df)} segments, {len(features)} features")
        print(f"  Target: {target_col}")
        print(f"  Target range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")
        print(f"  Target mean: {df[target_col].mean():.2f} ± {df[target_col].std():.2f}")
        
        # Prepare feature matrix and handle missing values
        X = df[features].fillna(df[features].median())
        y = df[target_col]
        groups = df['original_file']
        
        return X, y, groups
    
    def train_model(self, X, y, model_name='RandomForest', verbose=True):
        """
        Train the death proximity regression model.
        
        Args:
            X: Feature matrix
            y: Target values (continuous)
            model_name: Type of model to use
            verbose: Print training progress
        """
        if verbose:
            print(f"Training {model_name} regressor...")
        
        # Update feature names
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else list(range(X.shape[1]))
        
        # Create model pipeline
        model_configs = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42
            ),
            'Ridge': Ridge(
                alpha=1.0, random_state=42
            ),
            'Lasso': Lasso(
                alpha=1.0, random_state=42, max_iter=2000
            ),
            'MLP': MLPRegressor(
                hidden_layer_sizes=(32, 32, 32, 16), random_state=42, max_iter=500
            )
        }
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model_configs[model_name])
        ])
        
        # Convert to numpy arrays
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        # Train model
        self.model = model
        self.model.fit(X_array, y_array)
    
    def evaluate_model(self, X, y, plot_results=False, verbose=True):
        """
        Evaluate the trained regression model.
        
        Args:
            X: Feature matrix
            y: True target values
            plot_results: Whether to create evaluation plots
            verbose: Whether to print detailed results
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Convert to numpy array to match training format
        X_array = X.values if hasattr(X, 'values') else X
        
        # Predictions
        y_pred = self.model.predict(X_array)
        
        # Calculate regression metrics
        y_array = y.values if hasattr(y, 'values') else y
        results = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'predictions': y_pred,
            'y_true': y_array
        }
        
        if verbose:
            print("="*60)
            print("REGRESSION MODEL EVALUATION RESULTS")
            print("="*60)
            print(f"Mean Absolute Error (MAE):  {results['mae']:.3f}")
            print(f"Root Mean Squared Error:    {results['rmse']:.3f}")
            print(f"R² Score:                   {results['r2']:.3f}")
            
        if plot_results:
            self._plot_evaluation_results(y, y_pred)
        
        return results
    
    def _plot_evaluation_results(self, y_true, y_pred):
        """Create evaluation plots for regression."""
        # Determine axis labels based on target type
        if self.target_type == 'segments_from_end':
            target_label = 'Segments from End'
        elif self.target_type == 'life_percentage_remaining':
            target_label = 'Life Percentage Remaining (%)'
        else:
            target_label = 'Target Value'
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel(f'Actual {target_label}')
        axes[0].set_ylabel(f'Predicted {target_label}')
        axes[0].set_title('Predicted vs Actual')
        
        # Distribution of predictions vs actual
        axes[1].hist(y_true, bins=30, alpha=0.5, label='Actual', edgecolor='black')
        axes[1].hist(y_pred, bins=30, alpha=0.5, label='Predicted', edgecolor='black')
        axes[1].set_xlabel(target_label)
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution Comparison')
        axes[1].legend()
        
        # Error histogram
        abs_errors = np.abs(y_true - y_pred)
        axes[2].hist(abs_errors, bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(x=np.mean(abs_errors), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(abs_errors):.2f}')
        axes[2].set_xlabel('Absolute Error')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Absolute Error Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('death_proximity_regression_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_segments_remaining(self, X):
        """
        Predict segments remaining until death for new data.
        
        Args:
            X: Feature matrix for new segments
            
        Returns:
            Predicted segments remaining (continuous values)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Handle missing features
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0  # Default value for missing features
        
        # Ensure correct feature order
        X = X[self.feature_names]
        
        # Convert to numpy array to match training format
        X_array = X.values if hasattr(X, 'values') else X
        
        predictions = self.model.predict(X_array)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions


def main(model_name='RandomForest', target_type='segments_from_end', use_top_features=True):
    """
    Main function to demonstrate death proximity regression.
    
    Args:
        model_name: Type of regression model to use
        target_type: Type of target variable to predict
        use_top_features: Whether to use only top discriminative features (True) or all features (False)
    """    
    print(f"Target: {target_type}")
    print(f"Model: {model_name}")
    print(f"Using top features: {use_top_features}")
    print()
    
    print("Loading segment features data...")
    df = pd.read_csv('feature_data/segments_features.csv')
    
    regressor = DeathProximityRegressor(
        use_top_features=use_top_features,
        target_type=target_type
    )
    
    X, y, groups = regressor.prepare_data(df)
    
    loader = LPBSDataLoader()
    cv_splits = loader.create_cv_splits(X, y, groups, n_splits=5)
    
    fold_results = []
    for fold_idx, fold in enumerate(cv_splits):
        print(f"\nFold {fold_idx + 1}/{len(cv_splits)}:")
        print(f"  Train: {len(fold['X_train'])} segments from {len(fold['train_files'])} files")
        print(f"  Test:  {len(fold['X_test'])} segments from {len(fold['test_files'])} files")
        
        # Create and train fold regressor
        fold_regressor = DeathProximityRegressor(
            use_top_features=use_top_features,
            target_type=target_type
        )
        fold_regressor.feature_names = regressor.feature_names
        
        fold_regressor.train_model(
            fold['X_train'], fold['y_train'], 
            model_name=model_name, verbose=(fold_idx == 0)
        )
        
        # Evaluate on test set
        results = fold_regressor.evaluate_model(fold['X_test'], fold['y_test'], verbose=False)
        fold_results.append(results)
        
        print(f"    MAE: {results['mae']:.3f}, RMSE: {results['rmse']:.3f}, R²: {results['r2']:.3f}")
    
    # Calculate average metrics across folds
    metrics = {
        'mae': [r['mae'] for r in fold_results],
        'rmse': [r['rmse'] for r in fold_results],
        'r2': [r['r2'] for r in fold_results]
    }
    
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}
    
    # Determine unit label based on target type
    if target_type == 'segments_from_end':
        unit_label = 'segments'
    elif target_type == 'life_percentage_remaining':
        unit_label = '%'
    else:
        unit_label = 'units'
    
    # Print summary
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS (True Generalization Performance)")
    print("="*80)
    print(f"MAE:   {avg_metrics['mae']:.3f} ± {std_metrics['mae']:.3f} {unit_label}")
    print(f"RMSE:  {avg_metrics['rmse']:.3f} ± {std_metrics['rmse']:.3f} {unit_label}")
    print(f"R²:    {avg_metrics['r2']:.3f} ± {std_metrics['r2']:.3f}")
    
    # Create visualization from aggregated CV predictions
    print("\nGenerating evaluation plots from CV predictions...")
    _plot_cv_results(fold_results, df, target_type)
    
    return {
        'fold_results': fold_results,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics
    }


def _plot_cv_results(fold_results, df, target_type='segments_from_end'):
    """
    Create evaluation plots from aggregated CV fold predictions.
    
    Args:
        fold_results: List of result dictionaries from each CV fold,
                     each containing 'y_true' and 'predictions'
        df: Original dataframe with segment metadata (unused, kept for compatibility)
        target_type: Type of target variable being predicted
    """
    # Determine axis labels based on target type
    if target_type == 'segments_from_end':
        target_label = 'Segments from End'
    elif target_type == 'life_percentage_remaining':
        target_label = 'Life Percentage Remaining (%)'
    else:
        target_label = 'Target Value'
    
    # Aggregate all predictions and true values across folds
    all_y_true = []
    all_y_pred = []
    
    for result in fold_results:
        all_y_true.extend(result['y_true'])
        all_y_pred.extend(result['predictions'])
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Predicted vs Actual (aggregated across all CV folds)
    axes[0].scatter(all_y_true, all_y_pred, alpha=0.5)
    axes[0].plot([all_y_true.min(), all_y_true.max()], 
                 [all_y_true.min(), all_y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel(f'Actual {target_label}')
    axes[0].set_ylabel(f'Predicted {target_label}')
    axes[0].set_title('Predicted vs Actual (CV Test Sets)')
    
    # 2. Distribution comparison
    axes[1].hist(all_y_true, bins=30, alpha=0.5, label='Actual', edgecolor='black')
    axes[1].hist(all_y_pred, bins=30, alpha=0.5, label='Predicted', edgecolor='black')
    axes[1].set_xlabel(target_label)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution Comparison (CV Test Sets)')
    axes[1].legend()
    
    # 3. Error distribution
    abs_errors = np.abs(all_y_true - all_y_pred)
    axes[2].hist(abs_errors, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=np.mean(abs_errors), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(abs_errors):.2f}')
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Absolute Error Distribution (CV Test Sets)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Test with different models and feature sets
    print("\n" + "="*80)
    print("TESTING RANDOM FOREST REGRESSOR - TOP FEATURES")
    print("="*80)
    results1 = main(model_name='RandomForest', target_type='segments_from_end', use_top_features=True)
    
    print("\n" + "="*80)
    print("TESTING RANDOM FOREST REGRESSOR - TOP FEATURES")
    print("="*80)
    results2 = main(model_name='RandomForest', target_type='life_percentage_remaining', use_top_features=True)
    
    print(f"\nSegments from end - MAE:   {results1['avg_metrics']['mae']:.3f} ± {results1['std_metrics']['mae']:.3f} segments")
    print(f"Segments from end - RMSE:  {results1['avg_metrics']['rmse']:.3f} ± {results1['std_metrics']['rmse']:.3f} segments")
    print(f"Segments from end - R²:    {results1['avg_metrics']['r2']:.3f} ± {results1['std_metrics']['r2']:.3f}")
    
    print(f"\nLife percentage - MAE:   {results2['avg_metrics']['mae']:.3f} ± {results2['std_metrics']['mae']:.3f} %")
    print(f"Life percentage - RMSE:  {results2['avg_metrics']['rmse']:.3f} ± {results2['std_metrics']['rmse']:.3f} %")
    print(f"Life percentage - R²:    {results2['avg_metrics']['r2']:.3f} ± {results2['std_metrics']['r2']:.3f}")
    
    # Uncomment to test other models:
    # print("\n" + "="*80)
    # print("TESTING GRADIENT BOOSTING REGRESSOR")
    # print("="*80)
    # results_gb = main(model_name='GradientBoosting', target_type='segments_from_end', use_top_features=True)
    # 
    # print("\n" + "="*80)
    # print("TESTING MLP REGRESSOR")
    # print("="*80)
    # results_mlp = main(model_name='MLP', target_type='segments_from_end', use_top_features=True)


# - 'segments_from_end': Number of segments remaining (default)
# - 'life_percentage_remaining': Percentage of life remaining (0-100)