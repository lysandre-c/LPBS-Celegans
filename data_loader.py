import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional, Tuple, Dict, List, Union, Any
from sklearn.model_selection import StratifiedKFold
import re

# Constants for time series features

TIME_SERIES_FEATURES = ['x_coordinate', 'y_coordinate', 'speed', 'turning_angle']



def extract_features_and_labels(df):
    """Extract features and labels from dataframe."""
    metadata_columns = ['label', 'filename', 'relative_path', 'file', 'worm_id', 'segment_number', 'segment_index', 'original_file']
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    feature_columns = [col for col in numeric_columns if col not in metadata_columns]
    
    X = df[feature_columns].copy()
    y = df['label'].copy()
    
    # Extract base filename for proper grouping (segments from same worm grouped together)
    groups = df['filename'].apply(lambda x: extract_worm_and_segment_info(x)[0])
    
    return X, y, groups


def extract_worm_and_segment_info(filename):
    """Extract worm ID and segment number from filename."""
    # Handle segment files: filename-segment5.0-preprocessed.csv
    segment_match = re.search(r'segment(\d+)', filename)
    if segment_match:
        segment_num = int(segment_match.group(1))
        # Extract base filename without segment info
        worm_id = re.sub(r'-segment\d+.*', '', filename)
        return worm_id, segment_num
    
    # Handle full files or files without segment info
    worm_id = re.sub(r'-preprocessed.*', '', filename)
    return worm_id, 0


def create_death_proximity_labels(df_with_filenames, proximity_threshold=5):
    """
    Create labels where 1 = death is close (within last N segments), 0 = death is far.
    
    Args:
        df_with_filenames: DataFrame with 'filename' and 'original_file' columns
        proximity_threshold: Number of final segments to consider as "death is close"
    
    Returns:
        pandas Series with death proximity labels (1 if close to death, 0 otherwise)
    """
    # Extract segment numbers from filenames
    temp_df = df_with_filenames.copy()
    temp_df['segment_number'] = temp_df['filename'].apply(
        lambda x: int(re.search(r'segment(\d+)', x).group(1)) if re.search(r'segment(\d+)', x) else 0
    )
    
    # Find max segment number for each worm
    max_segments = temp_df.groupby('original_file')['segment_number'].max().reset_index()
    max_segments.columns = ['original_file', 'max_segment']
    
    # Merge back to get max segment for each row
    temp_df = temp_df.merge(max_segments, on='original_file')
    
    # Calculate if segment is within proximity_threshold of death
    # Label = 1 if segment_number > (max_segment - proximity_threshold), 0 otherwise
    temp_df['death_close'] = (temp_df['segment_number'] > (temp_df['max_segment'] - proximity_threshold)).astype(int)
    
    return temp_df['death_close']


def create_kfold_splits(X, y, groups, n_splits=5):
    """Create file-based k-fold splits to prevent data leakage."""
    # Extract unique files and their labels
    file_df = pd.DataFrame({'file': groups, 'label': y}).drop_duplicates('file')
    unique_files = file_df['file'].values
    file_labels = file_df['label'].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_splits = []
    for train_file_idx, test_file_idx in skf.split(unique_files, file_labels):
        train_files = unique_files[train_file_idx]
        test_files = unique_files[test_file_idx]
        
        train_mask = groups.isin(train_files)
        test_mask = groups.isin(test_files)
        
        X_train = X[train_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)
        groups_train = groups[train_mask].reset_index(drop=True)
        groups_test = groups[test_mask].reset_index(drop=True)
        
        fold_splits.append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'groups_train': groups_train,
            'groups_test': groups_test,
            'train_files': train_files,
            'test_files': test_files
        })
    
    return fold_splits



class LPBSDataLoader:
    """
    Comprehensive data loader for LPBS worm movement analysis.
    
    Features:
    - Load preprocessed trajectory data (time series) and extracted features
    - Support for both segment-level and full-trajectory data
    - File-based splitting to prevent data leakage
    - Easy filtering to first N segments per worm
    - Built-in cross-validation support
    - Data validation and statistics
    """
    
    def __init__(
        self,
        base_dir: str = ".",
        preprocessed_dir: str = "preprocessed_data",
        feature_dir: str = "feature_data"
    ):
        """
        Initialize the data loader.
        
        Args:
            base_dir: Base directory containing the project
            preprocessed_dir: Directory containing preprocessed trajectory data
            feature_dir: Directory containing extracted features
        """
        self.base_dir = Path(base_dir)
        self.preprocessed_dir = self.base_dir / preprocessed_dir
        self.feature_dir = self.base_dir / feature_dir
        
        self._segment_features = None
        self._full_features = None
        self._segment_timeseries = None
        self._full_timeseries = None
        self._metadata = {}
    
    def load_segment_features(self, force_reload: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load segment-level features."""
        if self._segment_features is None or force_reload:
            df = pd.read_csv(self.feature_dir / "segments_features.csv")
            X, y, groups = extract_features_and_labels(df)
            self._segment_features = {'X': X, 'y': y, 'groups': groups}
            
        return self._segment_features['X'], self._segment_features['y'], self._segment_features['groups']
    
    def load_segment_features_with_custom_labels(
        self, 
        labeling_strategy: str = "death_proximity", 
        proximity_threshold: int = 5,
        force_reload: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load segment-level features with custom labeling strategies.
        
        Args:
            labeling_strategy: Strategy for creating labels ('death_proximity', 'original')
            proximity_threshold: For death_proximity, number of final segments considered close to death
            force_reload: Force reload of data
            
        Returns:
            Tuple of (features, labels, groups)
        """
        # Load the raw data
        df_full = pd.read_csv(self.feature_dir / "segments_features.csv")
        
        # Extract features using existing method
        X, original_y, groups = extract_features_and_labels(df_full)
        
        if labeling_strategy == "original":
            # Use original labels from the data
            y = original_y
        elif labeling_strategy == "death_proximity":
            # Create death proximity labels
            segment_data = df_full[['filename', 'original_file']].copy()
            y = create_death_proximity_labels(segment_data, proximity_threshold)
        else:
            raise ValueError(f"Unknown labeling strategy: {labeling_strategy}")
        
        return X, y, groups
    
    def load_segment_features_filtered(
        self,
        n_segments: Optional[int] = None,
        labeling_strategy: str = "death_proximity",
        proximity_threshold: int = 5,
        force_reload: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load segment-level features with optional filtering to first N segments and custom labeling.
        
        Args:
            n_segments: If provided, filter to first N segments per worm
            labeling_strategy: Strategy for creating labels ('death_proximity', 'original')
            proximity_threshold: For death_proximity, number of final segments considered close to death
            force_reload: Force reload of data
            
        Returns:
            Tuple of (features, labels, groups)
        """
        # Load the raw data
        df_full = pd.read_csv(self.feature_dir / "segments_features.csv")
        
        # Extract segment numbers from filenames if filtering is needed
        if n_segments is not None:
            df_full['segment_number'] = df_full['filename'].apply(
                lambda x: int(re.search(r'segment(\d+)', x).group(1)) if re.search(r'segment(\d+)', x) else 0
            )
            # Filter to first N segments per worm
            filtered_mask = df_full['segment_number'] < n_segments
            df_filtered = df_full[filtered_mask].reset_index(drop=True)
        else:
            df_filtered = df_full
        
        # Extract features using existing method
        X, original_y, groups = extract_features_and_labels(df_filtered)
        
        if labeling_strategy == "original":
            # Use original labels from the data
            y = original_y
        elif labeling_strategy == "death_proximity":
            # Create death proximity labels
            segment_data = df_filtered[['filename', 'original_file']].copy()
            y = create_death_proximity_labels(segment_data, proximity_threshold)
        else:
            raise ValueError(f"Unknown labeling strategy: {labeling_strategy}")
        
        return X, y, groups
    
    def load_full_features(self, force_reload: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load full-trajectory features."""
        if self._full_features is None or force_reload:
            df = pd.read_csv(self.feature_dir / "full_features.csv")
            X, y, groups = extract_features_and_labels(df)
            self._full_features = {'X': X, 'y': y, 'groups': groups}
            
        return self._full_features['X'], self._full_features['y'], self._full_features['groups']
    
    def load_segment_timeseries(self, force_reload: bool = False) -> Tuple[List, np.ndarray, np.ndarray]:
        """Load segment-level time series data."""
        if self._segment_timeseries is None or force_reload:
            segments_dir = self.preprocessed_dir / "segments"
            metadata = pd.read_csv(segments_dir / "labels_and_metadata.csv")
            
            time_series_data = []
            labels = []
            groups = []
            
            for _, row in metadata.iterrows():
                try:
                    df = pd.read_csv(segments_dir / row['relative_path'] / row['file'])
                    ts_data = df[['x', 'y', 'speed', 'turning_angle']].fillna(0).values
                    time_series_data.append(ts_data)
                    labels.append(row['label'])
                    groups.append(row['file'])
                except:
                    continue
            
            self._segment_timeseries = {
                'X': time_series_data,
                'y': np.array(labels),
                'groups': np.array(groups)
            }
            
        return self._segment_timeseries['X'], self._segment_timeseries['y'], self._segment_timeseries['groups']
    
    def load_full_timeseries(self, force_reload: bool = False) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """Load full-trajectory time series data.""" 
        if self._full_timeseries is None or force_reload:
            full_dir = self.preprocessed_dir / "full"
            metadata = pd.read_csv(full_dir / 'labels_and_metadata.csv')
            
            time_series_data = []
            labels = []
            groups = []
            
            for _, row in metadata.iterrows():
                try:
                    df = pd.read_csv(full_dir / row['relative_path'] / row['file'])
                    ts_data = df[['x', 'y', 'speed', 'turning_angle']].fillna(0).values
                    time_series_data.append(ts_data)
                    labels.append(row['label'])
                    groups.append(row['file'])
                except:
                    continue
            
            self._full_timeseries = {
                'X': time_series_data,
                'y': np.array(labels),
                'groups': np.array(groups)
            }
            
        return self._full_timeseries['X'], self._full_timeseries['y'], self._full_timeseries['groups']
    
    def create_cv_splits(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray], 
        groups: Union[pd.Series, np.ndarray],
        n_splits: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Create cross-validation splits with file-based grouping.
        
        Args:
            X: Feature matrix or time series data
            y: Labels
            groups: File identifiers for grouping
            n_splits: Number of CV splits
            
        Returns:
            List of fold split dictionaries
        """
        return create_kfold_splits(X, y, groups, n_splits)
    
    def get_data_info(self, data_type: str = "segment_features") -> Dict[str, Any]:
        """
        Get information about loaded data.
        
        Args:
            data_type: Type of data to analyze
            
        Returns:
            Dictionary with data statistics
        """
        if data_type == "segment_features":
            if self._segment_features is None:
                self.load_segment_features()
            X, y, groups = self._segment_features['X'], self._segment_features['y'], self._segment_features['groups']
        elif data_type == "full_features":
            if self._full_features is None:
                self.load_full_features()
            X, y, groups = self._full_features['X'], self._full_features['y'], self._full_features['groups']
        elif data_type == "segment_timeseries":
            if self._segment_timeseries is None:
                self.load_segment_timeseries()
            X, y, groups = self._segment_timeseries['X'], self._segment_timeseries['y'], self._segment_timeseries['groups']
        elif data_type == "full_timeseries":
            if self._full_timeseries is None:
                self.load_full_timeseries()
            X, y, groups = self._full_timeseries['X'], self._full_timeseries['y'], self._full_timeseries['groups']
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        info = {
            'data_type': data_type,
            'n_samples': len(y),
            'n_features': X.shape[1] if len(X.shape) > 1 else 'Variable',
            'n_files': len(np.unique(groups)),
            'class_distribution': dict(pd.Series(y).value_counts()),
        }
        
        if data_type.endswith('_timeseries'):
            info['sequence_length'] = 'Variable'
            info['sequence_lengths'] = {
                'min': min(len(seq) for seq in X),
                'max': max(len(seq) for seq in X),
                'mean': np.mean([len(seq) for seq in X])
            }
        else:
            info['feature_names'] = list(X.columns)
        
        return info
    
    def print_data_summary(self):
        """Print a summary of available data."""
        print("LPBS DataLoader Summary:")
        
        data_types = [
            ("load_segment_features", "Segment Features"),
            ("load_full_features", "Full Features"), 
            ("load_segment_timeseries", "Segment Time Series"),
            ("load_full_timeseries", "Full Time Series")
        ]
        
        for method_name, display_name in data_types:
            try:
                method = getattr(self, method_name)
                X, y, groups = method()
                n_samples = len(X) if isinstance(X, list) else X.shape[0]
                n_features = "variable" if isinstance(X, list) else X.shape[1]
                print(f"  {display_name}: {n_samples} samples, {n_features} features")
            except:
                print(f"  {display_name}: Not available")
    
    def save_data_info(self, output_file: str = "data_info.json"):
        """Save basic data info to JSON file.""" 
        info = {}
        for name, display in [("segment_features", "Segment Features"), ("full_features", "Full Features")]:
            try:
                info[name] = self.get_data_info(name)
            except:
                info[name] = {"error": "Not available"}
        
        with open(self.base_dir / output_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._segment_features = None
        self._full_features = None
        self._segment_timeseries = None
        self._full_timeseries = None


# Example usage
if __name__ == "__main__":
    loader = LPBSDataLoader()
    loader.print_data_summary()
    
    # Test basic functionality
    X, y, groups = loader.load_segment_features()
    print(f"✓ Loaded {X.shape[0]:,} samples with {X.shape[1]} features")
    
    X_first3, y_first3, groups_first3 = loader.get_first_n_segments(3, "features")
    cv_splits = loader.create_cv_splits(X_first3, y_first3, groups_first3, n_splits=3)
    print(f"✓ Created {len(cv_splits)} CV splits")
    print("✅ DataLoader working correctly!")
