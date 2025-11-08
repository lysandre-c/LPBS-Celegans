import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from imblearn.over_sampling import SMOTE

from data_loader import LPBSDataLoader



class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.input_size = input_size
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, input_size):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        # LSTM expects (batch, sequence_length, features)
        # If input is transposed for CNN, we need to transpose it back
        # CNN format: (batch, features, seq_len) where features=4 and seq_len=900
        # LSTM format: (batch, seq_len, features) where seq_len=900 and features=4
        if x.dim() == 3 and x.size(1) == 4 and x.size(2) > x.size(1):  # Check if it's CNN format (batch, 4, 900)
            x = x.transpose(1, 2)  # Convert from (batch, features, seq_len) to (batch, seq_len, features)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        x = hidden[-1]  # Use the last hidden state for classification
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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

def get_model(model_name: str, input_size):
    """Get PyTorch models for time series classification."""
    if model_name == 'CNN':
        return CNNClassifier(input_size)
    elif model_name == 'LSTM':
        return LSTMClassifier(input_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def pytorch_accuracy(y_true, y_pred):
    """Calculate accuracy using numpy."""
    return np.mean(y_true == y_pred)

def pytorch_f1_score(y_true, y_pred):
    """Calculate F1 score using numpy."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def pytorch_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix using numpy."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def weighted_voting_classification(model, weight_strategy='confidence', epochs=25, batch_size=32, learning_rate=0.001, features=None, verbose=False):
    """Weighted voting classifier using pure PyTorch.
    
    Args:
        model: PyTorch model instance (CNN or LSTM).
        weight_strategy: Strategy for weighting segments during voting.
        epochs: Number of training epochs per fold.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        features: List of feature names to use. Options: ['x', 'y', 'speed', 'turning_angle'].
                  If None, uses all features. Example: ['speed', 'turning_angle']
        verbose: Whether to print progress information.
    
    Returns:
        dict: Results including accuracy, F1, confusion matrix, and vote analysis.
    """
    # Feature mapping
    FEATURE_MAP = {'x': 0, 'y': 1, 'speed': 2, 'turning_angle': 3}
    ALL_FEATURES = ['x', 'y', 'speed', 'turning_angle']
    
    # Use MPS (Apple Silicon GPU) if available, otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if verbose:
        print(f"Using device: {device}")
    
    loader = LPBSDataLoader()
    X, y, groups = loader.load_segment_timeseries()
    
    # Select features if specified
    if features is not None:
        if verbose:
            print(f"Selected features: {features}")
        feature_indices = [FEATURE_MAP[f] for f in features]
        X = [ts[:, feature_indices] for ts in X]
    else:
        if verbose:
            print(f"Using all features: {ALL_FEATURES}")
    
    # Pad/truncate time series to same length
    target_length = 100  
    X_padded = []
    for ts in X:
        if len(ts) >= target_length:
            # Sample evenly spaced indices to downsample to target_length
            indices = np.linspace(0, len(ts) - 1, target_length, dtype=int)
            X_padded.append(ts[indices])
        else:
            # Pad with zeros if needed
            padded = np.zeros((target_length, ts.shape[1]))
            padded[:len(ts)] = ts
            X_padded.append(padded)
    X_padded = np.array(X_padded)
    
    if verbose:
        print(f"Loaded: {len(X):,} segments from {len(np.unique(groups))} worms")
        print(f"Padded shape: {X_padded.shape}")
        print(f"Weight strategy: {weight_strategy}")

    # Convert to pandas for cv_splits compatibility
    y_series = pd.Series(y)
    groups_series = pd.Series(groups)
    
    # Create dummy DataFrame with indices for cv_splits
    X_df = pd.DataFrame({'dummy': range(len(X_padded))})
    cv_splits = loader.create_cv_splits(X_df, y_series, groups_series, n_splits=5)
    
    file_predictions, file_true_labels = [], []
    vote_analysis = []
    
    for fold_idx, fold in enumerate(tqdm(cv_splits)):
        if verbose:
            print(f"\nFold {fold_idx + 1}/5")
        
        # Get indices for train/test
        train_indices = fold['X_train'].index.values
        test_indices = fold['X_test'].index.values
        
        X_train = X_padded[train_indices]
        X_test = X_padded[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # Apply SMOTE to balance the training data
        if verbose and fold_idx == 0:
            print(f"  Before SMOTE - Class distribution: {np.bincount(y_train)}")
        
        # Flatten time series for SMOTE (SMOTE needs 2D data)
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_flat = X_train.reshape(n_samples, n_timesteps * n_features)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
        
        # Reshape back to time series format
        X_train_balanced = X_train_balanced.reshape(-1, n_timesteps, n_features)
        
        if verbose and fold_idx == 0:
            print(f"  After SMOTE - Class distribution: {np.bincount(y_train_balanced)}")
            print(f"  Samples: {len(y_train)} → {len(y_train_balanced)}")
        
        # Convert to PyTorch tensors
        # For CNN: (batch, features, sequence_length) 
        # For LSTM: (batch, sequence_length, features) - handled in LSTM forward()
        X_train_tensor = torch.FloatTensor(X_train_balanced).transpose(1, 2).to(device)
        X_test_tensor = torch.FloatTensor(X_test).transpose(1, 2).to(device)
        y_train_tensor = torch.LongTensor(y_train_balanced).to(device)
        
        # Re-initialize model for each fold (CRITICAL!)
        fold_model = type(model)(model.input_size)
        fold_model.to(device)
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Create DataLoader for training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train the model
        fold_model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = fold_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Show progress every 10 epochs
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch}, Loss: {epoch_loss/len(train_dataloader):.4f}")
        
        # Test on each worm in the test split
        fold_model.eval()
        with torch.no_grad():
            test_outputs = fold_model(X_test_tensor)
            test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
            test_preds = np.argmax(test_probs, axis=1)
        
        # Group predictions by worm
        for test_worm in fold['test_files']:
            worm_mask = fold['groups_test'] == test_worm
            worm_indices = fold['groups_test'][worm_mask].index.values
            
            # Map back to test indices
            test_worm_indices = []
            for idx in worm_indices:
                test_pos = np.where(test_indices == idx)[0]
                if len(test_pos) > 0:
                    test_worm_indices.append(test_pos[0])
            
            if len(test_worm_indices) == 0:
                continue
                
            worm_preds = test_preds[test_worm_indices]
            worm_probs = test_probs[test_worm_indices]
            worm_true_label = fold['y_test'][worm_mask].iloc[0]
            n_segments = len(worm_preds)
            
            # Calculate weights based on strategy
            weights = calculate_segment_weights(weight_strategy, n_segments, worm_probs)
            
            # Calculate weighted votes
            weighted_vote_0 = np.sum(weights[worm_preds == 0])
            weighted_vote_1 = np.sum(weights[worm_preds == 1])
            worm_pred = int(weighted_vote_1 > weighted_vote_0)
            
            # Calculate confidence
            total_weight = weighted_vote_0 + weighted_vote_1
            confidence = max(weighted_vote_0, weighted_vote_1) / total_weight if total_weight > 0 else 0.5
            
            vote_analysis.append({
                'n_segments': n_segments,
                'weighted_pred': worm_pred,
                'weighted_confidence': confidence,
                'avg_weight': weights.mean(),
                'weight_std': weights.std(),
                'true_label': worm_true_label,
                'weighted_correct': worm_pred == worm_true_label,
            })
            
            file_predictions.append(worm_pred)
            file_true_labels.append(worm_true_label)
    
    file_predictions = np.array(file_predictions)
    file_true_labels = np.array(file_true_labels)
    
    accuracy = pytorch_accuracy(file_true_labels, file_predictions)
    f1 = pytorch_f1_score(file_true_labels, file_predictions)
    cm = pytorch_confusion_matrix(file_true_labels, file_predictions)
    
    vote_df = pd.DataFrame(vote_analysis)
    
    if verbose:
        print(f"\nResults: {len(file_predictions)} worms, Acc: {accuracy:.3f}, F1: {f1:.3f}")
        
        # Per-class accuracy
        class_0_mask = file_true_labels == 0
        class_1_mask = file_true_labels == 1
        class_0_acc = pytorch_accuracy(file_true_labels[class_0_mask], file_predictions[class_0_mask]) if class_0_mask.sum() > 0 else 0
        class_1_acc = pytorch_accuracy(file_true_labels[class_1_mask], file_predictions[class_1_mask]) if class_1_mask.sum() > 0 else 0
        
        print(f"\nPer-Class Performance:")
        print(f"  Class 0 (Control): {class_0_acc:.3f} ({class_0_mask.sum()} samples)")
        print(f"  Class 1 (Treatment): {class_1_acc:.3f} ({class_1_mask.sum()} samples)")
        
        print(f"\nWeighted Voting Analysis:")
        print(f"  Weighted accuracy: {vote_df['weighted_correct'].mean():.3f}")
        print(f"  Average confidence: {vote_df['weighted_confidence'].mean():.3f}")
        print(f"  Average segments per worm: {vote_df['n_segments'].mean():.1f}")
        
    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "n_worms": len(file_predictions),
        "vote_analysis": vote_df,
        "weight_strategy": weight_strategy
    }


if __name__ == "__main__":

    model_name = 'CNN'
    weight_strategy = 'last_10_segments_confidence'
    
    # Feature selection: specify which features to use
    # Options: ['x', 'y', 'speed', 'turning_angle']
    # Examples:
    #   None                            → use all 4 features
    #   ['speed', 'turning_angle']      → use only speed and turning angle (2 features)
    #   ['x', 'y']                      → use only coordinates (2 features)
    # selected_features = None  # Use all features by default
    selected_features = ['speed', 'turning_angle']
    
    print("===== Time Series Weighted Voting Classification =====")
    
    # Get input size based on selected features
    if selected_features is None:
        input_size = 4  # All features: x, y, speed, turning_angle
    else:
        input_size = len(selected_features)
    
    print(f"Input size: {input_size} features")
    if selected_features:
        print(f"Using features: {selected_features}")
    else:
        print(f"Using all features: ['x', 'y', 'speed', 'turning_angle']")
    
    model = get_model(model_name, input_size)
    results = weighted_voting_classification(
        model, 
        weight_strategy=weight_strategy, 
        epochs=30,  # Using 30 epochs with SMOTE balancing
        batch_size=32, 
        learning_rate=0.001,
        features=selected_features,  # Pass selected features
        verbose=True
    )
    print("\n===== Results =====")
    print("Accuracy:", results['accuracy'])
    print("F1:", results['f1'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
