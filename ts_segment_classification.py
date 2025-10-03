import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from data_loader import LPBSDataLoader



class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.global_avg_pool(x)  # Global average pooling to handle variable lengths
        x = x.view(x.size(0), -1)   # Flatten
        x = self.fc1(x)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, input_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 2)
        
    def forward(self, x):
        # LSTM expects (batch, sequence_length, features)
        # If input is transposed for CNN, we need to transpose it back
        # CNN format: (batch, features, seq_len) where features=4 and seq_len=900
        # LSTM format: (batch, seq_len, features) where seq_len=900 and features=4
        if x.dim() == 3 and x.size(1) == 4 and x.size(2) > x.size(1):  # Check if it's CNN format (batch, 4, 900)
            x = x.transpose(1, 2)  # Convert from (batch, features, seq_len) to (batch, seq_len, features)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use the last hidden state for classification
        x = hidden[-1]  # Take the last layer's hidden state
        x = self.fc1(x)
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


def weighted_voting_classification(model, weight_strategy='confidence', epochs=25, batch_size=32, learning_rate=0.001, verbose=False):
    """Weighted voting classifier using pure PyTorch."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loader = LPBSDataLoader()
    X, y, groups = loader.load_segment_timeseries()
    
    # Extract base worm names from segment filenames for proper grouping
    base_groups = []
    for filename in groups:
        # Extract base worm name by removing segment info
        base_name = re.sub(r'-segment\d+\.0-preprocessed\.csv$', '', filename)
        base_groups.append(base_name)
    base_groups = np.array(base_groups)
    
    # Pad/truncate time series to same length for PyTorch
    target_length = 300  
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
        print(f"Loaded: {len(X):,} segments from {len(np.unique(base_groups))} worms")
        print(f"Padded shape: {X_padded.shape}")
        print(f"Weight strategy: {weight_strategy}")

    # Convert to pandas for cv_splits compatibility
    y_series = pd.Series(y)
    base_groups_series = pd.Series(base_groups)
    
    # Create dummy DataFrame with indices for cv_splits
    X_df = pd.DataFrame({'dummy': range(len(X_padded))})
    cv_splits = loader.create_cv_splits(X_df, y_series, base_groups_series, n_splits=5)
    
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
        
        # Convert to PyTorch tensors
        # For CNN: (batch, features, sequence_length) 
        # For LSTM: (batch, sequence_length, features) - handled in LSTM forward()
        X_train_tensor = torch.FloatTensor(X_train).transpose(1, 2).to(device)
        X_test_tensor = torch.FloatTensor(X_test).transpose(1, 2).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        
        # Reset model for each fold
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Create DataLoader for training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train the model
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}, Loss: {epoch_loss/len(train_dataloader):.4f}")
        
        # Test on each worm in the test split
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
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

    model_name = 'LSTM'
    weight_strategy = 'last_10_segments_confidence'
    
    print("===== Time Series Weighted Voting Classification =====")
    
    # Get input size from data
    loader = LPBSDataLoader()
    X, _, _ = loader.load_segment_timeseries()
    input_size = X[0].shape[1]  # Number of features (x, y, speed, turning_angle = 4)
    
    print(f"Input size: {input_size} features")
    
    model = get_model(model_name, input_size)
    results = weighted_voting_classification(
        model, 
        weight_strategy=weight_strategy, 
        epochs=25, 
        batch_size=32, 
        learning_rate=0.001,
        verbose=True
    )
    print("Accuracy:", results['accuracy'])
    print("F1:", results['f1'])
    print("Confusion Matrix:", results['confusion_matrix'])
