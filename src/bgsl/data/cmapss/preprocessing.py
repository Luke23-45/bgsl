import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger("bgsl.cmapss.preprocessing")

CMAPSS_COLUMNS = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]

def read_cmapss_file(filepath: str) -> pd.DataFrame:
    """Read a CMAPSS text file, handling trailing spaces safely."""
    df = pd.read_csv(filepath, sep=r'\s+', header=None)
    # The last two columns might be NaNs due to trailing spaces
    df = df.dropna(axis=1, how='all')
    
    # Assign standard names to columns up to the number of valid columns
    df.columns = CMAPSS_COLUMNS[:df.shape[1]]
    return df

def process_train_data(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    """Compute piece-wise linear RUL for training data."""
    rul = pd.DataFrame(df.groupby('unit_id')['cycle'].max()).reset_index()
    rul.columns = ['unit_id', 'max_cycle']
    df = df.merge(rul, on=['unit_id'], how='left')
    df['rul'] = df['max_cycle'] - df['cycle']
    df['rul'] = df['rul'].clip(upper=max_rul)
    df = df.drop(columns=['max_cycle'])
    return df

def process_test_data(df: pd.DataFrame, rul_filepath: str, max_rul: int) -> pd.DataFrame:
    """Compute RUL for test data using the provided true RUL file."""
    # RUL file has true RUL for the *last* cycle of each engine
    true_rul = pd.read_csv(rul_filepath, sep=r'\s+', header=None, names=['rul'])
    true_rul['unit_id'] = true_rul.index + 1
    
    # Get the max cycle (last recorded cycle) for each engine in test
    max_cycles = pd.DataFrame(df.groupby('unit_id')['cycle'].max()).reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    # Merge true RUL at the end
    rul_data = max_cycles.merge(true_rul, on='unit_id')
    # True RUL at cycle `t` = true_rul_at_end + (max_cycle - t)
    rul_data['max_rul_theoretical'] = rul_data['max_cycle'] + rul_data['rul']
    
    df = df.merge(rul_data[['unit_id', 'max_rul_theoretical']], on='unit_id', how='left')
    df['rul'] = df['max_rul_theoretical'] - df['cycle']
    df['rul'] = df['rul'].clip(upper=max_rul)
    df = df.drop(columns=['max_rul_theoretical'])
    return df

def find_zero_variance_features(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """Find features with zero variance on the training set."""
    zero_var_cols = []
    for col in feature_cols:
        if df[col].std() == 0 or df[col].nunique() == 1:
            zero_var_cols.append(col)
    return zero_var_cols

def create_sliding_windows(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    window_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sliding windows grouped by unit_id.
    Zero-pads sequences shorter than window_size and creates an attention mask.
    
    Returns:
    - inputs: [N, window_size, num_features]
    - ruls: [N]
    - masks: [N, window_size] (1 for valid timestep, 0 for pad)
    """
    inputs_list = []
    rul_list = []
    mask_list = []
    
    for unit_id, group in df.groupby('unit_id'):
        data = group[feature_cols].values
        ruls = group['rul'].values
        seq_len = len(data)
        
        for i in range(seq_len):
            # i is the index of the end of the window
            end_idx = i + 1
            start_idx = max(0, end_idx - window_size)
            
            # Slice the sequence
            seq = data[start_idx:end_idx]
            curr_rul = ruls[i]
            
            curr_len = len(seq)
            pad_len = window_size - curr_len
            
            # Pad if necessary
            if pad_len > 0:
                pad_arr = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
                seq = np.vstack([pad_arr, seq])
                mask = np.array([0]*pad_len + [1]*curr_len, dtype=np.float32)
            else:
                mask = np.ones(window_size, dtype=np.float32)
                
            inputs_list.append(seq)
            rul_list.append(curr_rul)
            mask_list.append(mask)
            
    # Convert to tensors
    inputs = torch.tensor(np.array(inputs_list, dtype=np.float32))
    ruls = torch.tensor(np.array(rul_list, dtype=np.float32))
    masks = torch.tensor(np.array(mask_list, dtype=np.float32))
    
    return inputs, ruls, masks

def build_dataset(cfg_path: str):
    """Main pipeline for preprocessing CMAPSS data."""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    raw_dir = Path(cfg['raw_data_path'])
    out_dir = Path(cfg['processed_data_path'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subsets = cfg.get('subsets', ["FD001", "FD002", "FD003", "FD004"])
    window_size = cfg['preprocessing']['window_size']
    max_rul = cfg['preprocessing']['max_rul']
    auto_drop = cfg['preprocessing']['auto_drop_constant']
    val_ratio = cfg['preprocessing'].get('val_ratio', 0.20)
    seed = cfg['preprocessing'].get('seed', 42)
    
    np.random.seed(seed)
    
    for subset in subsets:
        print(f"Processing {subset}...")
        
        train_file = raw_dir / f"train_{subset}.txt"
        test_file = raw_dir / f"test_{subset}.txt"
        rul_file = raw_dir / f"RUL_{subset}.txt"
        
        train_df = read_cmapss_file(str(train_file))
        test_df = read_cmapss_file(str(test_file))
        
        train_df = process_train_data(train_df, max_rul)
        test_df = process_test_data(test_df, str(rul_file), max_rul)
        
        # Split train_df into train and val by engine ID
        all_units = train_df['unit_id'].unique()
        num_val = int(len(all_units) * val_ratio)
        val_units = np.random.choice(all_units, num_val, replace=False)
        
        val_df = train_df[train_df['unit_id'].isin(val_units)].copy()
        train_df = train_df[~train_df['unit_id'].isin(val_units)].copy()
        
        feature_cols = [c for c in train_df.columns if c not in ['unit_id', 'cycle', 'rul']]
        
        if auto_drop:
            zero_vars = find_zero_variance_features(train_df, feature_cols)
            if zero_vars:
                print(f"  Dropping zero-variance features: {zero_vars}")
                train_df = train_df.drop(columns=zero_vars)
                val_df = val_df.drop(columns=zero_vars)
                test_df = test_df.drop(columns=zero_vars)
                feature_cols = [c for c in feature_cols if c not in zero_vars]
                
        # Normalization (Min-Max) strictly on train stats
        min_vals = train_df[feature_cols].min()
        max_vals = train_df[feature_cols].max()
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        train_df[feature_cols] = (train_df[feature_cols] - min_vals) / range_vals
        val_df[feature_cols] = (val_df[feature_cols] - min_vals) / range_vals
        test_df[feature_cols] = (test_df[feature_cols] - min_vals) / range_vals
        
        # Windowing
        train_inputs, train_rul, train_masks = create_sliding_windows(train_df, feature_cols, window_size)
        val_inputs, val_rul, val_masks = create_sliding_windows(val_df, feature_cols, window_size)
        test_inputs, test_rul, test_masks = create_sliding_windows(test_df, feature_cols, window_size)
        
        # Save to disk
        train_dir = out_dir / "train"
        val_dir = out_dir / "val"
        test_dir = out_dir / "test"
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        torch.save({'inputs': train_inputs, 'rul': train_rul, 'mask': train_masks}, train_dir / f"{subset}.pt")
        torch.save({'inputs': val_inputs, 'rul': val_rul, 'mask': val_masks}, val_dir / f"{subset}.pt")
        torch.save({'inputs': test_inputs, 'rul': test_rul, 'mask': test_masks}, test_dir / f"{subset}.pt")
        
        print(f"  {subset} Train: {train_inputs.shape}, Val: {val_inputs.shape}, Test: {test_inputs.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/bgsl/config/dataset/cmapss.yaml")
    args = parser.parse_args()
    build_dataset(args.config)
