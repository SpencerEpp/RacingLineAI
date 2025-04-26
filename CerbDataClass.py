#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     Defines the RacingLineDataset class used for training the Cerberus model.
#     Handles loading, scaling, and sequence generation for both positional
#     and control prediction targets. Supports global normalization across 
#     all training and testing layouts for improved generalization.
#
# File Overview:
#     - Load track layouts and fit global input/output scalers.
#     - Extract centered sequences for position and control targets.
#     - Prepare tensors for PyTorch DataLoader compatibility.
#
# Classes and Functions Included:
#     - RacingLineDataset: Dataset for generating (input sequence, position target, control target) triples.
#====================================================


# === Imports ===
import os
import numpy as np
import pandas as pd
import torch as t
from glob import glob
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from LSTMDataClass import add_contextual_features, is_circular_track, get_centered_sequence


# === RacingLineDataset ===
"""
    Dataset class for racing line prediction with both positional and control outputs.

    This dataset:
        - Loads training and testing layout files.
        - Fits global MinMaxScalers for inputs, position outputs, and control outputs.
        - Extracts centered sequences of a specified length around each frame.
        - Returns input sequences along with corresponding position and control targets.

    Args:
        config (dict): Configuration dictionary containing feature names, device, and paths.
        seq_len (int): Sequence length for centered input extraction.

    Attributes:
        inputs (torch.Tensor): Input sequences for the model (shape: [n_samples, seq_len, n_features]).
        pos_targets (torch.Tensor): Positional targets (shape: [n_samples, 3]).
        cont_targets (torch.Tensor): Control targets (shape: [n_samples, control_dim]).
        scaler_x (MinMaxScaler): Fitted scaler for input features.
        scaler_y (MinMaxScaler): Fitted scaler for positional outputs.
        scaler_z (MinMaxScaler): Fitted scaler for control outputs.

    Methods:
        __len__():
            Return the total number of samples in the dataset.

        __getitem__(idx):
            Return a single sample:
                - Input sequence (features x seq_len),
                - Position target,
                - Control target.
"""
class RacingLineDataset(Dataset):
    def __init__(self, config, seq_len):
        self.inputs, self.pos_targets, self.cont_targets = [], [], []
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaler_z = MinMaxScaler()
        all_X, all_Y, all_Z = [], [], []
        train_files = sorted(glob(os.path.join(config["train_data_path"], "*.csv")))
        test_files = sorted(glob(os.path.join(config["test_data_path"], "*.csv")))

        # === First pass: Collect data for global fitting ===
        for file in train_files:
            df = pd.read_csv(file)
            X = df[config["input_cols"]].values
            Y = df[config["output_pos_cols"]].values
            Z = df[config["output_cont_cols"]].values
            all_X.append(X)
            all_Y.append(Y)
            all_Z.append(Z)
        for file in test_files:
            df = pd.read_csv(file)
            X = df[config["input_cols"]].values
            Y = df[config["output_pos_cols"]].values
            Z = df[config["output_cont_cols"]].values
            all_X.append(X)
            all_Y.append(Y)
            all_Z.append(Z)
        all_X = np.vstack(all_X)
        all_Y = np.vstack(all_Y)
        all_Z = np.vstack(all_Z)
        self.scaler_x.fit(all_X)
        self.scaler_y.fit(all_Y)
        self.scaler_z.fit(all_Z)

        # === Second pass: Normalize and extract sequences ===
        for file in train_files:
            df = pd.read_csv(file)
            is_circular = is_circular_track(df, config["output_pos_cols"])
            X = self.scaler_x.transform(df[config["input_cols"]].values)
            Y = self.scaler_y.transform(df[config["output_pos_cols"]].values)
            Z = self.scaler_z.transform(df[config["output_cont_cols"]].values)

            for i in range(len(X)):
                self.inputs.append(get_centered_sequence(X, i, seq_len, is_circular))
                self.pos_targets.append(Y[i])
                self.cont_targets.append(Z[i])

        self.inputs = t.tensor(np.array(self.inputs), dtype=t.float32).to(config["device"])
        self.pos_targets = t.tensor(np.array(self.pos_targets), dtype=t.float32).to(config["device"])
        self.cont_targets = t.tensor(np.array(self.cont_targets), dtype=t.float32).to(config["device"])
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx].permute(1,0), self.pos_targets[idx], self.cont_targets[idx]