#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     Dataset preprocessing and loading utilities for Racing Line AI.
#     Includes contextual feature engineering, sequence extraction
#     for sequence-to-sequence learning, and a RacingLineDataset 
#     class that handles automatic augmentation and profiling.
#
# File Overview:
#     This file provides functions to compute additional track features,
#     create centered sequences around each point, and wrap preprocessed 
#     racing line data into a PyTorch Dataset for model training.
#
# Functions and Classes Included:
#     - is_circular_track(): Determine if a track is circular based on X/Z distance.
#     - get_centered_sequence(): Extract a centered window around a target point.
#     - add_contextual_features(): Add heading, distance, curvature, and track stats.
#     - RacingLineDataset: PyTorch Dataset that loads, normalizes, augments, and 
#       batches track layout data.
#====================================================


# === Imports ===
import numpy as np
import pandas as pd
import torch as t
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


# === Determine Track Circularity ===
"""
    Determine whether a track is circular based on X/Z distance between start and end points.

    Args:
        df (pandas.DataFrame): Track data containing output columns.
        output_cols (list of str): Columns representing output targets (e.g., x, y, z).
        threshold (float, optional): Distance threshold below which track is considered circular.
        profiler (Profiler, optional): Profiler object for timing.

    Returns:
        bool: True if track is circular, False otherwise.
"""
def is_circular_track(df, output_cols, threshold=5.0, profiler=None):
    if profiler: profiler.start("Is Track Circular")

    start = df[output_cols].iloc[0].values
    end = df[output_cols].iloc[-1].values
    dist = np.linalg.norm(start[[0, 2]] - end[[0, 2]])  # X and Z only

    if profiler: profiler.stop("Is Track Circular")
    return dist <= threshold


# === Extract Centered Sequences ===
"""
    Extract a centered sequence of points around a given index.

    Handles both circular (wrap-around) and non-circular tracks,
    with extrapolated padding for non-circular ends.

    Args:
        X (np.ndarray): Input array of track features.
        center_idx (int): Index to center the sequence around.
        seq_len (int): Desired sequence length.
        circular (bool): Whether the track is circular.
        profiler (Profiler, optional): Profiler object for timing.

    Returns:
        np.ndarray: Extracted sequence of shape (seq_len, feature_dim).
"""
def get_centered_sequence(X, center_idx, seq_len, circular, profiler=None):
    if profiler: profiler.start("Get Centered Sequences")

    half = seq_len // 2
    n = len(X)

    if circular:
        indices = [(center_idx - half + i) % n for i in range(seq_len)]
        seq = X[indices]
    else:
        left = center_idx - half
        right = center_idx + half + 1
        if left < 0:
            delta = X[1] - X[0]
            pre = [X[0] - delta * (i + 1) for i in reversed(range(-left))]
            seq = np.vstack(pre + [*X[0:right]])
        elif right > n:
            delta = X[-1] - X[-2]
            post = [X[-1] + delta * (i + 1) for i in range(right - n)]
            seq = np.vstack([*X[left:n]] + post)
        else:
            seq = X[left:right]

    if profiler: profiler.stop("Get Centered Sequences")
    return seq


# === Add Contextual Features ===
"""
    Compute and append contextual features to track data.

    Adds heading vectors, cumulative distance, curvature, 
    and global track statistics (widths, length, curvature).

    Args:
        df (pandas.DataFrame): Track data containing left and right edge coordinates.
        profiler (Profiler, optional): Profiler object for timing.

    Returns:
        pandas.DataFrame: Track data with additional feature columns.
"""
def add_contextual_features(df, profiler=None):
    if profiler: profiler.start("Add Contextual Features")

    coords = df[["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]].values
    left = coords[:, :3]
    right = coords[:, 3:]

    center = (left + right) / 2
    heading = np.diff(center, axis=0, prepend=center[0:1])
    heading = heading / (np.linalg.norm(heading, axis=1, keepdims=True) + 1e-8)

    distances = np.linalg.norm(np.diff(center, axis=0, prepend=center[0:1]), axis=1)
    cumulative_distance = np.cumsum(distances)

    dd = np.diff(heading, axis=0, prepend=heading[0:1])
    curvature = np.linalg.norm(dd, axis=1)

    avg_width = np.mean(np.linalg.norm(left - right, axis=1))
    max_width = np.max(np.linalg.norm(left - right, axis=1))
    min_width = np.min(np.linalg.norm(left - right, axis=1))
    total_length = cumulative_distance[-1]
    avg_curvature = np.mean(curvature)
    max_curvature = np.max(curvature)

    df["distance"] = cumulative_distance
    df["heading_x"], df["heading_y"], df["heading_z"] = heading.T
    df["curvature"] = curvature
    df["track_avg_width"] = avg_width
    df["track_min_width"] = min_width
    df["track_max_width"] = max_width
    df["track_total_length"] = total_length
    df["track_avg_curvature"] = avg_curvature
    df["track_max_curvature"] = max_curvature

    if profiler: profiler.stop("Add Contextual Features")
    return df


# === Racing Line Dataset Class ===
"""
    PyTorch Dataset for Racing Line training.

    Loads, normalizes, augments, and batches track layout sequences.
    Supports optional online profiling during dataset loading and sampling.

    Args:
        config (dict): Training configuration containing input/output feature names and sequence length.
        file_list (list of str): List of CSV file paths containing track layouts.
        enable_augmentation (bool, optional): Whether to apply flip/mirror augmentations.
        profiler (Profiler, optional): Profiler object for timing.

    Attributes:
        scaler_x (MinMaxScaler): Global input feature scaler.
        scaler_y (MinMaxScaler): Global output target scaler.
        inputs (torch.Tensor): All input sequences.
        targets (torch.Tensor): All corresponding output targets.

    Methods:
        __len__(): 
            Return the total number of samples in the dataset.
        
        __getitem__(idx): 
            Return a single sample (input sequence tensor, target tensor) given an index.
"""
class RacingLineDataset(Dataset):
    def __init__(self, config, file_list, enable_augmentation=True, profiler=None):
        self.config = config
        self.file_list = file_list
        self.enable_augmentation = enable_augmentation
        self.profiler = profiler
        self.seq_len = config["seq_len"]
        self.input_cols = config["input_cols"]
        self.output_cols = config["output_cols"]

        if self.profiler: self.profiler.start("RacingLineDataset Init")

        # === Fit scalers globally ===
        all_X, all_Y = [], []
        for path in self.file_list:
            df = add_contextual_features(pd.read_csv(path), profiler=self.profiler)
            all_X.append(df[self.input_cols].values)
            all_Y.append(df[self.output_cols].values)
            del df
        self.scaler_x = MinMaxScaler().fit(np.vstack(all_X))
        self.scaler_y = MinMaxScaler().fit(np.vstack(all_Y))
        del all_X, all_Y

        # === Count total sequences ===
        n_total = 0
        sequence_counts = []  # (path, count, is_circular)
        for path in self.file_list:
            df = add_contextual_features(pd.read_csv(path), profiler=self.profiler)
            is_circular = is_circular_track(df, self.output_cols, profiler=self.profiler)
            aug = len(df) * 2 if enable_augmentation else 0
            total = len(df) + aug
            sequence_counts.append((path, total, is_circular))
            n_total += total
            del df

        # === Preallocate memory ===
        self.inputs = np.empty((n_total, self.seq_len, len(self.input_cols)), dtype=np.float32)
        self.targets = np.empty((n_total, len(self.output_cols)), dtype=np.float32)

        # === Fill data ===
        index = 0
        for path, total, is_circular in sequence_counts:
            df = add_contextual_features(pd.read_csv(path), profiler=self.profiler)
            X = self.scaler_x.transform(df[self.input_cols].values)
            Y = self.scaler_y.transform(df[self.output_cols].values)

            for i in range(len(X)):
                seq = get_centered_sequence(X, i, self.seq_len, is_circular, profiler=self.profiler)
                target = Y[i]
                self.inputs[index] = seq
                self.targets[index] = target
                index += 1

                if enable_augmentation:
                    flipped = np.flip(seq, axis=0).copy()
                    self.inputs[index] = flipped
                    self.targets[index] = target
                    index += 1

                    mirrored = seq.copy()
                    mirrored[:, [0, 3]] *= -1
                    mirrored[:, [2, 5]] *= -1
                    mirrored_target = target.copy()
                    mirrored_target[[0, 2]] *= -1
                    self.inputs[index] = mirrored
                    self.targets[index] = mirrored_target
                    index += 1

            del df, X, Y

        # === Convert to tensors ===
        self.inputs = t.from_numpy(self.inputs)
        self.targets = t.from_numpy(self.targets)

        if self.profiler: self.profiler.stop("RacingLineDataset Init")

    def __len__(self):
        if self.profiler: self.profiler.start("Dataset __len__")
        length = len(self.inputs)
        if self.profiler: self.profiler.stop("Dataset __len__")
        return length

    def __getitem__(self, idx):
        if self.profiler: self.profiler.start("Dataset __getitem__")
        result = (self.inputs[idx], self.targets[idx])
        if self.profiler: self.profiler.stop("Dataset __getitem__")
        return result
    
# # === Data with Augmentations on the fly (Use if <16Gb RAM and/or < 8Gb of VRAM) ===
# class RacingLineDataset(Dataset):
#     def __init__(self, config, file_list, enable_augmentation=True, profiler=None):
#         self.config = config
#         self.file_list = file_list
#         self.enable_augmentation = enable_augmentation
#         self.index_map = []
#         self.data = []

#         if self.profiler: self.profiler.start("RacingLineDataset Init")

#         # === Fit scalers globally ===
#         self.scaler_x = MinMaxScaler()
#         self.scaler_y = MinMaxScaler()
#         all_X, all_Y = [], []

#         for path in file_list:
#             df = add_contextual_features(pd.read_csv(path))
#             all_X.append(df[config["input_cols"]].values)
#             all_Y.append(df[config["output_cols"]].values)
#         self.scaler_x.fit(np.vstack(all_X))
#         self.scaler_y.fit(np.vstack(all_Y))

#         # === Preload layout data ===
#         for idx, path in enumerate(file_list):
#             df = add_contextual_features(pd.read_csv(path))
#             is_circular = is_circular_track(df, config["output_cols"])
#             X = self.scaler_x.transform(df[config["input_cols"]].values)
#             Y = self.scaler_y.transform(df[config["output_cols"]].values)
#             self.data.append((X, Y, is_circular))

#             for i in range(len(X)):
#                 self.index_map.append((idx, i, 'orig'))
#                 if self.enable_augmentation:
#                     self.index_map.append((idx, i, 'flip'))
#                     self.index_map.append((idx, i, 'mirror'))

#         if self.profiler: self.profiler.stop("RacingLineDataset Init")

#     def __len__(self):
#         if self.profiler: self.profiler.start("Dataset __len__")
#         length =  len(self.index_map)
#         if self.profiler: self.profiler.stop("Dataset __len__")
#         return length

#     def __getitem__(self, idx):
#         if self.profiler: self.profiler.start("Dataset __getitem__")
#         file_idx, i, aug_type = self.index_map[idx]
#         X, Y, is_circular = self.data[file_idx]
#         seq = get_centered_sequence(X, i, self.config["seq_len"], is_circular)
#         target = Y[i]

#         if aug_type == "flip":
#             seq = np.flip(seq, axis=0).copy()
#         elif aug_type == "mirror":
#             seq = seq.copy()
#             seq[:, [0, 3]] *= -1
#             seq[:, [2, 5]] *= -1
#             target = target.copy()
#             target[[0, 2]] *= -1

#         if self.profiler: self.profiler.stop("Dataset __getitem__")
#         return t.tensor(seq, dtype=t.float32), t.tensor(target, dtype=t.float32)


# #=== Data with Augmentations (better have a >16Gb of VRAM) ===
# class RacingLineDataset(Dataset):
#     def __init__(self, config, file_list, enable_augmentation=True, profiler=None):
#         self.config = config
#         self.enable_augmentation = enable_augmentation
#         self.scaler_x = MinMaxScaler()
#         self.scaler_y = MinMaxScaler()
#         inputs, targets = [], []

#         if self.profiler: self.profiler.start("RacingLineDataset Init")

#         # === Fit global scalers first ===
#         all_X, all_Y = [], []
#         count = 0
#         for path in file_list:
#             count +=1
#             print(count)
#             df = add_contextual_features(pd.read_csv(path))
#             all_X.append(df[config["input_cols"]].values)
#             all_Y.append(df[config["output_cols"]].values)
#         self.scaler_x.fit(np.vstack(all_X))
#         self.scaler_y.fit(np.vstack(all_Y))
#         del all_X, all_Y

#         # === Process each layout individually ===
#         count, items = 0, 0
#         for path in file_list:
#             count +=1
#             print(f"Loop iters: {count} | Items in inputs {items}")
#             df = add_contextual_features(pd.read_csv(path))
#             is_circular = is_circular_track(df, config["output_cols"])
#             X = self.scaler_x.transform(df[config["input_cols"]].values)
#             Y = self.scaler_y.transform(df[config["output_cols"]].values)

#             for i in range(len(X)):
#                 items +=1
#                 seq = get_centered_sequence(X, i, config["seq_len"], is_circular)
#                 inputs.append(seq)
#                 targets.append(Y[i])

#                 if self.enable_augmentation:
#                     flipped = np.flip(seq, axis=0).copy()
#                     inputs.append(flipped)
#                     targets.append(Y[i])
#                     mirrored = seq.copy()
#                     mirrored[:, [0, 3]] *= -1
#                     mirrored[:, [2, 5]] *= -1
#                     mirrored_target = Y[i].copy()
#                     mirrored_target[[0, 2]] *= -1
#                     inputs.append(mirrored)
#                     targets.append(mirrored_target)

#             del df, X, Y

#         # === Finalize dataset ===
#         self.inputs = t.tensor(np.array(inputs), dtype=t.float32).to(config["device"])
#         self.targets = t.tensor(np.array(targets), dtype=t.float32).to(config["device"])
#         del inputs, targets

#         if self.profiler: self.profiler.stop("RacingLineDataset Init")

#     def __len__(self):
#         if self.profiler: self.profiler.start("Dataset __len__")
#         length = len(self.inputs)
#         if self.profiler: self.profiler.stop("Dataset __len__")
#         return length

#     def __getitem__(self, idx):
#         if self.profiler: self.profiler.start("Dataset __getitem__")
#         result = (self.inputs[idx], self.targets[idx])
#         if self.profiler: self.profiler.stop("Dataset __getitem__")
#         return result


# # Test code for sequencing data into sliding windows
# from sklearn.preprocessing import MinMaxScaler
# df = pd.read_csv("./data/testing_layouts/ks_barcelona_layout_gp_Processed_Data.csv")
# scaler_x = MinMaxScaler()
# scaler_y = MinMaxScaler()
# scaler_x.fit(df[config["input_cols"]])
# scaler_y.fit(df[config["output_cols"]])
# config["scaler_x"] = scaler_x
# config["scaler_y"] = scaler_y
# visualize_all_center_points(df, config)

# def visualize_all_center_points(df, config):
#     X = df[config["input_cols"]].values
#     Y = df[config["output_cols"]].values
#     X_scaled = config["scaler_x"].transform(X)
#     is_circular = is_circular_track(df, config["output_cols"])

#     seq_len = config["seq_len"]
#     centers_x, centers_z = [], []

#     for i in range(len(X_scaled)):
#         seq = get_centered_sequence(X_scaled, i, seq_len, is_circular)
#         left_x, left_z = seq[:, 0], seq[:, 2]
#         right_x, right_z = seq[:, 3], seq[:, 5]
#         center_x = (left_x + right_x) / 2
#         center_z = (left_z + right_z) / 2
#         mid = seq_len // 2
#         centers_x.append(center_x[mid])
#         centers_z.append(center_z[mid])

#     plt.figure(figsize=(10, 5))
#     plt.plot(centers_x, centers_z, label="Input Sequence Centers", linewidth=2)
#     plt.title("Sliding Window Center Points (Before Model)")
#     plt.xlabel("X")
#     plt.ylabel("Z")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.legend()
#     plt.show()