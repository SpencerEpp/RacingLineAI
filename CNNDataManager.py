#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     A toolset for extracting and parsing track metadata, AI racing lines,
#     and surface geometry from Assetto Corsa track files (.kn5, .ai).
#
# File Overview:
#     This file provides a PyTorch Dataset wrapper for track data, utilities 
#     for loading training datasets, and functions to prepare inference data 
#     from track images. Enables easy data access for model training and evaluation.
#
# Functions Included:
#     - TrackDataset: PyTorch dataset class for processed Assetto Corsa track data.
#     - load_training_dataset(): Split dataset into training/validation loaders.
#     - create_inference_data(): Generate input features from a track image for inference.
#     - create_training_dataset(): Process and save all track layouts for training.
#====================================================


# === Imports ===
import os
import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
from CNNCreateData import load_all_files, process_track_image, cnn_process_all_tracks


# === Dataset Definition ===
"""
    PyTorch Dataset for loading processed Assetto Corsa track data.

    Loads preprocessed centerlines, patches, normalized AI outputs, and 
    real-world AI driving data for supervised model training.

    Args:
        dataset_dir (str): Directory containing processed track data.
        device (torch.device): Device to move data to (currently unused).

    Attributes:
        center (torch.Tensor): Center point coordinates.
        patches (torch.Tensor): 64x64 cropped images along centerline.
        ai_norm (torch.Tensor): Normalized AI behavior outputs.
        ai_df (torch.Tensor): Real-world AI driving targets (x, z, speed, gas, brake).
"""
#TODO: do metaidx for track_normalization values.
class TrackDataset(Dataset):
    def __init__(self, dataset_dir, device):
        self.tracks = load_all_files(dataset_dir)
        
        self.center       = []
        self.patches      = []
        self.ai_norm      = []
        self.ai_df        = []
        self.metadata     = []


        for index, track in tqdm(enumerate(self.tracks), desc="Loading Tracks", total=len(self.tracks)):
            self.center.extend(track["center"])
            patches_np = np.array(track["track_patches"])          
            self.patches.extend(patches_np)
            self.ai_norm.extend(track["ai_norm"])
            self.ai_df.extend(track["ai_df"][["x", "z", "speed", "gas", "brake"]].values)
            self.metadata.extend(np.column_stack([
                track["metadata"]["distance"],
                track["metadata"]["heading_x"],
                track["metadata"]["heading_z"],
                track["metadata"]["curvature"],
                track["metadata"]["track_widths"],
                track["metadata"]["track_avg_width"] * np.ones_like(track["metadata"]["distance"]),
                track["metadata"]["track_min_width"] * np.ones_like(track["metadata"]["distance"]),
                track["metadata"]["track_max_width"] * np.ones_like(track["metadata"]["distance"]),
                track["metadata"]["track_total_length"] * np.ones_like(track["metadata"]["distance"]),
                track["metadata"]["track_avg_curvature"] * np.ones_like(track["metadata"]["distance"]),
                track["metadata"]["track_max_curvature"] * np.ones_like(track["metadata"]["distance"]),
            ]))

        self.center       = t.tensor(np.array(self.center), dtype=t.float32)
        self.patches      = t.tensor(np.array(self.patches), dtype=t.float32)
        self.ai_norm      = t.tensor(np.array(self.ai_norm), dtype=t.float32)
        self.ai_df        = t.tensor(np.array(self.ai_df), dtype=t.float32)
        self.metadata      = t.tensor(np.array(self.metadata), dtype=t.float32)

        # self.center.to(device)       
        # self.patches.to(device)      
        # self.ai_norm.to(device)      
        # self.ai_df.to(device)
    
    def __len__(self):
        return len(self.center)
    
    def __getitem__(self, idx):
        inputs = {
            "center": self.center[idx],
            "patch" : self.patches[idx],
            "metadata"  : self.metadata[idx]
        }

        outputs = {
            "ai_norm"     : self.ai_norm[idx],
            "ai_real"     : self.ai_df[idx, :2],
            "ai_cont"     : self.ai_df[idx, 2:],
        }
        return inputs, outputs


# === Dataset Loader ===
"""
    Load and split the dataset into training and validation DataLoaders.

    Args:
        dataset_dir (str): Directory containing the processed dataset.
        device (torch.device): Device to move data to (currently unused).
        split_ratio (float): Proportion of data for training (e.g., 0.8 for 80% train).
        batch_size (int): Batch size for DataLoader.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader) DataLoader objects.
"""
def load_training_dataset(dataset_dir, device, split_ratio, batch_size, seed=None):
    if seed is None:
        generator = t.Generator().seed()
    else:
        generator = t.Generator().manual_seed(seed)

    dataset = TrackDataset(dataset_dir, device)

    train_len = int(len(dataset) * split_ratio)
    val_len = len(dataset) - train_len

    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, val_loader


# === Inference Data Preparation ===
"""
    Create inference-ready input data from a single track image.

    Args:
        image_path (str): Path to the transparent track map image.

    Returns:
        dict: Dictionary containing:
            - "inner": Inner edge points
            - "outer": Outer edge points
            - "center": Centerline points
            - "track_image": Rendered image array
            - "patches": 64x64 cropped patches along centerline
            - "scale": Scaling factor for normalization
            - "min_xy": XY translation offset
            - "pad": Padding applied to the image
"""
def create_track_data(image_path):
    return process_track_image(image_path, (1024, 1024))


def create_inference_data(image_root):
    data = []
    for image in tqdm(os.listdir(image_root), desc="Creating Inference Data"):
        file = os.path.join(image_root, image)
        data.append(create_track_data(file))
    return data


# === Training Data Preparation ===
"""
    Process and generate a full training dataset from raw Assetto Corsa track layouts.

    Args:
        track_root (str): Root directory containing downloaded Assetto Corsa maps.
        output_root (str): Directory where the processed dataset should be saved.

    Returns:
        None
"""
def create_training_dataset(track_root, output_root):
    cnn_process_all_tracks(track_root, output_root)


