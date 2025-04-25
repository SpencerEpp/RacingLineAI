'''
Simple Wrapper for easily allowing data access

'''

import os
import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm.auto as tqdm
from create_data2 import load_all_files, process_track_image_two_electric_boogaloo as process_track, process_all_tracks

''' TrackDataset
PyTorch version of 

'''
#TODO: do metaidx for track_normalization values.
class TrackDataset(Dataset):
    def __init__(self, dataset_dir, device):
        self.tracks = load_all_files(dataset_dir)
        
        self.center       = []
        self.patches      = []
        self.ai_norm      = []
        self.ai_df        = []
       # self.metaidx      = []


        for index, track in tqdm(enumerate(self.tracks), desc="Loading Tracks"):
            self.center.extend(track["center"])
            patches_np = np.array(track["track_patches"])          
            self.patches.extend(patches_np)
            self.ai_norm.extend(track["ai_norm"])
            self.ai_df.extend(track["ai_df"][["x", "z", "speed", "gas", "brake"]].values)
            #self.metaidx.extend([index] * len(track['center'])) 

        self.center       = t.tensor(np.array(self.center), dtype=t.float32)
        self.patches      = t.tensor(np.array(self.patches), dtype=t.float32)
        self.ai_norm      = t.tensor(np.array(self.ai_norm), dtype=t.float32)
        self.ai_df        = t.tensor(np.array(self.ai_df), dtype=t.float32)
       # self.metaidx      = t.tensor(self.metaidx, dtype=t.float32)

        # self.center.to(device)       
        # self.patches.to(device)      
        # self.ai_norm.to(device)      
        # self.ai_df.to(device)
    
    def __len__(self):
        return len(self.center)
    
    def __getitem__(self, idx):
        inputs = {
            "center": self.center[idx],
            "patch" : self.patches[idx]
        }

        outputs = {
            "ai_norm"     : self.ai_norm[idx],
            "ai_real"     : self.ai_df[idx, :2],
            "ai_cont"     : self.ai_df[idx, 2:],
        }
        return inputs, outputs

''' load_training_dataset
 @dataset_dir - Directory containing processed Assetto Corsa Tracks
 @device      -
 @split_ratio -
 @batch_size  -
 @seed        -
'''
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

''' Create Inference Data
Given an image path, returns a dictionary with the following properties: 
    "inner"      : inner edge of the track 
    "outer"      : outer edge of the track
    "center"     : generated center line for the track
    "track_image": Converted rendered track image
    "patches"    : 64x64 patches generated centered on every center point
    "scale"      : scale used to normalize the track image
    "min_xy"     : xy offset used to normalize the track
    "pad"        : extra image padding to ensure the image is centered

    Used for input data for inference.
'''
def create_inference_data(image_path):
    return process_track(image_path, (1024,1024))


''' Create Training Dataset
@track root  - folder containing downloaded assetto corsa maps
@output root - dir to output generated data to

Simple helper function that allows one to easily create a dataset.
'''
def create_training_dataset(track_root, output_root):
    process_all_tracks(track_root, output_root)
