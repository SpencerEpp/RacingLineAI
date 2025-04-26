#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     A toolset for performing inference on racing line predictions 
#     using the Cerberus model architecture. This script loads processed 
#     track layouts (either from images or pre-extracted coordinate files), 
#     applies preprocessing, and predicts optimized racing lines.
#
# File Overview:
#     - Load a trained Cerberus model and associated scalers.
#     - Process and resample track data from images or CSV files.
#     - Predict the racing line across entire layouts.
#     - Visualize predicted lines against actual track boundaries.
#
# Functions Included:
#     - get_racing_line(): Run full inference pipeline for track prediction.
#====================================================


# === Imports ===
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.serialization
torch.serialization.add_safe_globals([MinMaxScaler])
import matplotlib.pyplot as plt
from CerbModel import Cerberus, load_model 
from LSTMInference import process_track_image
from LSTMDataClass import add_contextual_features, is_circular_track, get_centered_sequence


# === get_racing_line ===
"""
    Perform inference on unseen track layouts using the trained Cerberus model.

    Args:
        data_dir (str): Directory containing track layouts (images or coordinate CSVs).
        data_type (str): Type of input data ("image" or "coords").
        model_path (str): Path to the saved model checkpoint.
        seq_len (int): Length of the input sequence window for the model.

    Returns:
        None
"""
def get_racing_line(data_dir, data_type, model_path, seq_len):
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, scaler_x, scaler_y, scaler_z = load_model(model_path, device)
    model.eval()

    print("Loading unseen layouts from:", data_dir)
    match data_type:
            case "image":
                image_exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
                layout_files = []
                for ext in image_exts:
                    layout_files.extend(glob(os.path.join(data_dir, ext)))
                layout_files = sorted(layout_files)
            case "coords":
                layout_files = sorted(glob(os.path.join(data_dir, "*.csv")))
            case _:
                raise ValueError(f"Unknown data type: {data_type} - use 'image' or 'coords'.")
    total_layouts = len(layout_files)
    print(f"Found {total_layouts} layout files.\n")

    for layout_index, layout_path in enumerate(layout_files):
        layout_name = os.path.basename(layout_path)
        print(f"[{layout_index + 1}/{total_layouts}] Predicting layout: {layout_name}")

        match data_type:
            case "image":
                img = plt.imread(layout_path)
                df = process_track_image(layout_path)
            case "coords":
                df = pd.read_csv(layout_path)
            case _:
                raise ValueError(f"Unknown data type: {data_type} - use 'image' or 'coords'.")
        
        df = add_contextual_features(df)
        is_circular = is_circular_track(df, config["input_cols"])
        X = df[config["input_cols"]].values
        X_scaled = scaler_x.transform(X)
        n = len(X_scaled)
        preds_real = []
            
        for i in tqdm(range(n), desc=f"[{layout_index + 1}/{total_layouts}]"):
            seq = get_centered_sequence(X_scaled, i, seq_len, is_circular)
            X_tensor = torch.tensor(seq.reshape(1, seq_len, -1), dtype=torch.float32).to(config["device"])
            X_tensor = X_tensor.permute(0, 2, 1)
            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().squeeze().numpy()
                pred_real = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
            preds_real.append(pred_real)
        
        preds_real = np.array(preds_real)
        left_x, left_z = X[:, 0], X[:, 2]
        right_x, right_z = X[:, 3], X[:, 5]
        plt.figure(figsize=(12, 6))
        if data_type == "image":
            plt.imshow(img, extent=[left_x.min(), right_x.max(), left_z.max(), left_z.min()])
        plt.plot(left_x, left_z, label="Left Edge", linewidth=1, linestyle=":", alpha=1)
        plt.plot(right_x, right_z, label="Right Edge", linewidth=1, linestyle=":", alpha=1)
        plt.plot(preds_real[:, 0], preds_real[:, 2], label="Predicted Center", linewidth=1, linestyle="--")
        plt.title(f"Predicted vs Track Edges: {layout_name}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Z Coordinate")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()