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
#     This file provides functionality for running inference on processed
#     or raw track layouts, either from coordinate CSV files or from 
#     transparent map images. It handles preprocessing, loading models, 
#     making predictions, and plotting predicted racing lines.
#
# Functions Included:
#     - process_track_image(): Parse and resample transparent track images into left/right edges.
#                              Unique to inference, doesnt scale or transform edge data.
#     - print_feature_accuracy(): Print per-feature accuracy metrics comparing predictions to true values.
#     - get_racing_line(): Main entry point for inference on datasets using a trained LSTM model.
#====================================================


# === Imports ===
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch.serialization
torch.serialization.add_safe_globals([MinMaxScaler])
import matplotlib.pyplot as plt
from LSTMModel import RacingLineLSTMWithAttention, save_model, load_model 
from LSTMDataClass import RacingLineDataset, add_contextual_features, is_circular_track, get_centered_sequence
from LSTMCreateData import resample_edge_savitzky, same_direction, find_best_roll_mean


# === Process Transparent Track Image ===
"""
    Process a transparent map image to extract resampled left and right track edges.

    Args:
        image_path (str): Path to the transparent track image file.

    Returns:
        pandas.DataFrame: DataFrame with columns ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]
        containing resampled inner and outer track edges.
"""
def process_track_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    alpha_channel = image[:, :, 3]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    binary[alpha_channel < 255] = 0 # Transparent Image 

    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(sorted_contours) < 2:
        raise ValueError(f"Not enough contours in image: {image_path}")
    outer = sorted_contours[0].squeeze()
    inner = sorted_contours[1].squeeze()

    distances = np.sqrt(np.sum(np.diff(outer, axis=0) ** 2, axis=1))
    total_length = np.sum(distances)//2
    num_points = max(400, int(total_length))
    outer_resampled = resample_edge_savitzky(outer, num_points)
    inner_resampled = resample_edge_savitzky(inner, num_points)

    if not same_direction(outer_resampled[:20], inner_resampled[:20]):
        inner_resampled = np.flipud(inner_resampled)
    shift = find_best_roll_mean(outer_resampled, inner_resampled, sample_size=20)
    inner_resampled = np.roll(inner_resampled, shift, axis=0)

    return pd.DataFrame({
        "left_x": outer_resampled[:, 0],
        "left_y": 0,
        "left_z": outer_resampled[:, 1],
        "right_x": inner_resampled[:, 0],
        "right_y": 0,
        "right_z": inner_resampled[:, 1]
    })


# === Print Feature Accuracy ===
"""
    Print per-feature prediction accuracy compared to true values.

    Args:
        preds (np.ndarray): Predicted feature values.
        trues (np.ndarray): True feature values.
        scaler_y (MinMaxScaler): Output scaler used during model training.
        feature_names (list): Names of the predicted output features.

    Returns:
        None
"""
def print_feature_accuracy(preds, trues, scaler_y, feature_names):
    preds = np.array(preds)
    trues = np.array(trues)

    print(f"\nPer-Feature Accuracy (%):")
    print("-" * 60)
    for i, name in enumerate(feature_names):
        range_train = scaler_y.scale_[i]
        range_test = trues[:, i].max() - trues[:, i].min()

        if range_test == 0:
            print(f"{name:>16}: N/A (zero test range)")
            continue

        mean_error = np.mean(np.abs(preds[:, i] - trues[:, i]))
        acc_train = (1 - (mean_error / range_train)) * 100
        acc_test = (1 - (mean_error / range_test)) * 100

        acc_train = max(0.0, min(100.0, acc_train))
        acc_test = max(0.0, min(100.0, acc_test))

        print(f"{name:>16}: {acc_test:6.2f}% (layout-based)   {acc_train:6.2f}% (train-scale)")


# === Get Racing Line Prediction ===
"""
    Run inference on unseen track layouts to predict optimal racing lines.

    Args:
        data_dir (str): Path to the folder containing testing layouts (images or coordinate CSVs).
        data_type (str): Type of data to process ("image" or "coords").
        model_path (str): Path to the saved LSTM model checkpoint.

    Returns:
        None
"""
def get_racing_line(data_dir, data_type, model_path):
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, scaler_x, scaler_y = load_model(model_path, device)
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
            seq = get_centered_sequence(X_scaled, i, config["seq_len"], is_circular)
            X_tensor = torch.tensor(seq.reshape(1, config["seq_len"], -1), dtype=torch.float32).to(config["device"])
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