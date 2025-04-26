#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     A toolset for loading a trained RoboTurbosky CNN model 
#     and running inference on track images to predict 
#     optimal racing lines and control behavior (speed, gas, brake).
#
# File Overview:
#     This file provides functionality for running inference directly 
#     on raw track layouts from transparent map images. It handles 
#     preprocessing the images into model input patches, loading trained 
#     models, making predictions for centerline and control outputs, 
#     and plotting the predicted racing lines along with speed, gas, 
#     and brake levels.
#
# Functions Included:
#     - plot_colored_racing_line(): Visualizes predictions over the track map.
#     - get_racing_line(): Runs batch inference over a directory of track images.
#====================================================


# === Imports ===
import os
import torch as t
import numpy as np
from glob import glob
from CNNModel import load_model
from CNNDataManager import create_inference_data
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# === Utility Function: Plot Colored Racing Line ===
"""
    Plots the predicted centerline of the track colored 
    by a specified control variable (speed, gas, or brake).

    Args:
        track_image (ndarray): Background image of the track.
        segments (ndarray): Line segments connecting prediction points.
        color_values (ndarray): Values (e.g., speed, gas) used to color segments.
        cmap (str): Matplotlib colormap.
        label (str): Label for the colorbar.
        title (str): Title of the plot.
"""
def plot_colored_racing_line(track_image, segments, color_values, cmap, label, title):
    if color_values.max() == color_values.min():
        norm = plt.Normalize(vmin=0, vmax=1)
    else:
        norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(color_values[:-1])
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.imshow(track_image, cmap="gray")
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label=label)
    plt.title(title)
    plt.axis('off')
    plt.show()


# === CNN Inference Function ===
"""
    Runs inference on a directory of track images using a trained 
    RoboTurbosky CNN model. Predicts centerline positions and 
    control outputs (speed, gas, brake) for each layout, and visualizes them.

    Args:
        images_dir (str): Directory containing input track images.
        model_path (str): Path to the saved trained model file (.pt).
"""
def get_racing_line(images_dir, model_path):
    print("Loading model...")
    device = "cuda" if t.cuda.is_available() else "cpu"
    model, config = load_model(model_path, device)
    model.eval()

    print("Loading unseen layouts from:", images_dir)
    track_data = create_inference_data(images_dir)
    print(f"Found {len(track_data)} layout files.\n")

    for idx, track in tqdm(enumerate(track_data), desc="Predicting Layouts"):
        
        tqdm.write(f"[{idx}/{len(track_data)}] Predicting layout: {idx}")
        metadata = []
        metadata.extend(np.column_stack([
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

        patches = t.tensor(np.array(track["patches"]), dtype=t.float32).unsqueeze(1).to(device)
        center = t.tensor(np.array(track["center"]), dtype=t.float32).to(device)
        metadata = t.tensor(np.array(metadata), dtype=t.float32).to(device)

        with t.inference_mode():
            control_pred, position_pred = model(patches, center, metadata)

        control_pred = control_pred.cpu().numpy()
        position_pred = position_pred.cpu().numpy()

        speed_val = control_pred[:, 0]
        gas_val   = control_pred[:, 1]
        brake_val = control_pred[:, 2]

        points = position_pred.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        plot_colored_racing_line(track["track_image"], segments, speed_val, cmap="viridis",
                                label="Speed", title=f"Speed Overlay: {idx}")
        plot_colored_racing_line(track["track_image"], segments, gas_val, cmap="Greens",
                                label="Gas", title=f"Gas Pedal Overlay: {idx}")
        plot_colored_racing_line(track["track_image"], segments, brake_val, cmap="Reds",
                                label="Brake", title=f"Brake Pedal Overlay: {idx}")