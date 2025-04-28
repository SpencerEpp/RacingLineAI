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
from CNNCreateData import load_track_dataset
from CNNDataManager import create_inference_data
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
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


# === CNN Inference Functions ===
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
        

"""
    Plots the track image with both predicted and ground truth racing lines.

    If `pred_feature` and `gt_feature` are provided, the predicted line is colorized based on `pred_feature`
    values using the specified colormap, and a colorbar is added. The ground truth line is drawn in blue
    with a dashed style for comparison.

    If no features are provided, the predicted line is drawn in solid lime and the ground truth line in dashed blue.

    Args:
        track_image (ndarray): Grayscale track image to use as the background.
        segments_pred (ndarray): Predicted racing line segments (shape: [N, 2, 2]).
        segments_gt (ndarray): Ground truth racing line segments (shape: [N, 2, 2]).
        title (str): Title of the plot.
        pred_feature (ndarray or None): Values to colorize the predicted line (e.g., speed, throttle).
        gt_feature (ndarray or None): Values associated with the ground truth line (used for normalization only).
        feature_name (str or None): Label for the colorbar (if features are provided).
        cmap (str): Colormap to use for the predicted line when features are provided.
        linesize (float): Thickness of the lines.
"""
def plot_racing_lines(track_image, segments_pred, segments_gt, title, pred_feature=None, gt_feature=None, feature_name=None, cmap="plasma", linesize=1.5):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(track_image, cmap="gray")

    if pred_feature is not None and gt_feature is not None:
        if pred_feature.max() == pred_feature.min():
            norm = plt.Normalize(vmin=0, vmax=1)
        else:
            vmin = min(pred_feature.min(), gt_feature.min())
            vmax = max(pred_feature.max(), gt_feature.max())
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

        pred_lc = LineCollection(segments_pred, cmap=cmap, norm=norm, linewidth=linesize)
        pred_lc.set_array(pred_feature[:-1])
        ax.add_collection(pred_lc)

        gt_lc = LineCollection(segments_gt, cmap="Blues", norm=norm, linewidth=linesize, linestyle=(0, (10, 10)))
        gt_lc.set_array(gt_feature[:-1])
        ax.add_collection(gt_lc)

        cbar = plt.colorbar(pred_lc, ax=ax)
        if feature_name:
            cbar.set_label(feature_name)
        
    else:
        pred_lc = LineCollection(segments_pred, colors="lime", linewidth=linesize, linestyle="solid")
        gt_lc = LineCollection(segments_gt, colors="blue", linewidth=linesize, linestyle=(0, (10, 10)))
        ax.add_collection(pred_lc)
        ax.add_collection(gt_lc)

        legend_elements = [
            Line2D([0], [0], color="lime", lw=2, label="Predicted Line"),
            Line2D([0], [0], color="blue", lw=2, linestyle=(0, (10, 10)), label="Ground Truth")
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    plt.title(title)
    plt.axis('off')
    plt.show()


"""
    Loads full track dataset files (.h5) using load_track_dataset(), runs inference with the trained CNN model,
    and plots the track image, predicted racing line, and ground truth ideal line.
    
    Args:
        dataset_dir (str): Directory containing .h5 dataset files.
        model_path (str): Path to the trained model checkpoint.
"""
def get_racing_line_from_dataset(dataset_dir, model_path):
    print("Loading model...")
    device = "cuda" if t.cuda.is_available() else "cpu"
    model, config = load_model(model_path, device)
    model.eval()

    dataset_files = glob(os.path.join(dataset_dir, "*.h5py"))
    print(f"Found {len(dataset_files)} dataset files.\n")

    for idx, file_path in tqdm(enumerate(dataset_files), desc="Predicting Datasets"):

        tqdm.write(f"[{idx}/{len(dataset_files)}] Predicting on: {os.path.basename(file_path)}")

        track = load_track_dataset(file_path)
        track_image = track["track_image"]
        ai_norm = track["ai_norm"]
        ai_df = track["ai_df"]
        track_patches = track["track_patches"]
        center = track["center"]
        metadata = track["metadata"]
        metadata_arr = np.column_stack([
            metadata["distance"],
            metadata["heading_x"],
            metadata["heading_z"],
            metadata["curvature"],
            metadata["track_widths"],
            metadata["track_avg_width"] * np.ones_like(metadata["distance"]),
            metadata["track_min_width"] * np.ones_like(metadata["distance"]),
            metadata["track_max_width"] * np.ones_like(metadata["distance"]),
            metadata["track_total_length"] * np.ones_like(metadata["distance"]),
            metadata["track_avg_curvature"] * np.ones_like(metadata["distance"]),
            metadata["track_max_curvature"] * np.ones_like(metadata["distance"]),
        ])

        patches = t.tensor(np.array(track_patches), dtype=t.float32).unsqueeze(1).to(device)
        center = t.tensor(np.array(center), dtype=t.float32).to(device)
        metadata_tensor = t.tensor(np.array(metadata_arr), dtype=t.float32).to(device)

        with t.inference_mode():
            control_pred, position_pred = model(patches, center, metadata_tensor)

        control_pred = control_pred.cpu().numpy()
        position_pred = position_pred.cpu().numpy()

        speed_val = control_pred[:, 0]
        gas_val   = control_pred[:, 1]
        brake_val = control_pred[:, 2]

        points_pred = position_pred.reshape(-1, 1, 2)
        segments_pred = np.concatenate([points_pred[:-1], points_pred[1:]], axis=1)

        points_gt = ai_norm.reshape(-1, 1, 2)
        segments_gt = np.concatenate([points_gt[:-1], points_gt[1:]], axis=1)

        # plot_racing_lines(track_image=track_image, segments_pred=segments_pred,
        #                   segments_gt=segments_gt, title=f"Ideal vs Predicted Line: {idx}")
        # plot_racing_lines(track_image=track_image, segments_pred=segments_pred,
        #                   segments_gt=segments_gt, title=f"Speed Overlay: {idx}",
        #                   color_values=speed_val, cmap="viridis", label="Speed")
        # plot_racing_lines(track_image=track_image, segments_pred=segments_pred,
        #                   segments_gt=segments_gt, title=f"Gas Pedal Overlay: {idx}",
        #                   color_values=gas_val, cmap="Greens", label="Gas Pedal")
        # plot_racing_lines(track_image=track_image,segments_pred=segments_pred,
        #                   segments_gt=segments_gt, title=f"Brake Pedal Overlay: {idx}",
        #                   color_values=brake_val, cmap="Reds", label="Brake Pedal")

        plot_racing_lines(track_image, segments_pred, segments_gt, title="Predicted Line vs Ground Truth Line")
        plot_racing_lines(track_image, segments_pred, segments_gt, title="Speed Overlay", pred_feature=speed_val,
                          gt_feature=ai_df["speed"], feature_name="Speed (km/h)", cmap="viridis")
        plot_racing_lines(track_image, segments_pred, segments_gt, title="Throttle Overlay", pred_feature=gas_val,
                          gt_feature=ai_df["gas"], feature_name="Throttle (%)", cmap="plasma")
        plot_racing_lines(track_image, segments_pred, segments_gt, title="Brake Overlay", pred_feature=brake_val,
                          gt_feature=ai_df["brake"], feature_name="Brake (%)", cmap="inferno")