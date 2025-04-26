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
#     This file provides functions to preprocess track images by detecting 
#     left and right edges, aligning them to ideal racing lines, resampling 
#     them smoothly, and adding precise elevation data from track .kn5 files.
#     Supports batch processing for multiple tracks and layouts.
#
# Functions Included:
#     - same_direction(): Check if two sequences of points move in the same direction.
#     - scale_and_translate_edges(): Scale and translate edges to match centerline bounding box.
#     - find_best_roll_sum(): Find the best circular shift to align two curves (sum distance).
#     - find_best_roll_mean(): Find the best circular shift to align two curves (mean distance).
#     - resample_edge_savitzky(): Resample and smooth contours with arc-length interpolation.
#     - process_track_image(): Extract and align left/right track edges from map images.
#     - add_precise_elevation(): Add accurate Y-coordinates from kn5 vertices.
#     - process_track(): Full processing pipeline for a single track and its layouts.
#     - lstm_process_all_tracks(): Process all tracks inside a directory.
#
# Known Bugs:
#     - The edges arnt scaled such that the ideal line is contained between 
#       the inner and outter edges.
#     - Translation is affected by the scaling above, unknown if translation
#       is fully accurate.
#     - The edge indicies arnt aligned with the ideal line indicies. i.e. 
#       index 1 in ideal line might correspond to index 20 in inner edge and
#       index 24 in outter edge. Current roll method gets close to matching
#       the indicies such that they create a slice of the track but it isnt
#       perfect.
#====================================================


# === Imports ===
import os
import cv2
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from ParseGameFiles import find_kn5_file, find_ai_files, parse_kn5_road_vertices, parse_ideal_line


# === Track Image Preprocessing Helpers ===
"""
    Check if two sequences of 2D points have the same overall direction.

    Args:
        vec_a (np.ndarray): First set of points (N x 2).
        vec_b (np.ndarray): Second set of points (N x 2).

    Returns:
        bool: True if direction is consistent, False otherwise.
"""
def same_direction(vec_a, vec_b):
    direction_a = vec_a[1] - vec_a[0]
    direction_b = vec_b[1] - vec_b[0]
    return np.dot(direction_a, direction_b) > 0


"""
    Scale and translate inner/outer track edges to match a target centerline's bounding box.

    Args:
        inner (np.ndarray): Left edge points (N x 2).
        outer (np.ndarray): Right edge points (N x 2).
        target (np.ndarray): Target centerline points (N x 2).

    Returns:
        tuple: (scaled_inner, scaled_outer) arrays.
"""
def scale_and_translate_edges(inner, outer, target):
    combined = np.vstack([inner, outer])
    target_min = target.min(axis=0)
    target_max = target.max(axis=0)
    combined_min = combined.min(axis=0)
    combined_max = combined.max(axis=0)

    scale = (target_max - target_min) / (combined_max - combined_min)
    offset = target_min - combined_min * scale

    inner_aligned = inner * scale + offset
    outer_aligned = outer * scale + offset
    return inner_aligned, outer_aligned


"""
    Find the circular shift that best aligns two curves using total distance (sum method).

    Args:
        a (np.ndarray): First curve (N x 2).
        b (np.ndarray): Second curve (N x 2).
        sample_size (int): Number of points to compare for alignment.

    Returns:
        int: Best shift value.
"""
def find_best_roll_sum(a, b, sample_size=50):
    N = len(a)
    best_shift = 0
    min_total_dist = float("inf")

    for shift in range(N):
        b_shifted = np.roll(b, shift, axis=0)
        total_dist = np.linalg.norm(a[:sample_size] - b_shifted[:sample_size], axis=1).sum()
        if total_dist < min_total_dist:
            min_total_dist = total_dist
            best_shift = shift

    return best_shift


"""
    Find the circular shift that best aligns two curves using average distance (mean method).

    Args:
        left_edge (np.ndarray): Left edge points (N x 2).
        right_edge (np.ndarray): Right edge points (N x 2).
        sample_size (int): Number of points to compare for alignment.

    Returns:
        int: Best shift value.
"""
def find_best_roll_mean(left_edge, right_edge, sample_size=10):
    min_dist = float("inf")
    best_shift = 0

    for shift in range(len(right_edge)):
        rolled = np.roll(right_edge, shift, axis=0)
        dist = np.linalg.norm(left_edge[:sample_size] - rolled[:sample_size], axis=1).mean()
        if dist < min_dist:
            min_dist = dist
            best_shift = shift

    return best_shift


# === Contour Resampling and Alignment ===
"""
    Resample a contour along arc-length and smooth it using Savitzky-Golay filtering.

    Args:
        edge (np.ndarray): Contour points (N x 2).
        num_points (int): Number of output points.
        window_length (int): Window size for smoothing filter.
        polyorder (int): Polynomial order for smoothing.

    Returns:
        np.ndarray: Resampled and smoothed contour (num_points x 2).
"""
def resample_edge_savitzky(edge, num_points, window_length=15, polyorder=3):
    edge = np.vstack([edge, edge[0]])  # Close the loop
    distances = np.sqrt(np.diff(edge[:,0])**2 + np.diff(edge[:,1])**2)
    cumulative_distance = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_distance[-1]
    uniform_distances = np.linspace(0, total_length, num_points)

    interp_x = interp1d(cumulative_distance, edge[:,0], kind='linear')
    interp_z = interp1d(cumulative_distance, edge[:,1], kind='linear')

    x_resampled = interp_x(uniform_distances)
    z_resampled = interp_z(uniform_distances)

    x_smooth = savgol_filter(x_resampled, window_length, polyorder, mode='wrap')
    z_smooth = savgol_filter(z_resampled, window_length, polyorder, mode='wrap')

    return np.vstack([x_smooth, z_smooth]).T


# === Full Track Image Processing ===
"""
    Process a track map image to extract, align, and resample left/right track edges.

    Args:
        image_path (str): Path to the map.png image.
        ideal_df (pandas.DataFrame): Ideal racing line DataFrame.
        fast_df (pandas.DataFrame): Fast lane centerline DataFrame.
        window_length (int): Savitzky-Golay smoothing window length.
        polyorder (int): Polynomial order for smoothing.

    Returns:
        pandas.DataFrame: DataFrame containing left_x, left_y, left_z, right_x, right_y, right_z.
"""
def process_track_image(image_path, ideal_df, fast_df, window_length, polyorder):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    alpha_channel = image[:, :, 3]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    binary[alpha_channel < 255] = 0

    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(sorted_contours) < 2:
        raise ValueError(f"Not enough contours in image: {image_path}")

    outer = sorted_contours[0].squeeze()
    inner = sorted_contours[1].squeeze()

    num_points = len(ideal_df)
    outer_resampled = resample_edge_savitzky(outer, num_points, window_length, polyorder)
    inner_resampled = resample_edge_savitzky(inner, num_points, window_length, polyorder)

    if not same_direction(outer_resampled[:20], inner_resampled[:20]):
        inner_resampled = np.flipud(inner_resampled)

    shift = find_best_roll_mean(outer_resampled, inner_resampled, sample_size=20)
    inner_resampled = np.roll(inner_resampled, shift, axis=0)

    fast_line = fast_df[["x", "z"]].to_numpy()
    if len(fast_line) != len(ideal_df):
        interp_x = interp1d(np.linspace(0, 1, len(fast_line)), fast_line[:, 0])
        interp_z = interp1d(np.linspace(0, 1, len(fast_line)), fast_line[:, 1])
        resampled_fast_line = np.column_stack([
            interp_x(np.linspace(0, 1, len(ideal_df))),
            interp_z(np.linspace(0, 1, len(ideal_df)))
        ])
    else:
        resampled_fast_line = fast_line

    inner_resampled, outer_resampled = scale_and_translate_edges(inner_resampled, outer_resampled, resampled_fast_line)

    edge_centerline = (inner_resampled + outer_resampled) / 2
    ideal_line = ideal_df[["x", "z"]].to_numpy()
    best_shift = find_best_roll_mean(ideal_line, edge_centerline, sample_size=30)

    inner_resampled = np.roll(inner_resampled, best_shift, axis=0)
    outer_resampled = np.roll(outer_resampled, best_shift, axis=0)

    track_edges_df = pd.DataFrame({
        "left_x": outer_resampled[:, 0],
        "left_y": 0,
        "left_z": outer_resampled[:, 1],
        "right_x": inner_resampled[:, 0],
        "right_y": 0,
        "right_z": inner_resampled[:, 1]
    })

    return track_edges_df


"""
    Add precise Y-elevation to a track DataFrame using KD-tree nearest lookup.

    Args:
        combined_df (pandas.DataFrame): Track DataFrame with x/z coordinates.
        kn5_df (pandas.DataFrame): KN5 vertices DataFrame with x/y/z.

    Returns:
        pandas.DataFrame: Updated DataFrame with interpolated y values.
"""
def add_precise_elevation(combined_df, kn5_df):
    tree = KDTree(kn5_df[["x", "z"]].values)

    for prefix in ["left", "right", ""]:
        x_col = f"{prefix}_x" if prefix else "x"
        z_col = f"{prefix}_z" if prefix else "z"
        y_col = f"{prefix}_y" if prefix else "y"

        coords = combined_df[[x_col, z_col]].values
        _, nearest_idxs = tree.query(coords)
        combined_df[y_col] = kn5_df.iloc[nearest_idxs]["y"].values

    return combined_df


# === Full Track Processing ===
"""
    Process an entire Assetto Corsa track folder (all layouts) and save extracted data.

    Args:
        track_name (str): Name of the track.
        tracks_root (str): Path to the tracks folder.
        output_root (str): Path to output the processed CSVs.
        window_length (int): Smoothing window for edge resampling.
        polyorder (int): Polynomial order for smoothing.

    Returns:
        None
"""
def process_track(track_name, tracks_root, output_root, window_length=15, polyorder=3):
    track_path = os.path.join(tracks_root, track_name)
    if not os.path.isdir(track_path):
        return
    
    kn5_path = find_kn5_file(track_path, track_name)
    if not kn5_path:
        print(f"No KN5 file for {track_name}, skipping...")
        return

    layouts = find_ai_files(track_path)
    if not layouts:
        print(f"No valid layouts found for {track_name}, skipping...")
        return

    os.makedirs(output_root, exist_ok=True)
    print(f"\nProcessing {track_name} with {len(layouts)} layout(s)...")

    for i, (fast_path, ideal_path) in enumerate(layouts):
        layout_dir = os.path.dirname(os.path.dirname(ideal_path))
        layout_name = os.path.basename(layout_dir)
        output_filename = f"{track_name}_{layout_name}_Processed_Data.csv"
        output_path = os.path.join(output_root, output_filename)

        image_path = None
        if os.path.isfile(os.path.join(layout_dir, "map.png")):
            image_path = os.path.join(layout_dir, "map.png")
        elif os.path.isfile(os.path.join(track_path, "map.png")):
            image_path = os.path.join(track_path, "map.png")

        if not image_path:
            print(f"No track image found for {track_name} layout {layout_name}, skipping...")
            continue

        try:
            ideal_df = parse_ideal_line(ideal_path)
            fast_df = parse_ideal_line(fast_path)
            edges_df = process_track_image(image_path, ideal_df, fast_df, window_length, polyorder)

            if len(ideal_df) != len(edges_df):
                raise ValueError(f"Length mismatch: {len(ideal_df)} vs {len(edges_df)}")

            combined_df = pd.concat([edges_df.reset_index(drop=True), ideal_df.reset_index(drop=True)], axis=1)
            kn5_df = parse_kn5_road_vertices(kn5_path)
            combined_df = add_precise_elevation(combined_df, kn5_df)

            combined_df.to_csv(output_path, index=False)
            print(f"Saved {output_filename}.")
        except Exception as e:
            print(f"Failed to process {layout_name}: {e}")

    print(f"Finished {track_name}.")


# === All Tracks Batch Processor ===
"""
    Process all tracks inside a root directory and save output CSVs.

    Args:
        tracks_root (str): Path to all tracks.
        output_root (str): Path to output all processed CSVs.
        window_length (int): Smoothing window for edge resampling.
        polyorder (int): Polynomial order for smoothing.

    Returns:
        None
"""
def lstm_process_all_tracks(tracks_root, output_root, window_length=15, polyorder=3):
    for track_name in os.listdir(tracks_root):
        process_track(track_name, tracks_root, output_root, window_length, polyorder)
    print("Finished all tracks.")