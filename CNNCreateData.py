#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     Provides functionality for processing racetrack images and AI racing line data
#     into standardized datasets for CNN model training and inference.
#
# File Overview:
#     This file includes:
#       - Image preprocessing to extract and normalize track edges and centerlines.
#       - Patch extraction for CNN input.
#       - Contextual feature engineering (curvature, heading, width).
#       - Alignment of game AI data to generated centerlines.
#       - Saving and loading track datasets to/from HDF5 files.
#       - Batch processing of multiple tracks and layouts.
#
# Functions Included:
#     - process_track_image(): Process track map image into contours, centerline, patches, and metadata.
#     - normalize_ai(): Normalize AI coordinate data to match track image dimensions.
#     - align_ai_and_center(): Align AI ideal line data to generated centerline points.
#     - save_track_data_as_hdf5(): Save processed track dataset to HDF5.
#     - process_track(): End-to-end processing for a single track.
#     - restore_scale(): Reverse normalization of AI coordinates.
#     - load_track_dataset(): Load a single processed track from HDF5.
#     - load_all_files(): Load all processed track files from a directory.
#     - cnn_process_all_tracks(): Batch process all tracks from root directory.
#====================================================


# === Imports ===
import os
import cv2
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
import h5py
from ParseGameFiles import parse_ideal_line, find_ai_files


"""
    Process a track map image to extract contours, centerline, patches, and metadata.

    Args:
        image_path (str): Path to the track map image (expects transparency for mask).
        target_size (tuple): Desired (width, height) of output image and normalized data.

    Returns:
        dict:
            - inner (np.ndarray): Normalized inner contour.
            - outer (np.ndarray): Normalized outer contour.
            - center (np.ndarray): Resampled centerline coordinates.
            - track_image (np.ndarray): Rendered binary track image (white track on black background).
            - patches (np.ndarray): 64x64 patches centered on centerline points.
            - scale (float): Scaling factor used during normalization.
            - min_xy (np.ndarray): Minimum x, y coordinates before scaling.
            - pad (np.ndarray): Padding applied after scaling.
            - metadata (dict): Contextual track features (curvature, heading, width, etc.).
"""
def process_track_image(image_path, target_size=(256,256)):

    # === Load & Binarize Transparent Image ===
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    alpha_channel = image[:, :, 3]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    binary[alpha_channel < 255] = 0 # could also have alpha_channel = 0 for strictly transparent 

    # === Extract and sort contours ===
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(sorted_contours) < 2:
        raise ValueError(f"Not enough contours in image: {image_path}")

    outer = sorted_contours[0].squeeze()
    inner = sorted_contours[1].squeeze()

    # === Helper: Normalizes countours into a bounding box (places it at (0,0)) ===
    def normalize_contours(inner, outer, target_size=(256,256)):
        all_points = np.vstack((inner, outer))
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        bounding_width = max_x - min_x
        bounding_height = max_y - min_y

        scale_x = target_size[0] / bounding_width
        scale_y = target_size[1] / bounding_height

        scale = min(scale_x, scale_y)
        inner_norm = (inner - [min_x, min_y]) * scale
        outer_norm = (outer - [min_x, min_y]) * scale

        all_scaled = np.vstack((inner_norm, outer_norm))
        max_scaled = np.max(all_scaled, axis=0)
        min_scaled = np.min(all_scaled, axis=0)
        scaled_width, scaled_height = max_scaled - min_scaled

        pad_x = (target_size[0] - scaled_width) / 2
        pad_y = (target_size[1] - scaled_height) / 2
        pad = np.array([pad_x, pad_y])

        inner_centered = inner_norm + pad
        outer_centered = outer_norm + pad

        return inner_centered, outer_centered, scale, np.array([min_x, min_y]), pad
    
    inner, outer, scale, min_xy, pad = normalize_contours(inner, outer, target_size=target_size)
    
    # === Helper: Samples a Centerline Based on Nearest-Neighbour ===
    def generate_centerline(inner, outer, spacing = 1, smoothing = True):  
        outer_tree = KDTree(outer)
        center_points = []

        #nearest neighbour points of outer line to inner line
        for point in inner:
            _, idx = outer_tree.query(point)
            match = outer[idx]
            center = (point + match) / 2
            center_points.append(center)

        center_points = np.array(center_points)
        diffs = np.diff(center_points, axis=0)
        dists = np.hypot(diffs[:,0], diffs[:,1])
        arc_len = np.insert(np.cumsum(dists), 0, 0)

        total_len = arc_len[-1]
        arc = np.arange(0, total_len, spacing)

        lambdax = interp1d(arc_len, center_points[:, 0], kind='linear')
        lambday = interp1d(arc_len, center_points[:, 1], kind='linear')

        resamp_x = lambdax(arc)
        resamp_y = lambday(arc)

        if smoothing:
            resamp_x_len = len(resamp_x)
            window = min(21, resamp_x_len - (resamp_x_len % 2))
            if window > 4:
                resamp_x = savgol_filter(resamp_x, window_length = window, polyorder = 3)
                resamp_y = savgol_filter(resamp_y, window_length = window, polyorder = 3)
        
        return np.stack([resamp_x, resamp_y], axis =1)

    center = generate_centerline(inner, outer)


    '''Render Track
    creates an image of canvas_size (with expected scaled inputs) where the
    track is rendered as white, and the non-track is rendered as black.

    Yes, we did effectively just invert and move the input image. lmao
    '''
    def render_track(inner, outer, canvas_size = (256,256)):
        image = np.zeros(canvas_size, dtype=np.uint8)

        cv2.fillPoly(image, [outer.astype(np.int32).reshape(-1, 1, 2)], 255)
        cv2.fillPoly(image, [inner.astype(np.int32).reshape(-1, 1, 2)], 0)
        
        return image
    
    track_image = render_track(inner, outer, canvas_size=target_size)


    '''Extract patch
    Gets a 64x64 sample of the track centered on the center point.
    
    '''
    def extract_patch(center, track_image, patch_size = (64,64)):
        half_patch = np.array(patch_size) // 2

        top_left = center - half_patch

        #clip so we don't exceed image bounds
        top_left = np.clip(top_left, 0, np.array(track_image.shape) - patch_size).astype(int)
        bottom_right = top_left + patch_size

        patch = track_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        return patch

    patches = []
    for point in center:
        patch = extract_patch(point, track_image)
        patches.append(patch)

    patches = np.stack(patches)


    '''Add Contextual Features
    generates metadata like cumulative distance, heading, curvature, and 
    track width from the centerline and edges.

    Used to provide additional input features for models.
    '''
    def add_contextual_features(inner, outer, center, profiler=None):
        if profiler: profiler.start("Add Contextual Features")

        diff = np.diff(center, axis=0)
        diff_norm = np.linalg.norm(diff, axis=1)
        cumulative_distance = np.concatenate(([0], np.cumsum(diff_norm)))

        heading_x = np.concatenate(([diff[0, 0] / diff_norm[0]], diff[:, 0] / diff_norm))
        heading_z = np.concatenate(([diff[0, 1] / diff_norm[0]], diff[:, 1] / diff_norm))

        tan_angle = np.arctan2(diff[:, 1], diff[:, 0])
        curvature = np.diff(tan_angle)
        curvature = np.concatenate(([curvature[0]], [curvature[0]], curvature))
        avg_curvature = np.mean(curvature)
        max_curvature = np.max(curvature)

        #duplicate some data here. That's fine
        inner_tree = KDTree(inner)
        outer_tree = KDTree(outer)
        _, inner_index = inner_tree.query(center)
        _, outer_index = outer_tree.query(center)
        inner_aligned = inner[inner_index]
        outer_aligned = outer[outer_index]
        track_widths = np.linalg.norm(inner_aligned - outer_aligned, axis=1)

        avg_width = np.mean(track_widths)
        min_width = np.min(track_widths)
        max_width = np.max(track_widths)

        total_length = cumulative_distance[-1]
        
        meta = {
            "distance"           : cumulative_distance,
            "heading_x"          : heading_x,
            "heading_z"          : heading_z,
            "curvature"          : curvature,
            "track_widths"       : track_widths,
            "track_avg_width"    : avg_width,
            "track_min_width"    : min_width,
            "track_max_width"    : max_width,
            "track_total_length" : total_length,
            "track_avg_curvature": avg_curvature,
            "track_max_curvature": max_curvature
        }

        if profiler: profiler.stop("Add Contextual Features")
        return meta

    metadata = add_contextual_features(inner, outer, center)
    return {"inner"      : inner, 
            "outer"      : outer, 
            "center"     : center, 
            "track_image": track_image, 
            "patches"    : patches,
            "scale"      : scale,
            "min_xy"     : min_xy,
            "pad"        : pad,
            "metadata"   : metadata
            }


"""
    Normalize AI racing line coordinates to match the scaled track image dimensions.

    Args:
        ai_x (np.ndarray): Array of x coordinates from the AI line.
        ai_z (np.ndarray): Array of z coordinates from the AI line.
        target_size (tuple): Target size (width, height) for scaling.

    Returns:
        tuple:
            - ai_padded (np.ndarray): Normalized and padded AI coordinates.
            - scale (float): Scaling factor applied.
            - min_xy (np.ndarray): Minimum x, y before scaling.
            - pad (np.ndarray): Padding added after scaling.
"""
def normalize_ai(ai_x, ai_z, target_size=(256, 256)):
    ai_pos = np.stack([ai_x, ai_z], axis=1)

    min_xy = np.min(ai_pos, axis=0)
    max_xy = np.max(ai_pos, axis=0)
    width, height = max_xy - min_xy

    scale = min(target_size[0] / width, target_size[1] / height)
    ai_scaled = (ai_pos - min_xy) * scale

    pad_x = (target_size[0] - (width * scale)) / 2
    pad_y = (target_size[1] - (height * scale)) / 2
    ai_padded = ai_scaled + np.array([pad_x, pad_y])

    return ai_padded, scale, min_xy, np.array([pad_x, pad_y])


"""
    Align AI line data with generated centerline points based on nearest neighbors.

    Args:
        center (np.ndarray): Generated centerline coordinates.
        patches (np.ndarray): Extracted patches corresponding to centerline points.
        ai_normed (np.ndarray): Normalized AI coordinates.
        ai_df (pandas.DataFrame): Original AI data (speeds, controls, etc.).

    Returns:
        tuple:
            - final_ai_df (pandas.DataFrame): AI data aligned to centerline.
            - final_ai_normed (np.ndarray): Normalized AI points aligned to centerline.
            - final_center (np.ndarray): Updated centerline points.
            - final_patches (np.ndarray): Patches corresponding to updated centerline.
"""
def align_ai_and_center(center, patches, ai_normed, ai_df):
    diff = len(center) - len(ai_df)

    if diff >= 0:
        # More centers → query ai_tree with centers
        ai_tree = KDTree(ai_normed)
        distances, indices = ai_tree.query(center)
        _, unique_idx = np.unique(indices, return_index=True)

        final_center = center[unique_idx]
        final_patches = patches[unique_idx]
        final_ai_df = ai_df.iloc[indices[unique_idx]].reset_index(drop=True)
        final_ai_normed = ai_normed[indices[unique_idx]]

    else:
        # More ai points → query center_tree with ai points
        center_tree = KDTree(center)
        distances, indices = center_tree.query(ai_normed)
        _, unique_idx = np.unique(indices, return_index=True)

        final_center = center[indices[unique_idx]]
        final_patches = patches[indices[unique_idx]]
        final_ai_df = ai_df.iloc[unique_idx].reset_index(drop=True)
        final_ai_normed = ai_normed[unique_idx]

    return final_ai_df, final_ai_normed, final_center, final_patches


"""
    Save a fully processed track dataset to an HDF5 file.

    Args:
        track_dataset (dict): Dictionary containing track data and metadata.
        filename (str): Output filename for the HDF5 file.
        save_dir (str): Directory to save the output file.

    Returns:
        None
"""
def save_track_data_as_hdf5(track_dataset, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    with h5py.File(path, 'w') as file:
        file.create_dataset("inner_edge", data=track_dataset["inner"])
        file.create_dataset("outer_edge", data=track_dataset["outer"])
        file.create_dataset("centerline", data=track_dataset["center"])
        file.create_dataset("track_image", data=track_dataset["track_image"])
        file.create_dataset("track_patches", data=track_dataset["track_patches"])
        file.create_dataset("ai_norm", data=track_dataset["ai_norm"])

        # Save transforms (as attributes)
        file.attrs["track_transform_scale"] = track_dataset["track_transform"][0]
        file.attrs["track_transform_min_xy"] = track_dataset["track_transform"][1]
        file.attrs["track_transform_pad"] = track_dataset["track_transform"][2]

        for col in track_dataset["ai_df"].columns:
            file.create_dataset(f"ai_df/{col}", data=track_dataset["ai_df"][col].to_numpy())

        meta_group = file.create_group("meta")
        for key, value in track_dataset["metadata"].items():
            meta_group.create_dataset(key, data=value)

        # Save AI transform as attributes
        file.attrs["ai_transform_scale"] = track_dataset["ai_transform"][0]
        file.attrs["ai_transform_min_xy"] = track_dataset["ai_transform"][1]
        file.attrs["ai_transform_pad"] = track_dataset["ai_transform"][2]


"""
    End-to-end processing of a single track and its layouts.

    Args:
        track_name (str): Name of the track directory.
        tracks_root (str): Root directory containing all tracks.
        output_root (str): Directory where processed data will be saved.
        target_size (tuple): Target size for track normalization and patch generation.

    Returns:
        None
"""
def process_track(track_name, tracks_root, output_root, target_size=(256,256)):
    track_path = os.path.join(tracks_root, track_name)
    print(track_path)
    if not os.path.isdir(track_path):
        print("no track found")
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
        output_filename = f"{track_name}_{layout_name}_Processed_Data.h5py"

        image_path = None
        if os.path.isfile(os.path.join(layout_dir, "map.png")):
            image_path = os.path.join(layout_dir, "map.png")
        elif os.path.isfile(os.path.join(track_path, "map.png")):
            image_path = os.path.join(track_path, "map.png")

        if not image_path:
            print(f"No track image found for {track_name} layout {layout_name}, skipping...")
            continue

        try:
            fast_df = parse_ideal_line(fast_path)
            track_dict = process_track_image(image_path, target_size=target_size)
            ai_trans, ai_scale, ai_min_xy, ai_pad = normalize_ai(fast_df["x"], fast_df["z"], target_size=target_size)
            aligned_df, aligned_norm, new_center, patches = align_ai_and_center(track_dict["center"], track_dict["patches"],  ai_trans, fast_df)

            track_dataset = {
                #track information
                "inner"          : track_dict["inner"], #normalized inner line
                "outer"          : track_dict["outer"], #normalied outer line
                "center"         : new_center,          #generated center line
                "track_image"    : track_dict["track_image"], #track image of target size
                "track_patches"  : patches,              #track patches per center point, 64x64
                "track_transform": (track_dict["scale"], track_dict["min_xy"], track_dict["pad"]), #
                #ai information
                "ai_df"       : aligned_df,
                "ai_norm"     : aligned_norm,
                "ai_transform": (ai_scale, ai_min_xy, ai_pad),
                "metadata"    : track_dict["metadata"]
            }

            save_track_data_as_hdf5(track_dataset, output_filename, output_root)

        except Exception as e:
            print(f"Failed to process {layout_name}: {e}")


"""
    Reverse the normalization and padding to recover original AI coordinates.

    Args:
        ai_normed (np.ndarray): Normalized and padded AI coordinates.
        scale (float): Scaling factor originally applied.
        min_xy (np.ndarray): Minimum x, y before scaling.
        pad (np.ndarray): Padding added after scaling.

    Returns:
        np.ndarray: Restored AI coordinates in original space.
"""
def restore_scale(ai_normed, scale, min_xy, pad):
    ai_unpadded = ai_normed - pad
    ai_original = ai_unpadded / scale + min_xy
    return ai_original


"""
    Load a processed track dataset from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict:
            - inner (np.ndarray): Inner contour points.
            - outer (np.ndarray): Outer contour points.
            - center (np.ndarray): Centerline points.
            - track_image (np.ndarray): Rendered track image.
            - track_patches (np.ndarray): Patches extracted around centerline points.
            - track_transform (tuple): (scale, min_xy, pad) for track image.
            - ai_transform (tuple): (scale, min_xy, pad) for AI coordinates.
            - ai_df (pandas.DataFrame): DataFrame of AI controls and speeds.
            - ai_norm (np.ndarray): Normalized AI coordinates.
            - metadata (dict): Track feature metadata.
"""
def load_track_dataset(file_path):
    with h5py.File(file_path, "r") as file:
        # Load edges and image
        inner = file["inner_edge"][:]
        outer = file["outer_edge"][:]
        center = file["centerline"][:]
        track_image = file["track_image"][:]
        track_patches = file["track_patches"][:]
        ai_norm = file["ai_norm"][:]

        track_transform = (
            file.attrs["track_transform_scale"],
            file.attrs["track_transform_min_xy"],
            file.attrs["track_transform_pad"]
        )

        ai_transform = (
            file.attrs["ai_transform_scale"],
            file.attrs["ai_transform_min_xy"],
            file.attrs["ai_transform_pad"]
        )

        ai_df = pd.DataFrame({
            key.split("/")[-1]: file[f"ai_df/{key.split('/')[-1]}"][:]
            for key in file["ai_df"]
        })

        metadata_group = file["meta"]
        metadata = {
            key: metadata_group[key][()]
            for key in metadata_group
        }

    return {
        "inner": inner,
        "outer": outer,
        "center": center,
        "track_image": track_image,
        "track_patches": track_patches,
        "track_tranform": track_transform,
        "ai_transform": ai_transform,
        "ai_df": ai_df,
        "ai_norm": ai_norm,
        "metadata": metadata
    }


"""
    Load all processed track datasets from a directory.

    Args:
        file_path (str): Directory containing processed track HDF5 files.

    Returns:
        list: List of track dataset dictionaries loaded from files.
"""
def load_all_files(file_path):
    tracks = []
    for track_name in os.listdir(file_path):
        full_path = os.path.join(file_path, track_name)
        if not os.path.isfile(full_path):
            continue
        if not (full_path.endswith(".h5") or full_path.endswith(".h5py")):
            continue
        tracks.append(load_track_dataset(full_path))
    return tracks


"""
    Batch process all tracks in the given root directory and save outputs.

    Args:
        tracks_root (str): Root directory containing raw track folders.
        output_root (str): Directory where processed datasets will be saved.
        target_size (tuple): Target size for track normalization and patch extraction.

    Returns:
        None
"""
def cnn_process_all_tracks(tracks_root, output_root, target_size = (1024,1024)):
    for track_name in os.listdir(tracks_root):
        process_track(track_name, tracks_root, output_root, target_size=target_size)
    print("Finished all tracks")



# Test to see if data is aligned
# def show_patches(data, number=64):
#     indices = np.random.choice(len(data), number, replace=False)
#     #indices = range(100000,100064)
#     #indices = range(64)


#     images = []
#     xy_dat = []
#     center = []

#     for i in indices:
#         inputs, outputs = data[i]
#         images.append(inputs["patch"])
#         xy_dat.append(outputs["ai_norm"])
#         center.append(inputs["center"])

#     import matplotlib.pyplot as plt

#     grid_size = int(np.ceil(np.sqrt(number)))
#     fix, axes = plt.subplots(grid_size, grid_size, figsize=(10,10))
#     axes = axes.flatten()

#     for i in range(grid_size * grid_size):
#         if i < len(images):
#             axes[i].imshow(images[i].numpy(), cmap='gray')
#             axes[i].plot(xy_dat[i][0::2],xy_dat[i][1::2], 'r.', markersize=5)
#             axes[i].plot(center[i][0::2],center[i][1::2], 'b.', markersize=5)
#             axes[i].set_xlim(0,1024)
#             axes[i].set_ylim(1024,0)
#         else:
#             axes[i].remove()

#     plt.tight_layout()
#     plt.show()