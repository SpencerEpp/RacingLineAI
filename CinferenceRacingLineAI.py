# === Imports ===
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm.notebook import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import torch.serialization
torch.serialization.add_safe_globals([MinMaxScaler])
import matplotlib.pyplot as plt


# === Model ===
class Cerberus (torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        #Thanks Resnet!
        class ResBlock(torch.nn.Module):
            def __init__(self, in_chans, out_chans, kern_size, padding, dilation):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(in_chans, out_chans, kern_size, padding=padding, dilation=dilation)
                self.relu = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv1d(out_chans, out_chans, kern_size, padding=padding, dilation=dilation)
    
                self.shortcut = torch.nn.Identity()
                if in_chans != out_chans:
                    self.shortcut = nn.Conv1d(in_chans, out_chans, kernel_size=1)
                
            def forward(self, x):
                residual = self.shortcut(x)
                return self.relu(self.conv2(self.relu(self.conv1(x))) + residual)
            
        
        
        self.encoder = torch.nn.Sequential(
            ResBlock(config["input_size"], config["hidden1"], config["kern_size1"], config["padding1"], config["dilation1"]),
            ResBlock(config["hidden1"], config["hidden1"], config["kern_size2"], config["padding2"], config["dilation2"])
        )

        #x, y, z
        self.position_head = torch.nn.Sequential(
            torch.nn.Linear(config["hidden1"], config["pos_head_sz"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["pos_head_sz"], 3)
        )

        #speed, gas, brake, side-left, side-right
        self.control_head = torch.nn.Sequential(
            torch.nn.Linear(config["hidden1"] + 3, config["cont_head_sz"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["cont_head_sz"], 5)
        )


            
        

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.mean(encoded, dim=2)

        position = self.position_head(encoded)
        control_in = torch.cat([encoded, position], dim=1)
        control = self.control_head(control_in)

        
        return position, control


# === Inference Helper Functions ===
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    scaler_x = checkpoint["scaler_x"]
    scaler_y = checkpoint["scaler_y"]
    scaler_z = checkpoint["scaler_z"]
    model = Cerberus(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg["device"])
    return model, cfg, scaler_x, scaler_y, scaler_z


def add_contextual_features(df):
    coords = df[["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]].values
    left = coords[:, :3]
    right = coords[:, 3:]

    # === Centerline & heading ===
    center = (left + right) / 2
    heading = np.diff(center, axis=0, prepend=center[0:1])
    heading = heading / (np.linalg.norm(heading, axis=1, keepdims=True) + 1e-8)

    # === Cumulative distance along centerline ===
    distances = np.linalg.norm(np.diff(center, axis=0, prepend=center[0:1]), axis=1)
    cumulative_distance = np.cumsum(distances)

    # === Curvature ===
    dd = np.diff(heading, axis=0, prepend=heading[0:1])
    curvature = np.linalg.norm(dd, axis=1)

    # === Track-level metadata ===
    avg_width = np.mean(np.linalg.norm(left - right, axis=1))
    max_width = np.max(np.linalg.norm(left - right, axis=1))
    min_width = np.min(np.linalg.norm(left - right, axis=1))
    total_length = cumulative_distance[-1]
    avg_curvature = np.mean(curvature)
    max_curvature = np.max(curvature)

    # === Append as new columns ===
    df["distance"] = cumulative_distance
    df["heading_x"], df["heading_y"], df["heading_z"] = heading.T
    df["curvature"] = curvature

    # === Append constant metadata to all rows ===
    df["track_avg_width"] = avg_width
    df["track_min_width"] = min_width
    df["track_max_width"] = max_width
    df["track_total_length"] = total_length
    df["track_avg_curvature"] = avg_curvature
    df["track_max_curvature"] = max_curvature

    return df

def is_circular_track(df, input_cols, threshold=5.0):
    start = df[input_cols].iloc[0].values
    end = df[input_cols].iloc[-1].values
    dist = np.linalg.norm(start[[0, 2]] - end[[0, 2]])
    return dist <= threshold

def get_centered_sequence(X_scaled, center_idx, seq_len, circular):
    half = seq_len // 2
    n = len(X_scaled)

    if circular:
        indices = [(center_idx - half + i) % n for i in range(seq_len)]
        return X_scaled[indices]
    else:
        left = center_idx - half
        right = center_idx + half + 1
        if left < 0:
            delta = X_scaled[1] - X_scaled[0]
            pre = [X_scaled[0] - delta * (i + 1) for i in reversed(range(-left))]
            seq = np.vstack(pre + [*X_scaled[0:right]])
        elif right > n:
            delta = X_scaled[-1] - X_scaled[-2]
            post = [X_scaled[-1] + delta * (i + 1) for i in range(right - n)]
            seq = np.vstack([*X_scaled[left:n]] + post)
        else:
            seq = X_scaled[left:right]
        return seq

def same_direction(vec_a, vec_b):
    direction_a = vec_a[1] - vec_a[0]
    direction_b = vec_b[1] - vec_b[0]
    return np.dot(direction_a, direction_b) > 0

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

def resample_edge_savitzky(edge, num_points, window_length=15, polyorder=3):
    edge = np.vstack([edge, edge[0]])
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

# === Inference On Testing Tracks ===
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
