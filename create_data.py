# === Configuration ===
import os
import cv2
import struct
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import KDTree

# === File Parsing Code ===
def parse_ideal_line(path):
    with open(path, "rb") as f:
        version = struct.unpack("<i", f.read(4))[0]
        if version != 7:
            raise ValueError(f"Unsupported spline version: {version}")

        point_count = struct.unpack("<i", f.read(4))[0]
        lap_time = struct.unpack("<i", f.read(4))[0]
        sample_count = struct.unpack("<i", f.read(4))[0]

        # AiPoint: position (vec3), length, id
        points = []
        for _ in range(point_count):
            x, y, z = struct.unpack("<fff", f.read(12))
            length = struct.unpack("<f", f.read(4))[0]
            point_id = struct.unpack("<i", f.read(4))[0]
            points.append([x, y, z, length, point_id])

        extra_count = struct.unpack("<i", f.read(4))[0]
        if extra_count != point_count:
            raise ValueError("Mismatch between point count and extra data count.")

        # AiPointExtra: 18 floats = 72 bytes
        extras = []
        for _ in range(extra_count):
            data = struct.unpack("<" + "f" * 18, f.read(72))
            extras.append(list(data))

    columns = [
        "x", "y", "z", "length", "id",
        "speed", "gas", "brake", "obsolete_lat_g", "radius",
        "side_left", "side_right", "camber", "direction",
        "normal_x", "normal_y", "normal_z",
        "extra_length",
        "forward_x", "forward_y", "forward_z",
        "tag", "grade"
    ]

    df = pd.DataFrame([p + e for p, e in zip(points, extras)], columns=columns)
    return df

# === Preprocess Track Image ===
# Arc-length Resampling with Savitzky-Golay Smoothing (best? marginally)
def resample_edge_savitzky(edge, num_points, window_length=15, polyorder=3):
    # Close loop explicitly
    edge = np.vstack([edge, edge[0]])

    # Calculate cumulative distance (arc-length)
    distances = np.sqrt(np.diff(edge[:,0])**2 + np.diff(edge[:,1])**2)
    cumulative_distance = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_distance[-1]

    # Uniformly spaced points along the distance
    uniform_distances = np.linspace(0, total_length, num_points)

    # Interpolate x and z coordinates linearly
    interp_x = interp1d(cumulative_distance, edge[:,0], kind='linear')
    interp_z = interp1d(cumulative_distance, edge[:,1], kind='linear')

    x_resampled = interp_x(uniform_distances)
    z_resampled = interp_z(uniform_distances)

    # Mild smoothing using Savitzky-Golay (no directional drift)
    x_smooth = savgol_filter(x_resampled, window_length, polyorder, mode='wrap')
    z_smooth = savgol_filter(z_resampled, window_length, polyorder, mode='wrap')

    return np.vstack([x_smooth, z_smooth]).T

def process_track_image(image_path, ideal_df, window_length, polyorder):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert transparent image to binary (white track and black edges)
    alpha_channel = image[:, :, 3]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    binary[alpha_channel < 255] = 0 # could also have alpha_channel = 0 for strictly transparent 

    # Extract edges using contours
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(sorted_contours) < 2:
        raise ValueError(f"Not enough contours in image: {image_path}")

    outer_edge = sorted_contours[0].squeeze()
    inner_edge = sorted_contours[1].squeeze()

    # Resample edges to match ideal_line_df size
    num_points = len(ideal_df)
    outer_edge_resampled = resample_edge_savitzky(outer_edge, num_points, window_length, polyorder)
    inner_edge_resampled = resample_edge_savitzky(inner_edge, num_points, window_length, polyorder)

    # Create DataFrame (x, z plane; y is zero)
    track_edges_df = pd.DataFrame({
        'left_x': outer_edge_resampled[:, 0],
        'left_y': 0,
        'left_z': outer_edge_resampled[:, 1],
        'right_x': inner_edge_resampled[:, 0],
        'right_y': 0,
        'right_z': inner_edge_resampled[:, 1]
    })

    return track_edges_df

# === Parse kn5 for Y-Coords ===
def parse_kn5_road_vertices(kn5_path, road_keywords=None):
    """
    Extracts road-related mesh vertices from a .kn5 file.
    Filters meshes by material or shader names using keywords.
    """
    if road_keywords is None:
        road_keywords = ["road", "asphalt", "track", "surface", "pitlane", "curb"]

    with open(kn5_path, "rb") as f:
        magic = f.read(6)
        version = struct.unpack("<I", f.read(4))[0]

        if version > 5:
            f.read(4)  # skip extra header if present

        # TEXTURES
        texture_count = struct.unpack("<i", f.read(4))[0]
        for _ in range(texture_count):
            f.read(4)  # texture type
            name_len = struct.unpack("<i", f.read(4))[0]
            f.read(name_len)
            tex_size = struct.unpack("<i", f.read(4))[0]
            f.read(tex_size)

        # MATERIALS
        material_count = struct.unpack("<i", f.read(4))[0]
        materials = []
        for _ in range(material_count):
            name_len = struct.unpack("<i", f.read(4))[0]
            name = f.read(name_len).decode("utf-8").lower()
            shader_len = struct.unpack("<i", f.read(4))[0]
            shader = f.read(shader_len).decode("utf-8").lower()
            f.read(2)  # unknown short
            if version > 4:
                f.read(4)
            prop_count = struct.unpack("<i", f.read(4))[0]
            for _ in range(prop_count):
                pname_len = struct.unpack("<i", f.read(4))[0]
                f.read(pname_len)
                f.read(4)
                f.read(36)
            sample_count = struct.unpack("<i", f.read(4))[0]
            for _ in range(sample_count):
                sname_len = struct.unpack("<i", f.read(4))[0]
                f.read(sname_len)
                f.read(4)
                tname_len = struct.unpack("<i", f.read(4))[0]
                f.read(tname_len)
            materials.append((name, shader))

        # MESHES
        mesh_vertices = []

        def matches_road(mat_name, shader_name):
            return any(k in mat_name for k in road_keywords) or any(k in shader_name for k in road_keywords)

        def read_string():
            strlen = struct.unpack("<i", f.read(4))[0]
            return f.read(strlen).decode("utf-8")

        def read_vec3():
            return struct.unpack("<3f", f.read(12))

        def read_node():
            node_type = struct.unpack("<i", f.read(4))[0]
            name = read_string()
            child_count = struct.unpack("<i", f.read(4))[0]
            f.read(1)

            if node_type == 1:  # Dummy node
                f.read(64)
            elif node_type in [2, 3]:  # Mesh or Animated Mesh
                f.read(3)
                vertex_count = struct.unpack("<i", f.read(4))[0]
                positions = []
                for _ in range(vertex_count):
                    pos = read_vec3()
                    f.read(12 + 8 + 12)  # skip normals, UVs, tangents
                    positions.append(pos)
                idx_count = struct.unpack("<i", f.read(4))[0]
                f.read(idx_count * 2)  # indices
                mat_id = struct.unpack("<i", f.read(4))[0]
                f.read(29 if node_type == 2 else 12)

                if 0 <= mat_id < len(materials):
                    mat_name, shader = materials[mat_id]
                    if matches_road(mat_name, shader):
                        mesh_vertices.extend(positions)

            for _ in range(child_count):
                read_node()

        read_node()

    return pd.DataFrame(mesh_vertices, columns=["x", "y", "z"])

def add_precise_elevation(combined_df, kn5_df):
    tree = KDTree(kn5_df[["x", "z"]].values)

    for prefix in ["left", "right", ""]:  # handles left_x/z/y, right_x/z/y, and x/y/z
        x_col = f"{prefix}_x" if prefix else "x"
        z_col = f"{prefix}_z" if prefix else "z"
        y_col = f"{prefix}_y" if prefix else "y"

        coords = combined_df[[x_col, z_col]].values
        _, nearest_idxs = tree.query(coords)
        combined_df[y_col] = kn5_df.iloc[nearest_idxs]["y"].values

    return combined_df

# === Pipeline Helpers ===
def find_track_image(track_path, track_name):
    """Finds the track image whether in track root folder or layout folder"""

    # Case 1: One layout, map.png in base folder
    candidate = os.path.join(track_path, "map.png")
    if os.path.isfile(candidate):
        return candidate
    
    #Case 2: Multiple layouts, map.png found in layout folder per layout
    for sub in os.listdir(track_path):
        layout_dir = os.path.join(track_path, sub)
        if os.path.isdir(layout_dir):
            candidate = os.path.join(layout_dir, "map.png")
            if os.path.isfile(candidate):
                return candidate
            
    return None

def find_ai_files(track_path):
    """Finds all valid layout pairs: (fast_lane.ai, ideal_line.ai)"""

    layouts = []

    # Case 1: single-layout in root (ai/ and data/ inside track root)
    root_fast = os.path.join(track_path, "ai", "fast_lane.ai")
    root_ideal = os.path.join(track_path, "data", "ideal_line.ai")
    if os.path.isfile(root_fast) and os.path.isfile(root_ideal):
        layouts.append((root_fast, root_ideal))

    # Case 2: multiple layouts in subfolders
    for sub in os.listdir(track_path):
        layout_path = os.path.join(track_path, sub)
        if not os.path.isdir(layout_path):
            continue

        fast_path = os.path.join(layout_path, "ai", "fast_lane.ai")
        ideal_path = os.path.join(layout_path, "data", "ideal_line.ai")

        if os.path.isfile(fast_path) and os.path.isfile(ideal_path):
            layouts.append((fast_path, ideal_path))

    return layouts

def find_kn5_file(track_path, track_name):
    """Find the .kn5 file that matches the track name inside the track folder."""
    expected_kn5 = f"{track_name}.kn5"
    kn5_path = os.path.join(track_path, expected_kn5)
    return kn5_path if os.path.isfile(kn5_path) else None

# === Per-track processor ===
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
            edges_df = process_track_image(image_path, ideal_df, window_length, polyorder)

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

# === All Tracks Entry Point ===
def process_all_tracks(tracks_root, output_root, window_length=15, polyorder=3):
    for track_name in os.listdir(tracks_root):
        process_track(track_name, tracks_root, output_root, window_length, polyorder)
    print("Finished all tracks.")