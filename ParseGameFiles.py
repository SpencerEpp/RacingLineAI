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
#     This file provides helper functions for locating key track resources 
#     (images, AI files, KN5 files) and parsing structured data from them 
#     into pandas DataFrames for further analysis or machine learning use.
#
# Functions Included:
#     - find_track_image(): Locate 'map.png' images associated with track layouts.
#     - find_ai_files(): Find (fast_lane.ai, ideal_line.ai) pairs for layouts.
#     - find_kn5_file(): Find the main .kn5 model file for a given track.
#     - parse_ideal_line(): Parse 'ideal_line.ai' racing line data.
#     - parse_kn5_road_vertices(): Extract track surface vertices from .kn5 files.
#====================================================


# === Imports ===
import os
import struct
import pandas as pd


# === Locate Assetto Corsa Track Resources ===
"""
    Locate the 'map.png' track image for a given Assetto Corsa track.

    The function first checks the base track directory (for single-layout tracks),
    and then iterates through subdirectories (for multi-layout tracks) to find 'map.png'.

    Args:
        track_path (str): Path to the track folder.
        track_name (str): Name of the track.

    Returns:
        str or None: Path to the found map image, or None if not found.
"""
def find_track_image(track_path, track_name):
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

"""
    Locate valid layout pairs of AI files (fast_lane.ai and ideal_line.ai).

    The function checks both the track root directory (single-layout tracks) 
    and any subdirectories (multi-layout tracks) for AI file pairs.

    Args:
        track_path (str): Path to the track folder.

    Returns:
        list of tuples: List of (fast_lane_path, ideal_line_path) pairs found.
"""
def find_ai_files(track_path):
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

"""
    Locate the .kn5 file corresponding to a given track.

    Args:
        track_path (str): Path to the track folder.
        track_name (str): Expected track name (without extension).

    Returns:
        str or None: Path to the found .kn5 file, or None if not found.
"""
def find_kn5_file(track_path, track_name):
    expected_kn5 = f"{track_name}.kn5"
    kn5_path = os.path.join(track_path, expected_kn5)
    return kn5_path if os.path.isfile(kn5_path) else None


# === Parse .ai Files ===
"""
    Parse an Assetto Corsa ideal_line.ai file into a structured pandas DataFrame.

    Extracts positional data, lap data, and driving parameters such as speed, brake,
    gas, and curvature from the binary format.

    Args:
        path (str): Path to the ideal_line.ai file.

    Returns:
        pandas.DataFrame: DataFrame containing ideal line data with detailed columns.
"""
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


# === Parse kn5 Track Surface Vertices ===
"""
    Extract road surface vertices from a .kn5 3D model file.

    Filters mesh nodes based on material or shader names that match typical 
    road-related keywords ("road", "asphalt", etc.).

    Args:
        kn5_path (str): Path to the .kn5 track file.
        road_keywords (list of str, optional): List of keywords for identifying road meshes.

    Returns:
        pandas.DataFrame: DataFrame of extracted (x, y, z) vertex coordinates for the track surface.
"""
def parse_kn5_road_vertices(kn5_path, road_keywords=None):
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