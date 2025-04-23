import os
import cv2
import struct
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
import h5py


from create_data import parse_ideal_line, find_kn5_file, find_ai_files



def process_track_image_two_electric_boogaloo(image_path, target_size=(256,256)):

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


    #helper function. Normalizes countours into a bounding box, and places it at (0,0)
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
    
    #samples a centerline based on nearest-neighbour.
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


    #NOTE/TODO: Move below into a new function that is optional if needed?
    #This is some hot garbage, but it gives us scaled images to work off of.
    #Hooray for CNN's!

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

    #GET READY FOR SOME HOT GARBAGE
    patches = []
    for point in center:
        patch = extract_patch(point, track_image)
        patches.append(patch)

    patches = np.stack(patches)


    #NOTE/TODO: metadata functions. Curvature (off center), track width (off center), and covered dist

    return {"inner"      : inner, 
            "outer"      : outer, 
            "center"     : center, 
            "track_image": track_image, 
            "patches"    : patches,
            "scale"      : scale,
            "min_xy"     : min_xy,
            "pad"        : pad}


def transform_ai_positions(ai_x, ai_y, scale, min_xy, pad):
    ai_pos = np.stack([ai_x, ai_y], axis = 1)
    ai_scaled = (ai_pos - min_xy) * scale
    ai_transformed = ai_scaled + pad
    return ai_transformed

# def align_ai_and_center(center, ai_transformed):
#     center_tree = KDTree(center)
#     distances, indices = center_tree.query(ai_transformed)

#     aligned = center[indices]
#     return aligned, distances



'''align_ai_and_center
gets the ai information matching the generated center line.
Returns a DF that contains the ai information that matches each point
on the center line.

Note: There is some data loss here. It's intentional.
We have may more AI points than our generated center line.
'''
# def align_ai_and_center(center, ai_transformed, ai_df):
#     center_tree = KDTree(center)
#     distances, indices = center_tree.query(ai_transformed)

#     n_centers = len(center)
#     aligned_labels = [None] * n_centers

#     for ai_idx, center_idx in enumerate(indices):
#         aligned_labels[center_idx] = ai_df.iloc[ai_idx]
    
#     aligned_df = pd.DataFrame([row.to_dict() if row is not None else None for row in aligned_labels])

#     return aligned_df

def align_ai_and_center(ai_transformed, center, ai_df):
    center_tree = KDTree(center)
    distances, indices = center_tree.query(ai_transformed)

    # Create empty DataFrame with the same columns as ai_df, and one row per center point
    aligned_df = pd.DataFrame(index=range(len(center)), columns=ai_df.columns)

    # Fill rows in aligned_df at centerline indices with the corresponding ai_df rows
    for ai_idx, center_idx in enumerate(indices):
        aligned_df.iloc[center_idx] = ai_df.iloc[ai_idx]

    return aligned_df, distances


def save_track_data_as_hdf5(aligned_df, track_dict, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    with h5py.File(path, 'w') as file:
        file.create_dataset("inner_edge", data=track_dict["inner"])
        file.create_dataset("outer_edge", data=track_dict["outer"])
        file.create_dataset("ai_data", data=aligned_df)
        file.create_dataset("track_image", data=track_dict["track_image"])
        file.create_dataset("patches", data=track_dict["patches"])


def process_track(track_name, tracks_root, output_root, target_size=(256,256)):
    track_path = os.path.join(tracks_root, track_name)
    if not os.path.isdir(track_path):
        print("no track found")
        return
    
    # kn5_path = find_kn5_file(track_path, track_name)
    # if not kn5_path:
    #     print(f"No KN5 file for {track_name}, skipping...")
    #     return

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
            fast_df = parse_ideal_line(fast_path)
            print("Got here! 1")
            track_dict = process_track_image_two_electric_boogaloo(image_path, target_size=target_size)
            print("Got here! 2")
            ai_trans = transform_ai_positions(fast_df["x"], fast_df["y"], track_dict["scale"], track_dict["min_xy"], track_dict["pad"])
            print("Got here! 3")
            aligned = align_ai_and_center(track_dict["center"], ai_trans, fast_df)
            print("Got here! 4")

            print(aligned)
            import matplotlib.pyplot as plt

            #temp visual code
            track_image = track_dict["track_image"]
            plt.imshow(track_image, cmap='gray')
            plt.xlim(0, track_image.shape[1])
            plt.ylim(track_image.shape[1], 0)
            plt.plot(track_dict["center"][:, 0], track_dict["center"][:, 1], linestyle=":", color="red")
            plt.plot(aligned["x"][:,0], aligned["z"][:,1], linestyle=":", color="green")

            plt.show()



            #aligned has the df that has all the data points from fast_df aligned to the centerline.




        except Exception as e:
            print(f"Failed to process {layout_name}: {e}")


process_track("monaco", "./data/test/track", "./", (1024,1024))


# p_trk_img = process_track_image_two_electric_boogaloo("./data/testing_layouts/images/ks_barcelona_layout_gp.png", (512,512))

# import matplotlib.pyplot as plt

# track_image = p_trk_img["track_image"]
# plt.imshow(track_image, cmap='gray')
# plt.xlim(0, track_image.shape[1])
# plt.ylim(track_image.shape[1], 0)
# plt.plot(p_trk_img["center"][:, 0], p_trk_img["center"][:, 1], linestyle=":", color="red")
# plt.show()

# patches = p_trk_img["patches"]


# rand = np.random.choice(len(patches), size=5, replace=False)

# fig, axes = plt.subplots(1, 5, figsize=(15,3))
# for i, idx in enumerate(rand):
#     ax = axes[i]
#     img = patches[idx]
#     ax.imshow(img, cmap='gray')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

