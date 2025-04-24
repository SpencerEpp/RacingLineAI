# === Imports ===
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.serialization
torch.serialization.add_safe_globals([MinMaxScaler])
import matplotlib.pyplot as plt
import time
from torch.amp import GradScaler, autocast
import gc


class Profiler:
    def __init__(self):
        self.times = {}

    def start(self, label):
        self.times[label] = self.times.get(label, {"total": 0, "count": 0, "start": 0})
        self.times[label]["start"] = time.perf_counter()

    def stop(self, label):
        elapsed = time.perf_counter() - self.times[label]["start"]
        self.times[label]["total"] += elapsed
        self.times[label]["count"] += 1

    def report(self):
        print("\n=== Profiler Summary ===")
        for label, t in self.times.items():
            avg = t["total"] / max(1, t["count"])
            print(f"{label:25} | Total: {t['total']:.2f}s | Calls: {t['count']} | Avg: {avg:.4f}s")

def profiled(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler.start(name)
            result = func(*args, **kwargs)
            profiler.stop(name)
            return result
        return wrapper
    return decorator

profiler = Profiler()


# === Dataset ===
@profiled("Is Track Circular")
def is_circular_track(df, output_cols, threshold=5.0):
    start = df[output_cols].iloc[0].values
    end = df[output_cols].iloc[-1].values
    dist = np.linalg.norm(start[[0, 2]] - end[[0, 2]])  # Use X and Z
    return dist <= threshold

@profiled("Get Centered Sequences")
def get_centered_sequence(X, center_idx, seq_len, circular):
    half = seq_len // 2
    n = len(X)

    if circular:
        indices = [(center_idx - half + i) % n for i in range(seq_len)]
        return X[indices]
    else:
        left = center_idx - half
        right = center_idx + half + 1
        if left < 0:
            delta = X[1] - X[0]
            pre = [X[0] - delta * (i + 1) for i in reversed(range(-left))]
            seq = np.vstack(pre + [*X[0:right]])
        elif right > n:
            delta = X[-1] - X[-2]
            post = [X[-1] + delta * (i + 1) for i in range(right - n)]
            seq = np.vstack([*X[left:n]] + post)
        else:
            seq = X[left:right]
        return seq

# @profiled("Add Contextual Features")
# def add_contextual_features(df):
#     coords = df[["left_x","left_y","left_z","right_x","right_y","right_z"]].values
#     left = coords[:, :3]
#     right = coords[:, 3:]

#     # === Centerline & heading ===
#     center = (left + right) / 2
#     heading = np.diff(center, axis=0, prepend=center[0:1])
#     heading = heading / (np.linalg.norm(heading, axis=1, keepdims=True) + 1e-8)

#     # === Cumulative distance along centerline ===
#     distances = np.linalg.norm(np.diff(center, axis=0, prepend=center[0:1]), axis=1)
#     cumulative_distance = np.cumsum(distances)

#     # === Curvature ===
#     dd = np.diff(heading, axis=0, prepend=heading[0:1])
#     curvature = np.linalg.norm(dd, axis=1)

#     # === Append as new columns ===
#     df["distance"] = cumulative_distance
#     df["heading_x"], df["heading_y"], df["heading_z"] = heading.T
#     df["curvature"] = curvature

#     return df

@profiled("Add Contextual Features")
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


# === Data with Augmentations on the fly (Use if <16Gb RAM and/or < 8Gb of VRAM) ===
# class RacingLineDataset(Dataset):
#     def __init__(self, config, file_list, enable_augmentation=True):
#         self.config = config
#         self.file_list = file_list
#         self.enable_augmentation = enable_augmentation
#         self.index_map = []
#         self.data = []

#         # === Fit global scalers ===
#         self.scaler_x = MinMaxScaler()
#         self.scaler_y = MinMaxScaler()
#         all_X, all_Y = [], []

#         for path in file_list:
#             df = add_contextual_features(pd.read_csv(path))
#             all_X.append(df[config["input_cols"]].values)
#             all_Y.append(df[config["output_cols"]].values)
#         self.scaler_x.fit(np.vstack(all_X))
#         self.scaler_y.fit(np.vstack(all_Y))

#         # === Preload layout data ===
#         for idx, path in enumerate(file_list):
#             df = add_contextual_features(pd.read_csv(path))
#             is_circular = is_circular_track(df, config["output_cols"])
#             X = self.scaler_x.transform(df[config["input_cols"]].values)
#             Y = self.scaler_y.transform(df[config["output_cols"]].values)
#             self.data.append((X, Y, is_circular))

#             for i in range(len(X)):
#                 self.index_map.append((idx, i, 'orig'))
#                 if self.enable_augmentation:
#                     self.index_map.append((idx, i, 'flip'))
#                     self.index_map.append((idx, i, 'mirror'))

#     @profiled("Dataset __len__")
#     def __len__(self):
#         return len(self.index_map)

#     @profiled("Dataset __getitem__")
#     def __getitem__(self, idx):
#         file_idx, i, aug_type = self.index_map[idx]
#         X, Y, is_circular = self.data[file_idx]
#         seq = get_centered_sequence(X, i, self.config["seq_len"], is_circular)
#         target = Y[i]

#         if aug_type == "flip":
#             seq = np.flip(seq, axis=0).copy()
#         elif aug_type == "mirror":
#             seq = seq.copy()
#             seq[:, [0, 3]] *= -1
#             seq[:, [2, 5]] *= -1
#             target = target.copy()
#             target[[0, 2]] *= -1

#         return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# === Data with Augmentations (better have a >16Gb of VRAM) ===
# class RacingLineDataset(Dataset):
#     def __init__(self, config, file_list, enable_augmentation=True):
#         self.config = config
#         self.enable_augmentation = enable_augmentation
#         self.scaler_x = MinMaxScaler()
#         self.scaler_y = MinMaxScaler()
#         inputs, targets = [], []

#         # === Fit global scalers first ===
#         all_X, all_Y = [], []
#         count = 0
#         for path in file_list:
#             count +=1
#             print(count)
#             df = add_contextual_features(pd.read_csv(path))
#             all_X.append(df[config["input_cols"]].values)
#             all_Y.append(df[config["output_cols"]].values)
#         self.scaler_x.fit(np.vstack(all_X))
#         self.scaler_y.fit(np.vstack(all_Y))
#         del all_X, all_Y

#         # === Process each layout individually ===
#         count, items = 0, 0
#         for path in file_list:
#             count +=1
#             print(f"Loop iters: {count} | Items in inputs {items}")
#             df = add_contextual_features(pd.read_csv(path))
#             is_circular = is_circular_track(df, config["output_cols"])
#             X = self.scaler_x.transform(df[config["input_cols"]].values)
#             Y = self.scaler_y.transform(df[config["output_cols"]].values)

#             for i in range(len(X)):
#                 items +=1
#                 seq = get_centered_sequence(X, i, config["seq_len"], is_circular)
#                 inputs.append(seq)
#                 targets.append(Y[i])

#                 if self.enable_augmentation:
#                     flipped = np.flip(seq, axis=0).copy()
#                     inputs.append(flipped)
#                     targets.append(Y[i])
#                     mirrored = seq.copy()
#                     mirrored[:, [0, 3]] *= -1
#                     mirrored[:, [2, 5]] *= -1
#                     mirrored_target = Y[i].copy()
#                     mirrored_target[[0, 2]] *= -1
#                     inputs.append(mirrored)
#                     targets.append(mirrored_target)

#             del df, X, Y

#         # === Finalize dataset ===
#         self.inputs = torch.tensor(np.array(inputs), dtype=torch.float32).to(config["device"])
#         self.targets = torch.tensor(np.array(targets), dtype=torch.float32).to(config["device"])
#         del inputs, targets

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.targets[idx]


# === Data with Augmentations (Stored in RAM 16Gb req, also uses up to 8Gb VRAM) ===
class RacingLineDataset(Dataset):
    def __init__(self, config, file_list, enable_augmentation=True):
        self.config = config
        self.enable_augmentation = enable_augmentation
        self.seq_len = config["seq_len"]
        self.input_cols = config["input_cols"]
        self.output_cols = config["output_cols"]

        # === Fit scalers globally ===
        all_X, all_Y = [], []
        for path in file_list:
            df = add_contextual_features(pd.read_csv(path))
            all_X.append(df[self.input_cols].values)
            all_Y.append(df[self.output_cols].values)
            del df
        self.scaler_x = MinMaxScaler().fit(np.vstack(all_X))
        self.scaler_y = MinMaxScaler().fit(np.vstack(all_Y))
        del all_X, all_Y

        # === Count total sequences ===
        n_total = 0
        sequence_counts = []  # (path, count, is_circular)
        for path in file_list:
            df = add_contextual_features(pd.read_csv(path))
            is_circular = is_circular_track(df, self.output_cols)
            aug = len(df) * 2 if enable_augmentation else 0
            total = len(df) + aug
            sequence_counts.append((path, total, is_circular))
            n_total += total
            del df

        # === Preallocate memory ===
        self.inputs = np.empty((n_total, self.seq_len, len(self.input_cols)), dtype=np.float32)
        self.targets = np.empty((n_total, len(self.output_cols)), dtype=np.float32)

        # === Fill data ===
        index = 0
        for path, total, is_circular in sequence_counts:
            df = add_contextual_features(pd.read_csv(path))
            X = self.scaler_x.transform(df[self.input_cols].values)
            Y = self.scaler_y.transform(df[self.output_cols].values)

            for i in range(len(X)):
                seq = get_centered_sequence(X, i, self.seq_len, is_circular)
                target = Y[i]
                self.inputs[index] = seq
                self.targets[index] = target
                index += 1

                if enable_augmentation:
                    flipped = np.flip(seq, axis=0).copy()
                    self.inputs[index] = flipped
                    self.targets[index] = target
                    index += 1

                    mirrored = seq.copy()
                    mirrored[:, [0, 3]] *= -1
                    mirrored[:, [2, 5]] *= -1
                    mirrored_target = target.copy()
                    mirrored_target[[0, 2]] *= -1
                    self.inputs[index] = mirrored
                    self.targets[index] = mirrored_target
                    index += 1

            del df, X, Y

        # === Convert to tensors ===
        self.inputs = torch.from_numpy(self.inputs)
        self.targets = torch.from_numpy(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# === Model ===
class RacingLineLSTMWithAttention(nn.Module):
    def __init__(self, config, scaler_x=None, scaler_y=None):
        super().__init__()
        self.bidirectional = config["bidirectional"]
        self.hidden_size = config["hidden_size"]
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=self.hidden_size,
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.attn = nn.Linear(self.num_directions * self.hidden_size, 1)
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(self.num_directions * self.hidden_size, config["output_size"])
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        return self.fc(context)

    def get_attention_weights(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)
    

# === Loss Function ===
@profiled("MSE Loss")
def weighted_mse(preds, targets, w_xyz=(10, 1, 10)):
    loss = ((preds - targets) ** 2)
    x_loss = w_xyz[0] * loss[:, 0]
    y_loss = w_xyz[1] * loss[:, 1]
    z_loss = w_xyz[2] * loss[:, 2]
    return (x_loss + y_loss + z_loss).mean()

@profiled("Euclidean Loss")
def euclidean_loss(preds, targets):
    return torch.norm(preds - targets, dim=1).mean()


# === Evaluation Function ===
@profiled("Evaluate Model")
def evaluate_model(model, dataloader, criterion, config):
    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for X_batch, Y_batch, in dataloader:
            X_batch, Y_batch = X_batch.to(config["device"], non_blocking=True), Y_batch.to(config["device"], non_blocking=True)
            preds = model(X_batch).detach()
            loss = criterion(preds, Y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# === Save and Load Model ===
@profiled("Save Model")
def save_model(model, config, scaler_x, scaler_y):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
    }, config["model_save_path"])

@profiled("Load Model")
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    scaler_x = checkpoint["scaler_x"]
    scaler_y = checkpoint["scaler_y"]
    model = RacingLineLSTMWithAttention(cfg, scaler_x, scaler_y)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg["device"])
    return model, cfg, scaler_x, scaler_y


# === Training Function ===
@profiled("Train Model")
def train_model(model, train_loader, val_loader, config, scaler_x=None, scaler_y=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    criterion = euclidean_loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"])
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    early_stopping_patience = config["patience"]
    epochs_without_improvement = 0
    best_epoch = 0

    obar = tqdm(range(config["num_epochs"]), desc="Epochs")
    for epoch in obar:
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False)
        for X_batch, Y_batch, in pbar:

            profiler.start("to_device")
            X_batch, Y_batch = X_batch.to(config["device"], non_blocking=True), Y_batch.to(config["device"], non_blocking=True)
            profiler.stop("to_device")

            profiler.start("forward")
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            profiler.stop("forward")

            profiler.start("backward")
            optimizer.zero_grad()
            loss.backward()
            profiler.stop("backward")

            profiler.start("step")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            profiler.stop("step")

            running_loss += loss.item()
            pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})

        del X_batch, Y_batch, preds, loss
        torch.cuda.empty_cache()
        gc.collect()

        train_losses.append(running_loss / len(train_loader))
        val_loss = evaluate_model(model, val_loader, criterion, config)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config, scaler_x, scaler_y)
            best_epoch = epoch+1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        obar.set_postfix({"Train Loss": running_loss/len(train_loader), "Val Loss": val_loss, "Lr": scheduler.get_last_lr()[0], "best_epoch": best_epoch})

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return train_losses, val_losses


# === AMP Enabled Training Function (Probably broken doesnt learn well) ===
# @profiled("Train Model")
# def train_model(model, train_loader, val_loader, config, scaler_x=None, scaler_y=None):
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
#     criterion = euclidean_loss
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["scheduler_patience"])
    
#     scaler = GradScaler()
#     best_val_loss = float("inf")
#     train_losses = []
#     val_losses = []
#     early_stopping_patience = config["patience"]
#     epochs_without_improvement = 0
#     best_epoch = 0

#     obar = tqdm(range(config["num_epochs"]), desc="Epochs")
#     for epoch in obar:
#         model.train()
#         running_loss = 0.0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False)

#         for X_batch, Y_batch in pbar:
#             profiler.start("to_device")
#             X_batch, Y_batch = X_batch.to(config["device"], non_blocking=True), Y_batch.to(config["device"], non_blocking=True)
#             profiler.stop("to_device")

#             optimizer.zero_grad()

#             profiler.start("forward")
#             with autocast(device_type="cuda"):
#                 preds = model(X_batch)
#                 loss = criterion(preds, Y_batch)
#             profiler.stop("forward")

#             profiler.start("backward")
#             scaler.scale(loss).backward()
#             profiler.stop("backward")

#             profiler.start("step")
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             profiler.stop("step")

#             running_loss += loss.item()
#             pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})

#         train_losses.append(running_loss / len(train_loader))
#         val_loss = evaluate_model(model, val_loader, criterion, config)
#         val_losses.append(val_loss)
#         scheduler.step(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             save_model(model, config, scaler_x, scaler_y)
#             best_epoch = epoch + 1
#             epochs_without_improvement = 0
#         else:
#             epochs_without_improvement += 1

#         obar.set_postfix({
#             "Train Loss": running_loss/len(train_loader),
#             "Val Loss": val_loss,
#             "Lr": scheduler.get_last_lr()[0],
#             "best_epoch": best_epoch
#         })

#         if epochs_without_improvement >= early_stopping_patience:
#             print(f"Early stopping triggered after {epoch+1} epochs.")
#             break

#     return train_losses, val_losses


# === Validation Testing ===
@profiled("Evaluate On Val Set")
def evaluate_on_val_set(model, val_files, config, scaler_x, scaler_y):
    model.eval()
    layout_errors = []

    for layout_path in val_files:
        layout_name = os.path.basename(layout_path)
        df = pd.read_csv(layout_path)
        df = add_contextual_features(df)
        is_circular = is_circular_track(df, config["output_cols"])
        X = scaler_x.transform(df[config["input_cols"]].values)
        Y = scaler_y.transform(df[config["output_cols"]].values)
        preds = []

        for i in range(len(X)):
            seq = get_centered_sequence(X, i, config["seq_len"], is_circular)
            X_tensor = torch.tensor(seq.reshape(1, config["seq_len"], -1), dtype=torch.float32).to(config["device"])
            with torch.inference_mode():
                pred = model(X_tensor).detach().cpu().squeeze().numpy()
            preds.append(pred)

        preds = np.array(preds)
        errors = np.linalg.norm(preds - Y, axis=1)
        layout_errors.append((layout_name, errors.mean(), errors.max()))

    print("\n=== Per-Layout Validation Error ===")
    for name, mean_err, max_err in sorted(layout_errors, key=lambda x: x[1], reverse=True):
        print(f"{name:40s}  Mean: {mean_err:.4f}  Max: {max_err:.4f}")

@profiled("Input Ablation")
def run_input_ablation(model, val_dataset, config):
    model.eval()
    base_loss = evaluate_model(model, DataLoader(val_dataset, batch_size=config["batch_size"]), euclidean_loss, config)
    print(f"\nBaseline Val Loss: {base_loss:.4f}\n")

    input_cols = config["input_cols"]
    ablation_results = []

    for i, col in enumerate(input_cols):
        all_inputs, all_targets = [], []
        for j in range(len(val_dataset)):
            x, y = val_dataset[j]
            all_inputs.append(x.unsqueeze(0))
            all_targets.append(y.unsqueeze(0))
        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        ablated_inputs = all_inputs.clone()
        ablated_inputs[:, :, i] = 0.0  # zero out one feature

        ablated_loader = DataLoader(torch.utils.data.TensorDataset(ablated_inputs, all_targets), batch_size=config["batch_size"])
        loss = evaluate_model(model, ablated_loader, euclidean_loss, config)
        impact = loss - base_loss
        ablation_results.append((col, loss, impact))

    print("=== Input Ablation Results ===")
    for col, loss, delta in sorted(ablation_results, key=lambda x: -x[2]):
        print(f"{col:20s}  Loss: {loss:.4f}  ΔLoss: {delta:.4f}")


# === Full Pipeline ===
@profiled("Run Inference")
def run_pipeline(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Preparing dataset...")
    profiler.start("Gathering Files and Split")
    all_files = sorted(glob(os.path.join(config["train_data_path"], "*.csv")))
    train_files, val_files = train_test_split(
        all_files,
        test_size=(1 - config["train_split"]),
        random_state=config["seed"]
    )
    profiler.stop("Gathering Files and Split")

    profiler.start("Create Train Dataset")
    train_dataset = RacingLineDataset(config, file_list=train_files)
    profiler.stop("Create Train Dataset")

    profiler.start("Create Val Dataset")
    val_dataset = RacingLineDataset(config, file_list=val_files)
    profiler.stop("Create Val Dataset")

    scaler_x = train_dataset.scaler_x
    scaler_y = train_dataset.scaler_y
    print("Total sequences loaded:", (len(train_dataset)+len(val_dataset)))

    profiler.start("Create DataLoaders")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)
    torch.cuda.empty_cache()
    profiler.stop("Create DataLoaders")

    print(f"Initializing model...")
    profiler.start("Init Model")
    model = RacingLineLSTMWithAttention(config, scaler_x, scaler_y).to(config["device"]) 
    profiler.stop("Init Model")

    print("Training started...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, config, scaler_x, scaler_y)
    print(f"Training complete. Model saved to {config['model_save_path']}")

    torch.cuda.empty_cache()
    model, cfg, scaler_x, scaler_y = load_model(config["model_save_path"], config["device"])
    evaluate_on_val_set(model, val_files, cfg, scaler_x, scaler_y)
    run_input_ablation(model, val_dataset, cfg)

    profiler.report()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


# === TESTING CODE (make proper inference file) ===
# === Inference on Testing Track Layouts (from coordinates only, doesnt take image for inference) ===
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
def run_inference(path):
    print("Loading model and scalers...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, scaler_x, scaler_y = load_model(path, device)
    model.eval()

    print("Loading unseen layouts from:", config["test_data_path"])
    layout_files = sorted(glob(os.path.join(config["test_data_path"], "*.csv")))
    total_layouts = len(layout_files)
    print(f"Found {total_layouts} layout files.\n")

    for layout_index, layout_path in enumerate(layout_files):
        layout_name = os.path.basename(layout_path)
        print(f"[{layout_index + 1}/{total_layouts}] Predicting layout: {layout_name}")

        df = pd.read_csv(layout_path)
        df = add_contextual_features(df)
        is_circular = is_circular_track(df, config["output_cols"])
        X = df[config["input_cols"]].values
        Y = df[config["output_cols"]].values
        X_scaled = scaler_x.transform(X)
        n = len(X_scaled)
        preds_real = np.zeros_like(Y)
        trues_real = Y.copy()

        for i in tqdm(range(n), desc=f"[{layout_index + 1}/{total_layouts}]"):
            seq = get_centered_sequence(X_scaled, i, config["seq_len"], is_circular)
            X_tensor = torch.tensor(seq.reshape(1, config["seq_len"], -1), dtype=torch.float32).to(config["device"])

            with torch.inference_mode():
                pred_scaled = model(X_tensor).detach().cpu().squeeze().numpy()
                pred_real = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
            preds_real[i] = pred_real

        # === Plot X/Z comparison ===
        plt.figure(figsize=(12, 6))
        plt.plot(trues_real[:, 0], trues_real[:, 2], label="True", linewidth=2)
        plt.plot(preds_real[:, 0], preds_real[:, 2], label="Predicted", linewidth=2, linestyle="--")
        plt.title(f"X vs Z: {layout_name}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Z Coordinate")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

        # === Accuracy & Spatial Error (X/Z only) ===
        print_feature_accuracy(preds_real, trues_real, scaler_y, config["output_cols"])
        spatial_errors = np.linalg.norm(preds_real[:, [0, 2]] - trues_real[:, [0, 2]], axis=1)
        mean_spatial_error = np.mean(spatial_errors)
        max_spatial_error = np.max(spatial_errors)
        print(f"Mean X/Z spatial error: {mean_spatial_error:.2f}m, Max: {max_spatial_error:.2f}m\n")






# # Test code for sequencing data into sliding windows
# from sklearn.preprocessing import MinMaxScaler
# df = pd.read_csv("./data/testing_layouts/ks_barcelona_layout_gp_Processed_Data.csv")
# scaler_x = MinMaxScaler()
# scaler_y = MinMaxScaler()
# scaler_x.fit(df[config["input_cols"]])
# scaler_y.fit(df[config["output_cols"]])
# config["scaler_x"] = scaler_x
# config["scaler_y"] = scaler_y
# visualize_all_center_points(df, config)

# def visualize_all_center_points(df, config):
#     X = df[config["input_cols"]].values
#     Y = df[config["output_cols"]].values
#     X_scaled = config["scaler_x"].transform(X)
#     is_circular = is_circular_track(df, config["output_cols"])

#     seq_len = config["seq_len"]
#     centers_x, centers_z = [], []

#     for i in range(len(X_scaled)):
#         seq = get_centered_sequence(X_scaled, i, seq_len, is_circular)
#         left_x, left_z = seq[:, 0], seq[:, 2]
#         right_x, right_z = seq[:, 3], seq[:, 5]
#         center_x = (left_x + right_x) / 2
#         center_z = (left_z + right_z) / 2
#         mid = seq_len // 2
#         centers_x.append(center_x[mid])
#         centers_z.append(center_z[mid])

#     plt.figure(figsize=(10, 5))
#     plt.plot(centers_x, centers_z, label="Input Sequence Centers", linewidth=2)
#     plt.title("Sliding Window Center Points (Before Model)")
#     plt.xlabel("X")
#     plt.ylabel("Z")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.legend()
#     plt.show()