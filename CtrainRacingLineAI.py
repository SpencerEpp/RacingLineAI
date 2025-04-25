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
from sklearn.preprocessing import MinMaxScaler
import torch.serialization
torch.serialization.add_safe_globals([MinMaxScaler])
import matplotlib.pyplot as plt


# === Dataset ===
def is_circular_track(df, output_cols, threshold=5.0):
    start = df[output_cols].iloc[0].values
    end = df[output_cols].iloc[-1].values
    dist = np.linalg.norm(start[[0, 2]] - end[[0, 2]])  # Use X and Z
    return dist <= threshold

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

class RacingLineDataset(Dataset):
    def __init__(self, config, seq_len):
        self.inputs, self.pos_targets, self.cont_targets = [], [], []
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaler_z = MinMaxScaler()
        all_X, all_Y, all_Z = [], [], []
        train_files = sorted(glob(os.path.join(config["train_data_path"], "*.csv")))
        test_files = sorted(glob(os.path.join(config["test_data_path"], "*.csv")))

        # === First pass: Collect data for global fitting ===
        for file in train_files:
            df = pd.read_csv(file)
            X = df[config["input_cols"]].values
            Y = df[config["output_pos_cols"]].values
            Z = df[config["output_cont_cols"]].values
            all_X.append(X)
            all_Y.append(Y)
            all_Z.append(Z)
        for file in test_files:
            df = pd.read_csv(file)
            X = df[config["input_cols"]].values
            Y = df[config["output_pos_cols"]].values
            Z = df[config["output_cont_cols"]].values
            all_X.append(X)
            all_Y.append(Y)
            all_Z.append(Z)
        all_X = np.vstack(all_X)
        all_Y = np.vstack(all_Y)
        all_Z = np.vstack(all_Z)
        self.scaler_x.fit(all_X)
        self.scaler_y.fit(all_Y)
        self.scaler_z.fit(all_Z)

        # === Second pass: Normalize and extract sequences ===
        for file in train_files:
            df = pd.read_csv(file)
            is_circular = is_circular_track(df, config["output_pos_cols"])
            X = self.scaler_x.transform(df[config["input_cols"]].values)
            Y = self.scaler_y.transform(df[config["output_pos_cols"]].values)
            Z = self.scaler_z.transform(df[config["output_cont_cols"]].values)

            for i in range(len(X)):
                self.inputs.append(get_centered_sequence(X, i, seq_len, is_circular))
                self.pos_targets.append(Y[i])
                self.cont_targets.append(Z[i])

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32).to(config["device"])
        self.pos_targets = torch.tensor(np.array(self.pos_targets), dtype=torch.float32).to(config["device"])
        self.cont_targets = torch.tensor(np.array(self.cont_targets), dtype=torch.float32).to(config["device"])
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx].permute(1,0), self.pos_targets[idx], self.cont_targets[idx]
    

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
    

# === Loss Function ===
def weighted_mse(preds, targets, w_xyz=(10, 1, 10)):
    loss = ((preds - targets) ** 2)
    x_loss = w_xyz[0] * loss[:, 0]
    y_loss = w_xyz[1] * loss[:, 1]
    z_loss = w_xyz[2] * loss[:, 2]
    return (x_loss + y_loss + z_loss).mean()

class PositionLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(PositionLoss, self).__init__()
        self.mse_loss = weighted_mse

    def forward(self, predicted, target):
        loss = self.mse_loss(predicted, target)
        return loss

#we can add penalties to encourage smoother and more realistic driving here
class ControlLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(ControlLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        
    def forward(self, predicted, target):
        loss = self.mse_loss(predicted, target)
        return loss


# === Evaluation Function ===
def evaluate_model(model, dataloader, pos_crit, con_crit, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch, Z_batch in dataloader:
            X_batch, Y_batch, Z_batch = X_batch.to(config["device"]), Y_batch.to(config["device"]), Z_batch.to(config["device"])
            pos, cont = model(X_batch)
            pos_loss = pos_crit(pos, Y_batch)
            con_loss = con_crit(cont, Z_batch)
            
            loss = pos_loss + 0.5 * con_loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

# === Save and Load Model ===
def save_model(model, config, scalers):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler_x": scalers[0],
        "scaler_y": scalers[1],
        "scaler_z": scalers[2],
    }, config["model_save_path"])

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


# === Training Function ===
def train_model(model, long_dataset, medium_dataset, short_dataset, config, scalers):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # = hybrid_loss #nn.MSELoss()

    pos_crit = PositionLoss()
    con_crit = ControlLoss(weight=0.5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    early_stopping_patience = config["patience"]
    epochs_without_improvement = 0
    best_epoch = 0

    obar = tqdm(range(config["num_epochs"]), desc="Epochs")
    for epoch in obar:
        match epoch:
            case 0:
                train_loader = long_dataset[0]
                val_loader = long_dataset[1]
                
            case val if val == config["long_epochs"]:
                train_loader = medium_dataset[0]
                val_loader = medium_dataset[1]
            case val if val == (config["medium_epochs"] + config["long_epochs"]):
                train_loader = short_dataset[0]
                val_loader = short_dataset[1]
            case _:
                pass
                
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False)
        for X_batch, Y_batch, Z_batch in pbar:
            X_batch, Y_batch, Z_batch = X_batch.to(config["device"]), Y_batch.to(config["device"]), Z_batch.to(config["device"])

            
            optimizer.zero_grad()
            p_pos, p_cont = model(X_batch)
            pos_loss, con_loss = pos_crit(p_pos, Y_batch), con_crit(p_cont, Z_batch)
            loss = pos_loss + 0.5 * con_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})

        train_losses.append(running_loss / len(train_loader))
        
        val_loss = evaluate_model(model, val_loader, pos_crit, con_crit, config)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config, scalers)
            best_epoch = epoch+1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        obar.set_postfix({"Train Loss": running_loss/len(train_loader), "Val Loss": val_loss, "Lr": scheduler.get_last_lr()[0], "best_epoch": best_epoch})

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return train_losses, val_losses


# === Full Pipeline ===
def run_pipeline(config, plot=False):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Preparing dataset...")
    long_dataset = RacingLineDataset(
        config,
        config["long_seq_len"],
    )
    medium_dataset = RacingLineDataset(
        config,
        config["med_seq_len"],
    )
    short_dataset = RacingLineDataset(
        config,
        config["short_seq_len"],
    )

    datasets = [long_dataset, medium_dataset, short_dataset]
    scaler_x = long_dataset.scaler_x
    scaler_y = long_dataset.scaler_y
    scaler_z = long_dataset.scaler_z

    scalers = (scaler_x, scaler_y, scaler_z)
    print("Total sequences loaded:", len(long_dataset))
    
    loaders = []
    
    for dataset in datasets:
        train_len = int(len(dataset) * config["train_split"])
        val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(config["seed"])
        )
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
        loaders.append([train_loader, val_loader])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    print(f"Initializing model...")
    model = Cerberus(config=config).to(config["device"])

    print("Training started...")
    train_losses, val_losses = train_model(model, loaders[0], loaders[1], loaders[2], config, scalers)
    print(f"Training complete. Model saved to {config['model_save_path']}")

    if plot:
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
        is_circular = is_circular_track(df, config["output_pos_cols"])
        X = df[config["input_cols"]].values
        Y = df[config["output_pos_cols"]].values
        X_scaled = scaler_x.transform(X)
        n = len(X_scaled)
        preds_real = np.zeros_like(Y)
        trues_real = Y.copy()

        for i in tqdm(range(n), desc=f"[{layout_index + 1}/{total_layouts}]"):
            seq = get_centered_sequence(X_scaled, i, config["seq_len"], is_circular)
            X_tensor = torch.tensor(seq.reshape(1, config["seq_len"], -1), dtype=torch.float32).to(config["device"])

            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().squeeze().numpy()
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
        print_feature_accuracy(preds_real, trues_real, scaler_y, config["output_pos_cols"])
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