#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     Defines the Cerberus model for racing line and control prediction.
#     Cerberus uses a lightweight residual 1D convolutional encoder and 
#     separate MLP heads for position and control outputs.
#
# File Overview:
#     This file defines the Cerberus model class and provides functions 
#     to save and load trained model checkpoints.
#
# Classes and Functions Included:
#     - Cerberus: Neural network model with encoder, position head, and control head.
#     - save_model(): Save the model checkpoint along with config and scalers.
#     - load_model(): Load the model checkpoint and restore scalers.
#====================================================


# === Imports ===
import torch as t


# === Cerberus Model Definition ===
"""
    Neural network for simultaneous racing line (position) and control prediction.

    Architecture:
        - 1D convolutional encoder with residual blocks (inspired by ResNet).
        - Position head: MLP predicting (x, y, z) target position.
        - Control head: MLP predicting (speed, gas, brake, side-left, side-right) based on encoded features and position.

    Args:
        config (dict): Configuration dictionary specifying input size, hidden sizes, kernel sizes, etc.

    Methods:
        forward(x):
            Forward pass through the network.
            Args:
                x (torch.Tensor): Input tensor of shape [batch_size, n_features, seq_len].
            Returns:
                tuple: (position tensor, control tensor)
"""
class Cerberus (t.nn.Module):
    def __init__(self, config):
        super().__init__()

        #Thanks Resnet!
        class ResBlock(t.nn.Module):
            def __init__(self, in_chans, out_chans, kern_size, padding, dilation):
                super().__init__()
                self.conv1 = t.nn.Conv1d(in_chans, out_chans, kern_size, padding=padding, dilation=dilation)
                self.relu = t.nn.ReLU()
                self.conv2 = t.nn.Conv1d(out_chans, out_chans, kern_size, padding=padding, dilation=dilation)
    
                self.shortcut = t.nn.Identity()
                if in_chans != out_chans:
                    self.shortcut = t.nn.Conv1d(in_chans, out_chans, kernel_size=1)
                
            def forward(self, x):
                residual = self.shortcut(x)
                return self.relu(self.conv2(self.relu(self.conv1(x))) + residual)
    
        self.encoder = t.nn.Sequential(
            ResBlock(config["input_size"], config["hidden1"], config["kern_size1"], config["padding1"], config["dilation1"]),
            ResBlock(config["hidden1"], config["hidden1"], config["kern_size2"], config["padding2"], config["dilation2"])
        )

        #x, y, z
        self.position_head = t.nn.Sequential(
            t.nn.Linear(config["hidden1"], config["pos_head_sz"]),
            t.nn.ReLU(),
            t.nn.Linear(config["pos_head_sz"], 3)
        )

        #speed, gas, brake, side-left, side-right
        self.control_head = t.nn.Sequential(
            t.nn.Linear(config["hidden1"] + 3, config["cont_head_sz"]),
            t.nn.ReLU(),
            t.nn.Linear(config["cont_head_sz"], 5)
        )


    def forward(self, x):
        encoded = self.encoder(x)
        encoded = t.mean(encoded, dim=2)

        position = self.position_head(encoded)
        control_in = t.cat([encoded, position], dim=1)
        control = self.control_head(control_in)

        return position, control
    

# === save_model ===
"""
    Save the model checkpoint including weights, configuration, and scalers.

    Args:
        model (Cerberus): Trained model to save.
        config (dict): Configuration dictionary.
        scalers (tuple): Tuple containing (scaler_x, scaler_y, scaler_z).

    Returns:
"""
def save_model(model, config, scalers):
    t.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler_x": scalers[0],
        "scaler_y": scalers[1],
        "scaler_z": scalers[2],
    }, config["model_save_path"])


# === load_model ===
"""
    Load a saved model checkpoint including weights and scalers.

    Args:
        path (str): Path to the saved checkpoint file.
        device (torch.device or str): Device to map the model to (e.g., "cuda" or "cpu").

    Returns:
        tuple: (Cerberus model, config dictionary, scaler_x, scaler_y, scaler_z)
"""
def load_model(path, device):
    checkpoint = t.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    scaler_x = checkpoint["scaler_x"]
    scaler_y = checkpoint["scaler_y"]
    scaler_z = checkpoint["scaler_z"]
    model = Cerberus(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg["device"])
    return model, cfg, scaler_x, scaler_y, scaler_z