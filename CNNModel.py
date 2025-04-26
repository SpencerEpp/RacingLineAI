#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     Defines the RoboTurbosky convolutional neural network (CNN) 
#     model architecture for racing line prediction and control output 
#     (speed, gas, brake) estimation from track image patches and 
#     coordinate metadata.
#
# File Overview:
#     This file provides functionality to:
#       - Dynamically build the RoboTurbosky model based on a configuration dictionary.
#       - Save trained model weights and architecture config to disk.
#       - Load trained models for inference or retraining.
#
# Functions Included:
#     - RoboTurbosky: CNN model class for predicting racing line and control outputs.
#     - save_model(): Save model state and configuration to a file.
#     - load_model(): Load model state and configuration from a file.
#====================================================

# === Imports ===
import torch as t


# === Model Definition ===
"""
    RoboTurbosky CNN model for racing line prediction and control output estimation.

    Args:
        config (dict): Model configuration dictionary containing:
            - cnn_channels (list): Output channels for each CNN layer.
            - coord_dims (list): Hidden dimensions for the coordinate MLP.
            - combined_dim (int): Dimension of combined CNN + coordinate feature vector.
            - control_head_dim (int): Hidden size for control head.
            - position_head_dim (int): Hidden size for position head.

    Forward Inputs:
        patch (torch.Tensor): Track image patches of shape (batch_size, 1, H, W).
        coord (torch.Tensor): Centerline coordinates of shape (batch_size, 2).

    Forward Outputs:
        tuple:
            - control (torch.Tensor): Predicted speed, gas, brake (batch_size, 3).
            - position (torch.Tensor): Predicted x, z positions (batch_size, 2).
"""
class RoboTurbosky(t.nn.Module):
    def __init__(self, config):
        super().__init__()

        cnn_channels = config.get("cnn_channels", [32, 64, 128])
        coord_dims = config.get("coord_dims", [32, 64])
        combined_dim = config.get("combined_dim", 320)
        control_head_dim = config.get("control_head_dim", 128)
        position_head_dim = config.get("position_head_dim", 128)
        meta_dims = config.get("meta_dims", [32, 64])

        self.cnn = t.nn.Sequential(
            t.nn.Conv2d(1, cnn_channels[0], kernel_size=3, padding=1, stride=1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2),
            t.nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2),
            t.nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            t.nn.ReLU(),
            t.nn.AdaptiveAvgPool2d((1,1)),
            t.nn.Flatten()
        )

        self.coord = t.nn.Sequential(
            t.nn.Linear(2, coord_dims[0]),
            t.nn.ReLU(),
            t.nn.Linear(coord_dims[0], coord_dims[1]),
            t.nn.ReLU()
        )

        self.meta = t.nn.Sequential(
            t.nn.Linear(11, meta_dims[0]),
            t.nn.ReLU(),
            t.nn.Linear(meta_dims[0], meta_dims[1]),
            t.nn.ReLU(),
        )

        self.combined = t.nn.Sequential(
            t.nn.Linear(cnn_channels[2] + coord_dims[1] + meta_dims[1], combined_dim),
            t.nn.ReLU()
        )

        #speed, gas, brake,
        self.control_head = t.nn.Sequential(
            t.nn.Linear(combined_dim, control_head_dim),
            t.nn.ReLU(),
            t.nn.Linear(control_head_dim, 3)
        )

        #x, z
        self.position_head = t.nn.Sequential(
            t.nn.Linear(combined_dim + 3, position_head_dim),
            t.nn.ReLU(),
            t.nn.Linear(position_head_dim, 2)
        )

    def forward(self, patch, coord, meta=None):
        x_patch = self.cnn(patch)
        x_coord = self.coord(coord)
        x_meta = self.meta(meta)
        x = t.cat([x_patch, x_coord, x_meta], dim=1)
        x_combined = self.combined(x)

        control = self.control_head(x_combined)
        pos_in = t.cat([x_combined, control], dim=1)
        position = self.position_head(pos_in)

        return control, position
    

# === Save and Load Model ===
"""
    Save a trained model and its configuration to disk.

    Args:
        model (torch.nn.Module): Trained model to save.
        config (dict): Configuration dictionary associated with the model.
        temp_save (str, optional): Optional temporary save path. 
            If None, saves to config["model_save_path"].

    Returns:
        None
"""
def save_model(model, config, temp_save=None):
    if temp_save is None:
        save_name = config["model_save_path"]
    else:
        save_name = temp_save
    t.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, save_name)


"""
    Load a trained model and its configuration from disk.

    Args:
        path (str): Path to the saved model file (.pt).
        device (str, optional): Device to map the model to ("cuda" or "cpu").

    Returns:
        tuple:
            - model (torch.nn.Module): Loaded RoboTurbosky model.
            - config (dict): Configuration dictionary used to build the model.
"""
def load_model(path, device="cuda"):
    checkpoint = t.load(path, map_location=device)
    config = checkpoint["config"]
    model = RoboTurbosky(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config