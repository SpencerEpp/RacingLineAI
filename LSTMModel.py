#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     Neural network model and utilities for predicting optimal
#     racing lines based on track metadata. Implements an LSTM 
#     backbone with attention mechanism to focus on key parts 
#     of the input sequence.
#
# File Overview:
#     This file defines the RacingLineLSTMWithAttention model class
#     and provides functions to save and load trained model checkpoints.
#
# Classes and Functions Included:
#     - RacingLineLSTMWithAttention: LSTM model with attention mechanism.
#     - save_model(): Save model checkpoint with scalers and config.
#     - load_model(): Load model checkpoint and reconstruct model/scalers.
#====================================================


# === Imports ===
import torch as t


# === Model Definition ===
"""
    LSTM model with attention mechanism for racing line prediction.

    Args:
        config (dict): Model configuration dictionary containing:
            - input_size (int): Number of input features.
            - hidden_size (int): Hidden dimension size for LSTM.
            - num_layers (int): Number of LSTM layers.
            - dropout (float): Dropout probability.
            - output_size (int): Number of output features (typically 3: x, y, z).
            - bidirectional (bool): Whether to use bidirectional LSTM.
        scaler_x (MinMaxScaler, optional): Input feature scaler.
        scaler_y (MinMaxScaler, optional): Output feature scaler.
"""
class RacingLineLSTMWithAttention(t.nn.Module):
    def __init__(self, config, scaler_x=None, scaler_y=None):
        super().__init__()
        self.bidirectional = config["bidirectional"]
        self.hidden_size = config["hidden_size"]
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = t.nn.LSTM(
            input_size=config["input_size"],
            hidden_size=self.hidden_size,
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.attn = t.nn.Linear(self.num_directions * self.hidden_size, 1)
        self.dropout = t.nn.Dropout(config["dropout"])
        self.fc = t.nn.Linear(self.num_directions * self.hidden_size, config["output_size"])
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn(lstm_out)
        attn_weights = t.softmax(attn_scores, dim=1)
        context = t.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        return self.fc(context)

    def get_attention_weights(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn(lstm_out)
        attn_weights = t.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)
    

# === Save and Load Model ===
"""
    Save model checkpoint with associated config and scalers.

    Args:
        model (torch.t.nn.Module): Model to save.
        config (dict): Training config dictionary.
        scaler_x (MinMaxScaler): Input scaler.
        scaler_y (MinMaxScaler): Output scaler.
        profiler (Profiler, optional): Profiler for measuring save time.

    Returns:
        None
"""
def save_model(model, config, scaler_x, scaler_y, profiler=None):
    if profiler: profiler.start("Save Model")
    t.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
    }, config["model_save_path"])
    if profiler: profiler.stop("Save Model")


"""
    Load a model checkpoint and reconstruct model and scalers.

    Args:
        path (str): Path to the saved checkpoint.
        device (torch.device or str): Device to map model to.
        profiler (Profiler, optional): Profiler for measuring load time.

    Returns:
        tuple: (model, config, scaler_x, scaler_y)
"""
def load_model(path, device, profiler=None):
    if profiler: profiler.start("Load Model")
    checkpoint = t.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    scaler_x = checkpoint["scaler_x"]
    scaler_y = checkpoint["scaler_y"]

    model = RacingLineLSTMWithAttention(cfg, scaler_x, scaler_y)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg["device"])
    if profiler: profiler.stop("Load Model")
    return model, cfg, scaler_x, scaler_y