"""
Liquid Neural Network (LNN) model implementation
Uses ncps library for efficient continuous-time RNN
"""

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config
from .base import ModelBase


class LNNTradingModel(nn.Module, ModelBase):
    """
    Liquid Neural Network for trading predictions
    Uses CfC (Closed-form Continuous-time) for efficiency
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 output_size: int = 2, num_layers: int = 2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Create structured wiring for sparse connections
        # This makes the network interpretable and efficient
        self.wiring = AutoNCP(hidden_size, output_size)

        # CfC layer (Liquid Time-Constant Network)
        # Note: wiring is passed as positional arg, not keyword
        self.lnn = CfC(input_size, self.wiring)

        # Output layers for predictions
        self.fc_out = nn.Linear(output_size, output_size)

        # Confidence head (predicts uncertainty)
        self.fc_confidence = nn.Linear(output_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Optimizer (will be set during training)
        self.optimizer = None

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LNN
        Args:
            x: (batch, sequence_length, input_size)
            h: hidden state (batch, hidden_size) or None
        Returns:
            output: (batch, output_size) - [predicted_high, predicted_low]
            hidden: (batch, hidden_size) - final hidden state
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch_size, self.wiring.units, device=x.device)

        # CfC processes the entire sequence at once
        # Input: (batch, sequence, features), Hidden: (batch, units)
        # Output: (batch, sequence, output_size), Hidden: (batch, units)
        output, h = self.lnn(x, h)

        # Use the last timestep's output for prediction
        if len(output.shape) == 3:  # (batch, sequence, output_size)
            final_output = output[:, -1, :]  # Get last timestep
        else:
            final_output = output

        # Predict high/low
        predictions = self.fc_out(final_output)

        return predictions, h

    def predict(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Generate predictions with probabilities and confidence
        Args:
            x: (batch, sequence_length, input_size)
        Returns:
            dict with predictions, confidence, and probabilities
        """
        # Ensure input is on same device as model
        device = next(self.parameters()).device
        x = x.to(device)
        if h is not None:
            h = h.to(device)

        self.eval()
        with torch.no_grad():
            predictions, hidden = self.forward(x, h)

            # Confidence score
            confidence_logit = self.fc_confidence(predictions)
            confidence = self.sigmoid(confidence_logit).squeeze(-1)

            # Predicted high and low
            pred_high = predictions[:, 0]
            pred_low = predictions[:, 1]

            # Calculate prediction ranges and probabilities
            pred_center = (pred_high + pred_low) / 2
            pred_range = pred_high - pred_low

            return {
                'predicted_high': pred_high.cpu().numpy(),
                'predicted_low': pred_low.cpu().numpy(),
                'predicted_center': pred_center.cpu().numpy(),
                'predicted_range': pred_range.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'hidden_state': hidden
            }

    def save_checkpoint(self, path: str, metadata: Dict = None):
        """Save model checkpoint with metadata"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
        }

        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        if metadata:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, path)
        print(f"Model checkpoint saved to {path}")

    def load_checkpoint(self, path: str, device: Optional[torch.device] = None) -> Dict:
        """Load model checkpoint and return metadata

        Args:
            path: Path to checkpoint file
            device: Optional device to load model to (defaults to current device or CPU)
        """
        if device is None:
            # Try to use current device if model is already on one
            try:
                device = next(self.parameters()).device
            except:
                device = torch.device('cpu')

        # Load checkpoint with proper device mapping
        checkpoint = torch.load(path, map_location=device)

        # Verify architecture matches
        assert checkpoint['input_size'] == self.input_size, "Input size mismatch"
        assert checkpoint['hidden_size'] == self.hidden_size, "Hidden size mismatch"

        self.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move model to device
        self.to(device)

        # Log device information
        saved_device = checkpoint.get('metadata', {}).get('device', 'unknown')
        print(f"Model checkpoint loaded from {path}")
        print(f"  Saved on: {saved_device}")
        print(f"  Loaded to: {device}")

        return checkpoint.get('metadata', {})

    def update_online(self, x: torch.Tensor, y: torch.Tensor, lr: float = 0.0001):
        """
        Perform online learning update
        Used for incremental learning from new data
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.train()

        # Forward pass
        predictions, _ = self.forward(x)

        # Loss: MSE for high/low predictions
        loss = nn.functional.mse_loss(predictions, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class SelfSupervisedPretrainer:
    """
    Self-supervised pretraining for LNN
    Uses masking and reconstruction tasks
    """

    def __init__(self, model: nn.Module, mask_ratio: float = 0.15):
        self.model = model
        self.mask_ratio = mask_ratio

        # Reconstruction head - ensure it's on the same device as the model
        self.reconstruction_head = nn.Linear(model.output_size, model.input_size)

        # Move reconstruction head to same device as model
        device = next(model.parameters()).device
        self.reconstruction_head = self.reconstruction_head.to(device)

    def mask_sequences(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask portions of input sequences
        Returns: (masked_x, mask_indices)
        """
        batch_size, seq_len, feature_dim = x.shape

        # Create mask (1 = keep, 0 = mask) - ensure it's on same device as input
        mask = torch.rand(batch_size, seq_len, feature_dim, device=x.device) > self.mask_ratio

        masked_x = x.clone()
        masked_x[~mask] = 0  # Zero out masked positions

        return masked_x, mask

    def pretrain_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Single pretraining step using masking
        Model learns to reconstruct masked features
        """
        self.model.train()

        # Mask input
        masked_x, mask = self.mask_sequences(x)

        # Forward pass through model
        predictions, _ = self.model.forward(masked_x)

        # Reconstruct original features
        reconstructed = self.reconstruction_head(predictions)

        # Loss: only on masked positions
        # Average feature values across sequence for reconstruction target
        target = x.mean(dim=1)  # (batch, feature_dim)

        loss = nn.functional.mse_loss(reconstructed, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


class LSTMTradingModel(nn.Module, ModelBase):
    """
    Alternative LSTM-based model for comparison
    Can be swapped in via config
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 output_size: int = 2, num_layers: int = 2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)

        # Output layers
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.fc_confidence = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.optimizer = None

    def forward(self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through LSTM"""
        # LSTM forward
        output, (hidden, cell) = self.lstm(x, h)

        # Use final timestep output
        final_output = output[:, -1, :]

        # Predictions
        predictions = self.fc_out(final_output)

        return predictions, (hidden, cell)

    def predict(self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """Generate predictions"""
        # Ensure input is on same device as model
        device = next(self.parameters()).device
        x = x.to(device)
        if h is not None:
            h = (h[0].to(device), h[1].to(device))

        self.eval()
        with torch.no_grad():
            predictions, hidden = self.forward(x, h)

            # Confidence
            confidence_logit = self.fc_confidence(predictions)
            confidence = self.sigmoid(confidence_logit).squeeze(-1)

            pred_high = predictions[:, 0]
            pred_low = predictions[:, 1]
            pred_center = (pred_high + pred_low) / 2
            pred_range = pred_high - pred_low

            return {
                'predicted_high': pred_high.cpu().numpy(),
                'predicted_low': pred_low.cpu().numpy(),
                'predicted_center': pred_center.cpu().numpy(),
                'predicted_range': pred_range.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'hidden_state': hidden
            }

    def save_checkpoint(self, path: str, metadata: Dict = None):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
        }

        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        if metadata:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, device: Optional[torch.device] = None) -> Dict:
        """Load model checkpoint

        Args:
            path: Path to checkpoint file
            device: Optional device to load model to (defaults to current device or CPU)
        """
        if device is None:
            # Try to use current device if model is already on one
            try:
                device = next(self.parameters()).device
            except:
                device = torch.device('cpu')

        # Load checkpoint with proper device mapping
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move model to device
        self.to(device)

        # Log device information
        saved_device = checkpoint.get('metadata', {}).get('device', 'unknown')
        print(f"Model checkpoint loaded from {path}")
        print(f"  Saved on: {saved_device}")
        print(f"  Loaded to: {device}")

        return checkpoint.get('metadata', {})

    def update_online(self, x: torch.Tensor, y: torch.Tensor, lr: float = 0.0001):
        """Online learning update"""
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.train()
        predictions, _ = self.forward(x)
        loss = nn.functional.mse_loss(predictions, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def create_model(model_type: str = None, input_size: int = 50,
                hidden_size: int = None, **kwargs) -> ModelBase:
    """
    Factory function to create models
    Allows easy swapping between LNN and LSTM
    """
    model_type = model_type or config.ML_MODEL_TYPE
    hidden_size = hidden_size or config.LNN_HIDDEN_SIZE

    if model_type == "LNN":
        return LNNTradingModel(input_size, hidden_size, **kwargs)
    elif model_type == "LSTM":
        return LSTMTradingModel(input_size, hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
