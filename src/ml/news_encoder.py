"""
News encoder module using LFM2 foundation model.

Provides functionality to encode news headlines into fixed-size embeddings
for use in meta-model prediction. Supports two modes:
- backtest_no_news: Returns zero vectors (for backtesting without news)
- live_with_news: Encodes actual news using LFM2

Uses Liquid AI's LFM2-350M model as frozen encoder.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from datetime import datetime


try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class NewsEncoder:
    """
    Encodes news headlines using LFM2 foundation model.

    For backtest mode: Returns zero vectors (news disabled)
    For live mode: Fetches and encodes news using LFM2
    """

    def __init__(self,
                 model_name='LiquidAI/LFM2-350M',
                 mode='backtest_no_news',
                 device='cpu',
                 embedding_dim=768):
        """
        Initialize NewsEncoder.

        Args:
            model_name: Hugging Face model identifier for LFM2
            mode: 'backtest_no_news' or 'live_with_news'
            device: 'cpu', 'cuda', or 'mps'
            embedding_dim: Output embedding dimension (768 for LFM2-350M)
        """
        self.mode = mode
        self.device = device
        self.embedding_dim = embedding_dim
        self.model_name = model_name

        self.model = None
        self.tokenizer = None

        # Only load model if in live mode
        if mode == 'live_with_news':
            if not HAS_TRANSFORMERS:
                raise ImportError(
                    "transformers library required for live_with_news mode. "
                    "Install with: pip install transformers"
                )

            print(f"Loading LFM2 news encoder: {model_name}")
            try:
                # Load frozen LFM2 encoder
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model = self.model.to(device)
                self.model.eval()

                # Freeze all parameters (don't train LFM2)
                for param in self.model.parameters():
                    param.requires_grad = False

                print(f"  ✓ LFM2 loaded and frozen ({sum(p.numel() for p in self.model.parameters()):,} parameters)")

            except Exception as e:
                print(f"  ✗ Error loading LFM2: {e}")
                print(f"  Falling back to backtest_no_news mode")
                self.mode = 'backtest_no_news'

    def encode_headlines(self,
                        headlines: List[str],
                        timestamp: Optional[datetime] = None) -> Tuple[torch.Tensor, float]:
        """
        Encode list of news headlines into fixed-size embedding.

        Args:
            headlines: List of headline strings
            timestamp: Current prediction time (optional, for filtering)

        Returns:
            news_vec: [embedding_dim] tensor (e.g., 768-dim)
            news_mask: 1.0 if headlines exist, 0.0 otherwise
        """
        # Backtest mode or no headlines: return zeros
        if self.mode == 'backtest_no_news' or not headlines or len(headlines) == 0:
            return torch.zeros(self.embedding_dim, dtype=torch.float32), 0.0

        # Live mode with news
        try:
            # Concatenate headlines (take top 10 to avoid token limit)
            combined_text = " | ".join(headlines[:10])

            # Tokenize
            inputs = self.tokenizer(
                combined_text,
                return_tensors='pt',
                max_length=512,  # LFM2 context length
                truncation=True,
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings from LFM2
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Use CLS token embedding (or mean pooling)
                # LFM2 output: last_hidden_state shape [batch, seq_len, hidden_dim]
                if hasattr(outputs, 'last_hidden_state'):
                    # Option 1: Use CLS token (first token)
                    news_vec = outputs.last_hidden_state[:, 0, :]

                    # Option 2: Mean pooling (alternative)
                    # news_vec = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, 'pooler_output'):
                    news_vec = outputs.pooler_output
                else:
                    # Fallback: use first output
                    news_vec = outputs[0][:, 0, :]

                news_vec = news_vec.squeeze(0).cpu()  # [hidden_dim]

                # Ensure correct dimension
                if news_vec.shape[0] != self.embedding_dim:
                    # Project to target dimension if needed
                    if not hasattr(self, 'projection'):
                        self.projection = nn.Linear(news_vec.shape[0], self.embedding_dim)
                        self.projection = self.projection.to(self.device)
                    news_vec = self.projection(news_vec.unsqueeze(0)).squeeze(0).cpu()

            return news_vec, 1.0

        except Exception as e:
            print(f"Warning: Error encoding news: {e}")
            # Fallback to zeros on error
            return torch.zeros(self.embedding_dim, dtype=torch.float32), 0.0

    def encode_batch(self,
                    headlines_batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode batch of headline lists.

        Args:
            headlines_batch: List of headline lists [[h1, h2], [h3, h4], ...]

        Returns:
            news_vecs: [batch, embedding_dim] tensor
            news_masks: [batch] tensor
        """
        news_vecs = []
        news_masks = []

        for headlines in headlines_batch:
            vec, mask = self.encode_headlines(headlines)
            news_vecs.append(vec)
            news_masks.append(mask)

        news_vecs = torch.stack(news_vecs)  # [batch, embedding_dim]
        news_masks = torch.tensor(news_masks, dtype=torch.float32)  # [batch]

        return news_vecs, news_masks


class NewsGate(nn.Module):
    """
    Optional gating mechanism for news modality.

    Dynamically up-weights or down-weights news input based on
    learned importance. Can help model decide when to trust news signal.
    """

    def __init__(self, news_vec_dim=768, hidden_dim=64):
        """
        Initialize news gate.

        Args:
            news_vec_dim: Dimension of news embeddings
            hidden_dim: Hidden dimension for gating MLP
        """
        super().__init__()

        self.gate_mlp = nn.Sequential(
            nn.Linear(news_vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output 0-1 gate value
        )

    def forward(self, news_vec, news_mask):
        """
        Compute gate value and apply to news vector.

        Args:
            news_vec: [batch, news_vec_dim] - News embeddings
            news_mask: [batch, 1] - News availability (1 or 0)

        Returns:
            gated_news_vec: [batch, news_vec_dim] - Gated news embeddings
        """
        # Compute gate value
        gate = self.gate_mlp(news_vec)  # [batch, 1]

        # Apply mask (gate is 0 if no news available)
        gate = gate * news_mask

        # Apply gate to news vector
        gated_news_vec = news_vec * gate

        return gated_news_vec


# Convenience function
def create_news_encoder(mode='backtest_no_news', device='cpu'):
    """
    Factory function to create NewsEncoder.

    Args:
        mode: 'backtest_no_news' or 'live_with_news'
        device: Device to load model on

    Returns:
        NewsEncoder instance
    """
    return NewsEncoder(
        model_name='LiquidAI/LFM2-350M',
        mode=mode,
        device=device,
        embedding_dim=768
    )


__all__ = [
    'NewsEncoder',
    'NewsGate',
    'create_news_encoder'
]
