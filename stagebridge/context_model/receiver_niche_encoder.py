"""Receiver-centered local niche encoder per StageBridge doctrine.

This module implements the local neighborhood encoder as specified in
docs/architecture/typed_niche_context_model.md. The key principle is RECEIVER-CENTERING:
the focal cell (receiver) is the query, neighbors are keys/values,
and information flows TO the receiver.

Design principles enforced:
1. Receiver-centered architecture (receiver as query)
2. Distance-aware attention (explicit spatial modulation)
3. Sparsity/entropy regularization
4. Neighbor ablation for interpretability
5. Masked receiver reconstruction as self-supervised signal
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DistanceEncoding(StrEnum):
    """Distance encoding strategies."""

    RBF = "rbf"  # Radial basis function
    MLP = "mlp"  # Learned MLP
    SINUSOIDAL = "sinusoidal"  # Sinusoidal encoding


class SparsityType(StrEnum):
    """Attention sparsity strategies."""

    ENTROPY = "entropy"  # Entropy penalty in loss
    TOPK = "topk"  # Hard top-k selection
    SPARSEMAX = "sparsemax"  # Sparsemax projection


@dataclass(slots=True, frozen=True)
class ReceiverNicheOutput:
    """Output from the receiver-centered niche encoder.

    Attributes:
        context: [B, D] - What the receiver gets from its neighborhood
        attention_weights: [B, K] - Interpretable neighbor importance scores
        entropy_loss: Scalar - Attention entropy for regularization (if computed)
        receiver_reconstruction: [B, D] - Reconstructed receiver (if decoder present)
    """

    context: Tensor
    attention_weights: Tensor
    entropy_loss: Tensor | None = None
    receiver_reconstruction: Tensor | None = None


def _rbf_distance_encoding(
    distances: Tensor, num_rbf: int = 16, max_dist: float = 100.0
) -> Tensor:
    """Radial basis function encoding of distances.

    Args:
        distances: [B, K] pairwise distances
        num_rbf: Number of RBF centers
        max_dist: Maximum distance for RBF centers

    Returns:
        [B, K, num_rbf] RBF features
    """
    # RBF centers evenly spaced from 0 to max_dist
    centers = torch.linspace(0, max_dist, num_rbf, device=distances.device, dtype=distances.dtype)
    # Width of each RBF
    width = max_dist / num_rbf

    # [B, K, 1] - [num_rbf] -> [B, K, num_rbf]
    diff = distances.unsqueeze(-1) - centers
    rbf = torch.exp(-0.5 * (diff / width) ** 2)

    return rbf


def _sinusoidal_distance_encoding(distances: Tensor, dim: int = 16) -> Tensor:
    """Sinusoidal encoding of distances (like positional encoding).

    Args:
        distances: [B, K] pairwise distances
        dim: Encoding dimension

    Returns:
        [B, K, dim] sinusoidal features
    """
    half_dim = dim // 2
    freq = torch.exp(
        torch.arange(half_dim, device=distances.device, dtype=distances.dtype)
        * (-math.log(10000.0) / half_dim)
    )

    # [B, K, 1] * [half_dim] -> [B, K, half_dim]
    phase = distances.unsqueeze(-1) * freq
    encoding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

    return encoding


def _sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax activation (projects to simplex with sparsity).

    From "From Softmax to Sparsemax" (Martins & Astudillo, 2016).

    Args:
        logits: Input logits
        dim: Dimension to apply sparsemax

    Returns:
        Sparse probability distribution
    """
    # Sort in descending order
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)

    # Compute cumsum
    cumsum = torch.cumsum(sorted_logits, dim=dim)

    # Find k (number of non-zero elements)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    k = k.view([1] * (logits.dim() - 1) + [-1])

    # Check condition: 1 + k * z_k > cumsum
    condition = 1 + k * sorted_logits > cumsum

    # Find largest k satisfying condition
    k_max = condition.sum(dim=dim, keepdim=True).clamp(min=1)

    # Compute threshold tau
    cumsum_at_k = cumsum.gather(dim, (k_max - 1).long())
    tau = (cumsum_at_k - 1) / k_max.float()

    # Compute sparsemax output
    output = (logits - tau).clamp(min=0)

    return output


def _compute_attention_entropy(attention_weights: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute entropy of attention distribution for regularization.

    Lower entropy = more focused attention = encouraged by sparsity loss.

    Args:
        attention_weights: [B, K] attention probabilities
        eps: Small constant for numerical stability

    Returns:
        Scalar entropy averaged over batch
    """
    K = attention_weights.size(-1)

    # Handle edge case of single neighbor
    if K <= 1:
        return torch.tensor(0.0, device=attention_weights.device, dtype=attention_weights.dtype)

    # Entropy: -sum(p * log(p))
    log_attn = torch.log(attention_weights + eps)
    entropy = -torch.sum(attention_weights * log_attn, dim=-1)

    # Normalize by max entropy (uniform distribution)
    max_entropy = math.log(K)
    normalized_entropy = entropy / max_entropy

    return normalized_entropy.mean()


class DistanceEncoder(nn.Module):
    """Encode spatial distances into features for attention modulation."""

    def __init__(
        self,
        encoding_type: DistanceEncoding | str = DistanceEncoding.RBF,
        output_dim: int = 16,
        max_distance: float = 100.0,
    ):
        super().__init__()
        self.encoding_type = DistanceEncoding(encoding_type)
        self.output_dim = output_dim
        self.max_distance = max_distance

        if self.encoding_type == DistanceEncoding.MLP:
            self.mlp = nn.Sequential(
                nn.Linear(1, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
        elif self.encoding_type == DistanceEncoding.RBF:
            # RBF -> linear projection
            self.proj = nn.Linear(output_dim, output_dim)
        # Sinusoidal doesn't need learnable params

    def forward(self, distances: Tensor) -> Tensor:
        """Encode distances.

        Args:
            distances: [B, K] pairwise distances from receiver to neighbors

        Returns:
            [B, K, output_dim] distance features
        """
        if self.encoding_type == DistanceEncoding.MLP:
            return self.mlp(distances.unsqueeze(-1))
        elif self.encoding_type == DistanceEncoding.RBF:
            rbf = _rbf_distance_encoding(distances, self.output_dim, self.max_distance)
            return self.proj(rbf)
        else:  # SINUSOIDAL
            return _sinusoidal_distance_encoding(distances, self.output_dim)


class ReceiverCenteredAttention(nn.Module):
    """Cross-attention where receiver is query, neighbors are keys/values.

    This is the core of receiver-centered niche encoding. The receiver
    cell attends to its neighbors, with distance modulating attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        distance_encoding: DistanceEncoding | str = DistanceEncoding.RBF,
        sparsity_type: SparsityType | str = SparsityType.ENTROPY,
        topk: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sparsity_type = SparsityType(sparsity_type)
        self.topk = topk

        # Query projection for receiver
        self.q_proj = nn.Linear(dim, dim)
        # Key/value projections for neighbors
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Distance encoding
        self.distance_encoder = DistanceEncoder(
            encoding_type=distance_encoding,
            output_dim=num_heads,  # One bias per head
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        neighbor_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Receiver attends to neighbors with distance modulation.

        Args:
            receiver: [B, D] receiver cell embedding
            neighbors: [B, K, D] neighbor cell embeddings
            distances: [B, K] distances from receiver to each neighbor
            neighbor_mask: [B, K] boolean, True = valid neighbor, False = masked/ablated

        Returns:
            context: [B, D] aggregated context from neighborhood
            attention_weights: [B, K] interpretable attention weights
        """
        B, K, _ = neighbors.shape

        # Project receiver to query: [B, 1, D]
        q = self.q_proj(receiver).unsqueeze(1)
        # Project neighbors to keys and values: [B, K, D]
        k = self.k_proj(neighbors)
        v = self.v_proj(neighbors)

        # Reshape for multi-head attention
        # [B, 1, num_heads, head_dim] -> [B, num_heads, 1, head_dim]
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: [B, num_heads, 1, K]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add distance bias: [B, K, num_heads] -> [B, num_heads, 1, K]
        distance_bias = self.distance_encoder(distances)  # [B, K, num_heads]
        distance_bias = distance_bias.permute(0, 2, 1).unsqueeze(2)  # [B, num_heads, 1, K]
        attn_logits = attn_logits + distance_bias

        # Apply neighbor mask (ablation support)
        if neighbor_mask is not None:
            # Expand mask: [B, K] -> [B, 1, 1, K]
            mask = neighbor_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        # Apply sparsity mechanism
        if self.sparsity_type == SparsityType.TOPK:
            # Keep only top-k attention scores per head
            # attn_logits: [B, num_heads, 1, K]
            k_actual = min(self.topk, K)
            topk_values, topk_indices = torch.topk(attn_logits, k_actual, dim=-1)
            sparse_logits = torch.full_like(attn_logits, float("-inf"))
            sparse_logits.scatter_(-1, topk_indices, topk_values)
            attn_weights = F.softmax(sparse_logits, dim=-1)
        elif self.sparsity_type == SparsityType.SPARSEMAX:
            # Sparsemax for sparse attention - apply per head
            # Reshape: [B, num_heads, 1, K] -> [B*num_heads, K]
            logits_flat = attn_logits.squeeze(2).view(-1, K)
            sparse_flat = _sparsemax(logits_flat, dim=-1)
            attn_weights = sparse_flat.view(B, self.num_heads, 1, K)
        else:  # ENTROPY - standard softmax, regularize via loss
            attn_weights = F.softmax(attn_logits, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # Aggregate values: [B, num_heads, 1, head_dim]
        context = torch.matmul(attn_weights, v)

        # Reshape back: [B, 1, D]
        context = context.transpose(1, 2).contiguous().view(B, 1, self.dim)
        context = self.out_proj(context).squeeze(1)  # [B, D]

        # Return mean attention weights across heads for interpretability
        attn_weights_mean = attn_weights.squeeze(2).mean(dim=1)  # [B, K]

        return context, attn_weights_mean


class ReceiverCenteredNicheEncoder(nn.Module):
    """Receiver-centered local neighborhood encoder per doctrine.

    This encoder models "what does this cell receive from its neighbors?"
    by using the receiver cell as the attention query and neighbors as
    keys/values, with explicit distance modulation and sparsity regularization.

    Implements all requirements from typed_niche_context_model.md:
    - Receiver-centered architecture
    - Distance-aware attention
    - Sparsity/entropy regularization
    - Neighbor ablation interface
    - Optional masked receiver reconstruction

    Args:
        input_dim: Dimension of cell embeddings
        hidden_dim: Internal hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of attention layers
        max_neighbors: Maximum number of neighbors (for positional encoding)
        distance_encoding: How to encode distances ("rbf", "mlp", "sinusoidal")
        sparsity_type: Attention sparsity ("entropy", "topk", "sparsemax")
        sparsity_weight: Weight for entropy regularization loss
        topk: Number of neighbors for top-k sparsity
        dropout: Dropout rate
        use_reconstruction_head: Add decoder for masked receiver reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_neighbors: int = 20,
        distance_encoding: DistanceEncoding | str = DistanceEncoding.RBF,
        sparsity_type: SparsityType | str = SparsityType.ENTROPY,
        sparsity_weight: float = 0.01,
        topk: int = 5,
        dropout: float = 0.1,
        use_reconstruction_head: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_type = SparsityType(sparsity_type)
        self.sparsity_weight = sparsity_weight

        # Input projections
        self.receiver_proj = nn.Linear(input_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(input_dim, hidden_dim)

        # Receiver-centered attention layers
        self.attention_layers = nn.ModuleList(
            [
                ReceiverCenteredAttention(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    distance_encoding=distance_encoding,
                    sparsity_type=sparsity_type,
                    topk=topk,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norms for residual connections
        self.receiver_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Feed-forward networks
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Optional reconstruction head for self-supervised learning
        self.reconstruction_head = None
        if use_reconstruction_head:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

    def forward(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        neighbor_mask: Tensor | None = None,
        cell_type_hint: Tensor | None = None,
        return_reconstruction: bool = False,
    ) -> ReceiverNicheOutput:
        """Encode receiver's neighborhood context.

        Args:
            receiver: [B, D] receiver cell embedding
            neighbors: [B, K, D] neighbor cell embeddings
            distances: [B, K] distances from receiver to each neighbor
            neighbor_mask: [B, K] boolean, True = valid, False = ablated
            cell_type_hint: [B, D_type] optional cell type embedding (soft bias)
            return_reconstruction: Whether to compute receiver reconstruction

        Returns:
            ReceiverNicheOutput with context, attention weights, and optional losses
        """
        # Project to hidden dimension
        h_receiver = self.receiver_proj(receiver)  # [B, D]
        h_neighbors = self.neighbor_proj(neighbors)  # [B, K, D]

        # Optional cell type conditioning (soft bias, not rigid)
        if cell_type_hint is not None:
            h_receiver = h_receiver + cell_type_hint

        # Collect attention weights from all layers
        all_attention_weights = []

        # Apply receiver-centered attention layers
        for attn_layer, norm, ffn, ffn_norm in zip(
            self.attention_layers,
            self.receiver_norms,
            self.ffns,
            self.ffn_norms,
        ):
            # Cross-attention: receiver attends to neighbors
            context, attn_weights = attn_layer(h_receiver, h_neighbors, distances, neighbor_mask)
            all_attention_weights.append(attn_weights)

            # Residual + norm
            h_receiver = norm(h_receiver + context)

            # Feed-forward with residual
            h_receiver = ffn_norm(h_receiver + ffn(h_receiver))

        # Final output projection
        context = self.output_proj(h_receiver)

        # Average attention weights across layers for interpretability
        final_attention = torch.stack(all_attention_weights, dim=0).mean(dim=0)

        # Compute entropy loss if using entropy regularization
        entropy_loss = None
        if self.sparsity_type == SparsityType.ENTROPY and self.training:
            entropy_loss = self.sparsity_weight * _compute_attention_entropy(final_attention)

        # Optional reconstruction for self-supervised learning
        reconstruction = None
        if return_reconstruction and self.reconstruction_head is not None:
            reconstruction = self.reconstruction_head(context)

        return ReceiverNicheOutput(
            context=context,
            attention_weights=final_attention,
            entropy_loss=entropy_loss,
            receiver_reconstruction=reconstruction,
        )

    def compute_reconstruction_loss(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        neighbor_mask: Tensor | None = None,
        mask_ratio: float = 0.15,
    ) -> tuple[Tensor, ReceiverNicheOutput]:
        """Compute masked receiver reconstruction loss.

        This is the primary self-supervised signal for the niche encoder.
        Given neighbors, predict the receiver's masked features.

        Args:
            receiver: [B, D] receiver cell embedding (ground truth)
            neighbors: [B, K, D] neighbor embeddings
            distances: [B, K] distances
            neighbor_mask: [B, K] valid neighbor mask
            mask_ratio: Fraction of receiver features to mask

        Returns:
            loss: Scalar reconstruction loss
            output: Encoder output with reconstruction
        """
        B, D = receiver.shape
        device = receiver.device

        # Create random mask for receiver features
        mask = torch.rand(B, D, device=device) < mask_ratio

        # Mask receiver (replace masked positions with zeros or learned mask token)
        receiver_masked = receiver.clone()
        receiver_masked[mask] = 0.0

        # Forward pass with masked receiver
        output = self.forward(
            receiver_masked,
            neighbors,
            distances,
            neighbor_mask,
            return_reconstruction=True,
        )

        # Compute loss only on masked positions
        if output.receiver_reconstruction is not None:
            reconstruction_loss = F.mse_loss(
                output.receiver_reconstruction[mask],
                receiver[mask],
            )
        else:
            reconstruction_loss = torch.tensor(0.0, device=device)

        return reconstruction_loss, output

    def ablate_neighbor(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        ablate_idx: int,
        neighbor_mask: Tensor | None = None,
    ) -> ReceiverNicheOutput:
        """Ablate a specific neighbor to measure its influence.

        Args:
            receiver: [B, D] receiver embedding
            neighbors: [B, K, D] neighbor embeddings
            distances: [B, K] distances
            ablate_idx: Index of neighbor to ablate
            neighbor_mask: [B, K] existing mask

        Returns:
            Output with the specified neighbor ablated
        """
        B, K, _ = neighbors.shape

        # Create or update mask to ablate specified neighbor
        if neighbor_mask is None:
            neighbor_mask = torch.ones(B, K, dtype=torch.bool, device=neighbors.device)
        else:
            neighbor_mask = neighbor_mask.clone()

        neighbor_mask[:, ablate_idx] = False

        return self.forward(receiver, neighbors, distances, neighbor_mask)

    def compute_neighbor_importance(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        neighbor_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute importance scores for each neighbor via ablation.

        Measures how much the output changes when each neighbor is removed.

        Args:
            receiver: [B, D] receiver embedding
            neighbors: [B, K, D] neighbor embeddings
            distances: [B, K] distances
            neighbor_mask: [B, K] valid neighbor mask

        Returns:
            [B, K] importance scores (higher = more important)
        """
        B, K, _ = neighbors.shape

        # Get baseline output with all neighbors
        baseline_output = self.forward(receiver, neighbors, distances, neighbor_mask)
        baseline_context = baseline_output.context

        importance_scores = torch.zeros(B, K, device=neighbors.device)

        # Ablate each neighbor and measure change
        for k in range(K):
            ablated_output = self.ablate_neighbor(receiver, neighbors, distances, k, neighbor_mask)
            # Importance = L2 distance of context change
            diff = (baseline_context - ablated_output.context).norm(dim=-1)
            importance_scores[:, k] = diff

        # Normalize to [0, 1]
        importance_scores = importance_scores / (
            importance_scores.max(dim=-1, keepdim=True).values + 1e-8
        )

        return importance_scores


# =============================================================================
# ABLATION: Self-Attention Niche Encoder
# =============================================================================


class SelfAttentionNicheEncoder(nn.Module):
    """Self-attention niche encoder for ablation comparison.

    This encoder uses standard self-attention over ALL tokens (receiver + neighbors)
    instead of cross-attention where receiver queries neighbors.

    Key difference from ReceiverCenteredNicheEncoder:
    - ReceiverCentered: receiver is query, neighbors are key/value (cross-attention)
    - SelfAttention: all tokens attend to all tokens equally (self-attention)

    Same interface as ReceiverCenteredNicheEncoder for ablation comparison.
    This tests whether architectural enforcement of receiver-centrality helps,
    or whether self-attention can learn to focus on the receiver.

    Args:
        input_dim: Dimension of cell embeddings
        hidden_dim: Internal hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of self-attention layers
        dropout: Dropout rate
        use_reconstruction_head: Add decoder for masked receiver reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_neighbors: int = 20,  # For interface compatibility
        distance_encoding: DistanceEncoding | str = DistanceEncoding.RBF,  # Unused but for interface
        sparsity_type: SparsityType | str = SparsityType.ENTROPY,  # Unused but for interface
        sparsity_weight: float = 0.01,  # Unused but for interface
        topk: int = 5,  # Unused but for interface
        dropout: float = 0.1,
        use_reconstruction_head: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Unified projection for all tokens
        self.token_proj = nn.Linear(input_dim, hidden_dim)

        # Token type embeddings: 0 = receiver, 1 = neighbor
        self.token_type_embedding = nn.Embedding(2, hidden_dim)

        # Standard transformer self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Optional reconstruction head
        self.reconstruction_head = None
        if use_reconstruction_head:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

    def forward(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,  # Unused in self-attention (no distance modulation)
        neighbor_mask: Tensor | None = None,
        cell_type_hint: Tensor | None = None,  # Unused but for interface
        return_reconstruction: bool = False,
    ) -> ReceiverNicheOutput:
        """Encode receiver's neighborhood using self-attention.

        Args:
            receiver: [B, D] receiver cell embedding
            neighbors: [B, K, D] neighbor cell embeddings
            distances: [B, K] distances (unused - no distance modulation in self-attention)
            neighbor_mask: [B, K] boolean, True = valid, False = masked
            cell_type_hint: [B, D_type] optional (unused, for interface compatibility)
            return_reconstruction: Whether to compute receiver reconstruction

        Returns:
            ReceiverNicheOutput with context, attention weights, and optional reconstruction
        """
        B, K, D = neighbors.shape

        # Project all tokens to hidden dimension
        h_receiver = self.token_proj(receiver).unsqueeze(1)  # [B, 1, H]
        h_neighbors = self.token_proj(neighbors)  # [B, K, H]

        # Add token type embeddings
        receiver_type = torch.zeros(B, 1, dtype=torch.long, device=receiver.device)
        neighbor_type = torch.ones(B, K, dtype=torch.long, device=receiver.device)

        h_receiver = h_receiver + self.token_type_embedding(receiver_type)
        h_neighbors = h_neighbors + self.token_type_embedding(neighbor_type)

        # Concatenate: [receiver, neighbors] -> [B, K+1, H]
        tokens = torch.cat([h_receiver, h_neighbors], dim=1)

        # Build attention mask if needed
        # Mask format for nn.TransformerEncoder: True = IGNORE
        src_key_padding_mask = None
        if neighbor_mask is not None:
            # Receiver is always valid (False = attend), neighbors use provided mask
            receiver_valid = torch.zeros(B, 1, dtype=torch.bool, device=receiver.device)
            # Invert: neighbor_mask True = valid -> False for transformer
            neighbor_invalid = ~neighbor_mask
            src_key_padding_mask = torch.cat([receiver_valid, neighbor_invalid], dim=1)

        # Self-attention over all tokens
        h = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)

        # Extract receiver's representation (first token)
        h_receiver_out = h[:, 0, :]  # [B, H]

        # Output projection
        context = self.output_proj(h_receiver_out)

        # Compute pseudo-attention weights from final layer (for interpretability)
        # Self-attention doesn't have explicit receiver->neighbor weights,
        # so we report uniform weights as a placeholder
        attention_weights = torch.ones(B, K, device=receiver.device) / K
        if neighbor_mask is not None:
            attention_weights = attention_weights * neighbor_mask.float()
            attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Optional reconstruction
        reconstruction = None
        if return_reconstruction and self.reconstruction_head is not None:
            reconstruction = self.reconstruction_head(context)

        return ReceiverNicheOutput(
            context=context,
            attention_weights=attention_weights,
            entropy_loss=None,  # No entropy loss for self-attention
            receiver_reconstruction=reconstruction,
        )

    def compute_reconstruction_loss(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        neighbor_mask: Tensor | None = None,
        mask_ratio: float = 0.15,
    ) -> tuple[Tensor, ReceiverNicheOutput]:
        """Compute masked receiver reconstruction loss (same interface as ReceiverCentered)."""
        B, D = receiver.shape
        device = receiver.device

        mask = torch.rand(B, D, device=device) < mask_ratio
        receiver_masked = receiver.clone()
        receiver_masked[mask] = 0.0

        output = self.forward(
            receiver_masked,
            neighbors,
            distances,
            neighbor_mask,
            return_reconstruction=True,
        )

        if output.receiver_reconstruction is not None:
            reconstruction_loss = F.mse_loss(
                output.receiver_reconstruction[mask],
                receiver[mask],
            )
        else:
            reconstruction_loss = torch.tensor(0.0, device=device)

        return reconstruction_loss, output


# NOTE: ReceiverNicheEncoderWithDualReference was REMOVED (2026-04-09)
# It redundantly concatenated [fused | hlca | luca] = 80d when fused already contains hlca+luca.
# The correct approach: Use ReceiverCenteredNicheEncoder with fused embedding (40d).
# The Linear projection learns to weight HLCA vs LuCA features.
# See: docs/architecture/dual_reference_encoder.md
