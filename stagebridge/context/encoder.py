"""Receiver-centered niche encoder per StageBridge doctrine.

The key principle is RECEIVER-CENTERING: the focal cell (receiver) is the query,
neighbors are keys/values, and information flows TO the receiver.

Design principles (aligned with AMICI - Hong et al., bioRxiv 2025):
1. Receiver-centered architecture (receiver as query)
2. Distance-aware attention with MONOTONIC DECREASE (AMICI Eq. 1)
   - Attention = Softmax(phenotype_score - b1*distance || empty_token)
   - b1 enforced positive via Softplus -> guarantees decay with distance
3. Empty neighbor token allows attention to "escape" when no neighbor is informative
4. Sparsity regularization: entropy penalty + L1 on value matrix
5. Neighbor ablation for interpretability
6. Masked receiver reconstruction as self-supervised signal

Reference:
    Hong J, Desai K, Nguyen TD, Nazaret A, Levy N, Ergen C, Plitas G, Azizi E.
    AMICI: Attention Mechanism Interpretation of Cell-cell Interactions.
    bioRxiv 2025. doi:10.1101/2025.09.22.677860
    https://github.com/azizilab/amici

    License: CC BY-NC-ND 4.0. Patent pending (U.S. Serial No. 63/884,704).
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
    RBF = "rbf"
    MLP = "mlp"
    SINUSOIDAL = "sinusoidal"


class SparsityType(StrEnum):
    """Attention sparsity strategies."""
    ENTROPY = "entropy"
    TOPK = "topk"
    SPARSEMAX = "sparsemax"


@dataclass(slots=True, frozen=True)
class ReceiverNicheOutput:
    """Output from the receiver-centered niche encoder.

    Attributes:
        context: [B, D] - Pooled context from neighborhood
        context_tokens: [B, K, D] - Individual token representations for cross-attention
        attention_weights: [B, K] - Interpretable neighbor importance scores (includes empty token)
        entropy_loss: Scalar - Attention entropy for regularization
        value_l1_loss: Scalar - L1 penalty on value matrix (AMICI-style sparsity)
        empty_attention: [B] - How much attention went to empty token (interpretability)
        receiver_reconstruction: [B, D] - Reconstructed receiver (if decoder present)
        niche_prototype_composition: [B, K] - Soft assignment to niche archetypes (if prototypes enabled)
    """
    context: Tensor
    context_tokens: Tensor | None = None
    attention_weights: Tensor = None
    entropy_loss: Tensor | None = None
    value_l1_loss: Tensor | None = None
    empty_attention: Tensor | None = None
    receiver_reconstruction: Tensor | None = None
    niche_prototype_composition: Tensor | None = None


def _rbf_distance_encoding(distances: Tensor, num_rbf: int = 16, max_dist: float = 100.0) -> Tensor:
    """Radial basis function encoding of distances."""
    centers = torch.linspace(0, max_dist, num_rbf, device=distances.device, dtype=distances.dtype)
    width = max_dist / num_rbf
    diff = distances.unsqueeze(-1) - centers
    return torch.exp(-0.5 * (diff / width) ** 2)


def _sinusoidal_distance_encoding(distances: Tensor, dim: int = 16) -> Tensor:
    """Sinusoidal encoding of distances."""
    half_dim = dim // 2
    freq = torch.exp(
        torch.arange(half_dim, device=distances.device, dtype=distances.dtype)
        * (-math.log(10000.0) / half_dim)
    )
    phase = distances.unsqueeze(-1) * freq
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


def _sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax activation (Martins & Astudillo, 2016)."""
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    cumsum = torch.cumsum(sorted_logits, dim=dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    k = k.view([1] * (logits.dim() - 1) + [-1])
    condition = 1 + k * sorted_logits > cumsum
    k_max = condition.sum(dim=dim, keepdim=True).clamp(min=1)
    cumsum_at_k = cumsum.gather(dim, (k_max - 1).long())
    tau = (cumsum_at_k - 1) / k_max.float()
    return (logits - tau).clamp(min=0)


def _compute_attention_entropy(attention_weights: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute normalized entropy of attention distribution."""
    K = attention_weights.size(-1)
    if K <= 1:
        return torch.tensor(0.0, device=attention_weights.device, dtype=attention_weights.dtype)
    log_attn = torch.log(attention_weights + eps)
    entropy = -torch.sum(attention_weights * log_attn, dim=-1)
    return (entropy / math.log(K)).mean()


class DistanceEncoder(nn.Module):
    """Encode spatial distances for attention modulation."""

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
            self.proj = nn.Linear(output_dim, output_dim)

    def forward(self, distances: Tensor) -> Tensor:
        if self.encoding_type == DistanceEncoding.MLP:
            return self.mlp(distances.unsqueeze(-1))
        elif self.encoding_type == DistanceEncoding.RBF:
            rbf = _rbf_distance_encoding(distances, self.output_dim, self.max_distance)
            return self.proj(rbf)
        else:
            return _sinusoidal_distance_encoding(distances, self.output_dim)


class ReceiverCenteredAttention(nn.Module):
    """Cross-attention with AMICI-style distance modulation and empty neighbor token.

    Key AMICI features (Hong et al., bioRxiv 2025):
    1. Distance coefficient enforced POSITIVE via Softplus, then SUBTRACTED
       -> Guarantees attention monotonically decreases with distance
    2. Empty neighbor token with fixed score allows attention to "escape"
       -> Model can learn "no neighbor is informative here"
    3. L1 penalty on value vectors for sparse influence patterns
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        distance_encoding: DistanceEncoding | str = DistanceEncoding.RBF,
        sparsity_type: SparsityType | str = SparsityType.ENTROPY,
        topk: int = 5,
        use_empty_token: bool = True,
        empty_token_score: float = 3.0,
        distance_scale: float = 50.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparsity_type = SparsityType(sparsity_type)
        self.topk = topk
        self.use_empty_token = use_empty_token
        self.empty_token_score = empty_token_score
        self.distance_scale = distance_scale

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.distance_coef_mlp = nn.Sequential(
            nn.Linear(dim, num_heads),
        )
        self.distance_coef_offset = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        neighbor_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward with AMICI-style distance modulation.

        Returns:
            context: [B, D] aggregated context
            attn_weights: [B, K+1] attention weights (K neighbors + empty token)
            empty_attention: [B] attention to empty token
            value_l1: scalar L1 norm of value vectors
        """
        B, K, _ = neighbors.shape

        q = self.q_proj(receiver).unsqueeze(1)
        k = self.k_proj(neighbors)
        v = self.v_proj(neighbors)

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        phenotype_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        distance_coef_raw = self.distance_coef_mlp(receiver) + self.distance_coef_offset
        distance_coef = F.softplus(distance_coef_raw)

        normalized_dist = distances / self.distance_scale
        distance_penalty = distance_coef.unsqueeze(-1) * normalized_dist.unsqueeze(1)
        distance_penalty = distance_penalty.unsqueeze(2)

        attn_logits = phenotype_score - distance_penalty

        if neighbor_mask is not None:
            mask = neighbor_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        if self.use_empty_token:
            empty_score = torch.full(
                (B, self.num_heads, 1, 1),
                self.empty_token_score,
                device=attn_logits.device,
                dtype=attn_logits.dtype,
            )
            attn_logits = torch.cat([attn_logits, empty_score], dim=-1)

            empty_v = torch.zeros(
                (B, self.num_heads, 1, self.head_dim),
                device=v.device,
                dtype=v.dtype,
            )
            v = torch.cat([v, empty_v], dim=2)

        if self.sparsity_type == SparsityType.TOPK:
            total_k = attn_logits.size(-1)
            k_actual = min(self.topk, total_k)
            topk_values, topk_indices = torch.topk(attn_logits, k_actual, dim=-1)
            sparse_logits = torch.full_like(attn_logits, float("-inf"))
            sparse_logits.scatter_(-1, topk_indices, topk_values)
            attn_weights = F.softmax(sparse_logits, dim=-1)
        elif self.sparsity_type == SparsityType.SPARSEMAX:
            total_k = attn_logits.size(-1)
            logits_flat = attn_logits.squeeze(2).view(-1, total_k)
            sparse_flat = _sparsemax(logits_flat, dim=-1)
            attn_weights = sparse_flat.view(B, self.num_heads, 1, total_k)
        else:
            attn_weights = F.softmax(attn_logits, dim=-1)

        attn_weights_dropped = self.dropout(attn_weights)
        context = torch.matmul(attn_weights_dropped, v)
        context = context.transpose(1, 2).contiguous().view(B, 1, self.dim)
        context = self.out_proj(context).squeeze(1)

        attn_weights_mean = attn_weights.squeeze(2).mean(dim=1)

        if self.use_empty_token:
            empty_attention = attn_weights_mean[:, -1]
            neighbor_attention = attn_weights_mean[:, :-1]
        else:
            empty_attention = torch.zeros(B, device=attn_logits.device)
            neighbor_attention = attn_weights_mean

        value_l1 = v[:, :, :-1, :].abs().mean() if self.use_empty_token else v.abs().mean()

        return context, neighbor_attention, empty_attention, value_l1


class ReceiverCenteredNicheEncoder(nn.Module):
    """Receiver-centered local neighborhood encoder with AMICI features.

    Models "what does this cell receive from its neighbors?" by using the
    receiver cell as the attention query and neighbors as keys/values.

    AMICI-aligned features (Hong et al., bioRxiv 2025):
    - Monotonic distance decay: attention = phenotype_score - b1*distance
    - Empty neighbor token: allows attention to "escape" to nothing
    - L1 penalty on values: encourages sparse influence patterns
    - Entropy regularization: prevents uniform attention

    Args:
        input_dim: Dimension of cell embeddings
        hidden_dim: Internal hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of attention layers
        distance_encoding: How to encode distances (legacy, not used in AMICI mode)
        sparsity_type: Attention sparsity mechanism
        sparsity_weight: Weight for entropy regularization
        value_l1_weight: Weight for L1 penalty on values (AMICI default: 0.01)
        topk: Number of neighbors for top-k sparsity
        dropout: Dropout rate
        use_reconstruction_head: Add decoder for SSL
        use_token_type_embeddings: Add semantic token type embeddings
        use_empty_token: Enable empty neighbor token (AMICI feature)
        empty_token_score: Fixed score for empty token (AMICI default: 3.0)
        distance_scale: Distance normalization factor in microns
    """

    NUM_TOKEN_TYPES = 9

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
        value_l1_weight: float = 0.01,
        topk: int = 5,
        dropout: float = 0.1,
        use_reconstruction_head: bool = True,
        use_token_type_embeddings: bool = True,
        use_empty_token: bool = True,
        empty_token_score: float = 3.0,
        distance_scale: float = 50.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_type = SparsityType(sparsity_type)
        self.sparsity_weight = sparsity_weight
        self.value_l1_weight = value_l1_weight
        self.use_token_type_embeddings = use_token_type_embeddings
        self.use_empty_token = use_empty_token

        self.receiver_proj = nn.Linear(input_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(input_dim, hidden_dim)

        if use_token_type_embeddings:
            self.token_type_embedding = nn.Embedding(self.NUM_TOKEN_TYPES, hidden_dim)
        else:
            self.token_type_embedding = None

        self.attention_layers = nn.ModuleList([
            ReceiverCenteredAttention(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                distance_encoding=distance_encoding,
                sparsity_type=sparsity_type,
                topk=topk,
                use_empty_token=use_empty_token,
                empty_token_score=empty_token_score,
                distance_scale=distance_scale,
            )
            for _ in range(num_layers)
        ])

        self.receiver_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

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
        token_type_ids: Tensor | None = None,
    ) -> ReceiverNicheOutput:
        """Encode receiver's neighborhood context.

        Args:
            receiver: [B, D] receiver cell embedding
            neighbors: [B, K, D] neighbor embeddings (K=8 for 9-token structure)
            distances: [B, K] distances from receiver to each neighbor
            neighbor_mask: [B, K] boolean, True = valid
            token_type_ids: [B, K] optional explicit token types

        Returns:
            ReceiverNicheOutput
        """
        B, K, _ = neighbors.shape
        device = receiver.device

        h_receiver = self.receiver_proj(receiver)
        h_neighbors = self.neighbor_proj(neighbors)

        if self.use_token_type_embeddings and self.token_type_embedding is not None:
            receiver_type = torch.zeros(B, 1, dtype=torch.long, device=device)
            h_receiver = h_receiver + self.token_type_embedding(receiver_type).squeeze(1)

            if token_type_ids is None:
                token_type_ids = torch.arange(1, K + 1, dtype=torch.long, device=device)
                token_type_ids = token_type_ids.unsqueeze(0).expand(B, -1)
                token_type_ids = token_type_ids.clamp(max=self.NUM_TOKEN_TYPES - 1)
            h_neighbors = h_neighbors + self.token_type_embedding(token_type_ids)

        if cell_type_hint is not None:
            h_receiver = h_receiver + cell_type_hint

        all_attention_weights = []
        all_empty_attention = []
        all_value_l1 = []

        for attn_layer, norm, ffn, ffn_norm in zip(
            self.attention_layers, self.receiver_norms, self.ffns, self.ffn_norms
        ):
            attn_context, attn_weights, empty_attn, value_l1 = attn_layer(
                h_receiver, h_neighbors, distances, neighbor_mask
            )
            all_attention_weights.append(attn_weights)
            all_empty_attention.append(empty_attn)
            all_value_l1.append(value_l1)
            h_receiver = norm(h_receiver + attn_context)
            h_receiver = ffn_norm(h_receiver + ffn(h_receiver))

        context = self.output_proj(h_receiver)
        context_tokens = torch.cat([context.unsqueeze(1), self.output_proj(h_neighbors)], dim=1)
        assert context_tokens.shape[1] == neighbors.shape[1] + 1, (
            f"GRADIENT CONTRACT VIOLATED: context_tokens has {context_tokens.shape[1]} tokens, "
            f"expected {neighbors.shape[1] + 1} (neighbors + receiver). "
            f"Receiver must be included for gradients to flow through attention layers."
        )

        final_attention = torch.stack(all_attention_weights, dim=0).mean(dim=0)
        final_empty_attention = torch.stack(all_empty_attention, dim=0).mean(dim=0)

        entropy_loss = None
        if self.sparsity_type == SparsityType.ENTROPY and self.training:
            entropy_loss = self.sparsity_weight * _compute_attention_entropy(final_attention)

        value_l1_loss = None
        if self.training and self.value_l1_weight > 0:
            value_l1_loss = self.value_l1_weight * torch.stack(all_value_l1).mean()

        reconstruction = None
        if return_reconstruction and self.reconstruction_head is not None:
            reconstruction = self.reconstruction_head(context)

        return ReceiverNicheOutput(
            context=context,
            context_tokens=context_tokens,
            attention_weights=final_attention,
            entropy_loss=entropy_loss,
            value_l1_loss=value_l1_loss,
            empty_attention=final_empty_attention,
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
        """Compute masked receiver reconstruction loss."""
        B, D = receiver.shape
        device = receiver.device

        mask = torch.rand(B, D, device=device) < mask_ratio
        receiver_masked = receiver.clone()
        receiver_masked[mask] = 0.0

        output = self.forward(
            receiver_masked, neighbors, distances, neighbor_mask, return_reconstruction=True
        )

        if output.receiver_reconstruction is not None:
            reconstruction_loss = F.mse_loss(
                output.receiver_reconstruction[mask], receiver[mask]
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
        """Ablate a specific neighbor to measure its influence."""
        B, K, _ = neighbors.shape

        if neighbor_mask is None:
            neighbor_mask = torch.ones(B, K, dtype=torch.bool, device=neighbors.device)
        else:
            neighbor_mask = neighbor_mask.clone()

        neighbor_mask[:, ablate_idx] = False
        return self.forward(receiver, neighbors, distances, neighbor_mask)


class SelfAttentionNicheEncoder(nn.Module):
    """Self-attention niche encoder for ablation comparison.

    Uses standard self-attention over ALL tokens instead of cross-attention.
    Same interface as ReceiverCenteredNicheEncoder for fair comparison.
    """

    NUM_TOKEN_TYPES = 9

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
        num_distance_bins: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(distance_encoding, str):
            distance_encoding = DistanceEncoding(distance_encoding)
        self.distance_encoding = distance_encoding

        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.token_type_embedding = nn.Embedding(self.NUM_TOKEN_TYPES, hidden_dim)

        self.num_distance_bins = num_distance_bins
        if distance_encoding == DistanceEncoding.RBF:
            self.register_buffer("rbf_centers", torch.linspace(0.0, 1.0, num_distance_bins))
            self.rbf_sigma = 1.0 / num_distance_bins
            self.distance_proj = nn.Linear(num_distance_bins, hidden_dim)
        else:
            self.distance_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.reconstruction_head = None
        if use_reconstruction_head:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

    def _encode_distances(self, distances: Tensor) -> Tensor:
        if self.distance_encoding == DistanceEncoding.RBF:
            d = distances.unsqueeze(-1)
            c = self.rbf_centers.view(1, 1, -1)
            rbf = torch.exp(-((d - c) ** 2) / (2 * self.rbf_sigma ** 2))
            return self.distance_proj(rbf)
        else:
            return self.distance_mlp(distances.unsqueeze(-1))

    def forward(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        neighbor_mask: Tensor | None = None,
        cell_type_hint: Tensor | None = None,
        return_reconstruction: bool = False,
        token_type_ids: Tensor | None = None,
    ) -> ReceiverNicheOutput:
        B, K, D = neighbors.shape
        device = receiver.device

        h_receiver = self.token_proj(receiver).unsqueeze(1)
        h_neighbors = self.token_proj(neighbors)

        receiver_type = torch.zeros(B, 1, dtype=torch.long, device=device)
        h_receiver = h_receiver + self.token_type_embedding(receiver_type)

        if token_type_ids is None:
            token_type_ids = torch.arange(1, K + 1, dtype=torch.long, device=device)
            token_type_ids = token_type_ids.unsqueeze(0).expand(B, -1)
            token_type_ids = token_type_ids.clamp(max=self.NUM_TOKEN_TYPES - 1)
        h_neighbors = h_neighbors + self.token_type_embedding(token_type_ids)

        if distances is not None:
            distance_enc = self._encode_distances(distances)
            h_neighbors = h_neighbors + distance_enc

        tokens = torch.cat([h_receiver, h_neighbors], dim=1)

        src_key_padding_mask = None
        if neighbor_mask is not None:
            receiver_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
            neighbor_invalid = ~neighbor_mask
            src_key_padding_mask = torch.cat([receiver_valid, neighbor_invalid], dim=1)

        h = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)

        h_receiver_out = h[:, 0, :]
        h_neighbors_out = h[:, 1:, :]

        context = self.output_proj(h_receiver_out)
        context_tokens = self.output_proj(h_neighbors_out)

        attention_weights = self._compute_attention_weights(tokens, src_key_padding_mask, K)

        reconstruction = None
        if return_reconstruction and self.reconstruction_head is not None:
            reconstruction = self.reconstruction_head(context)

        return ReceiverNicheOutput(
            context=context,
            context_tokens=context_tokens,
            attention_weights=attention_weights,
            entropy_loss=None,
            receiver_reconstruction=reconstruction,
        )

    def _compute_attention_weights(self, tokens: Tensor, mask: Tensor | None, K: int) -> Tensor:
        layer = self.transformer.layers[0]
        with torch.no_grad():
            _, attn_weights = layer.self_attn(
                tokens, tokens, tokens,
                key_padding_mask=mask,
                need_weights=True,
                average_attn_weights=True,
            )
            return attn_weights[:, 0, 1:K+1]

    def compute_reconstruction_loss(
        self,
        receiver: Tensor,
        neighbors: Tensor,
        distances: Tensor,
        neighbor_mask: Tensor | None = None,
        mask_ratio: float = 0.15,
    ) -> tuple[Tensor, ReceiverNicheOutput]:
        B, D = receiver.shape
        device = receiver.device

        mask = torch.rand(B, D, device=device) < mask_ratio
        receiver_masked = receiver.clone()
        receiver_masked[mask] = 0.0

        output = self.forward(
            receiver_masked, neighbors, distances, neighbor_mask, return_reconstruction=True
        )

        if output.receiver_reconstruction is not None:
            reconstruction_loss = F.mse_loss(
                output.receiver_reconstruction[mask], receiver[mask]
            )
        else:
            reconstruction_loss = torch.tensor(0.0, device=device)

        return reconstruction_loss, output
