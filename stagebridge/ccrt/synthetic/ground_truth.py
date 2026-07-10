"""Independent synthetic teacher.

The teacher generates ground-truth context mechanisms from explicit mathematical
equations, coded directly in PyTorch. It is **mathematically independent** from
the CCRT student: it must not import ``sender_context``, ``operators``,
``transport``, or ``training``, and must not reference any student class. Teacher
parameters are fixed tensors (not ``nn.Parameter``) generated deterministically
from a dedicated ``torch.Generator``.

The teacher's hidden decomposition (context state, regulatory state, per-term
drift/growth) is evaluation-only truth — it is never fed to the student as
supervision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SyntheticSystemConfig
from .mechanisms import SyntheticMechanismSpec

__all__ = [
    "SyntheticTeacherParameters",
    "SyntheticGroundTruth",
    "SyntheticTeacher",
]


def _normed(gen: torch.Generator, *shape: int, scale: float = 1.0) -> torch.Tensor:
    """A deterministic normalized random tensor with controlled scale.

    Each matrix is scaled by 1/sqrt(fan_in) so downstream products stay O(1).
    """
    t = torch.randn(*shape, generator=gen, dtype=torch.float64)
    fan_in = shape[-2] if len(shape) >= 2 else 1
    return t * (scale / (fan_in ** 0.5))


@dataclass(frozen=True)
class SyntheticTeacherParameters:
    """Fixed (non-optimized) teacher parameter tensors, per transition edge."""

    sender_projection: torch.Tensor       # [E, T, D_S, D_C]
    sender_bias: torch.Tensor             # [E, T, D_C]
    distance_decay: torch.Tensor          # [E, T]  (positive)
    receiver_to_regulatory: torch.Tensor  # [D_R, D_REG]
    context_to_regulatory: torch.Tensor   # [D_C, D_REG]
    edge_regulatory_bias: torch.Tensor    # [E, D_REG]
    receiver_to_self_drift: torch.Tensor  # [E, D_R, D_Z]
    regulatory_to_drift: torch.Tensor     # [E, D_REG, D_Z]
    context_to_drift: torch.Tensor        # [E, D_C, D_Z]
    receiver_to_self_growth: torch.Tensor # [E, D_R, D_G]
    regulatory_to_growth: torch.Tensor    # [E, D_REG, D_G]
    context_to_growth: torch.Tensor       # [E, D_C, D_G]

    @classmethod
    def from_config(
        cls,
        *,
        system: SyntheticSystemConfig,
        mechanism: SyntheticMechanismSpec,
        seed: int,
    ) -> "SyntheticTeacherParameters":
        gen = torch.Generator().manual_seed(int(seed))
        E = system.num_transition_edges
        T = system.num_sender_context_types
        D_R, D_S, D_C = system.receiver_dim, system.sender_dim, system.context_dim
        D_REG, D_Z, D_G = system.regulatory_dim, system.semantic_dim, system.growth_dim

        # distance decay strictly positive in a controlled range (0.3 .. 1.3)
        decay = 0.3 + torch.rand(E, T, generator=gen, dtype=torch.float64)

        return cls(
            sender_projection=_normed(gen, E, T, D_S, D_C),
            sender_bias=_normed(gen, E, T, D_C, scale=0.1),
            distance_decay=decay,
            receiver_to_regulatory=_normed(gen, D_R, D_REG),
            context_to_regulatory=_normed(gen, D_C, D_REG),
            edge_regulatory_bias=_normed(gen, E, D_REG, scale=0.1),
            receiver_to_self_drift=_normed(gen, E, D_R, D_Z),
            regulatory_to_drift=_normed(gen, E, D_REG, D_Z),
            context_to_drift=_normed(gen, E, D_C, D_Z),
            receiver_to_self_growth=_normed(gen, E, D_R, D_G),
            regulatory_to_growth=_normed(gen, E, D_REG, D_G),
            context_to_growth=_normed(gen, E, D_C, D_G),
        )


@dataclass(frozen=True)
class SyntheticGroundTruth:
    """The teacher's full hidden decomposition (evaluation-only truth)."""

    sender_context_vectors: torch.Tensor   # [B, K, D_C]
    sender_distance_weights: torch.Tensor  # [B, K]
    context_state: torch.Tensor            # [B, D_C]
    regulatory_state: torch.Tensor         # [B, D_REG]
    self_drift: torch.Tensor               # [B, D_Z]
    regulatory_drift: torch.Tensor         # [B, D_Z]
    residual_drift: torch.Tensor           # [B, D_Z]
    context_delta_drift: torch.Tensor      # [B, D_Z]
    full_drift: torch.Tensor               # [B, D_Z]
    self_growth: torch.Tensor              # [B, D_G]
    regulatory_growth: torch.Tensor        # [B, D_G]
    residual_growth: torch.Tensor          # [B, D_G]
    context_delta_growth: torch.Tensor     # [B, D_G]
    full_growth: torch.Tensor              # [B, D_G]
    destination_semantic_features: torch.Tensor  # [B, D_Z]


class SyntheticTeacher:
    """Evaluates the explicit synthetic mechanism equations."""

    def __init__(
        self,
        *,
        system: SyntheticSystemConfig,
        mechanism: SyntheticMechanismSpec,
        parameters: SyntheticTeacherParameters,
    ) -> None:
        mechanism.validate_against_system(system)
        self.system = system
        self.mechanism = mechanism
        self.parameters = parameters
        self._type_scales = torch.tensor(
            mechanism.sender_type_effect_scales, dtype=torch.float64
        )
        self._edge_scales = torch.tensor(
            mechanism.transition_edge_effect_scales, dtype=torch.float64
        )

    def evaluate(
        self,
        *,
        receiver_features: torch.Tensor,
        sender_features: torch.Tensor,
        sender_mask: torch.Tensor,
        distance_to_receiver: torch.Tensor,
        sender_context_type_ids: torch.Tensor,
        transition_edge_index: torch.Tensor,
        source_semantic_features: torch.Tensor,
    ) -> SyntheticGroundTruth:
        sys = self.system
        p = self.parameters

        x = receiver_features.to(torch.float64)
        s = sender_features.to(torch.float64)
        dist = distance_to_receiver.to(torch.float64)
        mask = sender_mask.to(torch.float64)
        types = sender_context_type_ids.to(torch.long)
        edges = transition_edge_index.to(torch.long)
        z_src = source_semantic_features.to(torch.float64)

        # -- shape validation --
        if x.dim() != 2 or x.shape[1] != sys.receiver_dim:
            raise ValueError("receiver_features must be [B, D_R]")
        B = x.shape[0]
        if s.shape != (B, sys.senders_per_receiver, sys.sender_dim) and s.dim() != 3:
            raise ValueError("sender_features must be [B, K, D_S]")
        if s.dim() != 3 or s.shape[0] != B or s.shape[2] != sys.sender_dim:
            raise ValueError("sender_features must be [B, K, D_S]")
        K = s.shape[1]
        for name, t in (("sender_mask", mask), ("distance", dist), ("types", types)):
            if tuple(t.shape) != (B, K):
                raise ValueError(f"{name} must be [B, K]")
        if tuple(edges.shape) != (B,):
            raise ValueError("transition_edge_index must be [B]")
        if z_src.dim() != 2 or z_src.shape != (B, sys.semantic_dim):
            raise ValueError("source_semantic_features must be [B, D_Z]")
        if bool((dist < 0).any()):
            raise ValueError("distances must be non-negative")
        if int(types.max()) >= sys.num_sender_context_types or int(types.min()) < 0:
            raise ValueError("sender_context_type_ids out of range")
        if int(edges.max()) >= sys.num_transition_edges or int(edges.min()) < 0:
            raise ValueError("transition_edge_index out of range")
        for name, t in (("receiver", x), ("sender", s), ("source_semantic", z_src)):
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"{name} contains non-finite values")

        # -- per-(receiver, sender) sender latent h_ij = tanh(s W + b) --
        # gather edge- and type-specific projection/bias per token.
        edge_idx = edges.view(B, 1).expand(B, K)                  # [B, K]
        # W_sender[e, t]: [B, K, D_S, D_C]
        W = p.sender_projection[edge_idx, types]                  # [B, K, D_S, D_C]
        b = p.sender_bias[edge_idx, types]                        # [B, K, D_C]
        h = torch.tanh(torch.einsum("bks,bksc->bkc", s, W) + b)   # [B, K, D_C]

        # -- type/edge amplitude a_ij --
        a = self._type_scales[types] * self._edge_scales[edge_idx]  # [B, K]

        # -- distance weight w_ij --
        if self.mechanism.distance_dependent:
            lam = p.distance_decay[edge_idx, types]               # [B, K]
            w = torch.exp(-lam * dist)                            # [B, K]
        else:
            w = torch.ones(B, K, dtype=torch.float64)

        # -- masked sender contribution q_ij --
        q = mask.unsqueeze(-1) * a.unsqueeze(-1) * w.unsqueeze(-1) * h  # [B, K, D_C]

        # -- context state: mean over unmasked real senders --
        n_real = mask.sum(dim=1).clamp_min(1.0)                  # [B]
        c = q.sum(dim=1) / n_real.unsqueeze(-1)                  # [B, D_C]

        # -- regulatory state --
        reg = torch.tanh(
            x @ p.receiver_to_regulatory
            + self.mechanism.context_to_regulatory_strength * (c @ p.context_to_regulatory)
            + p.edge_regulatory_bias[edges]
        )                                                        # [B, D_REG]

        # -- drift terms --
        W_self_v = p.receiver_to_self_drift[edges]               # [B, D_R, D_Z]
        W_reg_v = p.regulatory_to_drift[edges]                   # [B, D_REG, D_Z]
        W_ctx_v = p.context_to_drift[edges]                      # [B, D_C, D_Z]
        self_drift = sys.self_drift_strength * torch.einsum("br,brz->bz", x, W_self_v)
        regulatory_drift = self.mechanism.regulatory_drift_strength * torch.einsum(
            "br,brz->bz", reg, W_reg_v
        )
        residual_drift = self.mechanism.direct_drift_strength * torch.einsum(
            "bc,bcz->bz", c, W_ctx_v
        )
        context_delta_drift = regulatory_drift + residual_drift
        full_drift = self_drift + context_delta_drift

        # -- growth terms --
        W_self_g = p.receiver_to_self_growth[edges]              # [B, D_R, D_G]
        W_reg_g = p.regulatory_to_growth[edges]                  # [B, D_REG, D_G]
        W_ctx_g = p.context_to_growth[edges]                     # [B, D_C, D_G]
        self_growth = sys.self_growth_strength * torch.einsum("br,brg->bg", x, W_self_g)
        regulatory_growth = self.mechanism.regulatory_growth_strength * torch.einsum(
            "br,brg->bg", reg, W_reg_g
        )
        residual_growth = self.mechanism.direct_growth_strength * torch.einsum(
            "bc,bcg->bg", c, W_ctx_g
        )
        context_delta_growth = regulatory_growth + residual_growth
        full_growth = self_growth + context_delta_growth

        destination = z_src + sys.delta_tau * full_drift

        return SyntheticGroundTruth(
            sender_context_vectors=h,
            sender_distance_weights=w,
            context_state=c,
            regulatory_state=reg,
            self_drift=self_drift,
            regulatory_drift=regulatory_drift,
            residual_drift=residual_drift,
            context_delta_drift=context_delta_drift,
            full_drift=full_drift,
            self_growth=self_growth,
            regulatory_growth=regulatory_growth,
            residual_growth=residual_growth,
            context_delta_growth=context_delta_growth,
            full_growth=full_growth,
            destination_semantic_features=destination,
        )
