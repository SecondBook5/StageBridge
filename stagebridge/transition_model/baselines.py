"""Baseline models for StageBridge benchmarking."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from stagebridge.context_model.set_encoder import SinusoidalTimeEmbedding
from stagebridge.transition_model.drift_network import FiLMConditioner
from stagebridge.utils.types import StageBridgeConfig


class DeepSetsEncoder(nn.Module):
    """Permutation-invariant Deep Sets encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x_set: Tensor, mask: Tensor | None = None) -> Tensor:
        if x_set.ndim == 2:
            x_set = x_set.unsqueeze(0)
        h = self.phi(x_set)
        if mask is not None:
            mask = mask.float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (h * mask).sum(dim=1) / denom
        else:
            pooled = h.mean(dim=1)
        return self.rho(pooled)


class DeepSetsFlowModel(nn.Module):
    """Deep Sets context encoder + same flow head as StageBridge."""

    def __init__(self, config: StageBridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DeepSetsEncoder(config.input_dim, config.hidden_dim, dropout=config.dropout)
        self.time_embedding = SinusoidalTimeEmbedding(config.time_embedding_dim)
        self.stage_embedding = nn.Embedding(config.num_stages * config.num_stages, config.stage_embedding_dim)
        cond_dim = config.hidden_dim + config.stage_embedding_dim
        self.film = FiLMConditioner(config.input_dim, cond_dim)
        vf_input_dim = (
            config.input_dim + config.time_embedding_dim + config.hidden_dim + config.stage_embedding_dim
        )
        self.vector_field = nn.Sequential(
            nn.Linear(vf_input_dim, config.vector_field_hidden_dim),
            nn.GELU(),
            nn.Linear(config.vector_field_hidden_dim, config.input_dim),
        )

    def encode_stage_pair(self, stage_src: int, stage_tgt: int) -> int:
        return int(stage_src * self.config.num_stages + stage_tgt)

    def encode_stage_pair_tensor(self, stage_src: int, stage_tgt: int, n: int, device: torch.device) -> Tensor:
        return torch.full((n,), self.encode_stage_pair(stage_src, stage_tgt), dtype=torch.long, device=device)

    def forward_set_context(self, x_set: Tensor, mask: Tensor | None = None, **kwargs: object) -> Tensor:
        return self.encoder(x_set, mask=mask)

    def forward_vector_field(self, x_t: Tensor, t: Tensor, c_s: Tensor, stage_pair_id: Tensor, wes_features: Tensor | None = None, **kwargs: object) -> Tensor:
        if c_s.ndim == 1:
            c_s = c_s.unsqueeze(0)
        if c_s.shape[0] == 1 and x_t.shape[0] > 1:
            c_s = c_s.expand(x_t.shape[0], -1)
        if stage_pair_id.ndim == 0:
            stage_pair_id = stage_pair_id.repeat(x_t.shape[0])
        stage_emb = self.stage_embedding(stage_pair_id)
        if not self.config.use_stage_embedding:
            stage_emb = torch.zeros_like(stage_emb)
        time_emb = self.time_embedding(t)
        cond = torch.cat([c_s, stage_emb], dim=-1)
        x_mod = self.film(x_t, cond)
        inp = torch.cat([x_mod, time_emb, c_s, stage_emb], dim=-1)
        return self.vector_field(inp)

    def integrate_euler(
        self,
        x0: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
    ) -> Tensor:
        """Integrate vector field with explicit Euler from t=0 to t=1."""
        x = x0
        dt = 1.0 / float(num_steps)
        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(x_t=x, t=t, c_s=c_s, stage_pair_id=stage_pair_id)
            x = x + dt * v
        return x

    def integrate_euler_maruyama(
        self,
        x0: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
        sigma: float = 0.0,
    ) -> Tensor:
        """Euler-Maruyama integration; sigma=0 recovers pure Euler."""
        x = x0
        dt = 1.0 / float(num_steps)
        sqrt_dt = dt ** 0.5
        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(x_t=x, t=t, c_s=c_s, stage_pair_id=stage_pair_id)
            x = x + dt * v
            if sigma > 0.0:
                x = x + sigma * sqrt_dt * torch.randn_like(x)
        return x


class NoContextFlowModel(nn.Module):
    """Ablation model with no set context (v(x,t) only)."""

    def __init__(self, config: StageBridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.time_embedding = SinusoidalTimeEmbedding(config.time_embedding_dim)
        self.vector_field = nn.Sequential(
            nn.Linear(config.input_dim + config.time_embedding_dim, config.vector_field_hidden_dim),
            nn.GELU(),
            nn.Linear(config.vector_field_hidden_dim, config.input_dim),
        )

    def encode_stage_pair(self, stage_src: int, stage_tgt: int) -> int:
        return 0

    def encode_stage_pair_tensor(self, stage_src: int, stage_tgt: int, n: int, device: torch.device) -> Tensor:
        return torch.zeros((n,), dtype=torch.long, device=device)

    def forward_set_context(self, x_set: Tensor, mask: Tensor | None = None, **kwargs: object) -> Tensor:
        if x_set.ndim == 2:
            return torch.zeros((1, self.config.hidden_dim), device=x_set.device, dtype=x_set.dtype)
        return torch.zeros((x_set.shape[0], self.config.hidden_dim), device=x_set.device, dtype=x_set.dtype)

    def forward_vector_field(self, x_t: Tensor, t: Tensor, c_s: Tensor, stage_pair_id: Tensor, wes_features: Tensor | None = None, **kwargs: object) -> Tensor:
        time_emb = self.time_embedding(t)
        inp = torch.cat([x_t, time_emb], dim=-1)
        return self.vector_field(inp)

    def integrate_euler(
        self,
        x0: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
    ) -> Tensor:
        """Integrate vector field with explicit Euler from t=0 to t=1."""
        x = x0
        dt = 1.0 / float(num_steps)
        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(x_t=x, t=t, c_s=c_s, stage_pair_id=stage_pair_id)
            x = x + dt * v
        return x

    def integrate_euler_maruyama(
        self,
        x0: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
        sigma: float = 0.0,
    ) -> Tensor:
        """Euler-Maruyama integration; sigma=0 recovers pure Euler."""
        x = x0
        dt = 1.0 / float(num_steps)
        sqrt_dt = dt ** 0.5
        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(x_t=x, t=t, c_s=c_s, stage_pair_id=stage_pair_id)
            x = x + dt * v
            if sigma > 0.0:
                x = x + sigma * sqrt_dt * torch.randn_like(x)
        return x


class LinearTransitionBaseline:
    """Linear one-step latent map fitted by least squares."""

    def __init__(self) -> None:
        self.weight: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, x_src: np.ndarray, y_tgt: np.ndarray, ridge: float = 1e-3) -> None:
        if x_src.ndim != 2 or y_tgt.ndim != 2:
            raise ValueError("x_src and y_tgt must be 2D arrays.")
        if x_src.shape[1] != y_tgt.shape[1]:
            raise ValueError("Input and output dimensions must match.")

        X = np.concatenate([x_src, np.ones((x_src.shape[0], 1), dtype=x_src.dtype)], axis=1)
        y = y_tgt
        d = X.shape[1]
        reg = ridge * np.eye(d, dtype=X.dtype)
        w = np.linalg.solve(X.T @ X + reg, X.T @ y)
        self.weight = w[:-1, :]
        self.bias = w[-1, :]

    def predict(self, x_src: np.ndarray) -> np.ndarray:
        if self.weight is None or self.bias is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return x_src @ self.weight + self.bias


def ot_barycentric_one_step(coupling: Tensor, x_tgt: Tensor) -> Tensor:
    """Compute barycentric mapped points from OT coupling."""
    row_sums = coupling.sum(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = coupling / row_sums
    return normalized @ x_tgt
