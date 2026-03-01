"""StageBridge transformer-first progression model."""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from stagebridge.models.layers import FiLMConditioner, ISAB, PMA, SAB, SinusoidalTimeEmbedding
from stagebridge.utils.types import StageBridgeConfig


class StageBridgeModel(nn.Module):
    """Set Transformer conditioned continuous vector field model."""

    def __init__(self, config: StageBridgeConfig) -> None:
        super().__init__()
        self.config = replace(config)

        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.isab1 = ISAB(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_inducing_points=config.num_inducing_points,
            dropout=config.dropout,
        )
        self.isab2 = ISAB(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_inducing_points=config.num_inducing_points,
            dropout=config.dropout,
        )
        self.sab = SAB(dim=config.hidden_dim, num_heads=config.num_heads, dropout=config.dropout)
        self.pma = PMA(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_seed_vectors=config.num_seed_vectors,
            dropout=config.dropout,
        )
        self.context_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )

        self.time_embedding = SinusoidalTimeEmbedding(config.time_embedding_dim)

        num_stage_pairs = config.num_stages * config.num_stages
        self.stage_pair_embedding = nn.Embedding(num_stage_pairs, config.stage_embedding_dim)

        condition_dim = config.hidden_dim + config.stage_embedding_dim
        self.film = FiLMConditioner(feature_dim=config.input_dim, condition_dim=condition_dim)

        vf_input_dim = (
            config.input_dim
            + config.time_embedding_dim
            + config.hidden_dim
            + config.stage_embedding_dim
        )
        self.vector_field = nn.Sequential(
            nn.Linear(vf_input_dim, config.vector_field_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.vector_field_hidden_dim, config.vector_field_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.vector_field_hidden_dim, config.input_dim),
        )

    def encode_stage_pair(self, stage_src: int, stage_tgt: int) -> int:
        """Encode directed stage pair to embedding id."""
        return int(stage_src * self.config.num_stages + stage_tgt)

    def encode_stage_pair_tensor(
        self,
        stage_src: int,
        stage_tgt: int,
        n: int,
        device: torch.device,
    ) -> Tensor:
        """Create repeated stage-pair id tensor for a batch of size ``n``."""
        pair_id = self.encode_stage_pair(stage_src=stage_src, stage_tgt=stage_tgt)
        return torch.full((n,), pair_id, dtype=torch.long, device=device)

    def forward_set_context(self, x_set: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode source-stage cell set into context embedding c_s."""
        squeeze = False
        if x_set.ndim == 2:
            x_set = x_set.unsqueeze(0)
            squeeze = True

        h = self.input_projection(x_set)
        h = self.isab1(h, mask=mask)
        h = self.isab2(h, mask=mask)
        h = self.sab(h, mask=mask)
        pooled = self.pma(h, mask=mask)  # (B, k, D)
        context = self.context_head(pooled[:, 0, :])

        if squeeze:
            return context[0:1]
        return context

    def _broadcast_condition(
        self,
        c_s: Tensor,
        stage_pair_id: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        if c_s.ndim == 1:
            c_s = c_s.unsqueeze(0)
        if c_s.shape[0] == 1 and batch_size > 1:
            c_s = c_s.expand(batch_size, -1)
        if stage_pair_id.ndim == 0:
            stage_pair_id = stage_pair_id.repeat(batch_size)
        if stage_pair_id.shape[0] == 1 and batch_size > 1:
            stage_pair_id = stage_pair_id.expand(batch_size)
        return c_s, stage_pair_id

    def forward_vector_field(
        self,
        x_t: Tensor,
        t: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
    ) -> Tensor:
        """Predict velocity v_phi(x_t, t, c_s, e_stage)."""
        n = x_t.shape[0]
        c_s, stage_pair_id = self._broadcast_condition(c_s=c_s, stage_pair_id=stage_pair_id, batch_size=n)

        stage_emb = self.stage_pair_embedding(stage_pair_id)
        if not self.config.use_stage_embedding:
            stage_emb = torch.zeros_like(stage_emb)
        time_emb = self.time_embedding(t)

        cond = torch.cat([c_s, stage_emb], dim=-1)
        x_mod = self.film(x_t, cond)

        vf_input = torch.cat([x_mod, time_emb, c_s, stage_emb], dim=-1)
        return self.vector_field(vf_input)

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

    def forward(
        self,
        x_set: Tensor,
        x_t: Tensor,
        t: Tensor,
        stage_src: int,
        stage_tgt: int,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Convenience forward path for training objectives."""
        c_s = self.forward_set_context(x_set=x_set, mask=mask)
        stage_pair = self.encode_stage_pair_tensor(
            stage_src=stage_src,
            stage_tgt=stage_tgt,
            n=x_t.shape[0],
            device=x_t.device,
        )
        return self.forward_vector_field(x_t=x_t, t=t, c_s=c_s, stage_pair_id=stage_pair)


def build_stagebridge_model(config: StageBridgeConfig) -> StageBridgeModel:
    """Factory helper to instantiate the canonical StageBridge model."""
    return StageBridgeModel(config=config)
