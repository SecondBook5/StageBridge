"""StageBridge transformer-first progression model."""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from stagebridge.models.layers import CrossAttentionDrift, FiLMConditioner, ISAB, PMA, SAB, SinusoidalTimeEmbedding
from stagebridge.utils.types import StageBridgeConfig

# Lazy import to avoid circular / heavy dependency at module load time
_DirichletStageHead = None


def _get_dirichlet_head_cls() -> type:
    global _DirichletStageHead
    if _DirichletStageHead is None:
        from stagebridge.models.dirichlet_stage import DirichletStageHead
        _DirichletStageHead = DirichletStageHead
    return _DirichletStageHead


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
            use_spatial_rpe=config.use_spatial_rpe,
            rpe_hidden=config.rpe_hidden_dim,
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

        # WES feature projection: (wes_feature_dim,) → (wes_hidden_dim,),
        # concatenated onto stage_emb before entering any drift head.
        if config.use_wes_features:
            self.wes_proj = nn.Sequential(
                nn.Linear(config.wes_feature_dim, config.wes_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.wes_hidden_dim),
            )
        else:
            self.wes_proj = None  # type: ignore[assignment]

        # LR signaling feature projection (same pattern as WES)
        if config.use_lr_features:
            self.lr_proj = nn.Sequential(
                nn.Linear(config.lr_feature_dim, config.lr_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.lr_hidden_dim),
            )
        else:
            self.lr_proj = None  # type: ignore[assignment]

        # Spatial niche composition conditioning: fuses averaged cell-type
        # composition from Tangram KNN into the set context representation.
        if config.use_spatial_niche:
            self.spatial_niche_proj = nn.Sequential(
                nn.Linear(config.spatial_niche_dim, config.spatial_niche_hidden),
                nn.GELU(),
                nn.LayerNorm(config.spatial_niche_hidden),
                nn.Linear(config.spatial_niche_hidden, config.hidden_dim),
            )
        else:
            self.spatial_niche_proj = None  # type: ignore[assignment]

        # Dirichlet stage assignment head
        if config.use_dirichlet_head:
            DirichletCls = _get_dirichlet_head_cls()
            self.dirichlet_head = DirichletCls(
                input_dim=config.input_dim,
                num_stages=config.num_stages,
                hidden_dim=config.dirichlet_hidden_dim,
            )
        else:
            self.dirichlet_head = None  # type: ignore[assignment]

        # Effective stage dim seen by the drift head
        # (stage_emb [+ wes_hidden] [+ lr_hidden])
        _eff_stage_dim = config.stage_embedding_dim
        if config.use_wes_features:
            _eff_stage_dim += config.wes_hidden_dim
        if config.use_lr_features:
            _eff_stage_dim += config.lr_hidden_dim

        condition_dim = config.hidden_dim + _eff_stage_dim
        self.film = FiLMConditioner(feature_dim=config.input_dim, condition_dim=condition_dim)

        if config.use_cross_attn_drift:
            # Cross-attention drift: x_t query attends over k niche context tokens.
            # The MLP vector_field is not built in this mode.
            self.drift_attn = CrossAttentionDrift(
                input_dim=config.input_dim,
                context_dim=config.hidden_dim,
                time_dim=config.time_embedding_dim,
                stage_dim=_eff_stage_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            self.vector_field = None  # type: ignore[assignment]
        else:
            # Original MLP drift.
            vf_input_dim = (
                config.input_dim
                + config.time_embedding_dim
                + config.hidden_dim
                + _eff_stage_dim
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
            self.drift_attn = None  # type: ignore[assignment]

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

    def forward_set_context(
        self,
        x_set: Tensor,
        mask: Tensor | None = None,
        niche_coords: Tensor | None = None,
        spatial_niche: Tensor | None = None,
    ) -> Tensor:
        """Encode source-stage cell set into context representation.

        Parameters
        ----------
        x_set : (N, dim) or (B, N, dim)
            Input token set (source cells ++ niche tokens when using spatial context).
        mask : optional validity mask
        niche_coords : (m_niche, 2) or (B, m_niche, 2), optional
            Spatial (x, y) coordinates for the niche token tail of ``x_set``.
            When provided and ``config.use_spatial_rpe=True``, ISAB1 receives
            spatial RPE bias.
        spatial_niche : (N, spatial_niche_dim) or (B, N, spatial_niche_dim), optional
            Per-cell spatial niche composition vectors (Tangram KNN).
            When provided and ``config.use_spatial_niche=True``, the mean
            niche vector is projected and added to the pooled context.

        Returns
        -------
        Tensor
            Shape ``(B, D)`` in the standard MLP-drift mode, or
            ``(B, num_seed_vectors, D)`` in cross-attention drift mode.
        """
        squeeze = False
        if x_set.ndim == 2:
            x_set = x_set.unsqueeze(0)
            squeeze = True

        h = self.input_projection(x_set)

        # Build full spatial coordinate tensor for Spatial RPE if enabled.
        coords_3d: Tensor | None = None
        n_src = 0
        if self.config.use_spatial_rpe and niche_coords is not None:
            m_niche = niche_coords.shape[-2]
            n_total = x_set.shape[1]
            n_src = n_total - m_niche
            B = x_set.shape[0]
            nc = niche_coords if niche_coords.ndim == 3 else niche_coords.unsqueeze(0).expand(B, -1, -1)
            src_zeros = torch.zeros(B, n_src, 2, device=x_set.device, dtype=x_set.dtype)
            coords_3d = torch.cat([src_zeros, nc.to(dtype=x_set.dtype)], dim=1)  # (B, n_total, 2)

        h = self.isab1(h, mask=mask, coords=coords_3d, n_src=n_src)
        h = self.isab2(h, mask=mask)
        h = self.sab(h, mask=mask)
        pooled = self.pma(h, mask=mask)  # (B, k, D)

        if self.config.use_cross_attn_drift:
            context = self.context_head(pooled)   # (B, k, D) — applied token-wise

            # Fuse spatial niche: project mean niche vector → (B, 1, D) and add
            if self.config.use_spatial_niche and spatial_niche is not None and self.spatial_niche_proj is not None:
                sn = spatial_niche if spatial_niche.ndim == 3 else spatial_niche.unsqueeze(0)
                niche_mean = sn.mean(dim=1, keepdim=True)  # (B, 1, niche_dim)
                niche_ctx = self.spatial_niche_proj(niche_mean)  # (B, 1, hidden_dim)
                context = context + niche_ctx  # broadcast over k tokens

            if squeeze:
                return context[0:1]
            return context
        else:
            context = self.context_head(pooled[:, 0, :])  # (B, D)

            # Fuse spatial niche: project mean niche vector → (B, D) and add
            if self.config.use_spatial_niche and spatial_niche is not None and self.spatial_niche_proj is not None:
                sn = spatial_niche if spatial_niche.ndim == 3 else spatial_niche.unsqueeze(0)
                niche_mean = sn.mean(dim=1)  # (B, niche_dim)
                niche_ctx = self.spatial_niche_proj(niche_mean)  # (B, hidden_dim)
                context = context + niche_ctx

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
        # 2D: (1, dim) → (B, dim); 3D tokens: (1, k, dim) → (B, k, dim)
        if c_s.shape[0] == 1 and batch_size > 1:
            c_s = c_s.expand(batch_size, *c_s.shape[1:])
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
        wes_features: Tensor | None = None,
        lr_features: Tensor | None = None,
    ) -> Tensor:
        """Predict velocity v_phi(x_t, t, c_s, e_stage[, wes][, lr]).

        ``c_s`` may be either:

        * shape ``(B, D)``   — single pooled vector (MLP drift mode)
        * shape ``(B, k, D)`` — k context tokens (cross-attention drift mode)

        ``wes_features``: optional ``(B, wes_feature_dim)`` somatic feature tensor.
        ``lr_features``: optional ``(B, lr_feature_dim)`` LR signaling tensor.
        """
        n = x_t.shape[0]
        c_s, stage_pair_id = self._broadcast_condition(c_s=c_s, stage_pair_id=stage_pair_id, batch_size=n)

        stage_emb = self.stage_pair_embedding(stage_pair_id)
        if not self.config.use_stage_embedding:
            stage_emb = torch.zeros_like(stage_emb)

        # Optionally augment stage_emb with projected WES features
        if self.config.use_wes_features:
            if wes_features is None:
                wes_h = torch.zeros(n, self.config.wes_hidden_dim, device=x_t.device, dtype=x_t.dtype)
            else:
                if wes_features.ndim == 1:
                    wes_features = wes_features.unsqueeze(0).expand(n, -1)
                wes_h = self.wes_proj(wes_features.to(x_t.dtype))
            stage_emb = torch.cat([stage_emb, wes_h], dim=-1)

        # Optionally augment stage_emb with projected LR signaling features
        if self.config.use_lr_features:
            if lr_features is None:
                lr_h = torch.zeros(n, self.config.lr_hidden_dim, device=x_t.device, dtype=x_t.dtype)
            else:
                if lr_features.ndim == 1:
                    lr_features = lr_features.unsqueeze(0).expand(n, -1)
                lr_h = self.lr_proj(lr_features.to(x_t.dtype))
            stage_emb = torch.cat([stage_emb, lr_h], dim=-1)

        time_emb = self.time_embedding(t)

        if c_s.ndim == 3:
            # Cross-attention drift: x_t query attends over niche context tokens.
            return self.drift_attn(
                x_t=x_t,
                time_emb=time_emb,
                context_tokens=c_s,
                stage_emb=stage_emb,
            )
        else:
            # Original FiLM-conditioned MLP drift.
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
        wes_features: Tensor | None = None,
        lr_features: Tensor | None = None,
    ) -> Tensor:
        """Integrate vector field with explicit Euler from t=0 to t=1."""
        x = x0
        dt = 1.0 / float(num_steps)
        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(
                x_t=x, t=t, c_s=c_s, stage_pair_id=stage_pair_id,
                wes_features=wes_features, lr_features=lr_features,
            )
            x = x + dt * v
        return x

    def integrate_euler_maruyama(
        self,
        x0: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 8,
        sigma: float = 0.0,
        wes_features: Tensor | None = None,
        lr_features: Tensor | None = None,
    ) -> Tensor:
        """Integrate with Euler-Maruyama (SDE) from t=0 to t=1.

        When ``sigma = 0`` output is identical to :meth:`integrate_euler`.
        When ``sigma > 0`` adds isotropic Brownian noise: ``x += sigma * sqrt(dt) * z``.
        """
        x = x0
        dt = 1.0 / float(num_steps)
        sqrt_dt = dt ** 0.5
        for k in range(num_steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device, dtype=x.dtype)
            v = self.forward_vector_field(
                x_t=x, t=t, c_s=c_s, stage_pair_id=stage_pair_id,
                wes_features=wes_features, lr_features=lr_features,
            )
            x = x + dt * v
            if sigma > 0.0:
                x = x + sigma * sqrt_dt * torch.randn_like(x)
        return x

    def predict_dirichlet(self, x: Tensor) -> Tensor | None:
        """Predict Dirichlet concentration parameters for stage assignment.

        Parameters
        ----------
        x : (B, input_dim) — raw cell latent representations.

        Returns
        -------
        Tensor or None
            Shape ``(B, num_stages)`` Dirichlet α if head is enabled, else None.
        """
        if self.dirichlet_head is None:
            return None
        return self.dirichlet_head(x)

    def forward(
        self,
        x_set: Tensor,
        x_t: Tensor,
        t: Tensor,
        stage_src: int,
        stage_tgt: int,
        mask: Tensor | None = None,
        wes_features: Tensor | None = None,
        lr_features: Tensor | None = None,
        niche_coords: Tensor | None = None,
        spatial_niche: Tensor | None = None,
    ) -> Tensor:
        """Convenience forward path for training objectives."""
        c_s = self.forward_set_context(
            x_set=x_set, mask=mask,
            niche_coords=niche_coords, spatial_niche=spatial_niche,
        )
        stage_pair = self.encode_stage_pair_tensor(
            stage_src=stage_src,
            stage_tgt=stage_tgt,
            n=x_t.shape[0],
            device=x_t.device,
        )
        return self.forward_vector_field(
            x_t=x_t, t=t, c_s=c_s, stage_pair_id=stage_pair,
            wes_features=wes_features, lr_features=lr_features,
        )


def build_stagebridge_model(config: StageBridgeConfig) -> StageBridgeModel:
    """Factory helper to instantiate the canonical StageBridge model."""
    return StageBridgeModel(config=config)
