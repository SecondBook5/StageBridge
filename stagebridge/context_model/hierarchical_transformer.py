"""Hierarchical typed transformer context encoder for StageBridge v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from stagebridge.context_model.set_encoder import FeedForwardBlock, ISAB, PMA, SAB, SetContextSummary


DATASET_TO_ID = {
    "luad_evo": 0,
    "brainmets": 1,
}


def dataset_name_to_id(name: str | None) -> int:
    """Map a dataset name into a stable embedding index."""
    key = str(name or "luad_evo").strip().lower()
    return int(DATASET_TO_ID.get(key, 0))


@dataclass(slots=True, frozen=True)
class GroupEncodingDiagnostics:
    """Compact diagnostics for one biological token group."""

    group_name: str
    token_count: int
    mean_confidence: float


class _GroupSetEncoder(nn.Module):
    """Small set encoder for one biological token group."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        num_inducing_points: int,
        num_summary_tokens: int,
        dropout: float,
        use_spatial_rpe: bool,
    ) -> None:
        super().__init__()
        self.isab = ISAB(
            dim=hidden_dim,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
            use_spatial_rpe=use_spatial_rpe,
        )
        self.sab = SAB(dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.pma = PMA(dim=hidden_dim, num_heads=num_heads, num_seed_vectors=num_summary_tokens, dropout=dropout)

    def forward(
        self,
        tokens: Tensor,
        *,
        coords: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if return_attention:
            h, isab_attention = self.isab(tokens, coords=coords, n_src=0, return_attention=True)
            h, sab_attention = self.sab(h, return_attention=True)
            pooled, pma_attention = self.pma(h, return_attention=True)
            return pooled, {
                "group_isab_inducing_to_tokens": isab_attention["inducing_to_tokens"],
                "group_isab_tokens_to_inducing": isab_attention["tokens_to_inducing"],
                "group_sab_attention": sab_attention,
                "group_pma_attention": pma_attention,
            }
        h = self.isab(tokens, coords=coords, n_src=0)
        h = self.sab(h)
        pooled = self.pma(h)
        return pooled, {}


class _ResidualContextHead(nn.Module):
    """Deeper context head with residual connection for richer final representation."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act2 = nn.GELU()
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.drop(self.act1(self.fc1(x)))
        h = self.ln1(h + x)
        h2 = self.drop(self.act2(self.fc2(h)))
        return self.ln2(h2 + h)


class TypedHierarchicalTransformerEncoder(nn.Module):
    """Hierarchical typed transformer with per-group encoders and fusion queries."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        *,
        num_heads: int = 4,
        num_inducing_points: int = 16,
        dropout: float = 0.1,
        num_token_types: int = 4,
        num_group_summary_tokens: int = 2,
        num_fusion_queries: int = 7,
        dataset_embedding_dim: int = 16,
        num_datasets: int = 4,
        num_edges: int = 8,
        use_spatial_rpe: bool = True,
        use_confidence_gate: bool = True,
        token_dropout_rate: float = 0.05,
        use_relation_tokens: bool = True,
        group_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_token_types = int(num_token_types)
        self.num_group_summary_tokens = int(num_group_summary_tokens)
        self.num_fusion_queries = int(num_fusion_queries)
        self.token_dropout_rate = float(token_dropout_rate)
        self.use_relation_tokens = bool(use_relation_tokens)
        self.group_names = tuple(group_names or [f"group_{idx}" for idx in range(self.num_token_types)])
        self.query_role_names = tuple(
            [
                "source_stage",
                "target_stage",
                "transition",
                *[str(name) for name in self.group_names],
            ][: self.num_fusion_queries]
        )
        while len(self.query_role_names) < self.num_fusion_queries:
            self.query_role_names = (*self.query_role_names, f"query_{len(self.query_role_names)}")

        self.input_projection = nn.Linear(int(input_dim), int(hidden_dim))
        self.token_type_embedding = nn.Embedding(self.num_token_types, int(hidden_dim))
        self.summary_role_embedding = nn.Embedding(self.num_group_summary_tokens, int(hidden_dim))
        self.coord_projection = nn.Sequential(
            nn.Linear(2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.confidence_gate = (
            nn.Sequential(
                nn.Linear(1, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.Sigmoid(),
            )
            if use_confidence_gate
            else None
        )
        self.group_embeddings = nn.Embedding(self.num_token_types, int(hidden_dim))
        self.dataset_embedding = nn.Embedding(int(num_datasets), int(hidden_dim))
        self.dataset_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.dataset_film = nn.Linear(int(hidden_dim), int(hidden_dim) * 2)
        self.edge_embedding = nn.Embedding(int(num_edges), int(hidden_dim))
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_fusion_queries, int(hidden_dim)) * 0.02)
        self.query_role_embedding = nn.Embedding(self.num_fusion_queries, int(hidden_dim))
        self.group_confidence_proj = nn.Sequential(
            nn.Linear(1, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.empty_group_tokens = nn.Parameter(
            torch.randn(self.num_token_types, self.num_group_summary_tokens, int(hidden_dim)) * 0.02
        )
        self.relation_pair_indices = [
            (left_idx, right_idx)
            for left_idx in range(self.num_token_types)
            for right_idx in range(left_idx + 1, self.num_token_types)
        ]
        self.relation_pair_names = tuple(
            f"{self.group_names[left_idx]}__{self.group_names[right_idx]}"
            for left_idx, right_idx in self.relation_pair_indices
        )

        self.group_encoders = nn.ModuleList(
            [
                _GroupSetEncoder(
                    int(hidden_dim),
                    num_heads=int(num_heads),
                    num_inducing_points=int(num_inducing_points),
                    num_summary_tokens=int(num_group_summary_tokens),
                    dropout=float(dropout),
                    use_spatial_rpe=bool(use_spatial_rpe),
                )
                for _ in range(self.num_token_types)
            ]
        )
        self.fusion_attn = nn.MultiheadAttention(int(hidden_dim), int(num_heads), dropout=float(dropout), batch_first=True)
        self.fusion_ln1 = nn.LayerNorm(int(hidden_dim))
        self.fusion_ff = FeedForwardBlock(dim=int(hidden_dim), dropout=float(dropout))
        self.fusion_ln2 = nn.LayerNorm(int(hidden_dim))
        self.fusion_sab = SAB(dim=int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout))
        self.fusion_sab2 = SAB(dim=int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout))
        self.relation_mlp = nn.Sequential(
            nn.Linear(int(hidden_dim) * 4, int(hidden_dim) * 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.context_head = _ResidualContextHead(int(hidden_dim), dropout=float(dropout))

    def _infer_token_type_ids(self, tokens: Tensor) -> Tensor:
        usable = min(int(tokens.shape[-1]), self.num_token_types)
        return tokens[..., :usable].argmax(dim=-1).long()

    def _normalize_by_group(self, tokens: Tensor, token_type_ids: Tensor) -> Tensor:
        normalized = tokens.clone()
        for group_idx in range(self.num_token_types):
            group_mask = token_type_ids == group_idx
            if not torch.any(group_mask):
                continue
            group_values = normalized[group_mask]
            group_mean = group_values.mean(dim=0, keepdim=True)
            group_std = group_values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            normalized[group_mask] = (group_values - group_mean) / group_std
        return normalized

    @staticmethod
    def _normalize_coords(coords: Tensor | None) -> Tensor | None:
        if coords is None:
            return None
        centered = coords - coords.mean(dim=-2, keepdim=True)
        scale = centered.norm(dim=-1).quantile(0.95).clamp_min(1e-4)
        return centered / scale

    def _apply_training_dropout(
        self,
        tokens: Tensor,
        token_type_ids: Tensor,
        token_confidence: Tensor | None,
        token_coords: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if not self.training or self.token_dropout_rate <= 0.0 or tokens.shape[-2] <= 1:
            return tokens, token_type_ids, token_confidence, token_coords
        keep_prob = 1.0 - self.token_dropout_rate
        keep_mask = torch.rand(tokens.shape[:-1], device=tokens.device) < keep_prob
        keep_mask[..., 0] = True
        dropped_tokens = tokens * keep_mask.unsqueeze(-1).to(tokens.dtype)
        dropped_confidence = None if token_confidence is None else token_confidence * keep_mask.to(token_confidence.dtype)
        dropped_coords = None if token_coords is None else token_coords * keep_mask.unsqueeze(-1).to(token_coords.dtype)
        return dropped_tokens, token_type_ids, dropped_confidence, dropped_coords

    def _encode_single_group(
        self,
        batch_tokens: Tensor,
        batch_type_ids: Tensor,
        *,
        batch_coords: Tensor | None,
        batch_confidence: Tensor | None,
        group_idx: int,
        return_attention: bool,
    ) -> tuple[Tensor, dict[str, Tensor], GroupEncodingDiagnostics]:
        group_mask = batch_type_ids == int(group_idx)
        if not torch.any(group_mask):
            summary = self.empty_group_tokens[int(group_idx)].unsqueeze(0)
            summary = summary + self.group_embeddings.weight[int(group_idx)].view(1, 1, -1)
            summary = summary + self.summary_role_embedding.weight.unsqueeze(0)
            diagnostics = GroupEncodingDiagnostics(
                group_name=self.group_names[int(group_idx)],
                token_count=0,
                mean_confidence=0.0,
            )
            return summary, {}, diagnostics

        tokens = batch_tokens[group_mask]
        token_h = self.input_projection(tokens)
        token_h = token_h + self.token_type_embedding.weight[int(group_idx)].view(1, -1)
        coords = None
        if batch_coords is not None:
            coords = batch_coords[group_mask]
            token_h = token_h + self.coord_projection(coords)
        mean_confidence = 1.0
        if batch_confidence is not None and self.confidence_gate is not None:
            confidence = batch_confidence[group_mask]
            gate = self.confidence_gate(confidence.unsqueeze(-1))
            token_h = token_h * gate
            mean_confidence = float(confidence.detach().mean().item())

        summary, attention = self.group_encoders[int(group_idx)](
            token_h.unsqueeze(0),
            coords=None if coords is None else coords.unsqueeze(0),
            return_attention=return_attention,
        )
        summary = summary + self.group_embeddings.weight[int(group_idx)].view(1, 1, -1)
        summary = summary + self.summary_role_embedding.weight.unsqueeze(0)
        if batch_confidence is not None:
            confidence_bias = self.group_confidence_proj(
                torch.tensor([[mean_confidence]], device=summary.device, dtype=summary.dtype)
            ).unsqueeze(1)
            summary = summary + confidence_bias
        diagnostics = GroupEncodingDiagnostics(
            group_name=self.group_names[int(group_idx)],
            token_count=int(group_mask.sum().item()),
            mean_confidence=float(mean_confidence),
        )
        return summary, attention, diagnostics

    def _apply_dataset_conditioning(self, tokens: Tensor, dataset_bias: Tensor) -> Tensor:
        gamma_beta = self.dataset_film(dataset_bias.squeeze(1)).unsqueeze(1)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        return gamma * tokens + beta

    def _build_relation_tokens(
        self,
        group_summaries: list[Tensor],
        *,
        group_diag_rows: list[dict[str, Any]],
    ) -> tuple[Tensor | None, dict[str, float]]:
        if not self.use_relation_tokens or len(group_summaries) < 2:
            return None, {}
        group_means = [summary.mean(dim=0) for summary in group_summaries]
        relation_tokens: list[Tensor] = []
        relation_scores: dict[str, float] = {}
        for relation_name, (left_idx, right_idx) in zip(self.relation_pair_names, self.relation_pair_indices, strict=False):
            left = group_means[left_idx]
            right = group_means[right_idx]
            relation_input = torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)
            relation_token = self.relation_mlp(relation_input)
            relation_tokens.append(relation_token)
            left_conf = float(group_diag_rows[left_idx]["mean_confidence"])
            right_conf = float(group_diag_rows[right_idx]["mean_confidence"])
            relation_scores[str(relation_name)] = float(0.5 * (left_conf + right_conf))
        if not relation_tokens:
            return None, relation_scores
        return torch.stack(relation_tokens, dim=0), relation_scores

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
        *,
        token_type_ids: Tensor | None = None,
        token_coords: Tensor | None = None,
        token_confidence: Tensor | None = None,
        dataset_ids: Tensor | None = None,
        edge_ids: Tensor | None = None,
        return_attention: bool = False,
    ) -> SetContextSummary:
        del mask  # unused for the hierarchical encoder; masking is carried by per-group selection.
        squeeze = False
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        if token_type_ids is None:
            token_type_ids = self._infer_token_type_ids(tokens)
        if token_type_ids.ndim == 1:
            token_type_ids = token_type_ids.unsqueeze(0)
        if token_coords is not None and token_coords.ndim == 2:
            token_coords = token_coords.unsqueeze(0)
        if token_confidence is not None and token_confidence.ndim == 1:
            token_confidence = token_confidence.unsqueeze(0)
        if dataset_ids is None:
            dataset_ids = torch.zeros((tokens.shape[0],), dtype=torch.long, device=tokens.device)
        elif dataset_ids.ndim == 0:
            dataset_ids = dataset_ids.unsqueeze(0)
        if edge_ids is None:
            edge_ids = torch.zeros((tokens.shape[0],), dtype=torch.long, device=tokens.device)
        elif edge_ids.ndim == 0:
            edge_ids = edge_ids.unsqueeze(0)

        normalized_coords = self._normalize_coords(token_coords)
        tokens, token_type_ids, token_confidence, normalized_coords = self._apply_training_dropout(
            tokens,
            token_type_ids.long(),
            token_confidence,
            normalized_coords,
        )
        normalized_tokens = torch.stack(
            [
                self._normalize_by_group(batch_tokens, batch_type_ids)
                for batch_tokens, batch_type_ids in zip(tokens, token_type_ids, strict=False)
            ],
            dim=0,
        )

        pooled_contexts: list[Tensor] = []
        fused_tokens_list: list[Tensor] = []
        group_token_list: list[Tensor] = []
        relation_token_list: list[Tensor] = []
        attention_maps: dict[str, list[Tensor]] = {}
        group_diagnostics: list[dict[str, Any]] = []

        for batch_idx in range(normalized_tokens.shape[0]):
            group_summaries: list[Tensor] = []
            group_diag_rows: list[dict[str, Any]] = []
            for group_idx in range(self.num_token_types):
                summary, group_attention, diag = self._encode_single_group(
                    normalized_tokens[batch_idx],
                    token_type_ids[batch_idx],
                    batch_coords=None if normalized_coords is None else normalized_coords[batch_idx],
                    batch_confidence=None if token_confidence is None else token_confidence[batch_idx],
                    group_idx=group_idx,
                    return_attention=return_attention,
                )
                group_summaries.append(summary[0])
                group_diag_rows.append(
                    {
                        "group_name": diag.group_name,
                        "token_count": diag.token_count,
                        "mean_confidence": diag.mean_confidence,
                    }
                )
                if return_attention:
                    for name, tensor in group_attention.items():
                        attention_maps.setdefault(f"{self.group_names[group_idx]}_{name}", []).append(tensor[0])

            summary_bank = torch.cat(group_summaries, dim=0).unsqueeze(0)
            relation_tokens, relation_scores = self._build_relation_tokens(
                group_summaries,
                group_diag_rows=group_diag_rows,
            )
            relation_token_count = 0
            if relation_tokens is not None:
                relation_token_count = int(relation_tokens.shape[0])
                summary_bank = torch.cat([summary_bank, relation_tokens.unsqueeze(0)], dim=1)
            dataset_bias = self.dataset_proj(self.dataset_embedding(dataset_ids[batch_idx])).view(1, 1, -1)
            edge_bias = self.edge_embedding(edge_ids[batch_idx]).view(1, 1, -1)
            query_roles = self.query_role_embedding(
                torch.arange(self.num_fusion_queries, device=tokens.device, dtype=torch.long)
            ).unsqueeze(0)
            summary_bank = self._apply_dataset_conditioning(summary_bank, dataset_bias)
            queries = self.query_tokens + query_roles + dataset_bias + edge_bias
            queries = self._apply_dataset_conditioning(queries, dataset_bias)
            fused, fusion_attention = self.fusion_attn(
                query=queries,
                key=summary_bank,
                value=summary_bank,
                need_weights=bool(return_attention),
                average_attn_weights=False,
            )
            fused = self.fusion_ln1(queries + fused)
            fused = self.fusion_ln2(fused + self.fusion_ff(fused))
            fused = self.fusion_sab(fused)
            fused = self.fusion_sab2(fused)
            pooled_context = self.context_head(fused[:, -1, :])
            pooled_contexts.append(pooled_context[0])
            fused_tokens_list.append(fused[0])
            group_token_list.append(torch.cat(group_summaries, dim=0))
            if relation_tokens is not None:
                relation_token_list.append(relation_tokens)
            if return_attention:
                attention_maps.setdefault("fusion_query_attention", []).append(fusion_attention[0])

            if return_attention and fusion_attention is not None:
                query_attention = fusion_attention[0].mean(dim=0)
                global_attention = query_attention[-1]
                group_attention_scores: dict[str, float] = {}
                for group_idx, group_name in enumerate(self.group_names):
                    start = group_idx * self.num_group_summary_tokens
                    stop = start + self.num_group_summary_tokens
                    group_attention_scores[str(group_name)] = float(global_attention[start:stop].mean().item())
                relation_attention_scores: dict[str, float] = {}
                if relation_token_count > 0:
                    offset = self.num_token_types * self.num_group_summary_tokens
                    for relation_idx, relation_name in enumerate(self.relation_pair_names[:relation_token_count]):
                        relation_attention_scores[str(relation_name)] = float(global_attention[offset + relation_idx].item())
                query_role_scores: dict[str, dict[str, float]] = {}
                for query_idx, query_name in enumerate(self.query_role_names):
                    query_weights = query_attention[query_idx]
                    query_role_scores[str(query_name)] = {
                        str(group_name): float(query_weights[group_idx * self.num_group_summary_tokens:(group_idx + 1) * self.num_group_summary_tokens].mean().item())
                        for group_idx, group_name in enumerate(self.group_names)
                    }
            else:
                group_attention_scores = {}
                relation_attention_scores = {}
                query_role_scores = {}

            group_diagnostics.append(
                {
                    "group_token_counts": {row["group_name"]: row["token_count"] for row in group_diag_rows},
                    "group_mean_confidence": {row["group_name"]: row["mean_confidence"] for row in group_diag_rows},
                    "fusion_attention_by_group": group_attention_scores,
                    "fusion_attention_by_relation": relation_attention_scores,
                    "query_attention_by_group": query_role_scores,
                    "relation_prior_confidence": relation_scores,
                    "dataset_id": int(dataset_ids[batch_idx].item()),
                    "edge_id": int(edge_ids[batch_idx].item()),
                }
            )

        stacked_context = torch.stack(pooled_contexts, dim=0)
        stacked_fused_tokens = torch.stack(fused_tokens_list, dim=0)
        stacked_group_tokens = torch.stack(group_token_list, dim=0)
        stacked_relation_tokens = None if not relation_token_list else torch.stack(relation_token_list, dim=0)
        reduced_attention = {name: torch.stack(values, dim=0) for name, values in attention_maps.items()}
        diagnostics: dict[str, Any] = {
            "dataset_ids": [int(item.item()) for item in dataset_ids],
            "edge_ids": [int(item.item()) for item in edge_ids],
            "group_diagnostics": group_diagnostics,
            "query_role_names": list(self.query_role_names),
            "relation_pair_names": list(self.relation_pair_names),
        }
        if squeeze:
            return SetContextSummary(
                pooled_context=stacked_context[0],
                token_embeddings=stacked_fused_tokens[0],
                context_tokens=stacked_fused_tokens[0],
                group_summary_tokens=stacked_group_tokens[0],
                relation_tokens=None if stacked_relation_tokens is None else stacked_relation_tokens[0],
                attention_maps={key: value[0] for key, value in reduced_attention.items()},
                token_type_ids=token_type_ids[0],
                token_confidence=None if token_confidence is None else token_confidence[0],
                token_coords=None if normalized_coords is None else normalized_coords[0],
                diagnostics=diagnostics,
            )
        return SetContextSummary(
            pooled_context=stacked_context,
            token_embeddings=stacked_fused_tokens,
            context_tokens=stacked_fused_tokens,
            group_summary_tokens=stacked_group_tokens,
            relation_tokens=stacked_relation_tokens,
            attention_maps=reduced_attention,
            token_type_ids=token_type_ids,
            token_confidence=token_confidence,
            token_coords=normalized_coords,
            diagnostics=diagnostics,
        )


__all__ = ["TypedHierarchicalTransformerEncoder", "dataset_name_to_id"]
