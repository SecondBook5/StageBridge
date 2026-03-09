"""Communication-relay transformer and baseline models for StageBridge."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from stagebridge.context_model.set_encoder import FeedForwardBlock, SAB
from stagebridge.utils.types import CommunicationBag, CommunicationBatch


@dataclass(slots=True, frozen=True)
class CommunicationRelayOutput:
    """Classifier and context outputs from one communication model forward pass."""

    query_embeddings: Tensor
    bag_embeddings: Tensor
    query_logits: Tensor
    bag_logits: Tensor
    context_tokens: Tensor | None = None
    attention_maps: dict[str, Tensor] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


def _masked_mean(tokens: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    total = (tokens * weights).sum(dim=dim)
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return total / denom


def _aggregate_bag_logits(query_logits: Tensor, bag_index: Tensor, num_bags: int) -> Tensor:
    bag_logits = []
    for bag_idx in range(int(num_bags)):
        mask = bag_index == int(bag_idx)
        if not torch.any(mask):
            bag_logits.append(torch.tensor(0.0, device=query_logits.device, dtype=query_logits.dtype))
        else:
            bag_logits.append(query_logits[mask].mean())
    return torch.stack(bag_logits, dim=0)


def collate_communication_bags(bags: list[CommunicationBag]) -> CommunicationBatch:
    """Pad variable-sized communication bags into one batch."""
    total_queries = sum(bag.num_examples for bag in bags)
    if total_queries <= 0:
        raise ValueError("Cannot collate an empty communication bag list.")
    receiver_dim = int(bags[0].examples[0].receiver_embedding.shape[0])
    program_dim = int(bags[0].examples[0].receiver_programs.shape[0])
    sender_dim = int(bags[0].examples[0].sender_embeddings.shape[1])
    lr_dim = int(bags[0].examples[0].lr_token_features.shape[1]) if bags[0].examples[0].lr_token_features.size else 10
    response_dim = int(bags[0].examples[0].response_token_features.shape[1]) if bags[0].examples[0].response_token_features.size else 5
    relay_dim = int(bags[0].examples[0].relay_token_features.shape[1]) if bags[0].examples[0].relay_token_features.size else 6
    wes_dim = 0
    target_dim = 0
    max_sender = max(example.sender_embeddings.shape[0] for bag in bags for example in bag.examples)
    max_lr = max(max(example.lr_token_features.shape[0], 1) for bag in bags for example in bag.examples)
    max_response = max(max(example.response_token_features.shape[0], 1) for bag in bags for example in bag.examples)
    max_relay = max(max(example.relay_token_features.shape[0], 1) for bag in bags for example in bag.examples)
    for bag in bags:
        for example in bag.examples:
            if example.wes_features is not None:
                wes_dim = max(wes_dim, int(example.wes_features.shape[0]))
            if example.target_latent is not None:
                target_dim = max(target_dim, int(example.target_latent.shape[0]))

    receiver_embedding = torch.zeros((total_queries, receiver_dim), dtype=torch.float32)
    receiver_programs = torch.zeros((total_queries, program_dim), dtype=torch.float32)
    sender_embeddings = torch.zeros((total_queries, max_sender, sender_dim), dtype=torch.float32)
    sender_types = torch.zeros((total_queries, max_sender), dtype=torch.long)
    sender_offsets = torch.zeros((total_queries, max_sender, 2), dtype=torch.float32)
    ring_ids = torch.zeros((total_queries, max_sender), dtype=torch.long)
    lr_token_features = torch.zeros((total_queries, max_lr, lr_dim), dtype=torch.float32)
    response_token_features = torch.zeros((total_queries, max_response, response_dim), dtype=torch.float32)
    relay_token_features = torch.zeros((total_queries, max_relay, relay_dim), dtype=torch.float32)
    query_mask = torch.ones((total_queries,), dtype=torch.bool)
    sender_mask = torch.zeros((total_queries, max_sender), dtype=torch.bool)
    lr_mask = torch.zeros((total_queries, max_lr), dtype=torch.bool)
    response_mask = torch.zeros((total_queries, max_response), dtype=torch.bool)
    relay_mask = torch.zeros((total_queries, max_relay), dtype=torch.bool)
    edge_ids = torch.zeros((total_queries,), dtype=torch.long)
    bag_index = torch.zeros((total_queries,), dtype=torch.long)
    weak_labels = torch.tensor([bag.weak_label for bag in bags], dtype=torch.float32)
    wes_features = None if wes_dim <= 0 else torch.zeros((total_queries, wes_dim), dtype=torch.float32)
    target_latent = None if target_dim <= 0 else torch.zeros((total_queries, target_dim), dtype=torch.float32)
    sample_ids = [bag.sample_id for bag in bags]
    donor_ids = [bag.donor_id for bag in bags]
    label_sources = [bag.label_source for bag in bags]
    receiver_cell_ids: list[str | None] = []

    query_idx = 0
    for bag_idx, bag in enumerate(bags):
        for example in bag.examples:
            receiver_embedding[query_idx] = torch.as_tensor(example.receiver_embedding, dtype=torch.float32)
            receiver_programs[query_idx] = torch.as_tensor(example.receiver_programs, dtype=torch.float32)
            n_sender = int(example.sender_embeddings.shape[0])
            sender_embeddings[query_idx, :n_sender] = torch.as_tensor(example.sender_embeddings, dtype=torch.float32)
            sender_types[query_idx, :n_sender] = torch.as_tensor(example.sender_types, dtype=torch.long)
            sender_offsets[query_idx, :n_sender] = torch.as_tensor(example.sender_offsets, dtype=torch.float32)
            ring_ids[query_idx, :n_sender] = torch.as_tensor(example.ring_ids, dtype=torch.long)
            sender_mask[query_idx, :n_sender] = True
            n_lr = int(example.lr_token_features.shape[0])
            if n_lr > 0:
                lr_token_features[query_idx, :n_lr] = torch.as_tensor(example.lr_token_features, dtype=torch.float32)
                lr_mask[query_idx, :n_lr] = True
            n_response = int(example.response_token_features.shape[0])
            if n_response > 0:
                response_token_features[query_idx, :n_response] = torch.as_tensor(example.response_token_features, dtype=torch.float32)
                response_mask[query_idx, :n_response] = True
            n_relay = int(example.relay_token_features.shape[0])
            if n_relay > 0:
                relay_token_features[query_idx, :n_relay] = torch.as_tensor(example.relay_token_features, dtype=torch.float32)
                relay_mask[query_idx, :n_relay] = True
            if wes_features is not None and example.wes_features is not None and example.wes_features.size > 0:
                wes_features[query_idx, : example.wes_features.shape[0]] = torch.as_tensor(example.wes_features, dtype=torch.float32)
            if target_latent is not None and example.target_latent is not None and example.target_latent.size > 0:
                target_latent[query_idx, : example.target_latent.shape[0]] = torch.as_tensor(example.target_latent, dtype=torch.float32)
            edge_ids[query_idx] = int(example.edge_id)
            bag_index[query_idx] = int(bag_idx)
            receiver_cell_ids.append(example.receiver_cell_id)
            query_idx += 1

    return CommunicationBatch(
        receiver_embedding=receiver_embedding,
        receiver_programs=receiver_programs,
        sender_embeddings=sender_embeddings,
        sender_types=sender_types,
        sender_offsets=sender_offsets,
        ring_ids=ring_ids,
        lr_token_features=lr_token_features,
        response_token_features=response_token_features,
        relay_token_features=relay_token_features,
        query_mask=query_mask,
        sender_mask=sender_mask,
        lr_mask=lr_mask,
        response_mask=response_mask,
        relay_mask=relay_mask,
        edge_ids=edge_ids,
        weak_labels=weak_labels,
        bag_index=bag_index,
        sample_ids=sample_ids,
        donor_ids=donor_ids,
        label_sources=label_sources,
        receiver_cell_ids=receiver_cell_ids,
        wes_features=wes_features,
        target_latent=target_latent,
    )


class SenderEncoder(nn.Module):
    """Context-aware sender encoder over local niche tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.block = SAB(dim=hidden_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, tokens: Tensor, mask: Tensor, *, return_attention: bool = False) -> tuple[Tensor, Tensor | None]:
        if return_attention:
            encoded, attn = self.block(tokens, mask=mask, return_attention=True)
            return encoded, attn
        return self.block(tokens, mask=mask), None


class CommunicationEncoder(nn.Module):
    """Cross-attention from LR proposal tokens into sender-aware memory."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForwardBlock(hidden_dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        lr_tokens: Tensor,
        sender_tokens: Tensor,
        sender_mask: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        attn_out, attn = self.attn(
            query=lr_tokens,
            key=sender_tokens,
            value=sender_tokens,
            key_padding_mask=~sender_mask.bool(),
            need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        h = self.ln1(lr_tokens + attn_out)
        h = self.ln2(h + self.ff(h))
        return h, attn


class RelayEncoder(nn.Module):
    """Self-attention over intracellular response and relay-memory tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.block = SAB(dim=hidden_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, tokens: Tensor, mask: Tensor, *, return_attention: bool = False) -> tuple[Tensor, Tensor | None]:
        if return_attention:
            encoded, attn = self.block(tokens, mask=mask, return_attention=True)
            return encoded, attn
        return self.block(tokens, mask=mask), None


class ReceiverQueryBlock(nn.Module):
    """Focal receiver query over all encoded communication memory tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForwardBlock(hidden_dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        attn_out, attn = self.attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=~memory_mask.bool(),
            need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        h = self.ln1(query + attn_out)
        h = self.ln2(h + self.ff(h))
        return h, attn


class StageBridgeCommunicationModel(nn.Module):
    """Full communication-relay transformer for weakly supervised edge classification."""

    def __init__(
        self,
        receiver_dim: int,
        receiver_program_dim: int,
        sender_dim: int,
        lr_dim: int,
        response_dim: int,
        relay_dim: int,
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_sender_types: int = 16,
        num_ring_ids: int = 4,
        num_edges: int = 4,
        wes_dim: int = 0,
        use_lr_tokens: bool = True,
        use_response_tokens: bool = True,
        use_relay_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_lr_tokens = bool(use_lr_tokens)
        self.use_response_tokens = bool(use_response_tokens)
        self.use_relay_tokens = bool(use_relay_tokens)

        self.receiver_projection = nn.Linear(int(receiver_dim + receiver_program_dim + max(wes_dim, 0)), int(hidden_dim))
        self.sender_projection = nn.Linear(int(sender_dim), int(hidden_dim))
        self.lr_projection = nn.Linear(int(lr_dim), int(hidden_dim))
        self.response_projection = nn.Linear(int(response_dim), int(hidden_dim))
        self.relay_projection = nn.Linear(int(relay_dim), int(hidden_dim))
        self.sender_type_embedding = nn.Embedding(int(num_sender_types), int(hidden_dim))
        self.ring_embedding = nn.Embedding(int(num_ring_ids), int(hidden_dim))
        self.offset_projection = nn.Sequential(
            nn.Linear(2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.edge_embedding = nn.Embedding(int(num_edges), int(hidden_dim))
        self.token_type_embedding = nn.Embedding(5, int(hidden_dim))
        self.sender_encoder = SenderEncoder(int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout))
        self.communication_encoder = CommunicationEncoder(int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout))
        self.relay_encoder = RelayEncoder(int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout))
        self.receiver_query = ReceiverQueryBlock(int(hidden_dim), num_heads=int(num_heads), dropout=float(dropout))
        self.query_norm = nn.LayerNorm(int(hidden_dim))
        self.edge_heads = nn.ModuleList([nn.Linear(int(hidden_dim), 1) for _ in range(int(num_edges))])

    def _receiver_token(self, batch: CommunicationBatch) -> Tensor:
        features = [batch.receiver_embedding, batch.receiver_programs]
        if batch.wes_features is not None:
            features.append(batch.wes_features)
        receiver_inputs = torch.cat(features, dim=-1)
        receiver_token = self.receiver_projection(receiver_inputs)
        receiver_token = receiver_token + self.edge_embedding(batch.edge_ids)
        receiver_token = receiver_token + self.token_type_embedding.weight[0].view(1, -1)
        return receiver_token

    def _sender_tokens(self, batch: CommunicationBatch) -> Tensor:
        sender_tokens = self.sender_projection(batch.sender_embeddings)
        sender_tokens = sender_tokens + self.sender_type_embedding(batch.sender_types.clamp_min(0))
        sender_tokens = sender_tokens + self.ring_embedding(batch.ring_ids.clamp_min(0))
        sender_tokens = sender_tokens + self.offset_projection(batch.sender_offsets)
        sender_tokens = sender_tokens + self.token_type_embedding.weight[1].view(1, 1, -1)
        return sender_tokens

    def _select_edge_logits(self, embeddings: Tensor, edge_ids: Tensor) -> Tensor:
        logits = embeddings.new_zeros((embeddings.shape[0],))
        for edge_id, head in enumerate(self.edge_heads):
            mask = edge_ids == int(edge_id)
            if torch.any(mask):
                logits[mask] = head(embeddings[mask]).squeeze(-1)
        return logits

    def forward(self, batch: CommunicationBatch, *, return_attention: bool = False) -> CommunicationRelayOutput:
        receiver_token = self._receiver_token(batch)
        sender_tokens = self._sender_tokens(batch)
        encoded_sender, sender_attention = self.sender_encoder(
            sender_tokens,
            batch.sender_mask,
            return_attention=return_attention,
        )
        memory_parts = [encoded_sender]
        memory_masks = [batch.sender_mask]
        attention_maps: dict[str, Tensor] = {}
        if return_attention and sender_attention is not None:
            attention_maps["sender_self_attention"] = sender_attention

        if self.use_lr_tokens:
            lr_tokens = self.lr_projection(batch.lr_token_features) + self.token_type_embedding.weight[2].view(1, 1, -1)
            encoded_lr, lr_attention = self.communication_encoder(
                lr_tokens,
                encoded_sender,
                batch.sender_mask,
                return_attention=return_attention,
            )
            encoded_lr = encoded_lr * batch.lr_mask.unsqueeze(-1).to(encoded_lr.dtype)
            memory_parts.append(encoded_lr)
            memory_masks.append(batch.lr_mask)
            if return_attention and lr_attention is not None:
                attention_maps["communication_attention"] = lr_attention

        relay_parts: list[Tensor] = []
        relay_masks: list[Tensor] = []
        if self.use_response_tokens:
            response_tokens = self.response_projection(batch.response_token_features) + self.token_type_embedding.weight[3].view(1, 1, -1)
            relay_parts.append(response_tokens)
            relay_masks.append(batch.response_mask)
        if self.use_relay_tokens:
            relay_tokens = self.relay_projection(batch.relay_token_features) + self.token_type_embedding.weight[4].view(1, 1, -1)
            relay_parts.append(relay_tokens)
            relay_masks.append(batch.relay_mask)
        if relay_parts:
            relay_bank = torch.cat(relay_parts, dim=1)
            relay_mask = torch.cat(relay_masks, dim=1)
            encoded_relay, relay_attention = self.relay_encoder(relay_bank, relay_mask, return_attention=return_attention)
            encoded_relay = encoded_relay * relay_mask.unsqueeze(-1).to(encoded_relay.dtype)
            memory_parts.append(encoded_relay)
            memory_masks.append(relay_mask)
            if return_attention and relay_attention is not None:
                attention_maps["relay_attention"] = relay_attention

        memory = torch.cat(memory_parts, dim=1)
        memory_mask = torch.cat(memory_masks, dim=1)
        query_out, receiver_attention = self.receiver_query(
            receiver_token.unsqueeze(1),
            memory,
            memory_mask,
            return_attention=return_attention,
        )
        query_embeddings = self.query_norm(query_out[:, 0, :])
        if return_attention and receiver_attention is not None:
            attention_maps["receiver_query_attention"] = receiver_attention
        query_logits = self._select_edge_logits(query_embeddings, batch.edge_ids)
        bag_logits = _aggregate_bag_logits(query_logits, batch.bag_index, len(batch.sample_ids))
        bag_embeddings = torch.stack(
            [_masked_mean(query_embeddings[batch.bag_index == idx], torch.ones_like(query_logits[batch.bag_index == idx], dtype=torch.bool), dim=0) for idx in range(len(batch.sample_ids))],
            dim=0,
        )
        diagnostics = {
            "sender_token_count_mean": float(batch.sender_mask.float().sum(dim=1).mean().item()),
            "lr_token_count_mean": float(batch.lr_mask.float().sum(dim=1).mean().item()),
            "response_token_count_mean": float(batch.response_mask.float().sum(dim=1).mean().item()),
            "relay_token_count_mean": float(batch.relay_mask.float().sum(dim=1).mean().item()),
        }
        return CommunicationRelayOutput(
            query_embeddings=query_embeddings,
            bag_embeddings=bag_embeddings,
            query_logits=query_logits,
            bag_logits=bag_logits,
            context_tokens=memory,
            attention_maps=attention_maps,
            diagnostics=diagnostics,
        )

    def counterfactual_edit(self, batch: CommunicationBatch, *, mask_lr: bool = False, mask_relay: bool = False) -> CommunicationRelayOutput:
        edited = CommunicationBatch(
            receiver_embedding=batch.receiver_embedding,
            receiver_programs=batch.receiver_programs,
            sender_embeddings=batch.sender_embeddings,
            sender_types=batch.sender_types,
            sender_offsets=batch.sender_offsets,
            ring_ids=batch.ring_ids,
            lr_token_features=torch.zeros_like(batch.lr_token_features) if mask_lr else batch.lr_token_features,
            response_token_features=batch.response_token_features,
            relay_token_features=torch.zeros_like(batch.relay_token_features) if mask_relay else batch.relay_token_features,
            query_mask=batch.query_mask,
            sender_mask=batch.sender_mask,
            lr_mask=torch.zeros_like(batch.lr_mask) if mask_lr else batch.lr_mask,
            response_mask=batch.response_mask,
            relay_mask=torch.zeros_like(batch.relay_mask) if mask_relay else batch.relay_mask,
            edge_ids=batch.edge_ids,
            weak_labels=batch.weak_labels,
            bag_index=batch.bag_index,
            sample_ids=batch.sample_ids,
            donor_ids=batch.donor_ids,
            label_sources=batch.label_sources,
            receiver_cell_ids=batch.receiver_cell_ids,
            wes_features=batch.wes_features,
            target_latent=batch.target_latent,
        )
        return self.forward(edited, return_attention=False)


class FocalCellMLP(nn.Module):
    """Receiver-only baseline."""

    def __init__(self, receiver_dim: int, receiver_program_dim: int, *, hidden_dim: int = 128, num_edges: int = 4, wes_dim: int = 0) -> None:
        super().__init__()
        self.edge_embedding = nn.Embedding(int(num_edges), int(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(receiver_dim + receiver_program_dim + max(wes_dim, 0)), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.edge_heads = nn.ModuleList([nn.Linear(int(hidden_dim), 1) for _ in range(int(num_edges))])

    def forward(self, batch: CommunicationBatch, *, return_attention: bool = False) -> CommunicationRelayOutput:
        del return_attention
        features = [batch.receiver_embedding, batch.receiver_programs]
        if batch.wes_features is not None:
            features.append(batch.wes_features)
        h = self.mlp(torch.cat(features, dim=-1)) + self.edge_embedding(batch.edge_ids)
        query_logits = h.new_zeros((h.shape[0],))
        for edge_id, head in enumerate(self.edge_heads):
            mask = batch.edge_ids == edge_id
            if torch.any(mask):
                query_logits[mask] = head(h[mask]).squeeze(-1)
        bag_logits = _aggregate_bag_logits(query_logits, batch.bag_index, len(batch.sample_ids))
        bag_embeddings = torch.stack([h[batch.bag_index == idx].mean(dim=0) for idx in range(len(batch.sample_ids))], dim=0)
        return CommunicationRelayOutput(query_embeddings=h, bag_embeddings=bag_embeddings, query_logits=query_logits, bag_logits=bag_logits)


class PooledNeighborhoodModel(nn.Module):
    """Focal + pooled neighbor baseline."""

    def __init__(
        self,
        receiver_dim: int,
        receiver_program_dim: int,
        sender_dim: int,
        lr_dim: int,
        response_dim: int,
        relay_dim: int,
        *,
        hidden_dim: int = 128,
        num_edges: int = 4,
        wes_dim: int = 0,
    ) -> None:
        super().__init__()
        input_dim = int(receiver_dim + receiver_program_dim + sender_dim + lr_dim + response_dim + relay_dim + max(wes_dim, 0))
        self.edge_embedding = nn.Embedding(int(num_edges), int(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.edge_heads = nn.ModuleList([nn.Linear(int(hidden_dim), 1) for _ in range(int(num_edges))])

    def forward(self, batch: CommunicationBatch, *, return_attention: bool = False) -> CommunicationRelayOutput:
        del return_attention
        sender_pool = _masked_mean(batch.sender_embeddings, batch.sender_mask, dim=1)
        lr_pool = _masked_mean(batch.lr_token_features, batch.lr_mask, dim=1)
        response_pool = _masked_mean(batch.response_token_features, batch.response_mask, dim=1)
        relay_pool = _masked_mean(batch.relay_token_features, batch.relay_mask, dim=1)
        parts = [batch.receiver_embedding, batch.receiver_programs, sender_pool, lr_pool, response_pool, relay_pool]
        if batch.wes_features is not None:
            parts.append(batch.wes_features)
        h = self.mlp(torch.cat(parts, dim=-1)) + self.edge_embedding(batch.edge_ids)
        query_logits = h.new_zeros((h.shape[0],))
        for edge_id, head in enumerate(self.edge_heads):
            mask = batch.edge_ids == edge_id
            if torch.any(mask):
                query_logits[mask] = head(h[mask]).squeeze(-1)
        bag_logits = _aggregate_bag_logits(query_logits, batch.bag_index, len(batch.sample_ids))
        bag_embeddings = torch.stack([h[batch.bag_index == idx].mean(dim=0) for idx in range(len(batch.sample_ids))], dim=0)
        return CommunicationRelayOutput(query_embeddings=h, bag_embeddings=bag_embeddings, query_logits=query_logits, bag_logits=bag_logits)


class DeepSetsCommunicationModel(nn.Module):
    """Permutation-invariant token pooling baseline on communication bags."""

    def __init__(
        self,
        receiver_dim: int,
        receiver_program_dim: int,
        sender_dim: int,
        lr_dim: int,
        response_dim: int,
        relay_dim: int,
        *,
        hidden_dim: int = 128,
        num_edges: int = 4,
        wes_dim: int = 0,
    ) -> None:
        super().__init__()
        max_dim = max(receiver_dim + receiver_program_dim + max(wes_dim, 0), sender_dim, lr_dim, response_dim, relay_dim)
        self.max_dim = int(max_dim)
        self.token_type_embedding = nn.Embedding(5, int(hidden_dim))
        self.proj = nn.Linear(int(max_dim), int(hidden_dim))
        self.phi = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.rho = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.edge_embedding = nn.Embedding(int(num_edges), int(hidden_dim))
        self.edge_heads = nn.ModuleList([nn.Linear(int(hidden_dim), 1) for _ in range(int(num_edges))])

    def _pad_last_dim(self, tensor: Tensor) -> Tensor:
        if tensor.shape[-1] == self.max_dim:
            return tensor
        out = torch.zeros((*tensor.shape[:-1], self.max_dim), device=tensor.device, dtype=tensor.dtype)
        out[..., : tensor.shape[-1]] = tensor
        return out

    def forward(self, batch: CommunicationBatch, *, return_attention: bool = False) -> CommunicationRelayOutput:
        del return_attention
        receiver_inputs = torch.cat(
            [batch.receiver_embedding, batch.receiver_programs, batch.wes_features if batch.wes_features is not None else batch.receiver_embedding.new_zeros((batch.receiver_embedding.shape[0], 0))],
            dim=-1,
        )
        receiver_tok = self.proj(self._pad_last_dim(receiver_inputs).unsqueeze(1)) + self.token_type_embedding.weight[0].view(1, 1, -1)
        sender_tok = self.proj(self._pad_last_dim(batch.sender_embeddings)) + self.token_type_embedding.weight[1].view(1, 1, -1)
        lr_tok = self.proj(self._pad_last_dim(batch.lr_token_features)) + self.token_type_embedding.weight[2].view(1, 1, -1)
        response_tok = self.proj(self._pad_last_dim(batch.response_token_features)) + self.token_type_embedding.weight[3].view(1, 1, -1)
        relay_tok = self.proj(self._pad_last_dim(batch.relay_token_features)) + self.token_type_embedding.weight[4].view(1, 1, -1)
        tokens = torch.cat([receiver_tok, sender_tok, lr_tok, response_tok, relay_tok], dim=1)
        mask = torch.cat(
            [
                torch.ones((batch.receiver_embedding.shape[0], 1), device=batch.receiver_embedding.device, dtype=torch.bool),
                batch.sender_mask,
                batch.lr_mask,
                batch.response_mask,
                batch.relay_mask,
            ],
            dim=1,
        )
        h = self.phi(tokens)
        pooled_mean = _masked_mean(h, mask, dim=1)
        pooled_max = (h.masked_fill(~mask.unsqueeze(-1), float("-inf"))).max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        query_embeddings = self.rho(torch.cat([pooled_mean, pooled_max], dim=-1)) + self.edge_embedding(batch.edge_ids)
        query_logits = query_embeddings.new_zeros((query_embeddings.shape[0],))
        for edge_id, head in enumerate(self.edge_heads):
            mask_edge = batch.edge_ids == edge_id
            if torch.any(mask_edge):
                query_logits[mask_edge] = head(query_embeddings[mask_edge]).squeeze(-1)
        bag_logits = _aggregate_bag_logits(query_logits, batch.bag_index, len(batch.sample_ids))
        bag_embeddings = torch.stack([query_embeddings[batch.bag_index == idx].mean(dim=0) for idx in range(len(batch.sample_ids))], dim=0)
        return CommunicationRelayOutput(query_embeddings=query_embeddings, bag_embeddings=bag_embeddings, query_logits=query_logits, bag_logits=bag_logits)


class LocalGraphSAGEBaseline(nn.Module):
    """Two-layer local receiver-sender graph aggregator."""

    def __init__(self, receiver_dim: int, receiver_program_dim: int, sender_dim: int, *, hidden_dim: int = 128, num_edges: int = 4, wes_dim: int = 0) -> None:
        super().__init__()
        receiver_input = int(receiver_dim + receiver_program_dim + max(wes_dim, 0))
        self.receiver_proj = nn.Linear(receiver_input, int(hidden_dim))
        self.sender_proj = nn.Linear(int(sender_dim), int(hidden_dim))
        self.recv_update1 = nn.Sequential(nn.Linear(int(hidden_dim) * 2, int(hidden_dim)), nn.GELU(), nn.LayerNorm(int(hidden_dim)))
        self.send_update1 = nn.Sequential(nn.Linear(int(hidden_dim) * 2, int(hidden_dim)), nn.GELU(), nn.LayerNorm(int(hidden_dim)))
        self.recv_update2 = nn.Sequential(nn.Linear(int(hidden_dim) * 2, int(hidden_dim)), nn.GELU(), nn.LayerNorm(int(hidden_dim)))
        self.send_update2 = nn.Sequential(nn.Linear(int(hidden_dim) * 2, int(hidden_dim)), nn.GELU(), nn.LayerNorm(int(hidden_dim)))
        self.edge_embedding = nn.Embedding(int(num_edges), int(hidden_dim))
        self.edge_heads = nn.ModuleList([nn.Linear(int(hidden_dim), 1) for _ in range(int(num_edges))])

    def forward(self, batch: CommunicationBatch, *, return_attention: bool = False) -> CommunicationRelayOutput:
        del return_attention
        recv_inputs = [batch.receiver_embedding, batch.receiver_programs]
        if batch.wes_features is not None:
            recv_inputs.append(batch.wes_features)
        recv = self.receiver_proj(torch.cat(recv_inputs, dim=-1)) + self.edge_embedding(batch.edge_ids)
        send = self.sender_proj(batch.sender_embeddings)
        neigh_mean1 = _masked_mean(send, batch.sender_mask, dim=1)
        recv1 = self.recv_update1(torch.cat([recv, neigh_mean1], dim=-1))
        send1 = self.send_update1(torch.cat([send, recv1.unsqueeze(1).expand_as(send)], dim=-1))
        neigh_mean2 = _masked_mean(send1, batch.sender_mask, dim=1)
        recv2 = self.recv_update2(torch.cat([recv1, neigh_mean2], dim=-1))
        _ = self.send_update2(torch.cat([send1, recv2.unsqueeze(1).expand_as(send1)], dim=-1))
        query_logits = recv2.new_zeros((recv2.shape[0],))
        for edge_id, head in enumerate(self.edge_heads):
            mask_edge = batch.edge_ids == edge_id
            if torch.any(mask_edge):
                query_logits[mask_edge] = head(recv2[mask_edge]).squeeze(-1)
        bag_logits = _aggregate_bag_logits(query_logits, batch.bag_index, len(batch.sample_ids))
        bag_embeddings = torch.stack([recv2[batch.bag_index == idx].mean(dim=0) for idx in range(len(batch.sample_ids))], dim=0)
        return CommunicationRelayOutput(query_embeddings=recv2, bag_embeddings=bag_embeddings, query_logits=query_logits, bag_logits=bag_logits)


class TransportHeadOTCFM(nn.Module):
    """Phase-3 latent transport head scaffold."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden_dim + latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, query_embeddings: Tensor, receiver_latent: Tensor) -> Tensor:
        return self.net(torch.cat([query_embeddings, receiver_latent], dim=-1))


def build_communication_model(
    model_name: str,
    *,
    receiver_dim: int,
    receiver_program_dim: int,
    sender_dim: int,
    lr_dim: int,
    response_dim: int,
    relay_dim: int,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
    num_edges: int,
    num_sender_types: int,
    num_ring_ids: int,
    wes_dim: int = 0,
) -> nn.Module:
    key = str(model_name)
    if key == "focal_only":
        return FocalCellMLP(receiver_dim, receiver_program_dim, hidden_dim=hidden_dim, num_edges=num_edges, wes_dim=wes_dim)
    if key == "pooled":
        return PooledNeighborhoodModel(
            receiver_dim,
            receiver_program_dim,
            sender_dim,
            lr_dim,
            response_dim,
            relay_dim,
            hidden_dim=hidden_dim,
            num_edges=num_edges,
            wes_dim=wes_dim,
        )
    if key == "deep_sets":
        return DeepSetsCommunicationModel(
            receiver_dim,
            receiver_program_dim,
            sender_dim,
            lr_dim,
            response_dim,
            relay_dim,
            hidden_dim=hidden_dim,
            num_edges=num_edges,
            wes_dim=wes_dim,
        )
    if key == "graphsage":
        return LocalGraphSAGEBaseline(
            receiver_dim,
            receiver_program_dim,
            sender_dim,
            hidden_dim=hidden_dim,
            num_edges=num_edges,
            wes_dim=wes_dim,
        )
    if key == "transformer_no_priors":
        return StageBridgeCommunicationModel(
            receiver_dim,
            receiver_program_dim,
            sender_dim,
            lr_dim,
            response_dim,
            relay_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_sender_types=num_sender_types,
            num_ring_ids=num_ring_ids,
            num_edges=num_edges,
            wes_dim=wes_dim,
            use_lr_tokens=False,
            use_response_tokens=False,
            use_relay_tokens=False,
        )
    if key == "transformer_no_relay":
        return StageBridgeCommunicationModel(
            receiver_dim,
            receiver_program_dim,
            sender_dim,
            lr_dim,
            response_dim,
            relay_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_sender_types=num_sender_types,
            num_ring_ids=num_ring_ids,
            num_edges=num_edges,
            wes_dim=wes_dim,
            use_lr_tokens=True,
            use_response_tokens=True,
            use_relay_tokens=False,
        )
    if key in {"transformer_classification", "stagebridge"}:
        return StageBridgeCommunicationModel(
            receiver_dim,
            receiver_program_dim,
            sender_dim,
            lr_dim,
            response_dim,
            relay_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_sender_types=num_sender_types,
            num_ring_ids=num_ring_ids,
            num_edges=num_edges,
            wes_dim=wes_dim,
            use_lr_tokens=True,
            use_response_tokens=True,
            use_relay_tokens=True,
        )
    raise ValueError(f"Unsupported communication model '{model_name}'.")


__all__ = [
    "CommunicationRelayOutput",
    "DeepSetsCommunicationModel",
    "FocalCellMLP",
    "LocalGraphSAGEBaseline",
    "PooledNeighborhoodModel",
    "StageBridgeCommunicationModel",
    "TransportHeadOTCFM",
    "build_communication_model",
    "collate_communication_bags",
]
