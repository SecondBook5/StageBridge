"""Training loop and data sampling utilities for StageBridge."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from stagebridge.context_model.token_builder import NicheTokenBank
from stagebridge.logging_utils import get_logger
from stagebridge.data.luad_evo.stages import (
    CANONICAL_STAGE_ORDER,
    ordered_transitions,
    stage_to_index,
)
from stagebridge.transition_model.infer import evaluate_transition
from stagebridge.transition_model.losses import flow_matching_loss, multihop_consistency_loss
from stagebridge.transition_model.relational_pretraining import (
    RelationalPretrainingConfig,
    RelationalPretrainingHeads,
    compute_relational_auxiliary_losses,
)
from stagebridge.utils.types import StageBatch, StageBridgeConfig

log = get_logger(__name__)

try:  # pragma: no cover - platform dependent
    import resource
except Exception:  # pragma: no cover - non-POSIX
    resource = None


@dataclass(slots=True)
class DonorSplit:
    """Donor-held-out split specification."""

    train_donors: list[str]
    val_donors: list[str]
    test_donors: list[str]


class TransitionSampler:
    """Stage transition sampler for train/val/test loops."""

    def __init__(
        self,
        latent: np.ndarray,
        obs_stage: np.ndarray,
        obs_donor: np.ndarray,
        donor_ids: list[str],
        batch_cells: int = 256,
        device: str = "cpu",
        stage_order: tuple[str, ...] = CANONICAL_STAGE_ORDER,
        transition_src: str | None = None,
        transition_tgt: str | None = None,
        niche_token_bank: NicheTokenBank | None = None,
        use_niche_tokens: bool = False,
        m_niche: int = 64,
        niche_sampling_strategy: str = "random_m",
        niche_fallback: str = "donor",
        niche_random_seed: int = 42,
        wes_lookup: "dict[tuple[str, str], np.ndarray] | None" = None,
        lr_lookup: "dict[tuple[str, str], np.ndarray] | None" = None,
        spatial_niche: np.ndarray | None = None,
    ) -> None:
        if int(batch_cells) <= 0:
            raise ValueError(f"batch_cells must be >= 1, got {batch_cells}.")

        self.device = torch.device(device)
        self.batch_cells = int(batch_cells)
        self.stage_order = stage_order
        latent_arr = np.asarray(latent, dtype=np.float32)
        if latent_arr.ndim != 2:
            raise ValueError(f"latent must be 2D, got shape={latent_arr.shape}.")
        self.input_dim = int(latent_arr.shape[1])
        self._rng = np.random.default_rng(int(niche_random_seed))
        self._token_projection: dict[int, Tensor] = {}

        self._niche_token_bank = niche_token_bank
        self._use_niche_tokens = bool(use_niche_tokens and niche_token_bank is not None)
        # WES lookup: (donor_id, stage_name) → feature np.ndarray
        self._wes_lookup: dict[tuple[str, str], np.ndarray] | None = wes_lookup
        # LR signaling lookup: (donor_id, stage_name) → feature np.ndarray
        self._lr_lookup: dict[tuple[str, str], np.ndarray] | None = lr_lookup
        self._m_niche = max(1, int(m_niche))
        self._niche_sampling_strategy = str(niche_sampling_strategy)
        self._niche_fallback = str(niche_fallback)

        obs_stage_arr = np.asarray(obs_stage, dtype=object)
        obs_donor_arr = np.asarray(obs_donor, dtype=object)
        if obs_stage_arr.shape[0] != latent_arr.shape[0]:
            raise ValueError(
                "obs_stage length must match latent rows: "
                f"{obs_stage_arr.shape[0]} != {latent_arr.shape[0]}."
            )
        if obs_donor_arr.shape[0] != latent_arr.shape[0]:
            raise ValueError(
                "obs_donor length must match latent rows: "
                f"{obs_donor_arr.shape[0]} != {latent_arr.shape[0]}."
            )
        if len(donor_ids) == 0:
            raise ValueError("donor_ids must be non-empty.")
        keep = np.isin(obs_donor_arr, np.asarray(donor_ids, dtype=object))
        keep_idx = np.where(keep)[0]
        if keep_idx.shape[0] == 0:
            raise ValueError("No cells found for requested donor_ids subset.")
        obs_stage_sub = obs_stage_arr[keep]
        obs_donor_sub = obs_donor_arr[keep]
        self._latent = latent_arr
        self.donor_ids = sorted({str(x) for x in obs_donor_sub.tolist()})

        # Spatial niche: per-cell composition vectors aligned to donor-filtered subset.
        self._spatial_niche_full: np.ndarray | None = None
        self._donor_to_spatial_rows: dict[str, np.ndarray] = {}
        if spatial_niche is not None:
            spatial_arr = np.asarray(spatial_niche, dtype=np.float32)
            if spatial_arr.ndim != 2:
                raise ValueError(f"Spatial niche array must be 2D, got shape={spatial_arr.shape}.")
            if spatial_arr.shape[0] == obs_donor_arr.shape[0]:
                spatial_arr = spatial_arr[keep]
            elif spatial_arr.shape[0] != keep_idx.shape[0]:
                raise ValueError(
                    "Spatial niche array must align to either full obs rows or donor-filtered rows. "
                    f"Got shape[0]={spatial_arr.shape[0]}, expected {obs_donor_arr.shape[0]} or {keep_idx.shape[0]}."
                )
            self._spatial_niche_full = spatial_arr

        self.stage_to_cells: dict[str, np.ndarray] = {}
        for stage in stage_order:
            idx = np.where(obs_stage_sub == stage)[0]
            if len(idx) == 0:
                continue
            self.stage_to_cells[stage] = keep_idx[idx]

        self.donor_stage_to_cells: dict[tuple[str, str], np.ndarray] = {}
        for donor in self.donor_ids:
            donor_mask = obs_donor_sub == donor
            for stage in stage_order:
                idx = np.where(donor_mask & (obs_stage_sub == stage))[0]
                if idx.shape[0] == 0:
                    continue
                self.donor_stage_to_cells[(donor, stage)] = keep_idx[idx]
            if self._spatial_niche_full is not None:
                self._donor_to_spatial_rows[donor] = np.where(donor_mask)[0]

        candidate_transitions: list[tuple[str, str]]
        if transition_src is not None or transition_tgt is not None:
            if transition_src is None or transition_tgt is None:
                raise ValueError("Both transition_src and transition_tgt must be set together.")
            candidate_transitions = [(str(transition_src), str(transition_tgt))]
        else:
            candidate_transitions = list(ordered_transitions(stage_order))

        self.transitions: list[tuple[str, str]] = []
        self.transition_to_src_donors: dict[tuple[str, str], list[str]] = {}
        for src, tgt in candidate_transitions:
            if src not in self.stage_to_cells or tgt not in self.stage_to_cells:
                continue
            src_donors = [
                donor for donor in self.donor_ids if (donor, src) in self.donor_stage_to_cells
            ]
            if not src_donors:
                continue
            self.transitions.append((src, tgt))
            self.transition_to_src_donors[(src, tgt)] = src_donors

        if not self.transitions:
            raise ValueError("No valid stage transitions available for donor subset.")

    @property
    def available_transitions(self) -> list[tuple[str, str]]:
        return list(self.transitions)

    def _sample_cells(self, pool_idx: np.ndarray, n: int) -> Tensor:
        """Sample latent rows by index and move the sampled batch to runtime device."""
        if pool_idx.shape[0] == 0:
            raise ValueError("Cannot sample from an empty index pool.")
        replace = pool_idx.shape[0] < n
        local_idx = self._rng.choice(pool_idx.shape[0], size=n, replace=replace)
        sampled_idx = pool_idx[local_idx]
        batch = torch.from_numpy(self._latent[sampled_idx])
        if self.device.type == "cpu":
            return batch
        return batch.to(device=self.device, non_blocking=True)

    def _project_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.shape[1] == self.input_dim:
            return tokens
        token_dim = int(tokens.shape[1])
        if token_dim not in self._token_projection:
            proj_np = self._rng.standard_normal((token_dim, self.input_dim)).astype(
                np.float32
            ) / np.sqrt(float(token_dim))
            self._token_projection[token_dim] = torch.tensor(
                proj_np,
                dtype=torch.float32,
                device=self.device,
            )
        proj = self._token_projection[token_dim]
        return tokens @ proj

    def _sample_niche_tokens(
        self, donor_id: str, stage_src: str
    ) -> tuple[Tensor | None, Tensor | None, dict[str, Any]]:
        if not self._use_niche_tokens or self._niche_token_bank is None:
            return None, None, {}
        tok_np, coord_np, meta = self._niche_token_bank.sample_tokens(
            donor_id=donor_id,
            stage=stage_src,
            m_niche=self._m_niche,
            strategy=self._niche_sampling_strategy,
            fallback=self._niche_fallback,
        )
        tok = torch.tensor(tok_np, dtype=torch.float32, device=self.device)
        tok = self._project_tokens(tok)
        coords: Tensor | None = None
        if coord_np is not None:
            coords = torch.tensor(coord_np, dtype=torch.float32, device=self.device)
        return tok, coords, meta

    def sample_batch(self) -> StageBatch:
        src_name, tgt_name = self.transitions[np.random.randint(len(self.transitions))]
        src_donors = self.transition_to_src_donors[(src_name, tgt_name)]
        donor_id = src_donors[np.random.randint(len(src_donors))]

        src_pool = self.donor_stage_to_cells[(donor_id, src_name)]
        tgt_pool = self.donor_stage_to_cells.get(
            (donor_id, tgt_name), self.stage_to_cells[tgt_name]
        )

        x_src = self._sample_cells(src_pool, self.batch_cells)
        x_tgt = self._sample_cells(tgt_pool, self.batch_cells)

        x_set = x_src
        sample_id = None
        niche_coords: Tensor | None = None
        niche_tokens, niche_coords_raw, token_meta = self._sample_niche_tokens(
            donor_id=donor_id, stage_src=src_name
        )
        if niche_tokens is not None and niche_tokens.shape[0] > 0:
            x_set = torch.cat([x_src, niche_tokens], dim=0)
            niche_coords = niche_coords_raw  # (m_niche, 2) or None
            samples_used = token_meta.get("samples_used", [])
            if samples_used:
                sample_id = str(samples_used[0])

        context_mask = torch.ones((1, x_set.shape[0]), dtype=torch.bool, device=self.device)

        wes_features: Tensor | None = None
        if self._wes_lookup is not None:
            key = (str(donor_id), src_name)
            feat_np = self._wes_lookup.get(key)
            if feat_np is None:
                feat_np = self._wes_lookup.get((str(donor_id), "LUAD"))
            if feat_np is not None:
                wes_features = torch.tensor(feat_np, dtype=torch.float32, device=self.device)

        lr_features: Tensor | None = None
        if self._lr_lookup is not None:
            key = (str(donor_id), src_name)
            feat_np = self._lr_lookup.get(key)
            if feat_np is None:
                feat_np = self._lr_lookup.get((str(donor_id), "LUAD"))
            if feat_np is not None:
                lr_features = torch.tensor(feat_np, dtype=torch.float32, device=self.device)

        # Spatial niche features: use mean over source cells as a batch-level feature.
        spatial_niche: Tensor | None = None
        if self._spatial_niche_full is not None:
            donor_rows = self._donor_to_spatial_rows.get(str(donor_id))
            if donor_rows is None or donor_rows.shape[0] == 0:
                sn_mean = self._spatial_niche_full.mean(axis=0)
            else:
                sn_mean = self._spatial_niche_full[donor_rows].mean(axis=0)
            spatial_niche = (
                torch.tensor(
                    sn_mean,
                    dtype=torch.float32,
                    device=self.device,
                )
                .unsqueeze(0)
                .expand(x_src.shape[0], -1)
            )  # (batch_cells, niche_dim)

        return StageBatch(
            x_src=x_src,
            x_tgt=x_tgt,
            x_set=x_set,
            context_mask=context_mask,
            stage_src=stage_to_index(src_name),
            stage_tgt=stage_to_index(tgt_name),
            donor_id=str(donor_id),
            sample_id=sample_id,
            wes_features=wes_features,
            niche_coords=niche_coords,
            lr_features=lr_features,
            spatial_niche=spatial_niche,
        )

    def sample_transition_pair(
        self,
        stage_src: str,
        stage_tgt: str,
        n_cells: int = 512,
        donor_id: str | None = None,
    ) -> tuple[Tensor, Tensor]:
        if donor_id is not None and (donor_id, stage_src) in self.donor_stage_to_cells:
            x_src_pool = self.donor_stage_to_cells[(donor_id, stage_src)]
        else:
            x_src_pool = self.stage_to_cells[stage_src]
        if donor_id is not None and (donor_id, stage_tgt) in self.donor_stage_to_cells:
            x_tgt_pool = self.donor_stage_to_cells[(donor_id, stage_tgt)]
        else:
            x_tgt_pool = self.stage_to_cells[stage_tgt]
        x_src = self._sample_cells(x_src_pool, n_cells)
        x_tgt = self._sample_cells(x_tgt_pool, n_cells)
        return x_src, x_tgt


def donors_with_min_stage_coverage(
    obs_stage: np.ndarray,
    obs_donor: np.ndarray,
    min_stages: int = 2,
) -> list[str]:
    """Return donors with at least ``min_stages`` unique stages."""
    donors = np.unique(obs_donor)
    out: list[str] = []
    for donor in donors:
        stages = np.unique(obs_stage[obs_donor == donor])
        if len(stages) >= min_stages:
            out.append(str(donor))
    return sorted(out)


def build_donor_holdout_splits(
    donor_ids: list[str],
    n_folds: int = 3,
    seed: int = 42,
) -> list[DonorSplit]:
    """Create donor-held-out cross-validation splits."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if not donor_ids:
        raise ValueError("donor_ids must contain at least one donor.")
    donor_ids = sorted(set(donor_ids))
    if len(donor_ids) < n_folds:
        raise ValueError(
            f"n_folds={n_folds} requires at least {n_folds} unique donors, got {len(donor_ids)}."
        )
    rng = np.random.default_rng(seed)
    shuffled = donor_ids.copy()
    rng.shuffle(shuffled)

    folds = np.array_split(np.asarray(shuffled, dtype=object), n_folds)
    splits: list[DonorSplit] = []
    for i in range(n_folds):
        test = folds[i].tolist()
        if n_folds == 2:
            pool = folds[(i + 1) % n_folds].tolist()
            n_val = max(1, len(pool) // 4)
            val = pool[:n_val]
            train = pool[n_val:]
            if not train:
                train = pool
                val = pool[:1]
        else:
            val = folds[(i + 1) % n_folds].tolist()
            train_parts = [folds[j] for j in range(n_folds) if j not in (i, (i + 1) % n_folds)]
            train = np.concatenate(train_parts).tolist() if train_parts else []
        splits.append(DonorSplit(train_donors=train, val_donors=val, test_donors=test))
    return splits


@dataclass(slots=True)
class TrainOutput:
    """Container for train artifacts and fold metrics."""

    best_checkpoint: Path
    history: list[dict[str, float]]
    best_val_loss: float
    benchmark_metrics: dict[str, float]
    profile_path: Path | None = None
    profile_summary: dict[str, float] | None = None


class StageBridgeTrainer:
    """Mixed-precision trainer for flow-matching objectives."""

    def __init__(self, model: nn.Module, config: StageBridgeConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.resolved_device())
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(config.mixed_precision and self.device.type == "cuda")
        )

    @staticmethod
    def _batch_tensor_mb(batch: StageBatch) -> float:
        total = 0
        tensors = [
            batch.x_src,
            batch.x_tgt,
            batch.x_set,
            batch.context_mask,
            batch.cell_type,
            batch.wes_features,
            batch.niche_coords,
            batch.lr_features,
            batch.spatial_niche,
            batch.stage_index,
        ]
        for tensor in tensors:
            if tensor is None:
                continue
            total += tensor.numel() * tensor.element_size()
        return float(total) / (1024.0 * 1024.0)

    @staticmethod
    def _cpu_rss_mb() -> float | None:
        if resource is None:
            return None
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rss <= 0:
            return None
        # macOS reports bytes, Linux reports KiB.
        if sys.platform == "darwin":  # pragma: no cover - CI is Linux
            return float(rss) / (1024.0 * 1024.0)
        return float(rss) / 1024.0

    @staticmethod
    def _summarize_profile_rows(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {}
        summary: dict[str, float] = {"n_profiled_steps": float(len(rows))}
        keys = sorted({k for row in rows for k in row.keys() if k != "step_index"})
        for key in keys:
            vals = np.array([float(row[key]) for row in rows if key in row], dtype=float)
            if vals.size == 0:
                continue
            summary[f"{key}_mean"] = float(np.nanmean(vals))
            summary[f"{key}_p50"] = float(np.nanpercentile(vals, 50))
            summary[f"{key}_p95"] = float(np.nanpercentile(vals, 95))
        return summary

    def _run_steps(
        self,
        sampler: TransitionSampler,
        n_steps: int,
        train: bool,
        *,
        profile_rows: list[dict[str, float]] | None = None,
        profile_limit: int = 0,
    ) -> float:
        self.model.train(mode=train)
        losses: list[float] = []

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        for step in range(n_steps):
            do_profile = bool(
                train
                and profile_rows is not None
                and len(profile_rows) < max(0, int(profile_limit))
            )
            step_t0 = time.perf_counter() if do_profile else 0.0
            if do_profile and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            sample_t0 = time.perf_counter() if do_profile else 0.0
            batch = sampler.sample_batch()
            sample_s = (time.perf_counter() - sample_t0) if do_profile else 0.0

            fwd_t0 = time.perf_counter() if do_profile else 0.0
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(
                    "cuda", enabled=(self.config.mixed_precision and self.device.type == "cuda")
                ):
                    loss, _, _ = flow_matching_loss(
                        batch=batch,
                        model=self.model,
                        ot_epsilon=self.config.ot_epsilon,
                        sinkhorn_iters=self.config.sinkhorn_iters,
                        num_ot_pairs=self.config.num_ot_pairs,
                        context_consistency_weight=self.config.context_consistency_weight,
                        use_ot=self.config.use_ot,
                        sigma=self.config.sigma if self.config.use_stochastic_bridge else 0.0,
                    )

                    # Multi-hop consistency loss for skip transitions
                    if (
                        self.config.use_multihop_consistency
                        and batch.stage_tgt - batch.stage_src >= 2
                    ):
                        n_mh = min(64, batch.x_src.shape[0])
                        mh_loss, _ = multihop_consistency_loss(
                            model=self.model,
                            x_src=batch.x_src[:n_mh],
                            stage_src=batch.stage_src,
                            stage_tgt=batch.stage_tgt,
                            num_stages=self.config.num_stages,
                            num_steps=8,
                            wes_features=batch.wes_features,
                        )
                        loss = loss + self.config.multihop_consistency_weight * mh_loss

                    loss_for_backward = loss / self.config.gradient_accumulation_steps
            fwd_bwd_s = (time.perf_counter() - fwd_t0) if do_profile else 0.0

            optim_t0 = time.perf_counter() if do_profile else 0.0
            if train:
                self.scaler.scale(loss_for_backward).backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            optim_s = (time.perf_counter() - optim_t0) if do_profile else 0.0

            if do_profile and profile_rows is not None:
                row: dict[str, float] = {
                    "step_index": float(len(profile_rows)),
                    "sample_s": float(sample_s),
                    "forward_backward_s": float(fwd_bwd_s),
                    "optimizer_s": float(optim_s),
                    "step_s": float(time.perf_counter() - step_t0),
                    "batch_tensor_mb": float(self._batch_tensor_mb(batch)),
                }
                if self.device.type == "cuda":
                    row["peak_cuda_alloc_mb"] = float(
                        torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
                    )
                    row["peak_cuda_reserved_mb"] = float(
                        torch.cuda.max_memory_reserved(self.device) / (1024.0 * 1024.0)
                    )
                else:
                    cpu_rss = self._cpu_rss_mb()
                    if cpu_rss is not None:
                        row["rss_mb"] = float(cpu_rss)
                profile_rows.append(row)

            losses.append(float(loss.detach().item()))

        if train and n_steps > 0 and (n_steps % self.config.gradient_accumulation_steps) != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return float(np.mean(losses)) if losses else float("nan")

    def _benchmark(self, sampler: TransitionSampler) -> dict[str, float]:
        metrics: list[dict[str, float]] = []
        self.model.eval()
        with torch.no_grad():
            for src, tgt in sampler.available_transitions:
                x_src, x_tgt = sampler.sample_transition_pair(src, tgt, n_cells=256)
                result = evaluate_transition(
                    model=self.model,
                    x_src=x_src,
                    x_tgt=x_tgt,
                    stage_src=stage_to_index(src),
                    stage_tgt=stage_to_index(tgt),
                    num_steps=8,
                    ot_epsilon=self.config.ot_epsilon,
                    sinkhorn_iters=self.config.sinkhorn_iters,
                )
                metrics.append(
                    {
                        "sinkhorn": result.sinkhorn,
                        "mmd_rbf": result.mmd_rbf,
                        "classifier_auc": result.classifier_auc,
                        "jsd_composition": result.jsd_composition,
                        "rank_consistency": result.rank_consistency,
                    }
                )

        if not metrics:
            return {}

        out: dict[str, float] = {}
        keys = metrics[0].keys()
        for key in keys:
            vals = np.array([m[key] for m in metrics], dtype=float)
            out[f"{key}_mean"] = float(np.nanmean(vals))
            out[f"{key}_std"] = float(np.nanstd(vals))
        return out

    def fit(
        self,
        train_sampler: TransitionSampler,
        val_sampler: TransitionSampler,
        test_sampler: TransitionSampler,
        output_dir: Path,
        run_name: str,
    ) -> TrainOutput:
        output_dir = Path(output_dir)
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")
        best_ckpt = ckpt_dir / f"{run_name}_best.pt"
        no_improve = 0
        history: list[dict[str, float]] = []
        profile_rows: list[dict[str, float]] = []
        profile_limit = max(0, int(self.config.profile_train_steps))

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._run_steps(
                train_sampler,
                self.config.steps_per_epoch,
                train=True,
                profile_rows=profile_rows if profile_limit > 0 else None,
                profile_limit=profile_limit,
            )
            val_loss = self._run_steps(val_sampler, self.config.val_steps, train=False)

            row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            history.append(row)
            log.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
                epoch,
                self.config.max_epochs,
                train_loss,
                val_loss,
            )

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": asdict(self.config),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    best_ckpt,
                )
            else:
                no_improve += 1

            if no_improve >= self.config.patience:
                log.info(
                    "Early stopping triggered after %d epochs without improvement.", no_improve
                )
                break

        payload = torch.load(best_ckpt, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])

        benchmark = self._benchmark(test_sampler)

        history_path = output_dir / f"history_{run_name}.json"
        with history_path.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        profile_path: Path | None = None
        profile_summary: dict[str, float] | None = None
        if profile_rows:
            profile_path = output_dir / f"profile_{run_name}.json"
            profile_summary = self._summarize_profile_rows(profile_rows)
            with profile_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {"run_name": run_name, "summary": profile_summary, "steps": profile_rows},
                    fh,
                    indent=2,
                )

        return TrainOutput(
            best_checkpoint=best_ckpt,
            history=history,
            best_val_loss=float(best_val),
            benchmark_metrics=benchmark,
            profile_path=profile_path,
            profile_summary=profile_summary,
        )


def build_samplers_from_anndata(
    adata: Any,
    split: DonorSplit,
    latent_key: str = "X_hlca",
    stage_col: str = "stage",
    donor_col: str = "patient_id",
    batch_cells: int = 256,
    device: str = "cpu",
    transition_src: str | None = None,
    transition_tgt: str | None = None,
    niche_token_bank: NicheTokenBank | None = None,
    use_niche_tokens: bool = False,
    m_niche: int = 64,
    niche_sampling_strategy: str = "random_m",
    niche_fallback: str = "donor",
    niche_random_seed: int = 42,
    wes_lookup: "dict[tuple[str, str], np.ndarray] | None" = None,
    lr_lookup: "dict[tuple[str, str], np.ndarray] | None" = None,
    spatial_niche_key: str | None = None,
) -> tuple[TransitionSampler, TransitionSampler, TransitionSampler]:
    """Create train/val/test transition samplers from AnnData and donor split."""
    if latent_key not in adata.obsm:
        raise KeyError(f"Missing latent key '{latent_key}' in adata.obsm")
    if stage_col not in adata.obs.columns:
        raise KeyError(f"Missing stage column '{stage_col}' in adata.obs")
    if donor_col not in adata.obs.columns:
        raise KeyError(f"Missing donor column '{donor_col}' in adata.obs")

    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    obs_stage = np.asarray(adata.obs[stage_col].astype(str).values)
    obs_donor = np.asarray(adata.obs[donor_col].astype(str).values)

    # Extract spatial niche features if available
    spatial_niche_arr: np.ndarray | None = None
    if spatial_niche_key is not None and spatial_niche_key in adata.obsm:
        spatial_niche_arr = np.asarray(adata.obsm[spatial_niche_key], dtype=np.float32)

    def _build_sampler(
        donor_subset: list[str],
        *,
        strict_transition: bool,
    ) -> TransitionSampler:
        common_kwargs: dict[str, Any] = dict(
            latent=latent,
            obs_stage=obs_stage,
            obs_donor=obs_donor,
            donor_ids=donor_subset,
            batch_cells=batch_cells,
            device=device,
            niche_token_bank=niche_token_bank,
            use_niche_tokens=use_niche_tokens,
            m_niche=m_niche,
            niche_sampling_strategy=niche_sampling_strategy,
            niche_fallback=niche_fallback,
            niche_random_seed=niche_random_seed,
            wes_lookup=wes_lookup,
            lr_lookup=lr_lookup,
            spatial_niche=spatial_niche_arr,
        )
        try:
            return TransitionSampler(
                transition_src=transition_src,
                transition_tgt=transition_tgt,
                **common_kwargs,
            )
        except ValueError:
            if strict_transition or transition_src is None or transition_tgt is None:
                raise
            log.warning(
                "Requested transition %s->%s unavailable for donor subset (n=%d). "
                "Falling back to all available transitions.",
                transition_src,
                transition_tgt,
                len(donor_subset),
            )
            return TransitionSampler(
                transition_src=None,
                transition_tgt=None,
                **common_kwargs,
            )

    train = _build_sampler(split.train_donors, strict_transition=True)
    val = _build_sampler(split.val_donors, strict_transition=False)
    test = _build_sampler(split.test_donors, strict_transition=False)
    return train, val, test


@dataclass(slots=True)
class EdgeSplitSummary:
    """Visible donor split diagnostics for one directed edge."""

    stage_src: str
    stage_tgt: str
    source_train_donors: list[str]
    source_test_donors: list[str]
    target_train_donors: list[str]
    target_test_donors: list[str]
    overlap_donors: list[str]
    split_strategy: str
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class EdgeTrainingResult:
    """Minimal training bundle for the Mission 3 active edge path."""

    model: nn.Module
    history: list[dict[str, float]]
    split_summary: EdgeSplitSummary
    sigma: float
    diffusion_weight: float


def build_stagewise_edge_split(
    src_obs: Any,
    tgt_obs: Any,
    *,
    donor_col: str = "donor_id",
    holdout_fraction: float = 0.25,
    min_intersection_donors: int = 4,
    stage_src: str,
    stage_tgt: str,
) -> EdgeSplitSummary:
    """Build an honest donor split summary for one edge.

    The split falls back to stagewise donor holdout when same-donor overlap
    across the edge is too small to support a meaningful paired holdout.
    """
    src_donors = sorted({str(value) for value in src_obs[donor_col].astype(str).tolist()})
    tgt_donors = sorted({str(value) for value in tgt_obs[donor_col].astype(str).tolist()})
    overlap = sorted(set(src_donors) & set(tgt_donors))

    def _split(donors: list[str]) -> tuple[list[str], list[str]]:
        if len(donors) <= 1:
            return donors, []
        n_test = max(1, int(round(len(donors) * float(holdout_fraction))))
        n_test = min(n_test, len(donors) - 1)
        return donors[:-n_test], donors[-n_test:]

    notes: list[str] = []
    if len(overlap) >= int(min_intersection_donors):
        train_overlap, test_overlap = _split(overlap)
        source_train = [donor for donor in src_donors if donor in train_overlap]
        source_test = [donor for donor in src_donors if donor in test_overlap]
        target_train = [donor for donor in tgt_donors if donor in train_overlap]
        target_test = [donor for donor in tgt_donors if donor in test_overlap]
        strategy = "same_donor_overlap_holdout"
    else:
        source_train, source_test = _split(src_donors)
        target_train, target_test = _split(tgt_donors)
        strategy = "stagewise_donor_holdout"
        notes.append(
            "Insufficient same-donor overlap across the edge for paired donor holdout; "
            "using stagewise donor holdout instead."
        )
        notes.append(f"same_donor_overlap_n={len(overlap)}")

    return EdgeSplitSummary(
        stage_src=stage_src,
        stage_tgt=stage_tgt,
        source_train_donors=source_train,
        source_test_donors=source_test,
        target_train_donors=target_train,
        target_test_donors=target_test,
        overlap_donors=overlap,
        split_strategy=strategy,
        notes=notes,
    )


def _sample_rows(x: Tensor, n: int, *, seed: int, step: int) -> tuple[Tensor, Tensor]:
    if x.shape[0] <= n:
        idx = torch.arange(x.shape[0], device=x.device)
        return x, idx
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed) + int(step))
    idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:n]
    return x[idx], idx


def _clone_trainable_parameters(module: nn.Module) -> list[Tensor]:
    return [param.detach().cpu().clone() for param in module.parameters() if param.requires_grad]


def _mean_parameter_delta(before: list[Tensor], module: nn.Module) -> float:
    after = [param.detach().cpu() for param in module.parameters() if param.requires_grad]
    if not before or not after:
        return 0.0
    deltas = [
        (after_param - before_param).abs().mean()
        for before_param, after_param in zip(before, after, strict=False)
    ]
    if not deltas:
        return 0.0
    return float(torch.stack(deltas).mean().item())


def _forward_context_encoder(
    context_encoder: nn.Module,
    context_tokens: Tensor,
    *,
    context_graph: Any | None = None,
    token_type_ids: Tensor | None = None,
    token_coords: Tensor | None = None,
    token_confidence: Tensor | None = None,
    dataset_ids: Tensor | None = None,
    edge_ids: Tensor | None = None,
    return_attention: bool = False,
) -> Any:
    if context_graph is not None:
        return context_encoder(context_tokens, context_graph)
    kwargs: dict[str, Any] = {}
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids
    if token_coords is not None:
        kwargs["token_coords"] = token_coords
    if token_confidence is not None:
        kwargs["token_confidence"] = token_confidence
    if dataset_ids is not None:
        kwargs["dataset_ids"] = dataset_ids
    if edge_ids is not None:
        kwargs["edge_ids"] = edge_ids
    if return_attention:
        kwargs["return_attention"] = True
    try:
        return context_encoder(context_tokens, **kwargs)
    except TypeError:
        reduced_kwargs = dict(kwargs)
        reduced_kwargs.pop("dataset_ids", None)
        reduced_kwargs.pop("edge_ids", None)
        try:
            return context_encoder(context_tokens, **reduced_kwargs)
        except TypeError:
            return context_encoder(context_tokens)


def _infer_token_type_ids_from_tokens(
    tokens: Tensor, num_token_types: int | None = None
) -> Tensor | None:
    if num_token_types is None:
        num_token_types = int(tokens.shape[-1])
    if tokens.ndim != 2 or tokens.shape[-1] < 1:
        return None
    usable = min(int(num_token_types), int(tokens.shape[-1]))
    return tokens[:, :usable].argmax(dim=-1).long()


def _build_context_negative_controls(
    *,
    context_tokens: Tensor,
    shuffled_context_tokens: Tensor,
    context_coords: Tensor | None,
    shuffled_context_coords: Tensor | None,
    context_confidence: Tensor | None,
    shuffled_context_confidence: Tensor | None,
    token_type_ids: Tensor | None,
    shuffled_token_type_ids: Tensor | None,
    dataset_ids: Tensor | None = None,
) -> list[dict[str, Tensor | None]]:
    negatives: list[dict[str, Tensor | None]] = [
        {
            "tokens": shuffled_context_tokens,
            "coords": shuffled_context_coords,
            "confidence": shuffled_context_confidence,
            "token_type_ids": shuffled_token_type_ids,
            "dataset_ids": dataset_ids,
            "label": "token_shuffle",
        }
    ]
    if context_tokens.ndim != 2:
        return negatives

    group_permuted_tokens = torch.roll(context_tokens, shifts=1, dims=-1)
    group_permuted_types = _infer_token_type_ids_from_tokens(
        group_permuted_tokens,
        token_type_ids.max().item() + 1 if token_type_ids is not None else None,
    )
    negatives.append(
        {
            "tokens": group_permuted_tokens,
            "coords": context_coords,
            "confidence": context_confidence,
            "token_type_ids": group_permuted_types,
            "dataset_ids": dataset_ids,
            "label": "group_permutation",
        }
    )

    if context_coords is not None:
        negatives.append(
            {
                "tokens": context_tokens,
                "coords": torch.roll(context_coords, shifts=1, dims=0),
                "confidence": context_confidence
                if context_confidence is None
                else torch.flip(context_confidence, dims=[0]),
                "token_type_ids": token_type_ids,
                "dataset_ids": dataset_ids,
                "label": "coordinate_permutation",
            }
        )
    return negatives


def train_edgewise_transition_model(
    *,
    model: nn.Module,
    x_src_train: Tensor,
    x_tgt_train: Tensor,
    context: Tensor,
    context_tokens: Tensor | None,
    edge_id: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    steps_per_epoch: int,
    batch_cells: int,
    sigma: float,
    diffusion_weight: float,
    epsilon: float,
    sinkhorn_iters: int,
    num_ot_pairs: int,
    seed: int,
    extra_cost: Tensor | None = None,
) -> list[dict[str, float]]:
    """Train the active edge-wise transition model on one directed edge."""
    from stagebridge.transition_model.schrodinger_bridge import edgewise_schrodinger_bridge_loss

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    history: list[dict[str, float]] = []
    edge_ids_full = torch.full(
        (x_src_train.shape[0],), int(edge_id), dtype=torch.long, device=x_src_train.device
    )

    for epoch in range(int(max_epochs)):
        losses: list[float] = []
        drift_losses: list[float] = []
        diffusion_losses: list[float] = []
        for step in range(int(steps_per_epoch)):
            batch_src, src_idx = _sample_rows(
                x_src_train,
                int(batch_cells),
                seed=seed,
                step=epoch * steps_per_epoch + step,
            )
            batch_tgt, tgt_idx = _sample_rows(
                x_tgt_train,
                int(batch_cells),
                seed=seed + 10_000,
                step=epoch * steps_per_epoch + step,
            )

            batch_extra: Tensor | None = None
            if extra_cost is not None:
                batch_extra = extra_cost.index_select(0, src_idx).index_select(1, tgt_idx)

            loss, diagnostics, _ = edgewise_schrodinger_bridge_loss(
                model,
                x_src=batch_src,
                x_tgt=batch_tgt,
                context=context,
                context_tokens=context_tokens,
                edge_ids=edge_ids_full[: batch_src.shape[0]],
                epsilon=float(epsilon),
                sinkhorn_iters=int(sinkhorn_iters),
                num_ot_pairs=int(num_ot_pairs),
                sigma=float(sigma),
                diffusion_weight=float(diffusion_weight),
                extra_cost=batch_extra,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.detach().item()))
            drift_losses.append(float(diagnostics["loss_drift"]))
            diffusion_losses.append(float(diagnostics["loss_diffusion"]))

        history.append(
            {
                "epoch": float(epoch + 1),
                "loss_total": float(np.mean(losses)),
                "loss_drift": float(np.mean(drift_losses)),
                "loss_diffusion": float(np.mean(diffusion_losses)),
            }
        )
    return history


def train_edgewise_transition_model_with_context_encoder(
    *,
    model: nn.Module,
    context_encoder: nn.Module,
    context_tokens: Tensor,
    shuffled_context_tokens: Tensor,
    context_coords: Tensor | None,
    shuffled_context_coords: Tensor | None,
    context_confidence: Tensor | None,
    shuffled_context_confidence: Tensor | None,
    x_src_train: Tensor,
    x_tgt_train: Tensor,
    edge_id: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    steps_per_epoch: int,
    batch_cells: int,
    sigma: float,
    diffusion_weight: float,
    epsilon: float,
    sinkhorn_iters: int,
    num_ot_pairs: int,
    seed: int,
    extra_cost: Tensor | None = None,
    context_graph: Any | None = None,
    token_type_ids: Tensor | None = None,
    shuffled_token_type_ids: Tensor | None = None,
    dataset_ids: Tensor | None = None,
    edge_ids: Tensor | None = None,
    context_negative_controls: list[dict[str, Any]] | None = None,
    provider_views: list[dict[str, Any]] | None = None,
    capture_attention: bool = False,
    auxiliary_loss_weight: float = 0.1,
    pretraining_config: RelationalPretrainingConfig | None = None,
    pretraining_heads: RelationalPretrainingHeads | None = None,
) -> dict[str, Any]:
    """Train the context encoder jointly with the edge-wise transition model."""
    from stagebridge.transition_model.schrodinger_bridge import edgewise_schrodinger_bridge_loss

    with torch.no_grad():
        probe_summary = _forward_context_encoder(
            context_encoder,
            context_tokens,
            context_graph=context_graph,
            token_type_ids=token_type_ids,
            token_coords=context_coords,
            token_confidence=context_confidence,
            dataset_ids=dataset_ids,
            edge_ids=edge_ids,
            return_attention=False,
        )
    probe_context = probe_summary.pooled_context
    context_dim = int(probe_context.shape[-1]) if probe_context.ndim > 0 else 1
    use_relational_aux = pretraining_heads is not None
    compatibility_head: nn.Module | None = None
    if not use_relational_aux:
        compatibility_head = nn.Sequential(
            nn.Linear(context_dim, max(context_dim // 2, 16)),
            nn.GELU(),
            nn.Linear(max(context_dim // 2, 16), 1),
        ).to(x_src_train.device)
    param_groups = [
        {"params": list(context_encoder.parameters()), "lr": float(learning_rate) * 0.3},
        {"params": list(model.parameters()), "lr": float(learning_rate)},
    ]
    if compatibility_head is not None:
        param_groups.append(
            {"params": list(compatibility_head.parameters()), "lr": float(learning_rate)}
        )
    if pretraining_heads is not None:
        param_groups.append(
            {"params": list(pretraining_heads.parameters()), "lr": float(learning_rate) * 0.5}
        )
    parameters = [param for group in param_groups for param in group["params"]]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(weight_decay),
    )
    history: list[dict[str, float]] = []
    edge_ids_full = torch.full(
        (x_src_train.shape[0],), int(edge_id), dtype=torch.long, device=x_src_train.device
    )
    before = _clone_trainable_parameters(context_encoder)
    negative_controls = _build_context_negative_controls(
        context_tokens=context_tokens,
        shuffled_context_tokens=shuffled_context_tokens,
        context_coords=context_coords,
        shuffled_context_coords=shuffled_context_coords,
        context_confidence=context_confidence,
        shuffled_context_confidence=shuffled_context_confidence,
        token_type_ids=token_type_ids,
        shuffled_token_type_ids=shuffled_token_type_ids,
        dataset_ids=dataset_ids,
    )
    if context_negative_controls:
        negative_controls.extend(context_negative_controls)
    provider_views = list(provider_views or [])
    if pretraining_config is None:
        pretraining_config = RelationalPretrainingConfig(
            masked_token_weight=0.05,
            ranking_weight=float(auxiliary_loss_weight),
            provider_consistency_weight=0.05 if provider_views else 0.0,
            coordinate_corruption_weight=0.0,
            group_relation_weight=0.0,
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            seed=int(seed),
        )

    for epoch in range(int(max_epochs)):
        losses: list[float] = []
        drift_losses: list[float] = []
        diffusion_losses: list[float] = []
        context_norms: list[float] = []
        auxiliary_losses: list[float] = []
        auxiliary_accuracies: list[float] = []
        provider_cosines: list[float] = []
        drift_context_gates: list[float] = []
        for step in range(int(steps_per_epoch)):
            batch_src, src_idx = _sample_rows(
                x_src_train,
                int(batch_cells),
                seed=seed,
                step=epoch * steps_per_epoch + step,
            )
            batch_tgt, tgt_idx = _sample_rows(
                x_tgt_train,
                int(batch_cells),
                seed=seed + 10_000,
                step=epoch * steps_per_epoch + step,
            )

            batch_extra: Tensor | None = None
            if extra_cost is not None:
                batch_extra = extra_cost.index_select(0, src_idx).index_select(1, tgt_idx)

            context_summary = _forward_context_encoder(
                context_encoder,
                context_tokens,
                context_graph=context_graph,
                token_type_ids=token_type_ids,
                token_coords=context_coords,
                token_confidence=context_confidence,
                dataset_ids=dataset_ids,
                edge_ids=edge_ids,
                return_attention=False,
            )
            context = context_summary.pooled_context
            context_tokens_for_drift = getattr(context_summary, "context_tokens", None)
            loss, diagnostics, _ = edgewise_schrodinger_bridge_loss(
                model,
                x_src=batch_src,
                x_tgt=batch_tgt,
                context=context,
                context_tokens=context_tokens_for_drift,
                edge_ids=edge_ids_full[: batch_src.shape[0]],
                epsilon=float(epsilon),
                sinkhorn_iters=int(sinkhorn_iters),
                num_ot_pairs=int(num_ot_pairs),
                sigma=float(sigma),
                diffusion_weight=float(diffusion_weight),
                extra_cost=batch_extra,
            )
            if use_relational_aux and pretraining_heads is not None:
                auxiliary_loss, _, relational_metrics, _ = compute_relational_auxiliary_losses(
                    context_encoder=context_encoder,
                    heads=pretraining_heads,
                    context_tokens=context_tokens,
                    token_type_ids=token_type_ids
                    if token_type_ids is not None
                    else torch.zeros(
                        context_tokens.shape[0], dtype=torch.long, device=context_tokens.device
                    ),
                    token_coords=context_coords,
                    token_confidence=context_confidence,
                    dataset_ids=dataset_ids,
                    edge_ids=edge_ids,
                    negative_controls=negative_controls,
                    provider_views=provider_views,
                    config=pretraining_config,
                    seed=seed + epoch * 1_000 + step,
                    include_masked_token=bool(pretraining_config.masked_token_weight > 0.0),
                    include_provider_consistency=bool(
                        pretraining_config.provider_consistency_weight > 0.0 and provider_views
                    ),
                    include_coordinate_corruption=False,
                    include_group_relation=False,
                    return_attention=False,
                )
                total_loss = loss + auxiliary_loss
                aux_accuracy = float(relational_metrics.get("ranking_accuracy", float("nan")))
                provider_cos = relational_metrics.get("provider_consistency_cosine", float("nan"))
                if np.isfinite(provider_cos):
                    provider_cosines.append(float(provider_cos))
            else:
                context_for_head = context.unsqueeze(0) if context.ndim == 1 else context
                negative_contexts: list[Tensor] = []
                for control in negative_controls:
                    negative_summary = _forward_context_encoder(
                        context_encoder,
                        control["tokens"],
                        context_graph=context_graph,
                        token_type_ids=control["token_type_ids"],
                        token_coords=control["coords"],
                        token_confidence=control["confidence"],
                        dataset_ids=control.get("dataset_ids"),
                        edge_ids=edge_ids,
                        return_attention=False,
                    )
                    negative_context = negative_summary.pooled_context
                    negative_contexts.append(
                        negative_context.unsqueeze(0)
                        if negative_context.ndim == 1
                        else negative_context
                    )
                shuffled_for_head = torch.cat(negative_contexts, dim=0)
                assert compatibility_head is not None
                positive_score = compatibility_head(context_for_head)
                negative_scores = compatibility_head(shuffled_for_head)
                margin = torch.tensor(0.2, device=context.device, dtype=context.dtype)
                auxiliary_loss = torch.relu(margin - positive_score + negative_scores).mean()
                total_loss = loss + float(auxiliary_loss_weight) * auxiliary_loss
                aux_accuracy = float(
                    (positive_score.detach() > negative_scores.detach()).float().mean().item()
                )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            losses.append(float(total_loss.detach().item()))
            drift_losses.append(float(diagnostics["loss_drift"]))
            diffusion_losses.append(float(diagnostics["loss_diffusion"]))
            context_norms.append(float(context.detach().norm().item()))
            auxiliary_losses.append(float(auxiliary_loss.detach().item()))
            auxiliary_accuracies.append(aux_accuracy)
            if getattr(model, "cross_attention_drift", None) is not None:
                drift_context_gates.append(
                    float(getattr(model.cross_attention_drift, "last_context_gate_mean", 0.0))
                )

        history.append(
            {
                "epoch": float(epoch + 1),
                "loss_total": float(np.mean(losses)),
                "loss_drift": float(np.mean(drift_losses)),
                "loss_diffusion": float(np.mean(diffusion_losses)),
                "context_norm": float(np.mean(context_norms)),
                "loss_context_shuffle": float(np.mean(auxiliary_losses)),
                "context_shuffle_accuracy": float(np.mean(auxiliary_accuracies)),
                "provider_consistency_cosine": float(np.mean(provider_cosines))
                if provider_cosines
                else float("nan"),
                "drift_context_gate": float(np.mean(drift_context_gates))
                if drift_context_gates
                else float("nan"),
            }
        )

    with torch.no_grad():
        if use_relational_aux and pretraining_heads is not None:
            final_aux_loss, _, final_aux_metrics, final_context_summary = (
                compute_relational_auxiliary_losses(
                    context_encoder=context_encoder,
                    heads=pretraining_heads,
                    context_tokens=context_tokens,
                    token_type_ids=token_type_ids
                    if token_type_ids is not None
                    else torch.zeros(
                        context_tokens.shape[0], dtype=torch.long, device=context_tokens.device
                    ),
                    token_coords=context_coords,
                    token_confidence=context_confidence,
                    dataset_ids=dataset_ids,
                    edge_ids=edge_ids,
                    negative_controls=negative_controls,
                    provider_views=provider_views,
                    config=pretraining_config,
                    seed=seed + 99_999,
                    include_masked_token=bool(pretraining_config.masked_token_weight > 0.0),
                    include_provider_consistency=bool(
                        pretraining_config.provider_consistency_weight > 0.0 and provider_views
                    ),
                    include_coordinate_corruption=False,
                    include_group_relation=False,
                    return_attention=bool(capture_attention),
                )
            )
            first_negative = negative_controls[0]
            final_shuffled_summary = _forward_context_encoder(
                context_encoder,
                first_negative["tokens"],
                context_graph=context_graph,
                token_type_ids=first_negative.get("token_type_ids"),
                token_coords=first_negative.get("coords"),
                token_confidence=first_negative.get("confidence"),
                dataset_ids=first_negative.get("dataset_ids"),
                edge_ids=edge_ids,
                return_attention=False,
            )
            final_aux_accuracy = float(final_aux_metrics.get("ranking_accuracy", float("nan")))
            separation_score = float(final_aux_metrics.get("provider_consistency_cosine", 0.0))
            negative_control_scores = {
                str(key): float(value)
                for key, value in (
                    final_aux_metrics.get("negative_control_scores", {}) or {}
                ).items()
            }
        else:
            final_context_summary = _forward_context_encoder(
                context_encoder,
                context_tokens,
                context_graph=context_graph,
                token_type_ids=token_type_ids,
                token_coords=context_coords,
                token_confidence=context_confidence,
                dataset_ids=dataset_ids,
                edge_ids=edge_ids,
                return_attention=bool(capture_attention),
            )
            final_negative_summaries = [
                _forward_context_encoder(
                    context_encoder,
                    control["tokens"],
                    context_graph=context_graph,
                    token_type_ids=control["token_type_ids"],
                    token_coords=control["coords"],
                    token_confidence=control["confidence"],
                    dataset_ids=control.get("dataset_ids"),
                    edge_ids=edge_ids,
                    return_attention=False,
                )
                for control in negative_controls
            ]
            final_shuffled_summary = final_negative_summaries[0]
            final_context = final_context_summary.pooled_context
            final_negative_contexts = [
                summary.pooled_context for summary in final_negative_summaries
            ]
            final_context_for_head = (
                final_context.unsqueeze(0) if final_context.ndim == 1 else final_context
            )
            final_shuffled_for_head = torch.cat(
                [neg.unsqueeze(0) if neg.ndim == 1 else neg for neg in final_negative_contexts],
                dim=0,
            )
            assert compatibility_head is not None
            final_positive_score = compatibility_head(final_context_for_head)
            final_negative_scores = compatibility_head(final_shuffled_for_head)
            final_margin = torch.tensor(
                0.2, device=final_positive_score.device, dtype=final_positive_score.dtype
            )
            final_aux_loss = torch.relu(
                final_margin - final_positive_score + final_negative_scores
            ).mean()
            final_aux_accuracy = float(
                (final_positive_score > final_negative_scores).float().mean().item()
            )
            separation_score = float(
                torch.norm(
                    final_context_for_head.mean(dim=0) - final_shuffled_for_head.mean(dim=0),
                    p=2,
                ).item()
            )
            negative_control_scores = {}
            for idx, control in enumerate(negative_controls):
                label = str(control.get("label", f"negative_{idx}"))
                negative_control_scores[label] = float(final_negative_scores[idx].mean().item())

    drift_context_gate = float(
        getattr(getattr(model, "cross_attention_drift", None), "last_context_gate_mean", 0.0)
    )
    drift_context_attention_entropy = float(
        getattr(
            getattr(model, "cross_attention_drift", None), "last_context_attention_entropy", 0.0
        )
    )

    return {
        "history": history,
        "context_summary": final_context_summary,
        "shuffled_context_summary": final_shuffled_summary,
        "encoder_parameter_delta": _mean_parameter_delta(before, context_encoder),
        "auxiliary_metrics": {
            "loss": float(final_aux_loss.item()),
            "accuracy": final_aux_accuracy,
            "separation_score": separation_score,
            "n_negative_controls": len(negative_controls),
            "task": "relational_pretraining_finetune"
            if use_relational_aux
            else "context_match_ranking",
            "margin": float(
                pretraining_config.ranking_margin if use_relational_aux else final_margin.item()
            ),
            "positive_score": float(final_aux_metrics.get("positive_score", 0.0))
            if use_relational_aux
            else float(final_positive_score.mean().item()),
            "negative_control_scores": negative_control_scores,
            "provider_consistency_cosine": float(
                final_aux_metrics.get("provider_consistency_cosine", float("nan"))
            )
            if use_relational_aux
            else float("nan"),
            "masked_token_count": int(final_aux_metrics.get("masked_token_count", 0))
            if use_relational_aux
            else 0,
            "coordinate_corruption_accuracy": float(
                final_aux_metrics.get("coordinate_corruption_accuracy", float("nan"))
            )
            if use_relational_aux
            else float("nan"),
            "group_relation_accuracy": float(
                final_aux_metrics.get("group_relation_accuracy", float("nan"))
            )
            if use_relational_aux
            else float("nan"),
            "loss_components": {
                "masked_token": float(final_aux_metrics.get("loss_masked_token", 0.0))
                if use_relational_aux
                else 0.0,
                "ranking": float(final_aux_metrics.get("loss_ranking", 0.0))
                if use_relational_aux
                else float(final_aux_loss.item()),
                "provider_consistency": float(
                    final_aux_metrics.get("loss_provider_consistency", 0.0)
                )
                if use_relational_aux
                else 0.0,
                "coordinate_corruption": float(
                    final_aux_metrics.get("loss_coordinate_corruption", 0.0)
                )
                if use_relational_aux
                else 0.0,
                "group_relation": float(final_aux_metrics.get("loss_group_relation", 0.0))
                if use_relational_aux
                else 0.0,
            },
            "drift_context_gate": drift_context_gate,
            "drift_context_attention_entropy": drift_context_attention_entropy,
        },
        "pretraining_heads": pretraining_heads,
    }
