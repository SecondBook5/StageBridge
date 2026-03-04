"""Training loop and data sampling utilities for StageBridge."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from stagebridge.io.niche_tokens import NicheTokenBank
from stagebridge.logging_utils import get_logger
from stagebridge.preprocessing.stage_ontology import CANONICAL_STAGE_ORDER, ordered_transitions, stage_to_index
from stagebridge.training.eval import evaluate_transition
from stagebridge.training.losses import flow_matching_loss
from stagebridge.utils.types import StageBatch, StageBridgeConfig

log = get_logger(__name__)


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
    ) -> None:
        self.device = torch.device(device)
        self.batch_cells = int(batch_cells)
        self.stage_order = stage_order
        self.input_dim = int(latent.shape[1])
        self._rng = np.random.default_rng(int(niche_random_seed))
        self._token_projection: dict[int, Tensor] = {}

        self._niche_token_bank = niche_token_bank
        self._use_niche_tokens = bool(use_niche_tokens and niche_token_bank is not None)
        self._m_niche = max(1, int(m_niche))
        self._niche_sampling_strategy = str(niche_sampling_strategy)
        self._niche_fallback = str(niche_fallback)

        keep = np.isin(obs_donor, np.asarray(donor_ids, dtype=object))
        latent = latent[keep]
        obs_stage = obs_stage[keep]
        obs_donor = obs_donor[keep]
        self.donor_ids = sorted({str(x) for x in obs_donor.tolist()})

        self.stage_to_cells: dict[str, Tensor] = {}
        for stage in stage_order:
            idx = np.where(obs_stage == stage)[0]
            if len(idx) == 0:
                continue
            arr = torch.tensor(latent[idx], dtype=torch.float32, device=self.device)
            self.stage_to_cells[stage] = arr

        self.donor_stage_to_cells: dict[tuple[str, str], Tensor] = {}
        for donor in self.donor_ids:
            donor_mask = obs_donor == donor
            for stage in stage_order:
                idx = np.where(donor_mask & (obs_stage == stage))[0]
                if idx.shape[0] == 0:
                    continue
                self.donor_stage_to_cells[(donor, stage)] = torch.tensor(
                    latent[idx],
                    dtype=torch.float32,
                    device=self.device,
                )

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

    def _sample_cells(self, x: Tensor, n: int) -> Tensor:
        if x.shape[0] >= n:
            idx = torch.randperm(x.shape[0], device=x.device)[:n]
            return x[idx]
        idx = torch.randint(0, x.shape[0], (n,), device=x.device)
        return x[idx]

    def _project_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.shape[1] == self.input_dim:
            return tokens
        token_dim = int(tokens.shape[1])
        if token_dim not in self._token_projection:
            proj_np = (
                self._rng.standard_normal((token_dim, self.input_dim)).astype(np.float32)
                / np.sqrt(float(token_dim))
            )
            self._token_projection[token_dim] = torch.tensor(
                proj_np,
                dtype=torch.float32,
                device=self.device,
            )
        proj = self._token_projection[token_dim]
        return tokens @ proj

    def _sample_niche_tokens(self, donor_id: str, stage_src: str) -> tuple[Tensor | None, dict[str, Any]]:
        if not self._use_niche_tokens or self._niche_token_bank is None:
            return None, {}
        tok_np, meta = self._niche_token_bank.sample_tokens(
            donor_id=donor_id,
            stage=stage_src,
            m_niche=self._m_niche,
            strategy=self._niche_sampling_strategy,
            fallback=self._niche_fallback,
        )
        tok = torch.tensor(tok_np, dtype=torch.float32, device=self.device)
        tok = self._project_tokens(tok)
        return tok, meta

    def sample_batch(self) -> StageBatch:
        src_name, tgt_name = self.transitions[np.random.randint(len(self.transitions))]
        src_donors = self.transition_to_src_donors[(src_name, tgt_name)]
        donor_id = src_donors[np.random.randint(len(src_donors))]

        src_pool = self.donor_stage_to_cells[(donor_id, src_name)]
        tgt_pool = self.donor_stage_to_cells.get((donor_id, tgt_name), self.stage_to_cells[tgt_name])

        x_src = self._sample_cells(src_pool, self.batch_cells)
        x_tgt = self._sample_cells(tgt_pool, self.batch_cells)

        x_set = x_src
        sample_id = None
        niche_tokens, token_meta = self._sample_niche_tokens(donor_id=donor_id, stage_src=src_name)
        if niche_tokens is not None and niche_tokens.shape[0] > 0:
            x_set = torch.cat([x_src, niche_tokens], dim=0)
            samples_used = token_meta.get("samples_used", [])
            if samples_used:
                sample_id = str(samples_used[0])

        context_mask = torch.ones((1, x_set.shape[0]), dtype=torch.bool, device=self.device)
        return StageBatch(
            x_src=x_src,
            x_tgt=x_tgt,
            x_set=x_set,
            context_mask=context_mask,
            stage_src=stage_to_index(src_name),
            stage_tgt=stage_to_index(tgt_name),
            donor_id=str(donor_id),
            sample_id=sample_id,
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
    donor_ids = sorted(set(donor_ids))
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
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.mixed_precision and self.device.type == "cuda"))

    def _run_steps(self, sampler: TransitionSampler, n_steps: int, train: bool) -> float:
        self.model.train(mode=train)
        losses: list[float] = []

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        for step in range(n_steps):
            batch = sampler.sample_batch()
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=(self.config.mixed_precision and self.device.type == "cuda")):
                    loss, _, _ = flow_matching_loss(
                        batch=batch,
                        model=self.model,
                        ot_epsilon=self.config.ot_epsilon,
                        sinkhorn_iters=self.config.sinkhorn_iters,
                        num_ot_pairs=self.config.num_ot_pairs,
                        context_consistency_weight=self.config.context_consistency_weight,
                        use_ot=self.config.use_ot,
                    )
                    loss_for_backward = loss / self.config.gradient_accumulation_steps

            if train:
                self.scaler.scale(loss_for_backward).backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().item()))

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

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._run_steps(train_sampler, self.config.steps_per_epoch, train=True)
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
                log.info("Early stopping triggered after %d epochs without improvement.", no_improve)
                break

        payload = torch.load(best_ckpt, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])

        benchmark = self._benchmark(test_sampler)

        history_path = output_dir / f"history_{run_name}.json"
        with history_path.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        return TrainOutput(
            best_checkpoint=best_ckpt,
            history=history,
            best_val_loss=float(best_val),
            benchmark_metrics=benchmark,
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

    def _build_sampler(
        donor_subset: list[str],
        *,
        strict_transition: bool,
    ) -> TransitionSampler:
        try:
            return TransitionSampler(
                latent=latent,
                obs_stage=obs_stage,
                obs_donor=obs_donor,
                donor_ids=donor_subset,
                batch_cells=batch_cells,
                device=device,
                transition_src=transition_src,
                transition_tgt=transition_tgt,
                niche_token_bank=niche_token_bank,
                use_niche_tokens=use_niche_tokens,
                m_niche=m_niche,
                niche_sampling_strategy=niche_sampling_strategy,
                niche_fallback=niche_fallback,
                niche_random_seed=niche_random_seed,
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
                latent=latent,
                obs_stage=obs_stage,
                obs_donor=obs_donor,
                donor_ids=donor_subset,
                batch_cells=batch_cells,
                device=device,
                transition_src=None,
                transition_tgt=None,
                niche_token_bank=niche_token_bank,
                use_niche_tokens=use_niche_tokens,
                m_niche=m_niche,
                niche_sampling_strategy=niche_sampling_strategy,
                niche_fallback=niche_fallback,
                niche_random_seed=niche_random_seed,
            )

    train = _build_sampler(split.train_donors, strict_transition=True)
    val = _build_sampler(split.val_donors, strict_transition=False)
    test = _build_sampler(split.test_donors, strict_transition=False)
    return train, val, test
