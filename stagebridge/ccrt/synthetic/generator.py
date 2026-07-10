"""Synthetic biological system spec + observable data generation.

Builds a generic ``BiologicalSystemSpec`` (through the real grammar layer) and
generates deterministic factual + null-context training/validation/test batches.
The student sees only observable tensors in ``CCRTTrainingBatch``; the teacher's
hidden decomposition is kept separately in ``SyntheticGroundTruth`` and is never
placed in batch metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.indexing import CCRTIndexRegistry
from ..grammar import (
    GROWTH_MASS,
    SEMANTIC_DRIFT,
    BiologicalSystemSpec,
    CounterfactualPerturbation,
    ReceiverBehavior,
    ReceiverState,
    RegulatoryMediator,
    SenderContextType,
    SignalProgram,
    TransitionEdge,
)
from ..training.batch import CCRTTrainingBatch
from .config import SyntheticSystemConfig
from .ground_truth import (
    SyntheticGroundTruth,
    SyntheticTeacher,
    SyntheticTeacherParameters,
)
from .mechanisms import SyntheticMechanismSpec, build_synthetic_mechanism_spec

__all__ = [
    "SyntheticExample",
    "SyntheticDatasetBundle",
    "build_synthetic_biological_system_spec",
    "generate_synthetic_dataset",
]

_DTYPE = torch.float64


def build_synthetic_biological_system_spec(
    system: SyntheticSystemConfig,
) -> BiologicalSystemSpec:
    """Build a generic synthetic ``BiologicalSystemSpec`` via the grammar layer."""
    # Enough receiver states to define all edges (edge k: state k -> state k+1).
    n_states = system.num_transition_edges + 1
    receiver_states = tuple(
        ReceiverState(f"receiver_state_{i}", order=float(i)) for i in range(n_states)
    )
    transition_edges = tuple(
        TransitionEdge(f"transition_edge_{k}", f"receiver_state_{k}", f"receiver_state_{k + 1}")
        for k in range(system.num_transition_edges)
    )
    sender_context_types = tuple(
        SenderContextType(f"sender_type_{t}", signal_program_ids=("synthetic_signal_program_0",))
        for t in range(system.num_sender_context_types)
    )
    signal_programs = (SignalProgram("synthetic_signal_program_0"),)
    receiver_behaviors = (ReceiverBehavior(SEMANTIC_DRIFT), ReceiverBehavior(GROWTH_MASS))
    regulatory_mediators = (
        RegulatoryMediator(
            "synthetic_regulatory_mediator",
            signal_program_ids=("synthetic_signal_program_0",),
        ),
    )
    counterfactuals = [
        CounterfactualPerturbation(
            "remove_all_sender_context", perturbation_kind="remove_sender_context"
        )
    ]
    for t in range(system.num_sender_context_types):
        counterfactuals.append(
            CounterfactualPerturbation(
                f"remove_sender_type_{t}",
                perturbation_kind="remove_sender_context",
                target_sender_context_type_ids=(f"sender_type_{t}",),
            )
        )

    spec = BiologicalSystemSpec(
        biological_system_id=system.biological_system_id,
        receiver_states=receiver_states,
        transition_edges=transition_edges,
        sender_context_types=sender_context_types,
        signal_programs=signal_programs,
        receiver_behaviors=receiver_behaviors,
        regulatory_mediators=regulatory_mediators,
        counterfactual_perturbations=tuple(counterfactuals),
        metadata={"hypothesis": "synthetic grammar validation system"},
    )
    spec.validate()
    return spec


@dataclass(frozen=True)
class SyntheticExample:
    """A factual batch + its null-context counterfactual, with hidden truth."""

    factual_batch: CCRTTrainingBatch
    factual_truth: SyntheticGroundTruth
    null_context_batch: CCRTTrainingBatch
    null_context_truth: SyntheticGroundTruth
    split: str
    batch_index: int
    seed: int


@dataclass(frozen=True)
class SyntheticDatasetBundle:
    """A full synthetic dataset for one scenario."""

    system_config: SyntheticSystemConfig
    mechanism: SyntheticMechanismSpec
    system_spec: BiologicalSystemSpec
    index_registry: CCRTIndexRegistry
    teacher: SyntheticTeacher
    train: tuple[SyntheticExample, ...]
    validation: tuple[SyntheticExample, ...]
    test: tuple[SyntheticExample, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.validation or not self.test:
            raise ValueError("train/validation/test must all be non-empty")


def _balanced_indices(gen: torch.Generator, n: int, num_classes: int) -> torch.Tensor:
    """Approximately balanced class indices of length n, deterministically shuffled."""
    base = torch.arange(n, dtype=torch.long) % num_classes
    perm = torch.randperm(n, generator=gen)
    return base[perm]


def _generate_observables(
    gen: torch.Generator, system: SyntheticSystemConfig, registry: CCRTIndexRegistry
) -> dict:
    """Generate one batch of observable teacher inputs (integer type/edge ids)."""
    B, K = system.batch_size, system.senders_per_receiver
    receiver = torch.randn(B, system.receiver_dim, generator=gen, dtype=_DTYPE)
    source_sem = system.source_semantic_scale * torch.randn(
        B, system.semantic_dim, generator=gen, dtype=_DTYPE
    )
    sender = torch.randn(B, K, system.sender_dim, generator=gen, dtype=_DTYPE)
    distance = system.max_distance * torch.rand(B, K, generator=gen, dtype=_DTYPE)

    # per-token sender-context type ids (local integer 0..T-1), balanced.
    local_types = _balanced_indices(gen, B * K, system.num_sender_context_types).view(B, K)
    # per-receiver edge ids, balanced.
    local_edges = _balanced_indices(gen, B, system.num_transition_edges)

    # sender mask: random drops, but guarantee >= 1 real sender per receiver.
    drop = torch.rand(B, K, generator=gen, dtype=_DTYPE) < system.sender_mask_probability
    mask = ~drop
    for i in range(B):
        if not bool(mask[i].any()):
            mask[i, 0] = True
    return dict(
        receiver=receiver,
        source_sem=source_sem,
        sender=sender,
        distance=distance,
        local_types=local_types,
        local_edges=local_edges,
        mask=mask,
    )


def _to_global_type_index(
    system: SyntheticSystemConfig, registry: CCRTIndexRegistry, local_types: torch.Tensor
) -> torch.Tensor:
    """Map local type ids [B,K] to global registry indices [B,K]."""
    B, K = local_types.shape
    out = torch.empty(B, K, dtype=torch.long)
    for t in range(system.num_sender_context_types):
        gidx = registry.encode_sender_context_type(
            system.biological_system_id, f"sender_type_{t}"
        )
        out[local_types == t] = gidx
    return out


def _to_global_edge_index(
    system: SyntheticSystemConfig, registry: CCRTIndexRegistry, local_edges: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(local_edges)
    for k in range(system.num_transition_edges):
        gidx = registry.encode_transition_edge(
            system.biological_system_id, f"transition_edge_{k}"
        )
        out[local_edges == k] = gidx
    return out


def _make_batch(
    *,
    system: SyntheticSystemConfig,
    obs: dict,
    mask: torch.Tensor,
    target_sem: torch.Tensor,
    growth_targets: torch.Tensor,
    global_types: torch.Tensor,
    global_edges: torch.Tensor,
    scenario_id: str,
    split: str,
    batch_index: int,
    seed: int,
) -> CCRTTrainingBatch:
    batch = CCRTTrainingBatch(
        receiver_features=obs["receiver"],
        sender_features=obs["sender"],
        sender_mask=mask.to(torch.bool),
        distance_to_receiver=obs["distance"],
        sender_context_type_ids=global_types,
        transition_edge_index=global_edges,
        source_semantic_features=obs["source_sem"],
        target_semantic_features=target_sem,
        growth_targets=growth_targets,
        metadata={
            "synthetic_scenario_id": scenario_id,
            "synthetic_split": split,
            "synthetic_batch_index": batch_index,
            "synthetic_seed": seed,
        },
    )
    batch.validate()
    return batch


def _teacher_local_ids(system, obs):
    """Teacher uses LOCAL type/edge ids (its scales are indexed locally)."""
    return obs["local_types"], obs["local_edges"]


def _generate_split(
    *,
    system: SyntheticSystemConfig,
    mechanism: SyntheticMechanismSpec,
    registry: CCRTIndexRegistry,
    teacher: SyntheticTeacher,
    scenario_id: str,
    split: str,
    n_batches: int,
    split_seed: int,
) -> tuple[SyntheticExample, ...]:
    examples: list[SyntheticExample] = []
    for bi in range(n_batches):
        batch_seed = split_seed + bi
        gen = torch.Generator().manual_seed(batch_seed)
        obs = _generate_observables(gen, system, registry)
        local_types, local_edges = _teacher_local_ids(system, obs)
        global_types = _to_global_type_index(system, registry, local_types)
        global_edges = _to_global_edge_index(system, registry, local_edges)

        # -- factual truth --
        factual_truth = teacher.evaluate(
            receiver_features=obs["receiver"],
            sender_features=obs["sender"],
            sender_mask=obs["mask"],
            distance_to_receiver=obs["distance"],
            sender_context_type_ids=local_types,
            transition_edge_index=local_edges,
            source_semantic_features=obs["source_sem"],
        )
        noise_gen = torch.Generator().manual_seed(batch_seed + 100_000)
        tgt_noise = system.target_noise_std * torch.randn(
            *factual_truth.destination_semantic_features.shape,
            generator=noise_gen, dtype=_DTYPE,
        )
        target_sem = factual_truth.destination_semantic_features + tgt_noise
        # target population may be permuted; the model must not rely on order.
        perm = torch.randperm(target_sem.shape[0], generator=noise_gen)
        target_sem = target_sem[perm]
        g_noise = system.growth_noise_std * torch.randn(
            *factual_truth.full_growth.shape, generator=noise_gen, dtype=_DTYPE
        )
        growth_targets = factual_truth.full_growth + g_noise

        factual_batch = _make_batch(
            system=system, obs=obs, mask=obs["mask"], target_sem=target_sem,
            growth_targets=growth_targets, global_types=global_types,
            global_edges=global_edges, scenario_id=scenario_id, split=split,
            batch_index=bi, seed=batch_seed,
        )

        # -- null-context counterfactual: same observables, mask all false --
        null_mask = torch.zeros_like(obs["mask"], dtype=torch.bool)
        null_truth = teacher.evaluate(
            receiver_features=obs["receiver"],
            sender_features=obs["sender"],
            sender_mask=null_mask,
            distance_to_receiver=obs["distance"],
            sender_context_type_ids=local_types,
            transition_edge_index=local_edges,
            source_semantic_features=obs["source_sem"],
        )
        null_noise_gen = torch.Generator().manual_seed(batch_seed + 200_000)
        null_tgt_noise = system.target_noise_std * torch.randn(
            *null_truth.destination_semantic_features.shape,
            generator=null_noise_gen, dtype=_DTYPE,
        )
        null_target = null_truth.destination_semantic_features + null_tgt_noise
        null_perm = torch.randperm(null_target.shape[0], generator=null_noise_gen)
        null_target = null_target[null_perm]
        null_g_noise = system.growth_noise_std * torch.randn(
            *null_truth.full_growth.shape, generator=null_noise_gen, dtype=_DTYPE
        )
        null_growth = null_truth.full_growth + null_g_noise

        # rebuild obs dict with the null mask for the null batch
        null_batch = _make_batch(
            system=system, obs=obs, mask=null_mask, target_sem=null_target,
            growth_targets=null_growth, global_types=global_types,
            global_edges=global_edges, scenario_id=scenario_id, split=split,
            batch_index=bi, seed=batch_seed,
        )

        examples.append(
            SyntheticExample(
                factual_batch=factual_batch,
                factual_truth=factual_truth,
                null_context_batch=null_batch,
                null_context_truth=null_truth,
                split=split,
                batch_index=bi,
                seed=batch_seed,
            )
        )
    return tuple(examples)


def generate_synthetic_dataset(
    *,
    system: SyntheticSystemConfig,
    scenario_id: str,
) -> SyntheticDatasetBundle:
    """Generate a full synthetic dataset (independent train/val/test splits)."""
    mechanism = build_synthetic_mechanism_spec(scenario_id, system=system)
    system_spec = build_synthetic_biological_system_spec(system)
    registry = CCRTIndexRegistry.from_system_specs([system_spec])

    # Teacher parameters are identical across splits (same seed).
    params = SyntheticTeacherParameters.from_config(
        system=system, mechanism=mechanism, seed=system.seed
    )
    teacher = SyntheticTeacher(system=system, mechanism=mechanism, parameters=params)

    # Independent split seeds (well-separated ranges).
    train = _generate_split(
        system=system, mechanism=mechanism, registry=registry, teacher=teacher,
        scenario_id=scenario_id, split="train", n_batches=system.train_batches,
        split_seed=system.seed + 1_000,
    )
    validation = _generate_split(
        system=system, mechanism=mechanism, registry=registry, teacher=teacher,
        scenario_id=scenario_id, split="validation",
        n_batches=system.validation_batches, split_seed=system.seed + 2_000,
    )
    test = _generate_split(
        system=system, mechanism=mechanism, registry=registry, teacher=teacher,
        scenario_id=scenario_id, split="test", n_batches=system.test_batches,
        split_seed=system.seed + 3_000,
    )

    return SyntheticDatasetBundle(
        system_config=system,
        mechanism=mechanism,
        system_spec=system_spec,
        index_registry=registry,
        teacher=teacher,
        train=train,
        validation=validation,
        test=test,
    )
