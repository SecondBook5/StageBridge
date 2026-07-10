"""End-to-end synthetic benchmark: train the CCRT student, measure recovery.

Builds an independent student (operator + semantic loss + composite objective +
trainer), trains it on observable synthetic data only, and evaluates how well it
recovers the teacher's hidden context mechanism via factual-minus-null
counterfactuals plus scenario-specific diagnostics (distance, sender-type,
transition-edge, negative-control).

The student is never initialized from teacher parameters, never supervised on the
hidden decomposition, and never trained on test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..operators.model import (
    ContextResidualTransportConfig,
    ContextResidualTransportOperator,
)
from ..representations.semantic import SemanticGeometryConfig
from ..training.batch import CCRTTrainingBatch
from ..training.objective import CompositeCCRTObjective, CompositeCCRTObjectiveConfig
from ..training.optim import OptimizerConfig, build_optimizer
from ..training.trainer import CCRTTrainer, EpochMetrics, TrainerConfig
from ..transport.native_sinkhorn import SinkhornConfig
from ..transport.semantic_loss import SemanticTransportLoss, SemanticTransportLossConfig
from .config import SyntheticBenchmarkConfig, SyntheticSystemConfig
from .counterfactuals import (
    attach_teacher_targets,
    remove_sender_context_type,
    replace_transition_edge,
    set_sender_distances,
)
from .generator import SyntheticDatasetBundle, generate_synthetic_dataset
from .ground_truth import SyntheticGroundTruth, SyntheticTeacher
from .mechanisms import (
    DISTANCE_DEPENDENT,
    REGULATORY_MEDIATED,
    SENDER_TYPE_SPECIFIC,
    SYNTHETIC_SCENARIO_IDS,
    TRANSITION_EDGE_SPECIFIC,
    WRONG_CONTEXT_NEGATIVE_CONTROL,
)
from .metrics import (
    CounterfactualRecoveryMetrics,
    mean_cosine_recovery,
    mean_effect_norm,
    pearson_recovery,
    rank_order_recovery,
    relative_root_mean_squared_error,
    root_mean_squared_error,
)

__all__ = [
    "SyntheticPrediction",
    "SyntheticBenchmarkComponents",
    "SyntheticScenarioResult",
    "SyntheticBenchmarkOutput",
    "predict_synthetic_batch",
    "evaluate_context_counterfactual",
    "build_synthetic_benchmark_components",
    "build_synthetic_training_sequence",
    "run_synthetic_scenario_benchmark",
    "run_synthetic_benchmark_matrix",
]

_EPS = 1e-8
_DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class SyntheticPrediction:
    full_drift: torch.Tensor
    full_growth: torch.Tensor
    regulatory_drift: torch.Tensor
    residual_drift: torch.Tensor
    regulatory_growth: torch.Tensor
    residual_growth: torch.Tensor
    regulatory_state: torch.Tensor
    attention_weights: torch.Tensor


@dataclass(frozen=True)
class SyntheticBenchmarkComponents:
    model: ContextResidualTransportOperator
    semantic_loss: SemanticTransportLoss
    objective: CompositeCCRTObjective
    optimizer: torch.optim.Optimizer
    trainer: CCRTTrainer


@dataclass(frozen=True)
class SyntheticScenarioResult:
    scenario_id: str
    initial_test_loss: float
    final_test_loss: float
    best_training_loss: float
    drift_rmse: float
    drift_relative_rmse: float
    drift_cosine: float
    drift_pearson: float
    growth_rmse: float
    growth_pearson: float
    predicted_context_drift_norm: float
    true_context_drift_norm: float
    predicted_context_growth_norm: float
    true_context_growth_norm: float
    regulatory_drift_fraction: float
    regulatory_growth_fraction: float
    distance_response_correlation: float | None
    sender_type_rank_recovery: float | None
    sender_type_top_effect_correct: bool | None
    transition_edge_contrast_cosine: float | None
    negative_control_effect_ratio: float | None
    all_metrics_finite: bool


@dataclass(frozen=True)
class SyntheticBenchmarkOutput:
    result: SyntheticScenarioResult
    history: tuple[EpochMetrics, ...]
    model: ContextResidualTransportOperator


# ---------------------------------------------------------------------------
# Prediction + counterfactual recovery
# ---------------------------------------------------------------------------


def predict_synthetic_batch(
    *,
    model: ContextResidualTransportOperator,
    batch: CCRTTrainingBatch,
) -> SyntheticPrediction:
    """Run the model in eval/no-grad mode; do not mutate model or batch."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(
                receiver_features=batch.receiver_features,
                sender_features=batch.sender_features,
                sender_mask=batch.sender_mask,
                distance_to_receiver=batch.distance_to_receiver,
                sender_context_type_ids=batch.sender_context_type_ids,
                transition_edge_index=batch.transition_edge_index,
                uncertainty=batch.uncertainty,
            )
    finally:
        if was_training:
            model.train()
    return SyntheticPrediction(
        full_drift=out.full_drift,
        full_growth=out.full_growth,
        regulatory_drift=out.regulatory_drift,
        residual_drift=out.residual_drift,
        regulatory_growth=out.regulatory_growth,
        residual_growth=out.residual_growth,
        regulatory_state=out.regulatory_state,
        attention_weights=out.attention_weights,
    )


def evaluate_context_counterfactual(
    *,
    model: ContextResidualTransportOperator,
    factual_batch: CCRTTrainingBatch,
    factual_truth: SyntheticGroundTruth,
    counterfactual_batch: CCRTTrainingBatch,
    counterfactual_truth: SyntheticGroundTruth,
) -> CounterfactualRecoveryMetrics:
    """Recovery of factual-minus-counterfactual context effect (drift + growth)."""
    fp = predict_synthetic_batch(model=model, batch=factual_batch)
    cp = predict_synthetic_batch(model=model, batch=counterfactual_batch)

    pred_drift = fp.full_drift - cp.full_drift
    pred_growth = fp.full_growth - cp.full_growth
    true_drift = (
        factual_truth.full_drift - counterfactual_truth.full_drift
    ).to(pred_drift.dtype)
    true_growth = (
        factual_truth.full_growth - counterfactual_truth.full_growth
    ).to(pred_growth.dtype)

    return CounterfactualRecoveryMetrics(
        drift_rmse=root_mean_squared_error(pred_drift, true_drift),
        drift_relative_rmse=relative_root_mean_squared_error(pred_drift, true_drift),
        drift_cosine=mean_cosine_recovery(pred_drift, true_drift),
        drift_pearson=pearson_recovery(pred_drift, true_drift),
        growth_rmse=root_mean_squared_error(pred_growth, true_growth),
        growth_pearson=pearson_recovery(pred_growth, true_growth),
        predicted_drift_effect_norm=mean_effect_norm(pred_drift),
        true_drift_effect_norm=mean_effect_norm(true_drift),
        predicted_growth_effect_norm=mean_effect_norm(pred_growth),
        true_growth_effect_norm=mean_effect_norm(true_growth),
    )


# ---------------------------------------------------------------------------
# Student construction
# ---------------------------------------------------------------------------


def build_synthetic_benchmark_components(
    *,
    dataset: SyntheticDatasetBundle,
    benchmark: SyntheticBenchmarkConfig,
) -> SyntheticBenchmarkComponents:
    """Build an independent student (native transport) for the dataset."""
    system = dataset.system_config
    registry = dataset.index_registry

    model = ContextResidualTransportOperator(
        ContextResidualTransportConfig(
            receiver_dim=system.receiver_dim,
            sender_dim=system.sender_dim,
            hidden_dim=benchmark.hidden_dim,
            num_heads=benchmark.num_heads,
            num_sender_context_types=registry.num_sender_context_types,
            empty_sender_context_type_id=registry.empty_sender_context_type_index,
            regulatory_dim=system.regulatory_dim,
            drift_dim=system.semantic_dim,
            growth_dim=system.growth_dim,
            num_transition_edges=registry.num_transition_edges,
        )
    )

    semantic_loss = SemanticTransportLoss(
        geometry=SemanticGeometryConfig(metric="squared_euclidean", normalization="none"),
        native_sinkhorn=SinkhornConfig(
            epsilon=benchmark.sinkhorn_epsilon,
            max_iterations=benchmark.sinkhorn_iterations,
            early_stopping=False,
        ),
        loss=SemanticTransportLossConfig(
            delta_tau=system.delta_tau,
            displacement_weight=benchmark.displacement_weight,
            direction_weight=benchmark.direction_weight,
            distribution_weight=benchmark.distribution_weight,
            distribution_backend="native",
        ),
    )

    objective = CompositeCCRTObjective(
        semantic_transport_loss=semantic_loss,
        config=CompositeCCRTObjectiveConfig(
            semantic_weight=1.0,
            attention_entropy_weight=benchmark.attention_entropy_weight,
            sender_effect_l1_weight=benchmark.sender_effect_l1_weight,
            regulatory_l1_weight=benchmark.regulatory_l1_weight,
            residual_drift_l2_weight=benchmark.residual_drift_l2_weight,
            residual_growth_l2_weight=benchmark.residual_growth_l2_weight,
            growth_supervision_weight=benchmark.growth_supervision_weight,
        ),
    )

    optimizer = build_optimizer(
        model.parameters(),
        OptimizerConfig(
            learning_rate=benchmark.learning_rate, weight_decay=benchmark.weight_decay
        ),
    )

    trainer = CCRTTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        config=TrainerConfig(
            epochs=benchmark.epochs,
            device=benchmark.device,
            dtype=benchmark.dtype,
            gradient_clip_norm=benchmark.gradient_clip_norm,
            seed=benchmark.seed,
        ),
    )
    return SyntheticBenchmarkComponents(
        model=trainer.model,  # trainer moved it to device/dtype
        semantic_loss=semantic_loss,
        objective=objective,
        optimizer=optimizer,
        trainer=trainer,
    )


def build_synthetic_training_sequence(
    dataset: SyntheticDatasetBundle,
) -> tuple[CCRTTrainingBatch, ...]:
    """Ordered training batches (alternating factual/null when configured)."""
    seq: list[CCRTTrainingBatch] = []
    include_null = dataset.system_config.include_null_context_training_pairs
    for ex in dataset.train:
        seq.append(ex.factual_batch)
        if include_null:
            seq.append(ex.null_context_batch)
    return tuple(seq)


def _validation_sequence(dataset: SyntheticDatasetBundle) -> tuple[CCRTTrainingBatch, ...]:
    seq: list[CCRTTrainingBatch] = []
    include_null = dataset.system_config.include_null_context_training_pairs
    for ex in dataset.validation:
        seq.append(ex.factual_batch)
        if include_null:
            seq.append(ex.null_context_batch)
    return tuple(seq)


# ---------------------------------------------------------------------------
# Scenario-specific diagnostics
# ---------------------------------------------------------------------------


def _regulatory_fractions(
    model: ContextResidualTransportOperator, dataset: SyntheticDatasetBundle
) -> tuple[float, float]:
    """Fraction of predicted context effect routed through the regulatory path."""
    reg_d, res_d, reg_g, res_g = 0.0, 0.0, 0.0, 0.0
    n = 0
    for ex in dataset.test:
        fp = predict_synthetic_batch(model=model, batch=ex.factual_batch)
        cp = predict_synthetic_batch(model=model, batch=ex.null_context_batch)
        reg_d += float((fp.regulatory_drift - cp.regulatory_drift).norm(dim=-1).mean())
        res_d += float((fp.residual_drift - cp.residual_drift).norm(dim=-1).mean())
        reg_g += float((fp.regulatory_growth - cp.regulatory_growth).norm(dim=-1).mean())
        res_g += float((fp.residual_growth - cp.residual_growth).norm(dim=-1).mean())
        n += 1
    n = max(n, 1)
    reg_d, res_d, reg_g, res_g = reg_d / n, res_d / n, reg_g / n, res_g / n
    drift_frac = reg_d / (reg_d + res_d + _EPS)
    growth_frac = reg_g / (reg_g + res_g + _EPS)
    return drift_frac, growth_frac


def _distance_response(
    model: ContextResidualTransportOperator, dataset: SyntheticDatasetBundle
) -> float | None:
    """Pearson corr between predicted and true distance-response curves."""
    ex = dataset.test[0]
    teacher = dataset.teacher
    distances = [0.25, 0.75, 1.50, 3.00]
    pred_curve, true_curve = [], []
    for d in distances:
        fb = set_sender_distances(ex.factual_batch, d, real_senders_only=True)
        fb, ftruth = attach_teacher_targets(batch=fb, teacher=teacher, seed=999)
        nb = set_sender_distances(ex.null_context_batch, d, real_senders_only=True)
        nb, ntruth = attach_teacher_targets(batch=nb, teacher=teacher, seed=998)
        fp = predict_synthetic_batch(model=model, batch=fb)
        cp = predict_synthetic_batch(model=model, batch=nb)
        pred_curve.append(float((fp.full_drift - cp.full_drift).norm(dim=-1).mean()))
        true_curve.append(float((ftruth.full_drift - ntruth.full_drift).norm(dim=-1).mean()))
    pred_t = torch.tensor(pred_curve, dtype=torch.float64)
    true_t = torch.tensor(true_curve, dtype=torch.float64)
    return float(pearson_recovery(pred_t, true_t))


def _sender_type_effects(
    model: ContextResidualTransportOperator, dataset: SyntheticDatasetBundle
) -> tuple[float, bool]:
    """Rank recovery + top-effect correctness across real sender types."""
    system = dataset.system_config
    ex = dataset.test[0]
    teacher = dataset.teacher
    pred_mags, true_mags = [], []
    for t in range(system.num_sender_context_types):
        cf = remove_sender_context_type(ex.factual_batch, t)
        cf, ctruth = attach_teacher_targets(batch=cf, teacher=teacher, seed=500 + t)
        fp = predict_synthetic_batch(model=model, batch=ex.factual_batch)
        cp = predict_synthetic_batch(model=model, batch=cf)
        pred_mags.append(float((fp.full_drift - cp.full_drift).norm(dim=-1).mean()))
        true_mags.append(float((ex.factual_truth.full_drift - ctruth.full_drift).norm(dim=-1).mean()))
    pred_t = torch.tensor(pred_mags, dtype=torch.float64)
    true_t = torch.tensor(true_mags, dtype=torch.float64)
    rank = float(rank_order_recovery(pred_t, true_t))
    top_correct = bool(int(pred_t.argmax()) == int(true_t.argmax()))
    return rank, top_correct


def _edge_contrast(
    model: ContextResidualTransportOperator, dataset: SyntheticDatasetBundle
) -> float:
    """Cosine recovery of the edge-0 vs edge-1 full-drift contrast."""
    ex = dataset.test[0]
    teacher = dataset.teacher
    b0 = replace_transition_edge(ex.factual_batch, 0)
    b1 = replace_transition_edge(ex.factual_batch, 1)
    b0, t0 = attach_teacher_targets(batch=b0, teacher=teacher, seed=600)
    b1, t1 = attach_teacher_targets(batch=b1, teacher=teacher, seed=601)
    p0 = predict_synthetic_batch(model=model, batch=b0)
    p1 = predict_synthetic_batch(model=model, batch=b1)
    pred_contrast = p0.full_drift - p1.full_drift
    true_contrast = (t0.full_drift - t1.full_drift).to(pred_contrast.dtype)
    return float(mean_cosine_recovery(pred_contrast, true_contrast))


def _negative_control_ratio(
    model: ContextResidualTransportOperator, dataset: SyntheticDatasetBundle
) -> float:
    """Predicted removal-effect ratio: negative-control / active (smaller better)."""
    ex = dataset.test[0]
    mech = dataset.mechanism
    active = mech.active_sender_context_type_ids[0]
    negctl = mech.negative_control_sender_context_type_ids[0]
    fp = predict_synthetic_batch(model=model, batch=ex.factual_batch)

    cf_active = remove_sender_context_type(ex.factual_batch, active)
    pa = predict_synthetic_batch(model=model, batch=cf_active)
    active_effect = float((fp.full_drift - pa.full_drift).norm(dim=-1).mean())

    cf_neg = remove_sender_context_type(ex.factual_batch, negctl)
    pn = predict_synthetic_batch(model=model, batch=cf_neg)
    neg_effect = float((fp.full_drift - pn.full_drift).norm(dim=-1).mean())
    return neg_effect / (active_effect + _EPS)


# ---------------------------------------------------------------------------
# Scenario benchmark runner
# ---------------------------------------------------------------------------


def _mean_test_objective(
    components: SyntheticBenchmarkComponents, dataset: SyntheticDatasetBundle
) -> float:
    metrics = [
        components.trainer.evaluate_step(ex.factual_batch) for ex in dataset.test
    ]
    return sum(m.total_loss for m in metrics) / max(len(metrics), 1)


def run_synthetic_scenario_benchmark(
    *,
    system: SyntheticSystemConfig,
    scenario_id: str,
    benchmark: SyntheticBenchmarkConfig,
) -> SyntheticBenchmarkOutput:
    """Generate data, train the student, and measure mechanism recovery."""
    dataset = generate_synthetic_dataset(system=system, scenario_id=scenario_id)
    components = build_synthetic_benchmark_components(dataset=dataset, benchmark=benchmark)
    trainer = components.trainer
    model = components.model

    initial_test_loss = _mean_test_objective(components, dataset)

    train_seq = build_synthetic_training_sequence(dataset)
    val_seq = _validation_sequence(dataset)
    history = trainer.fit(train_batches=train_seq, validation_batches=val_seq)

    final_test_loss = _mean_test_objective(components, dataset)
    best_training_loss = min(h.train["total_loss"] for h in history)

    # -- aggregate factual/null counterfactual recovery over the test split --
    drift_rmse = drift_rel = drift_cos = drift_pear = 0.0
    growth_rmse = growth_pear = 0.0
    pred_dn = true_dn = pred_gn = true_gn = 0.0
    n = 0
    for ex in dataset.test:
        m = evaluate_context_counterfactual(
            model=model,
            factual_batch=ex.factual_batch,
            factual_truth=ex.factual_truth,
            counterfactual_batch=ex.null_context_batch,
            counterfactual_truth=ex.null_context_truth,
        )
        drift_rmse += float(m.drift_rmse)
        drift_rel += float(m.drift_relative_rmse)
        drift_cos += float(m.drift_cosine)
        drift_pear += float(m.drift_pearson)
        growth_rmse += float(m.growth_rmse)
        growth_pear += float(m.growth_pearson)
        pred_dn += float(m.predicted_drift_effect_norm)
        true_dn += float(m.true_drift_effect_norm)
        pred_gn += float(m.predicted_growth_effect_norm)
        true_gn += float(m.true_growth_effect_norm)
        n += 1
    n = max(n, 1)

    reg_drift_frac, reg_growth_frac = _regulatory_fractions(model, dataset)

    # -- scenario-specific diagnostics --
    distance_corr = None
    sender_rank = None
    sender_top = None
    edge_cos = None
    neg_ratio = None
    if scenario_id == DISTANCE_DEPENDENT:
        distance_corr = _distance_response(model, dataset)
    if scenario_id in (SENDER_TYPE_SPECIFIC, WRONG_CONTEXT_NEGATIVE_CONTROL):
        sender_rank, sender_top = _sender_type_effects(model, dataset)
    if scenario_id == TRANSITION_EDGE_SPECIFIC:
        edge_cos = _edge_contrast(model, dataset)
    if scenario_id == WRONG_CONTEXT_NEGATIVE_CONTROL:
        neg_ratio = _negative_control_ratio(model, dataset)

    values = [
        initial_test_loss, final_test_loss, best_training_loss,
        drift_rmse / n, drift_rel / n, drift_cos / n, drift_pear / n,
        growth_rmse / n, growth_pear / n, pred_dn / n, true_dn / n,
        pred_gn / n, true_gn / n, reg_drift_frac, reg_growth_frac,
    ]
    optional = [distance_corr, sender_rank, edge_cos, neg_ratio]
    all_finite = all(
        v == v and abs(v) != float("inf") for v in values
    ) and all(
        (v is None) or (v == v and abs(v) != float("inf")) for v in optional
    )

    result = SyntheticScenarioResult(
        scenario_id=scenario_id,
        initial_test_loss=initial_test_loss,
        final_test_loss=final_test_loss,
        best_training_loss=best_training_loss,
        drift_rmse=drift_rmse / n,
        drift_relative_rmse=drift_rel / n,
        drift_cosine=drift_cos / n,
        drift_pearson=drift_pear / n,
        growth_rmse=growth_rmse / n,
        growth_pearson=growth_pear / n,
        predicted_context_drift_norm=pred_dn / n,
        true_context_drift_norm=true_dn / n,
        predicted_context_growth_norm=pred_gn / n,
        true_context_growth_norm=true_gn / n,
        regulatory_drift_fraction=reg_drift_frac,
        regulatory_growth_fraction=reg_growth_frac,
        distance_response_correlation=distance_corr,
        sender_type_rank_recovery=sender_rank,
        sender_type_top_effect_correct=sender_top,
        transition_edge_contrast_cosine=edge_cos,
        negative_control_effect_ratio=neg_ratio,
        all_metrics_finite=all_finite,
    )
    return SyntheticBenchmarkOutput(result=result, history=history, model=model)


def run_synthetic_benchmark_matrix(
    *,
    system: SyntheticSystemConfig,
    benchmark: SyntheticBenchmarkConfig,
    scenario_ids=SYNTHETIC_SCENARIO_IDS,
) -> tuple[SyntheticScenarioResult, ...]:
    """Run several scenarios in order, each with an independent student."""
    for sid in scenario_ids:
        if sid not in SYNTHETIC_SCENARIO_IDS:
            raise ValueError(f"unsupported scenario_id '{sid}'")
    results = []
    for position, sid in enumerate(scenario_ids):
        # deterministic per-scenario seed derived from position (no hash()).
        scenario_benchmark = SyntheticBenchmarkConfig(
            **{**benchmark.__dict__, "seed": benchmark.seed + 1_000 * (position + 1)}
        )
        out = run_synthetic_scenario_benchmark(
            system=system, scenario_id=sid, benchmark=scenario_benchmark
        )
        results.append(out.result)
    return tuple(results)
