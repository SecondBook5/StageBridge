"""Tests for the independent synthetic teacher."""

from __future__ import annotations

import torch

from stagebridge.ccrt.synthetic import (
    SyntheticSystemConfig,
    SyntheticTeacher,
    SyntheticTeacherParameters,
    build_synthetic_mechanism_spec,
)

SYS = SyntheticSystemConfig()


def make_teacher(scenario_id, seed=7):
    mech = build_synthetic_mechanism_spec(scenario_id, system=SYS)
    params = SyntheticTeacherParameters.from_config(system=SYS, mechanism=mech, seed=seed)
    return SyntheticTeacher(system=SYS, mechanism=mech, parameters=params)


def make_obs(seed=1, all_masked=False, B=5):
    g = torch.Generator().manual_seed(seed)
    K = SYS.senders_per_receiver
    return dict(
        receiver_features=torch.randn(B, SYS.receiver_dim, generator=g, dtype=torch.float64),
        sender_features=torch.randn(B, K, SYS.sender_dim, generator=g, dtype=torch.float64),
        sender_mask=(torch.zeros(B, K) if all_masked else torch.ones(B, K)),
        distance_to_receiver=torch.rand(B, K, generator=g, dtype=torch.float64) * SYS.max_distance,
        sender_context_type_ids=torch.randint(0, SYS.num_sender_context_types, (B, K), generator=g),
        transition_edge_index=torch.randint(0, SYS.num_transition_edges, (B,), generator=g),
        source_semantic_features=torch.randn(B, SYS.semantic_dim, generator=g, dtype=torch.float64),
    )


def test_deterministic_parameters():
    mech = build_synthetic_mechanism_spec("mixed_drift_growth", system=SYS)
    p1 = SyntheticTeacherParameters.from_config(system=SYS, mechanism=mech, seed=3)
    p2 = SyntheticTeacherParameters.from_config(system=SYS, mechanism=mech, seed=3)
    assert torch.equal(p1.sender_projection, p2.sender_projection)
    assert torch.equal(p1.context_to_drift, p2.context_to_drift)


def test_changed_seed_changes_parameters():
    mech = build_synthetic_mechanism_spec("mixed_drift_growth", system=SYS)
    p1 = SyntheticTeacherParameters.from_config(system=SYS, mechanism=mech, seed=3)
    p2 = SyntheticTeacherParameters.from_config(system=SYS, mechanism=mech, seed=4)
    assert not torch.equal(p1.sender_projection, p2.sender_projection)


def test_output_shapes_and_finite():
    teacher = make_teacher("mixed_drift_growth")
    obs = make_obs()
    gt = teacher.evaluate(**obs)
    B = obs["receiver_features"].shape[0]
    assert gt.full_drift.shape == (B, SYS.semantic_dim)
    assert gt.full_growth.shape == (B, SYS.growth_dim)
    assert gt.context_state.shape == (B, SYS.context_dim)
    assert gt.regulatory_state.shape == (B, SYS.regulatory_dim)
    for name in ("full_drift", "full_growth", "context_state", "regulatory_state",
                 "destination_semantic_features"):
        assert bool(torch.isfinite(getattr(gt, name)).all())


def test_decomposition_arithmetic_exact():
    teacher = make_teacher("regulatory_mediated")
    gt = teacher.evaluate(**make_obs())
    assert torch.allclose(gt.context_delta_drift, gt.regulatory_drift + gt.residual_drift, atol=1e-12)
    assert torch.allclose(gt.full_drift, gt.self_drift + gt.context_delta_drift, atol=1e-12)
    assert torch.allclose(gt.context_delta_growth, gt.regulatory_growth + gt.residual_growth, atol=1e-12)
    assert torch.allclose(gt.full_growth, gt.self_growth + gt.context_delta_growth, atol=1e-12)


def test_destination_arithmetic_exact():
    teacher = make_teacher("mixed_drift_growth")
    obs = make_obs()
    gt = teacher.evaluate(**obs)
    expected = obs["source_semantic_features"] + SYS.delta_tau * gt.full_drift
    assert torch.allclose(gt.destination_semantic_features, expected, atol=1e-12)


def test_null_context_zero_effect():
    teacher = make_teacher("null_context")
    gt = teacher.evaluate(**make_obs())
    assert float(gt.context_delta_drift.abs().max()) == 0.0
    assert float(gt.context_delta_growth.abs().max()) == 0.0


def test_drift_only_isolation():
    teacher = make_teacher("drift_only")
    gt = teacher.evaluate(**make_obs())
    assert float(gt.context_delta_drift.norm()) > 0
    assert float(gt.context_delta_growth.abs().max()) == 0.0


def test_growth_only_isolation():
    teacher = make_teacher("growth_only")
    gt = teacher.evaluate(**make_obs())
    assert float(gt.context_delta_growth.norm()) > 0
    assert float(gt.context_delta_drift.abs().max()) == 0.0


def test_mixed_both_nonzero():
    teacher = make_teacher("mixed_drift_growth")
    gt = teacher.evaluate(**make_obs())
    assert float(gt.context_delta_drift.norm()) > 0
    assert float(gt.context_delta_growth.norm()) > 0


def test_regulatory_mediated_direct_residual_zero():
    teacher = make_teacher("regulatory_mediated")
    gt = teacher.evaluate(**make_obs())
    assert float(gt.residual_drift.abs().max()) == 0.0
    assert float(gt.residual_growth.abs().max()) == 0.0
    assert float(gt.regulatory_drift.norm()) > 0
    assert float(gt.regulatory_growth.norm()) > 0


def test_all_masked_context_exactly_zero():
    teacher = make_teacher("mixed_drift_growth")
    gt = teacher.evaluate(**make_obs(all_masked=True))
    assert float(gt.context_state.abs().max()) == 0.0
    assert float(gt.context_delta_drift.abs().max()) == 0.0


def test_distance_dependent_monotonic_decrease():
    teacher = make_teacher("distance_dependent")
    # single receiver, single sender, vary distance
    B, K = 1, SYS.senders_per_receiver
    base = make_obs(B=B)
    base["sender_mask"] = torch.zeros(B, K)
    base["sender_mask"][0, 0] = 1.0  # one real sender
    norms = []
    for d in (0.1, 0.5, 1.5, 3.0):
        obs = dict(base)
        dist = base["distance_to_receiver"].clone()
        dist[0, 0] = d
        obs["distance_to_receiver"] = dist
        gt = teacher.evaluate(**obs)
        norms.append(float(gt.context_delta_drift.norm()))
    # strictly non-increasing with distance
    for a, b in zip(norms, norms[1:]):
        assert b <= a + 1e-9
    assert norms[0] > norms[-1]  # near > far


def test_teacher_source_has_no_student_imports():
    # ground_truth.py must not reference student classes
    import stagebridge.ccrt.synthetic.ground_truth as gt_mod
    import inspect
    src = inspect.getsource(gt_mod)
    for forbidden in (
        "ContextResidualTransportOperator", "TypedSenderContextAttention",
        "RegulatoryBottleneck", "DriftHead", "GrowthHead",
        "SemanticTransportLoss", "CCRTTrainer",
    ):
        assert forbidden not in src, f"teacher references student class {forbidden}"
