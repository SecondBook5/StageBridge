from __future__ import annotations

import numpy as np
import torch

from stagebridge.transition_model.gaussian_init import (
    build_gaussian_bridge,
    estimate_diagonal_gaussian,
    interpolate_bridge_moments,
    sample_bridge_state,
)


def test_estimate_diagonal_gaussian_returns_mean_and_variance() -> None:
    x = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=torch.float32,
    )

    moments = estimate_diagonal_gaussian(x, min_variance=1e-5)

    assert torch.allclose(moments.mean, torch.tensor([2.0, 3.0]))
    assert torch.allclose(moments.variance, torch.tensor([8.0 / 3.0, 8.0 / 3.0]))


def test_interpolate_bridge_moments_matches_endpoints_and_adds_midpoint_noise() -> None:
    x_src = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x_tgt = torch.tensor(
        [
            [4.0, 4.0],
            [4.0, 6.0],
            [6.0, 4.0],
        ],
        dtype=torch.float32,
    )

    bridge = build_gaussian_bridge(x_src, x_tgt, sigma=0.4, min_variance=1e-5)
    at_start = interpolate_bridge_moments(bridge, 0.0)
    at_mid = interpolate_bridge_moments(bridge, 0.5)
    at_end = interpolate_bridge_moments(bridge, 1.0)

    assert torch.allclose(at_start.mean, bridge.source.mean)
    assert torch.allclose(at_start.variance, bridge.source.variance)
    assert torch.allclose(at_end.mean, bridge.target.mean)
    assert torch.allclose(at_end.variance, bridge.target.variance)
    assert torch.allclose(at_mid.mean, 0.5 * (bridge.source.mean + bridge.target.mean))
    assert torch.all(at_mid.variance > 0.0)
    assert torch.all(at_mid.variance >= 0.25 * (bridge.source.variance + bridge.target.variance))


def test_sample_bridge_state_returns_requested_batch_shape() -> None:
    x_src = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    x_tgt = x_src + 1.5

    bridge = build_gaussian_bridge(x_src, x_tgt, sigma=0.2, min_variance=1e-4)
    samples = sample_bridge_state(bridge, t=0.35, n_samples=16)

    assert samples.shape == (16, 3)
    assert torch.isfinite(samples).all()


def test_build_gaussian_bridge_accepts_numpy_inputs() -> None:
    x_src = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )
    x_tgt = x_src + 2.0

    bridge = build_gaussian_bridge(x_src, x_tgt, sigma=0.2, min_variance=1e-4)

    assert bridge.source.mean.shape == (2,)
    assert bridge.target.mean.shape == (2,)


def test_interpolate_bridge_moments_supports_batched_times() -> None:
    x_src = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    x_tgt = x_src + 2.0
    bridge = build_gaussian_bridge(x_src, x_tgt, sigma=0.1, min_variance=1e-4)

    moments = interpolate_bridge_moments(bridge, torch.tensor([0.0, 0.5, 1.0]))

    assert moments.mean.shape == (3, 2)
    assert moments.variance.shape == (3, 2)
    assert torch.allclose(moments.mean[0], bridge.source.mean)
    assert torch.allclose(moments.mean[-1], bridge.target.mean)
