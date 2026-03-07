import numpy as np
import pytest
import torch

from stagebridge.training.trainer import TransitionSampler


def _toy_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latent = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
            [2.0, 2.1, 2.2],
            [2.3, 2.4, 2.5],
        ],
        dtype=np.float32,
    )
    obs_stage = np.array(["Normal", "Normal", "AAH", "AAH", "Normal", "AAH"], dtype=object)
    obs_donor = np.array(["D1", "D1", "D1", "D1", "D2", "D2"], dtype=object)
    spatial_niche = np.array(
        [
            [1.0, 10.0],
            [1.0, 10.0],
            [3.0, 30.0],
            [3.0, 30.0],
            [9.0, 90.0],
            [9.0, 90.0],
        ],
        dtype=np.float32,
    )
    return latent, obs_stage, obs_donor, spatial_niche


def test_transition_sampler_spatial_niche_accepts_full_matrix_and_uses_donor_mean() -> None:
    latent, obs_stage, obs_donor, spatial_niche = _toy_inputs()
    sampler = TransitionSampler(
        latent=latent,
        obs_stage=obs_stage,
        obs_donor=obs_donor,
        donor_ids=["D1"],
        batch_cells=4,
        device="cpu",
        spatial_niche=spatial_niche,
    )
    batch = sampler.sample_batch()
    assert batch.spatial_niche is not None
    expected = torch.tensor([2.0, 20.0], dtype=torch.float32)
    assert batch.spatial_niche.shape == (4, 2)
    assert torch.allclose(batch.spatial_niche[0], expected)
    assert torch.allclose(batch.spatial_niche.mean(dim=0), expected)


def test_transition_sampler_spatial_niche_accepts_prefiltered_matrix() -> None:
    latent, obs_stage, obs_donor, spatial_niche = _toy_inputs()
    prefiltered = spatial_niche[:4]
    sampler = TransitionSampler(
        latent=latent,
        obs_stage=obs_stage,
        obs_donor=obs_donor,
        donor_ids=["D1"],
        batch_cells=3,
        device="cpu",
        spatial_niche=prefiltered,
    )
    x_src, x_tgt = sampler.sample_transition_pair("Normal", "AAH", n_cells=5, donor_id="D1")
    assert x_src.shape == (5, 3)
    assert x_tgt.shape == (5, 3)
    batch = sampler.sample_batch()
    assert batch.spatial_niche is not None
    expected = torch.tensor([2.0, 20.0], dtype=torch.float32)
    assert torch.allclose(batch.spatial_niche[0], expected)


def test_transition_sampler_rejects_empty_donor_subset() -> None:
    latent, obs_stage, obs_donor, _ = _toy_inputs()
    with pytest.raises(ValueError, match="No cells found"):
        TransitionSampler(
            latent=latent,
            obs_stage=obs_stage,
            obs_donor=obs_donor,
            donor_ids=["missing_donor"],
            batch_cells=4,
            device="cpu",
        )
