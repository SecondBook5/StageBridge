"""Tests for the safe checkpoint contract (uses tmp_path only)."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.contracts import CCRTValidationError
from stagebridge.ccrt.training import (
    CHECKPOINT_SCHEMA_VERSION,
    OptimizerConfig,
    build_optimizer,
    build_checkpoint_state,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)


def make_model_and_opt():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    opt = build_optimizer(model.parameters(), OptimizerConfig(learning_rate=1e-2))
    return model, opt


def make_state():
    model, opt = make_model_and_opt()
    # take one step so optimizer state is populated
    out = model(torch.randn(2, 4)).sum()
    out.backward()
    opt.step()
    state = build_checkpoint_state(
        model=model, optimizer=opt, epoch=2, global_step=5,
        extra={"note": "unit-test"},
    )
    return model, opt, state


def test_state_contains_only_state_dicts():
    _, _, state = make_state()
    assert set(state.keys()) == {
        "schema_version", "metadata", "model_state_dict",
        "optimizer_state_dict", "scheduler_state_dict",
    }
    # no full objects
    assert not isinstance(state["model_state_dict"], torch.nn.Module)
    assert isinstance(state["model_state_dict"], dict)
    assert isinstance(state["optimizer_state_dict"], dict)


def test_metadata_correct():
    _, _, state = make_state()
    meta = state["metadata"]
    assert meta["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert meta["epoch"] == 2
    assert meta["global_step"] == 5
    assert "Linear" in meta["model_class"]
    assert "AdamW" in meta["optimizer_class"]
    assert meta["scheduler_class"] is None
    assert meta["extra"] == {"note": "unit-test"}


def test_save_load_round_trip_and_restore(tmp_path):
    model, opt, state = make_state()
    path = tmp_path / "ckpt.pt"
    saved = save_checkpoint(path, state)
    assert saved.exists()

    loaded = load_checkpoint(path)
    fresh_model, fresh_opt = make_model_and_opt()
    # perturb fresh model so restoration is observable
    with torch.no_grad():
        for p in fresh_model.parameters():
            p.add_(1.0)
    meta = restore_checkpoint(state=loaded, model=fresh_model, optimizer=fresh_opt)
    assert meta.epoch == 2
    # parameters now match the saved model exactly
    for p_saved, p_restored in zip(model.parameters(), fresh_model.parameters()):
        assert torch.allclose(p_saved, p_restored, atol=1e-8)


def test_scheduler_round_trip(tmp_path):
    model, opt = make_model_and_opt()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    opt.step()
    sched.step()
    state = build_checkpoint_state(
        model=model, optimizer=opt, epoch=1, global_step=1, scheduler=sched
    )
    path = tmp_path / "ckpt.pth"
    save_checkpoint(path, state)
    loaded = load_checkpoint(path)

    model2, opt2 = make_model_and_opt()
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5)
    meta = restore_checkpoint(
        state=loaded, model=model2, optimizer=opt2, scheduler=sched2
    )
    assert "CosineAnnealingLR" in meta.scheduler_class


def test_scheduler_restore_without_state_fails(tmp_path):
    model, opt, state = make_state()  # no scheduler
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, state)
    loaded = load_checkpoint(path)
    model2, opt2 = make_model_and_opt()
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5)
    with pytest.raises(CCRTValidationError):
        restore_checkpoint(state=loaded, model=model2, scheduler=sched2)


def test_schema_mismatch_fails(tmp_path):
    _, _, state = make_state()
    state = dict(state)
    state["schema_version"] = "bogus"
    path = tmp_path / "ckpt.pt"
    with pytest.raises(CCRTValidationError):
        save_checkpoint(path, state)


def test_missing_file_fails(tmp_path):
    with pytest.raises(CCRTValidationError):
        load_checkpoint(tmp_path / "does_not_exist.pt")


def test_unsupported_suffix_fails(tmp_path):
    _, _, state = make_state()
    with pytest.raises(CCRTValidationError):
        save_checkpoint(tmp_path / "ckpt.bin", state)


def test_metadata_leakage_key_fails():
    model, opt = make_model_and_opt()
    with pytest.raises(CCRTValidationError):
        build_checkpoint_state(
            model=model, optimizer=opt, epoch=0, global_step=0,
            extra={"outcome_label": "x"},
        )


def test_parent_must_exist(tmp_path):
    _, _, state = make_state()
    with pytest.raises(CCRTValidationError):
        save_checkpoint(tmp_path / "nope" / "ckpt.pt", state)


def test_atomic_save_leaves_no_temp_file(tmp_path):
    _, _, state = make_state()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, state)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []
