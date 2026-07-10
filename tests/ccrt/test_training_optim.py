"""Tests for optimizer and scheduler factories."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.training import (
    OptimizerConfig,
    SchedulerConfig,
    build_optimizer,
    build_scheduler,
)


def test_optimizer_config_validation():
    with pytest.raises(ValueError):
        OptimizerConfig(name="sgd")
    with pytest.raises(ValueError):
        OptimizerConfig(learning_rate=0.0)
    with pytest.raises(ValueError):
        OptimizerConfig(weight_decay=-1.0)
    with pytest.raises(ValueError):
        OptimizerConfig(beta1=1.0)
    with pytest.raises(ValueError):
        OptimizerConfig(eps=0.0)


def test_build_adamw():
    lin = torch.nn.Linear(3, 2)
    opt = build_optimizer(lin.parameters(), OptimizerConfig(learning_rate=1e-3))
    assert isinstance(opt, torch.optim.AdamW)


def test_trainable_parameters_filtered():
    lin = torch.nn.Linear(3, 2)
    lin.bias.requires_grad_(False)
    opt = build_optimizer(lin.parameters(), OptimizerConfig())
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == 1  # only the weight


def test_no_trainable_parameters_fails():
    lin = torch.nn.Linear(3, 2)
    for p in lin.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError):
        build_optimizer(lin.parameters(), OptimizerConfig())


def test_scheduler_config_validation():
    with pytest.raises(ValueError):
        SchedulerConfig(name="step")
    with pytest.raises(ValueError):
        SchedulerConfig(t_max=0)
    with pytest.raises(ValueError):
        SchedulerConfig(eta_min=-1.0)


def test_none_scheduler_returns_none():
    lin = torch.nn.Linear(3, 2)
    opt = build_optimizer(lin.parameters(), OptimizerConfig())
    assert build_scheduler(opt, SchedulerConfig(name="none")) is None


def test_cosine_scheduler_returned_and_changes_lr():
    lin = torch.nn.Linear(3, 2)
    opt = build_optimizer(lin.parameters(), OptimizerConfig(learning_rate=0.1))
    sched = build_scheduler(opt, SchedulerConfig(name="cosine", t_max=10))
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    lr0 = opt.param_groups[0]["lr"]
    opt.step()
    sched.step()
    lr1 = opt.param_groups[0]["lr"]
    assert lr1 != lr0
