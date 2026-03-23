"""Transition model components for StageBridge.

This module provides the stochastic transition dynamics layer (Layer D/E),
implementing flow matching with optimal transport coupling for modeling
cell-state transitions under spatial and niche context.

Key components:
- StageBridgeModel: Main transformer-conditioned flow model
- DriftNetwork variants: Edge-conditioned, cross-attention, biological baseline
- DiffusionNetwork: State-dependent diffusion for uncertainty
- WESRegularizer: Evolutionary compatibility constraints
- RelationalPretraining: SSL pretraining objectives

Training modes:
- SSL Pretraining: Masked receiver reconstruction, niche discrimination
- Transition Training: OT-CFM flow matching with niche conditioning
"""
