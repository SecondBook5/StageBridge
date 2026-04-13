"""
Layer A: Dual-Reference Latent Mapping

Maps cells to a shared Euclidean latent space using dual references:
- HLCA (Healthy Lung Cell Atlas) - normal reference
- LuCA (Lung Cancer Atlas) - disease reference

V1 uses Euclidean geometry with code structure ready for V2 non-Euclidean upgrade.

Architecture:
1. Map cell to HLCA reference → z_hlca
2. Map cell to LuCA reference → z_luca
3. Fuse via learned combination → z_fused
4. Project to isometric latent space

For V1 synthetic data: Can use pre-computed embeddings.
For V1 real data: Will use reference mapping (scanvi, scVI, etc.)
"""

import torch
import torch.nn as nn


class DualReferenceMapper(nn.Module):
    """
    Dual-reference latent mapping with Euclidean geometry.

    V1: Euclidean latent space
    V2: Extensible to hyperbolic/spherical geometry

    Args:
        input_dim: Gene expression dimensionality
        latent_dim: Target latent space dimension
        hlca_dim: HLCA reference embedding dimension
        luca_dim: LuCA reference embedding dimension
        fusion_mode: How to fuse references ('concat', 'attention', 'gate')
        use_projection: Project to isometric space
    """

    def __init__(
        self,
        input_dim: int = 2000,
        latent_dim: int = 40,
        hlca_dim: int = 16,
        luca_dim: int = 16,
        fusion_mode: str = "attention",
        use_projection: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hlca_dim = hlca_dim
        self.luca_dim = luca_dim
        self.fusion_mode = fusion_mode
        self.use_projection = use_projection

        # Reference encoders
        self.hlca_encoder = self._build_encoder(input_dim, hlca_dim)
        self.luca_encoder = self._build_encoder(input_dim, luca_dim)

        # Fusion mechanism
        if fusion_mode == "concat":
            fusion_input_dim = hlca_dim + luca_dim
            self.fusion = nn.Linear(fusion_input_dim, latent_dim)

        elif fusion_mode == "attention":
            # Attention-weighted fusion
            self.query = nn.Linear(hlca_dim + luca_dim, latent_dim)
            self.key_hlca = nn.Linear(hlca_dim, latent_dim)
            self.key_luca = nn.Linear(luca_dim, latent_dim)
            self.value_hlca = nn.Linear(hlca_dim, latent_dim)
            self.value_luca = nn.Linear(luca_dim, latent_dim)

        elif fusion_mode == "gate":
            # Gated fusion (FiLM-style)
            self.gate = nn.Sequential(
                nn.Linear(hlca_dim + luca_dim, latent_dim),
                nn.Sigmoid(),
            )
            self.hlca_proj = nn.Linear(hlca_dim, latent_dim)
            self.luca_proj = nn.Linear(luca_dim, latent_dim)

        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # Optional: Project to isometric space
        if use_projection:
            self.projector = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )

    def _build_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build encoder network for reference mapping."""
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Map cells to dual-reference latent space.

        Args:
            x: Cell expression profiles (batch_size, input_dim)
            return_intermediates: Return z_hlca, z_luca in addition to z_fused

        Returns:
            z_fused: Fused latent embedding (batch_size, latent_dim)
            If return_intermediates: (z_fused, z_hlca, z_luca)
        """
        # Encode to each reference
        z_hlca = self.hlca_encoder(x)  # (batch_size, hlca_dim)
        z_luca = self.luca_encoder(x)  # (batch_size, luca_dim)

        # Fuse references
        if self.fusion_mode == "concat":
            z_concat = torch.cat([z_hlca, z_luca], dim=-1)
            z_fused = self.fusion(z_concat)

        elif self.fusion_mode == "attention":
            # Attention-weighted combination
            z_concat = torch.cat([z_hlca, z_luca], dim=-1)
            query = self.query(z_concat)  # (batch_size, latent_dim)

            key_h = self.key_hlca(z_hlca)  # (batch_size, latent_dim)
            key_l = self.key_luca(z_luca)  # (batch_size, latent_dim)

            # Compute attention scores
            attn_h = torch.sum(query * key_h, dim=-1, keepdim=True)  # (batch_size, 1)
            attn_l = torch.sum(query * key_l, dim=-1, keepdim=True)  # (batch_size, 1)

            attn_weights = torch.softmax(
                torch.cat([attn_h, attn_l], dim=-1), dim=-1
            )  # (batch_size, 2)

            value_h = self.value_hlca(z_hlca)  # (batch_size, latent_dim)
            value_l = self.value_luca(z_luca)  # (batch_size, latent_dim)

            z_fused = attn_weights[:, 0:1] * value_h + attn_weights[:, 1:2] * value_l

        elif self.fusion_mode == "gate":
            # Gated fusion
            z_concat = torch.cat([z_hlca, z_luca], dim=-1)
            gate = self.gate(z_concat)  # (batch_size, latent_dim)

            h_proj = self.hlca_proj(z_hlca)  # (batch_size, latent_dim)
            l_proj = self.luca_proj(z_luca)  # (batch_size, latent_dim)

            z_fused = gate * h_proj + (1 - gate) * l_proj

        # Optional projection
        if self.use_projection:
            z_fused = self.projector(z_fused)

        if return_intermediates:
            return z_fused, z_hlca, z_luca
        else:
            return z_fused

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights between HLCA and LuCA references.

        Useful for interpretability: how much does each reference contribute?

        Returns:
            weights: (batch_size, 2) - [hlca_weight, luca_weight]
        """
        assert self.fusion_mode == "attention", "Only available for attention fusion"

        z_hlca = self.hlca_encoder(x)
        z_luca = self.luca_encoder(x)

        z_concat = torch.cat([z_hlca, z_luca], dim=-1)
        query = self.query(z_concat)

        key_h = self.key_hlca(z_hlca)
        key_l = self.key_luca(z_luca)

        attn_h = torch.sum(query * key_h, dim=-1, keepdim=True)
        attn_l = torch.sum(query * key_l, dim=-1, keepdim=True)

        attn_weights = torch.softmax(torch.cat([attn_h, attn_l], dim=-1), dim=-1)

        return attn_weights


class PrecomputedDualReference(nn.Module):
    """
    Passthrough module for pre-computed dual-reference embeddings.

    For V1 synthetic data or when embeddings are pre-computed offline,
    this module simply returns the provided embeddings without additional
    computation.

    This allows the same training pipeline to work with both:
    - Live reference mapping (DualReferenceMapper)
    - Pre-computed embeddings (this class)

    Args:
        latent_dim: Dimensionality of embeddings
    """

    def __init__(self, latent_dim: int = 40):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(
        self,
        z_fused: torch.Tensor | None = None,
        z_hlca: torch.Tensor | None = None,
        z_luca: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Pass through pre-computed embeddings.

        Args:
            z_fused: Pre-computed fused embedding (batch_size, latent_dim)
            z_hlca: Pre-computed HLCA embedding (batch_size, latent_dim)
            z_luca: Pre-computed LuCA embedding (batch_size, latent_dim)
            return_intermediates: Whether to return all three embeddings

        Returns:
            z_fused or (z_fused, z_hlca, z_luca)
        """
        if z_fused is None:
            raise ValueError("z_fused must be provided for PrecomputedDualReference")

        if return_intermediates:
            if z_hlca is None or z_luca is None:
                raise ValueError("z_hlca and z_luca required for return_intermediates")
            return z_fused, z_hlca, z_luca
        else:
            return z_fused


def create_dual_reference_mapper(
    mode: str = "precomputed",
    latent_dim: int = 40,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create appropriate dual-reference mapper.

    Args:
        mode: 'precomputed' or 'learned'
        latent_dim: Latent space dimensionality
        **kwargs: Additional args for DualReferenceMapper

    Returns:
        Mapper module
    """
    if mode == "precomputed":
        return PrecomputedDualReference(latent_dim=latent_dim)
    elif mode == "learned":
        return DualReferenceMapper(latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


class DualReferenceAligner(nn.Module):
    """
    Align HLCA and LuCA references in shared space.

    Optional component for V1 that learns optimal alignment between
    the two reference atlases before fusion. Can improve transition
    structure by ensuring geometric consistency.

    Uses Procrustes-style alignment with learnable rotation/scaling.

    Args:
        latent_dim: Embedding dimensionality
        align_mode: 'procrustes', 'affine', or 'none'
    """

    def __init__(
        self,
        latent_dim: int = 40,
        align_mode: str = "affine",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.align_mode = align_mode

        if align_mode == "procrustes":
            # Learnable rotation matrix (orthogonal)
            self.rotation = nn.Parameter(torch.eye(latent_dim))

        elif align_mode == "affine":
            # Learnable affine transformation
            self.affine = nn.Linear(latent_dim, latent_dim, bias=True)

        elif align_mode == "none":
            pass  # No alignment

        else:
            raise ValueError(f"Unknown align_mode: {align_mode}")

    def forward(
        self,
        z_hlca: torch.Tensor,
        z_luca: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align HLCA and LuCA embeddings.

        Args:
            z_hlca: HLCA embeddings (batch_size, latent_dim)
            z_luca: LuCA embeddings (batch_size, latent_dim)

        Returns:
            z_hlca_aligned: Aligned HLCA embeddings
            z_luca: LuCA embeddings (unchanged, serves as anchor)
        """
        if self.align_mode == "none":
            return z_hlca, z_luca

        elif self.align_mode == "procrustes":
            # Apply orthogonal rotation to HLCA
            # (LuCA is anchor space)
            R = self._orthogonalize(self.rotation)
            z_hlca_aligned = z_hlca @ R

        elif self.align_mode == "affine":
            # Apply affine transformation to HLCA
            z_hlca_aligned = self.affine(z_hlca)

        return z_hlca_aligned, z_luca

    def _orthogonalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Orthogonalize matrix using SVD projection."""
        U, _, Vt = torch.linalg.svd(matrix, full_matrices=False)
        return U @ Vt


if __name__ == "__main__":
    # Test dual-reference mapper
    print("Testing DualReferenceMapper...")

    batch_size = 16
    input_dim = 2000
    latent_dim = 40

    # Test learned mapper
    mapper = DualReferenceMapper(
        input_dim=input_dim,
        latent_dim=latent_dim,
        fusion_mode="attention",
    )

    x = torch.randn(batch_size, input_dim)
    z_fused, z_hlca, z_luca = mapper(x, return_intermediates=True)

    print(f"Input shape: {x.shape}")
    print(f"z_fused shape: {z_fused.shape}")
    print(f"z_hlca shape: {z_hlca.shape}")
    print(f"z_luca shape: {z_luca.shape}")

    # Test attention weights
    weights = mapper.get_attention_weights(x)
    print(f"Attention weights shape: {weights.shape}")
    print(f"Sample weights: {weights[0]}")

    # Test precomputed mode
    print("\nTesting PrecomputedDualReference...")
    precomputed = PrecomputedDualReference(latent_dim=latent_dim)

    z_fused_in = torch.randn(batch_size, latent_dim)
    z_hlca_in = torch.randn(batch_size, latent_dim)
    z_luca_in = torch.randn(batch_size, latent_dim)

    z_out = precomputed(
        z_fused=z_fused_in,
        z_hlca=z_hlca_in,
        z_luca=z_luca_in,
        return_intermediates=False,
    )

    print(f"Output shape: {z_out.shape}")
    assert torch.allclose(z_out, z_fused_in), "Passthrough failed"

    # Test aligner
    print("\nTesting DualReferenceAligner...")
    aligner = DualReferenceAligner(latent_dim=latent_dim, align_mode="affine")

    z_hlca_aligned, z_luca_out = aligner(z_hlca_in, z_luca_in)
    print(f"Aligned HLCA shape: {z_hlca_aligned.shape}")

    print("\n All tests passed!")
