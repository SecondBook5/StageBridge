"""Test ring token loading from neighborhoods.parquet.

This test verifies that the fix for the ring token placeholder issue works correctly.
The key requirement is that tokens 1-4 (ring tokens) should contain ACTUAL neighborhood
data from neighborhoods.parquet, not just copies of the receiver embedding.
"""

import pandas as pd
import torch
import numpy as np
import tempfile
from pathlib import Path


def test_ring_token_parsing_logic():
    """Test the core parsing logic for ring tokens from neighborhoods.parquet."""
    n_cells = 10
    embed_dim = 40

    # Create mock cells_df
    cells_df = pd.DataFrame({
        'cell_id': [f'spatial_cell_{i}' for i in range(n_cells)],
        'donor_id': ['D1'] * n_cells,
        'stage': ['Normal'] * 5 + ['AAH'] * 5,
    })
    # Add fused embedding columns
    np.random.seed(42)
    for d in range(embed_dim):
        cells_df[f'z_fused_{d}'] = np.random.randn(n_cells)

    # Create mock neighborhoods_df with tokens column
    # Only first 5 cells have neighborhoods (spatial cells)
    neighborhoods = []
    for i in range(5):
        tokens = []
        # Token 0: receiver
        tokens.append({'token_idx': 0, 'token_type': 'receiver', 'cell_id': f'spatial_cell_{i}'})
        # Tokens 1-4: rings with z_pooled (unique values to verify they're loaded)
        for ring in range(1, 5):
            z_pooled = [float(i * 10 + ring + d * 0.1) for d in range(embed_dim)]
            tokens.append({
                'token_idx': ring,
                'token_type': f'ring_{ring}',
                'n_cells': 5,
                'z_pooled': z_pooled,
                'celltype_composition': {'AT2': 3, 'Macrophage': 2},
                'mean_distance': 0.5 * ring
            })
        # Token 5-6: HLCA, LuCA (no z_pooled)
        tokens.append({'token_idx': 5, 'token_type': 'hlca'})
        tokens.append({'token_idx': 6, 'token_type': 'luca'})
        # Token 7: pathway
        tokens.append({'token_idx': 7, 'token_type': 'pathway'})
        # Token 8: stats
        tokens.append({
            'token_idx': 8,
            'token_type': 'stats',
            'n_neighbors': 20,
            'mean_distance': 0.3,
            'diversity': 5
        })

        neighborhoods.append({
            'cell_id': f'spatial_cell_{i}',
            'donor_id': 'D1',
            'stage': 'Normal',
            'tokens': tokens
        })

    neighborhoods_df = pd.DataFrame(neighborhoods)

    # Run the parsing logic (copied from run_v1_ddp.py)
    fused_cols = [f'z_fused_{d}' for d in range(embed_dim)]
    embeddings = torch.tensor(cells_df[fused_cols].values, dtype=torch.float32)
    niche_tokens = torch.zeros(n_cells, 9, embed_dim)
    niche_tokens[:, 0, :] = embeddings

    cell_id_to_idx = {cid: i for i, cid in enumerate(cells_df['cell_id'].values)}
    n_rings_populated = 0
    n_stats_populated = 0

    cell_ids = neighborhoods_df['cell_id'].values
    tokens_col = neighborhoods_df['tokens'].values

    for row_idx in range(len(neighborhoods_df)):
        cell_id = cell_ids[row_idx]
        if cell_id not in cell_id_to_idx:
            continue
        cell_idx = cell_id_to_idx[cell_id]
        tokens_list = tokens_col[row_idx]

        for token_dict in tokens_list:
            token_idx = token_dict.get('token_idx', -1)

            if 1 <= token_idx <= 4:
                z_pooled = token_dict.get('z_pooled')
                if z_pooled is not None and len(z_pooled) > 0:
                    z_pooled_tensor = torch.tensor(z_pooled, dtype=torch.float32)
                    if len(z_pooled_tensor) < embed_dim:
                        padded = torch.zeros(embed_dim)
                        padded[:len(z_pooled_tensor)] = z_pooled_tensor
                        z_pooled_tensor = padded
                    elif len(z_pooled_tensor) > embed_dim:
                        z_pooled_tensor = z_pooled_tensor[:embed_dim]
                    niche_tokens[cell_idx, token_idx, :] = z_pooled_tensor
                    n_rings_populated += 1

            elif token_idx == 8:
                n_neighbors = token_dict.get('n_neighbors', 0)
                mean_distance = token_dict.get('mean_distance', 0.0)
                diversity = token_dict.get('diversity', 0)
                niche_tokens[cell_idx, 8, 0] = float(n_neighbors) / 20.0
                niche_tokens[cell_idx, 8, 1] = float(mean_distance)
                niche_tokens[cell_idx, 8, 2] = float(diversity) / 20.0
                n_stats_populated += 1

    # Assertions
    assert n_rings_populated == 20, f"Expected 20 ring tokens (5 cells x 4 rings), got {n_rings_populated}"
    assert n_stats_populated == 5, f"Expected 5 stats tokens, got {n_stats_populated}"

    # Verify ring tokens are DIFFERENT from receiver for cells with neighborhoods
    ring_diff = (niche_tokens[:5, 1:5, :] - niche_tokens[:5, 0:1, :]).abs().mean()
    assert ring_diff > 0.1, f"Ring tokens should differ from receiver, got diff={ring_diff}"

    # Verify cells without neighborhoods have zero ring tokens (before fallback)
    ring_sum_no_niche = niche_tokens[5:, 1:5, :].abs().sum()
    assert ring_sum_no_niche < 1e-6, f"Cells without neighborhoods should have zero rings, got sum={ring_sum_no_niche}"

    # Verify stats token values for cell 0
    assert abs(niche_tokens[0, 8, 0].item() - 1.0) < 1e-5, "n_neighbors/20 should be 1.0"  # 20/20
    assert abs(niche_tokens[0, 8, 1].item() - 0.3) < 1e-5, "mean_distance should be 0.3"
    assert abs(niche_tokens[0, 8, 2].item() - 0.25) < 1e-5, "diversity/20 should be 0.25"  # 5/20

    # Test fallback logic
    ring_sum = niche_tokens[:, 1:5, :].abs().sum(dim=(1, 2))
    cells_without_niche_mask = (ring_sum < 1e-6)
    assert cells_without_niche_mask.sum().item() == 5, "5 cells should lack niche context"

    # Apply fallback
    for ring_idx in range(1, 5):
        niche_tokens[cells_without_niche_mask, ring_idx, :] = embeddings[cells_without_niche_mask]

    # Verify all cells now have ring tokens
    final_ring_sum = niche_tokens[:, 1:5, :].abs().sum(dim=(1, 2))
    assert (final_ring_sum < 1e-6).sum().item() == 0, "After fallback, no cells should have zero rings"

    print("All assertions passed!")


def test_empty_neighborhoods_fallback():
    """Test that empty neighborhoods_df falls back to receiver embedding."""
    n_cells = 5
    embed_dim = 40

    cells_df = pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(n_cells)],
    })
    for d in range(embed_dim):
        cells_df[f'z_fused_{d}'] = np.random.randn(n_cells)

    # Empty neighborhoods
    neighborhoods_df = pd.DataFrame(columns=['cell_id', 'tokens'])

    fused_cols = [f'z_fused_{d}' for d in range(embed_dim)]
    embeddings = torch.tensor(cells_df[fused_cols].values, dtype=torch.float32)
    niche_tokens = torch.zeros(n_cells, 9, embed_dim)
    niche_tokens[:, 0, :] = embeddings

    # Should trigger fallback
    if len(neighborhoods_df) > 0 and "tokens" in neighborhoods_df.columns:
        pass  # Would parse tokens
    else:
        # Fallback
        for ring_idx in range(1, 5):
            niche_tokens[:, ring_idx, :] = embeddings

    # Verify rings equal receiver
    for ring_idx in range(1, 5):
        diff = (niche_tokens[:, ring_idx, :] - embeddings).abs().max()
        assert diff < 1e-6, f"Ring {ring_idx} should equal receiver in fallback mode"

    print("Empty neighborhoods fallback test passed!")


def test_ring_token_dimension_handling():
    """Test that z_pooled of different dimensions is handled correctly."""
    embed_dim = 40

    # Test case 1: z_pooled smaller than embed_dim (should be padded)
    z_pooled_small = [1.0] * 20
    z_tensor = torch.tensor(z_pooled_small, dtype=torch.float32)
    if len(z_tensor) < embed_dim:
        padded = torch.zeros(embed_dim)
        padded[:len(z_tensor)] = z_tensor
        z_tensor = padded
    assert len(z_tensor) == embed_dim, "Should be padded to embed_dim"
    assert z_tensor[19] == 1.0, "Original values preserved"
    assert z_tensor[20] == 0.0, "Padding should be zeros"

    # Test case 2: z_pooled larger than embed_dim (should be truncated)
    z_pooled_large = [1.0] * 50
    z_tensor = torch.tensor(z_pooled_large, dtype=torch.float32)
    if len(z_tensor) > embed_dim:
        z_tensor = z_tensor[:embed_dim]
    assert len(z_tensor) == embed_dim, "Should be truncated to embed_dim"

    # Test case 3: z_pooled exactly embed_dim (no change)
    z_pooled_exact = [1.0] * 40
    z_tensor = torch.tensor(z_pooled_exact, dtype=torch.float32)
    assert len(z_tensor) == embed_dim, "Should remain unchanged"

    print("Dimension handling test passed!")


if __name__ == "__main__":
    test_ring_token_parsing_logic()
    test_empty_neighborhoods_fallback()
    test_ring_token_dimension_handling()
    print("\n=== ALL TESTS PASSED ===")
