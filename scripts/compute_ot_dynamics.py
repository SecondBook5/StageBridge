#!/usr/bin/env python3
"""
Compute OT dynamics (flux, curl, divergence) from model velocity predictions.

Takes displacement vectors from inference and computes:
- Flux: magnitude of velocity field
- Divergence: sources/sinks (∂vx/∂x + ∂vy/∂y)
- Curl: rotation/vorticity (∂vy/∂x - ∂vx/∂y)

Usage:
    python scripts/compute_ot_dynamics.py \
        --data-dir /data1/chaunzt1/stagebridge/processed/luad_evo/canonical \
        --model-dir /data1/chaunzt1/stagebridge/outputs/v1.1 \
        --output-dir /data1/chaunzt1/stagebridge/processed/luad_evo/canonical/progression
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import sobel
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def load_umap_and_velocities(
    data_dir: Path,
    model_dir: Path
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load UMAP coordinates and velocity vectors from inference outputs."""

    # Load UMAP embedding
    umap_path = data_dir / 'embeddings' / 'umap_embedding.parquet'
    if not umap_path.exists():
        raise FileNotFoundError(f"UMAP not found: {umap_path}")

    umap_df = pd.read_parquet(umap_path)
    if 'UMAP1' in umap_df.columns:
        umap = umap_df[['UMAP1', 'UMAP2']].values
    elif 'umap_1' in umap_df.columns:
        umap = umap_df[['umap_1', 'umap_2']].values
    else:
        umap = umap_df.iloc[:, :2].values

    print(f"Loaded UMAP: {umap.shape}")

    # Load cells for metadata
    cells_path = data_dir / 'cells.parquet'
    cells_df = pd.read_parquet(cells_path)
    print(f"Loaded cells: {len(cells_df)}")

    # Collect displacements from all inference runs
    inf_dir = model_dir / 'inference' / 'full'
    all_displacements = []
    all_cell_ids = []

    if inf_dir.exists():
        for fold_dir in sorted(inf_dir.glob('fold_*')):
            for seed_dir in sorted(fold_dir.glob('seed_*')):
                disp_path = seed_dir / 'displacements.npy'
                pred_path = seed_dir / 'predictions.parquet'

                if disp_path.exists() and pred_path.exists():
                    disp = np.load(disp_path)
                    pred_df = pd.read_parquet(pred_path)

                    print(f"  {fold_dir.name}/{seed_dir.name}: {disp.shape}")
                    all_displacements.append(disp)
                    all_cell_ids.append(pred_df['cell_id'].values)

    if not all_displacements:
        raise FileNotFoundError(
            f"No displacement files found in {inf_dir}. "
            "Run inference with updated infer.py first."
        )

    # Average across runs for cells that appear multiple times
    # (each fold has different test cells, so we aggregate)
    cell_velocities = {}
    for disp, cell_ids in zip(all_displacements, all_cell_ids):
        for i, cid in enumerate(cell_ids):
            if cid not in cell_velocities:
                cell_velocities[cid] = []
            cell_velocities[cid].append(disp[i])

    # Average velocities per cell
    avg_velocities = {cid: np.mean(vels, axis=0) for cid, vels in cell_velocities.items()}
    print(f"Aggregated velocities for {len(avg_velocities)} cells")

    return umap, avg_velocities, cells_df


def project_velocities_to_umap(
    velocities: dict,
    cells_df: pd.DataFrame,
    umap: np.ndarray,
    n_neighbors: int = 30
) -> np.ndarray:
    """
    Project high-dimensional velocities to UMAP space.

    Uses local PCA in UMAP neighborhood to project velocity vectors.
    This is similar to scVelo's velocity projection.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA

    n_cells = len(cells_df)
    velocity_2d = np.zeros((n_cells, 2))
    has_velocity = np.zeros(n_cells, dtype=bool)

    # Get cell IDs from cells_df
    if 'cell_id' in cells_df.columns:
        cell_ids = cells_df['cell_id'].values
    else:
        cell_ids = cells_df.index.values

    # Build neighbor graph in UMAP space
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(umap)

    # For each cell with velocity, project to local tangent space
    for i, cid in enumerate(cell_ids):
        if cid in velocities:
            vel = velocities[cid]
            has_velocity[i] = True

            # Get neighbors in UMAP space
            _, neighbor_idx = nn.kneighbors([umap[i]])
            neighbor_idx = neighbor_idx[0]

            # Local displacement in UMAP = difference to neighbors
            umap_local = umap[neighbor_idx] - umap[i]

            # Project velocity using local geometry
            # Simple approach: use mean direction weighted by velocity magnitude
            if np.linalg.norm(vel) > 0:
                # Normalize and use magnitude
                vel_mag = np.linalg.norm(vel)

                # Use PCA on local UMAP neighborhood to get principal directions
                if len(umap_local) > 2:
                    pca = PCA(n_components=2)
                    pca.fit(umap_local)

                    # Project velocity onto PCA components (using first 2 dims of vel)
                    vel_proj = vel[:2] if len(vel) >= 2 else np.concatenate([vel, [0]])
                    velocity_2d[i] = vel_proj[:2] * vel_mag / (np.linalg.norm(vel_proj[:2]) + 1e-8)

    print(f"Projected velocities for {has_velocity.sum()} cells")
    return velocity_2d, has_velocity


def compute_vector_field_on_grid(
    umap: np.ndarray,
    velocities: np.ndarray,
    mask: np.ndarray,
    grid_size: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate velocity field onto regular grid."""

    # Only use cells with velocities
    umap_sub = umap[mask]
    vel_sub = velocities[mask]

    # Create grid
    x_min, x_max = umap[:, 0].min(), umap[:, 0].max()
    y_min, y_max = umap[:, 1].min(), umap[:, 1].max()

    # Add padding
    pad = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= pad * x_range
    x_max += pad * x_range
    y_min -= pad * y_range
    y_max += pad * y_range

    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate velocity components
    vx_grid = griddata(umap_sub, vel_sub[:, 0], (xi_grid, yi_grid), method='linear')
    vy_grid = griddata(umap_sub, vel_sub[:, 1], (xi_grid, yi_grid), method='linear')

    # Fill NaN with nearest
    vx_nearest = griddata(umap_sub, vel_sub[:, 0], (xi_grid, yi_grid), method='nearest')
    vy_nearest = griddata(umap_sub, vel_sub[:, 1], (xi_grid, yi_grid), method='nearest')

    vx_grid = np.where(np.isnan(vx_grid), vx_nearest, vx_grid)
    vy_grid = np.where(np.isnan(vy_grid), vy_nearest, vy_grid)

    return xi_grid, yi_grid, vx_grid, vy_grid


def compute_divergence_curl(
    vx_grid: np.ndarray,
    vy_grid: np.ndarray,
    dx: float,
    dy: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute divergence and curl of vector field."""

    # Use Sobel filters for smooth derivatives
    dvx_dx = sobel(vx_grid, axis=1) / (8 * dx)
    dvy_dy = sobel(vy_grid, axis=0) / (8 * dy)
    dvx_dy = sobel(vx_grid, axis=0) / (8 * dy)
    dvy_dx = sobel(vy_grid, axis=1) / (8 * dx)

    # Divergence = ∂vx/∂x + ∂vy/∂y
    divergence = dvx_dx + dvy_dy

    # Curl (z-component in 2D) = ∂vy/∂x - ∂vx/∂y
    curl = dvy_dx - dvx_dy

    return divergence, curl


def interpolate_to_cells(
    xi_grid: np.ndarray,
    yi_grid: np.ndarray,
    field: np.ndarray,
    umap: np.ndarray
) -> np.ndarray:
    """Interpolate gridded field back to cell positions."""

    from scipy.interpolate import RegularGridInterpolator

    xi = xi_grid[0, :]
    yi = yi_grid[:, 0]

    interp = RegularGridInterpolator(
        (yi, xi), field,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    # Interpolate at cell positions
    cell_values = interp(umap[:, ::-1])  # Note: y, x order for RegularGridInterpolator

    return cell_values


def main():
    parser = argparse.ArgumentParser(description="Compute OT dynamics from model velocities")
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--grid-size', type=int, default=50)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir / 'progression'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("COMPUTING OT DYNAMICS")
    print("="*60)

    # Load data
    try:
        umap, velocities, cells_df = load_umap_and_velocities(args.data_dir, args.model_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nTo generate displacements, run inference with the updated infer.py:")
        print("  bash scripts/run_inference_all.sh")
        return

    # Project velocities to UMAP space
    print("\nProjecting velocities to UMAP space...")
    vel_2d, has_vel = project_velocities_to_umap(velocities, cells_df, umap)

    # Compute flux (magnitude)
    flux = np.linalg.norm(vel_2d, axis=1)

    # Interpolate to grid
    print(f"\nInterpolating to {args.grid_size}x{args.grid_size} grid...")
    xi, yi, vx, vy = compute_vector_field_on_grid(umap, vel_2d, has_vel, args.grid_size)

    # Grid spacing
    dx = xi[0, 1] - xi[0, 0]
    dy = yi[1, 0] - yi[0, 0]

    # Compute divergence and curl
    print("Computing divergence and curl...")
    div_grid, curl_grid = compute_divergence_curl(vx, vy, dx, dy)

    # Interpolate back to cells
    divergence = interpolate_to_cells(xi, yi, div_grid, umap)
    curl = interpolate_to_cells(xi, yi, curl_grid, umap)

    # Create output dataframe
    result_df = pd.DataFrame({
        'flux': flux,
        'divergence': divergence,
        'curl': curl,
        'velocity_x': vel_2d[:, 0],
        'velocity_y': vel_2d[:, 1],
        'has_velocity': has_vel,
    })

    # Add cell_id if available
    if 'cell_id' in cells_df.columns:
        result_df['cell_id'] = cells_df['cell_id'].values

    # Save
    output_path = args.output_dir / 'ot_dynamics.parquet'
    result_df.to_parquet(output_path)
    print(f"\nSaved: {output_path}")

    # Summary stats
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Cells with velocity: {has_vel.sum()} / {len(has_vel)}")
    print(f"Flux range: [{flux[has_vel].min():.4f}, {flux[has_vel].max():.4f}]")
    print(f"Divergence range: [{np.nanmin(divergence):.4f}, {np.nanmax(divergence):.4f}]")
    print(f"Curl range: [{np.nanmin(curl):.4f}, {np.nanmax(curl):.4f}]")

    # Also save grid data for visualization
    grid_data = {
        'xi': xi,
        'yi': yi,
        'vx': vx,
        'vy': vy,
        'divergence': div_grid,
        'curl': curl_grid,
    }
    np.savez(args.output_dir / 'ot_grid.npz', **grid_data)
    print(f"Saved grid data: {args.output_dir / 'ot_grid.npz'}")


if __name__ == "__main__":
    main()
