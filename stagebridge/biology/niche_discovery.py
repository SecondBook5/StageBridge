"""NMF-based niche archetype discovery from deconvolution data.

Discovers interpretable niche archetypes from DestVI/cell2location gammas
(cell type proportion vectors) using Non-negative Matrix Factorization.

Key outputs:
- Niche archetypes defined by cell type loadings
- Soft archetype assignments per spot (W matrix)
- Stage enrichment analysis
- Archetype characterization (dominant cell types, stage associations)

Why NMF over clustering:
- Compositional data (proportions sum to 1) - NMF respects non-negativity
- Soft assignments - spots can be mixtures of archetypes
- Interpretable factors - each archetype is a cell type mixture
- No spatial smoothing - niches ARE heterogeneous (different types together)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


@dataclass
class NicheArchetype:
    """A discovered niche archetype."""

    index: int
    name: str
    top_cell_types: list[tuple[str, float]]  # (cell_type, weight) pairs
    stage_enrichment: dict[str, float]  # stage -> % of spots with this dominant archetype
    n_spots_dominant: int  # spots where this is the dominant archetype

    def __str__(self) -> str:
        top_3 = ", ".join([f"{ct}({w:.2f})" for ct, w in self.top_cell_types[:3]])
        return f"Archetype {self.index} ({self.name}): {top_3}"


@dataclass
class NicheDiscoveryResult:
    """Results from NMF niche discovery."""

    n_archetypes: int
    archetypes: list[NicheArchetype]
    W: np.ndarray  # spots x archetypes (soft assignments)
    H: np.ndarray  # archetypes x cell_types (loadings)
    cell_type_names: list[str]
    reconstruction_error: float
    spot_assignments: pd.DataFrame  # spot_id, dominant_archetype, weights
    stage_distribution: pd.DataFrame  # stage x archetype crosstab

    def get_archetype_by_name(self, name: str) -> NicheArchetype | None:
        """Get archetype by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for arch in self.archetypes:
            if name_lower in arch.name.lower():
                return arch
        return None

    def summarize(self) -> str:
        """Return summary string."""
        lines = [
            f"NMF Niche Discovery: {self.n_archetypes} archetypes",
            f"Reconstruction error: {self.reconstruction_error:.2f}",
            "",
            "Archetypes:"
        ]
        for arch in self.archetypes:
            lines.append(f"  {arch}")
        return "\n".join(lines)


def discover_niches(
    proportions: pd.DataFrame,
    n_archetypes: int = 8,
    stage_col: str | None = "stage",
    random_state: int = 42,
    max_iter: int = 200,
    auto_name: bool = True,
) -> NicheDiscoveryResult:
    """Discover niche archetypes via NMF on cell type proportions.

    Args:
        proportions: DataFrame with cell type proportions per spot.
                    Index = spot IDs, columns = cell types (+ optional 'stage', 'sample')
        n_archetypes: Number of archetypes to discover
        stage_col: Column name for stage labels (None to skip stage analysis)
        random_state: Random seed
        max_iter: Maximum NMF iterations
        auto_name: Automatically name archetypes from top cell type

    Returns:
        NicheDiscoveryResult with archetypes, assignments, and stage distribution
    """
    # Identify cell type columns (exclude metadata)
    meta_cols = {'stage', 'sample', 'sample_id', 'spot_id', 'cell_id', 'patient', 'donor'}
    cell_type_cols = [c for c in proportions.columns if c.lower() not in meta_cols]

    X = proportions[cell_type_cols].values

    # Verify it's compositional (rows sum to ~1)
    row_sums = X.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.01):
        print(f"Warning: Row sums not ~1 (mean={row_sums.mean():.3f}). Normalizing.")
        X = X / row_sums[:, np.newaxis]

    # Run NMF
    nmf = NMF(
        n_components=n_archetypes,
        init='nndsvda',
        max_iter=max_iter,
        random_state=random_state,
    )
    W = nmf.fit_transform(X)  # spots x archetypes
    H = nmf.components_        # archetypes x cell_types

    # Build archetype objects
    archetypes = []
    for i in range(n_archetypes):
        # Top cell types for this archetype
        top_idx = np.argsort(H[i])[::-1][:5]
        top_types = [(cell_type_cols[j], float(H[i, j])) for j in top_idx]

        # Auto-name from top cell type
        if auto_name:
            name = _clean_cell_type_name(top_types[0][0])
        else:
            name = f"Archetype_{i}"

        archetypes.append(NicheArchetype(
            index=i,
            name=name,
            top_cell_types=top_types,
            stage_enrichment={},  # filled below
            n_spots_dominant=0,   # filled below
        ))

    # Dominant archetype per spot
    dominant = W.argmax(axis=1)

    # Count dominant assignments
    for i, arch in enumerate(archetypes):
        arch.n_spots_dominant = int((dominant == i).sum())

    # Stage distribution
    stage_distribution = None
    if stage_col and stage_col in proportions.columns:
        stages = proportions[stage_col].values
        stage_df = pd.DataFrame({
            'stage': stages,
            'archetype': dominant,
        })
        stage_distribution = pd.crosstab(
            stage_df['stage'],
            stage_df['archetype'],
            normalize='index'
        ) * 100

        # Rename columns to archetype names
        stage_distribution.columns = [archetypes[i].name for i in range(n_archetypes)]

        # Fill stage enrichment in archetypes
        for i, arch in enumerate(archetypes):
            arch.stage_enrichment = stage_distribution.iloc[:, i].to_dict()

    # Build spot assignments dataframe
    spot_assignments = pd.DataFrame({
        'spot_id': proportions.index,
        'dominant_archetype': dominant,
        'dominant_name': [archetypes[i].name for i in dominant],
    })
    for i, arch in enumerate(archetypes):
        spot_assignments[f'weight_{arch.name}'] = W[:, i]

    return NicheDiscoveryResult(
        n_archetypes=n_archetypes,
        archetypes=archetypes,
        W=W,
        H=H,
        cell_type_names=cell_type_cols,
        reconstruction_error=nmf.reconstruction_err_,
        spot_assignments=spot_assignments,
        stage_distribution=stage_distribution,
    )


def _clean_cell_type_name(name: str) -> str:
    """Clean cell type name for use as archetype name."""
    # Take first word, capitalize
    parts = name.replace('-', ' ').replace('_', ' ').split()
    if len(parts) >= 2:
        # Handle "pulmonary alveolar type 2 cell" -> "AT2"
        if 'alveolar' in name.lower() and 'type' in name.lower():
            if '1' in name:
                return 'AT1'
            if '2' in name:
                return 'AT2'
        # Handle "malignant cell" -> "Malignant"
        return parts[0].capitalize()
    return name[:12].capitalize()


def select_n_archetypes(
    proportions: pd.DataFrame,
    k_range: Sequence[int] = (4, 6, 8, 10, 12),
    random_state: int = 42,
) -> dict[int, float]:
    """Evaluate different numbers of archetypes via reconstruction error.

    Args:
        proportions: DataFrame with cell type proportions
        k_range: Numbers of archetypes to try
        random_state: Random seed

    Returns:
        Dict of k -> reconstruction_error
    """
    meta_cols = {'stage', 'sample', 'sample_id', 'spot_id', 'cell_id', 'patient', 'donor'}
    cell_type_cols = [c for c in proportions.columns if c.lower() not in meta_cols]
    X = proportions[cell_type_cols].values

    errors = {}
    for k in k_range:
        nmf = NMF(n_components=k, init='nndsvda', max_iter=200, random_state=random_state)
        nmf.fit(X)
        errors[k] = nmf.reconstruction_err_
        print(f"k={k}: error={nmf.reconstruction_err_:.2f}")

    return errors


def plot_niche_discovery(
    result: NicheDiscoveryResult,
    output_path: str | Path,
    stage_order: list[str] | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Generate niche discovery figure.

    Args:
        result: NicheDiscoveryResult from discover_niches
        output_path: Path to save figure (without extension)
        stage_order: Order of stages for plotting
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)

    if stage_order is None:
        stage_order = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel A: Archetype composition heatmap
    ax = axes[0, 0]
    # Show top cell types
    top_per_arch = 4
    top_types_idx = set()
    for i in range(result.n_archetypes):
        top_types_idx.update(np.argsort(result.H[i])[-top_per_arch:])
    top_types_idx = sorted(top_types_idx)

    H_show = result.H[:, top_types_idx]
    type_labels = [result.cell_type_names[i][:20] for i in top_types_idx]
    arch_names = [a.name for a in result.archetypes]

    sns.heatmap(H_show, ax=ax, cmap='YlOrRd',
                xticklabels=type_labels, yticklabels=arch_names,
                cbar_kws={'label': 'NMF weight'})
    ax.set_title(f'A. Niche Archetypes (k={result.n_archetypes})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Archetype')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # Panel B: Stage distribution stacked bar
    ax = axes[0, 1]
    if result.stage_distribution is not None:
        stage_dist = result.stage_distribution.copy()
        # Reorder stages if possible
        available_stages = [s for s in stage_order if s in stage_dist.index]
        if available_stages:
            stage_dist = stage_dist.reindex(available_stages)

        stage_dist.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', width=0.8)
        ax.set_title('B. Niche Distribution by Stage', fontsize=12, fontweight='bold')
        ax.set_xlabel('Disease Stage')
        ax.set_ylabel('% of Spots')
        ax.legend(title='Niche', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    else:
        ax.text(0.5, 0.5, 'No stage data', ha='center', va='center')
        ax.set_title('B. Niche Distribution by Stage', fontsize=12, fontweight='bold')

    # Panel C: Archetype sizes
    ax = axes[1, 0]
    sizes = [a.n_spots_dominant for a in result.archetypes]
    names = [a.name for a in result.archetypes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    ax.barh(names, sizes, color=colors)
    ax.set_xlabel('Number of Spots (dominant)')
    ax.set_title('C. Archetype Sizes', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # Panel D: Progression trends for key archetypes
    ax = axes[1, 1]
    if result.stage_distribution is not None:
        stage_dist = result.stage_distribution.copy()
        available_stages = [s for s in stage_order if s in stage_dist.index]
        if available_stages:
            stage_dist = stage_dist.reindex(available_stages)

        # Find interesting archetypes (high variance across stages)
        variance = stage_dist.var()
        top_archs = variance.nlargest(4).index.tolist()

        colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
        for arch, color in zip(top_archs, colors):
            vals = stage_dist[arch].values
            ax.plot(range(len(available_stages)), vals, 'o-',
                   label=arch, color=color, linewidth=2, markersize=8)

        ax.set_xticks(range(len(available_stages)))
        ax.set_xticklabels(available_stages)
        ax.set_xlabel('Disease Stage')
        ax.set_ylabel('% of Spots')
        ax.set_title('D. Niche Remodeling During Progression', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No stage data', ha='center', va='center')
        ax.set_title('D. Progression Trends', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}.png and .pdf")


def run_niche_discovery(
    proportions_path: str | Path,
    output_dir: str | Path,
    n_archetypes: int = 8,
    reference: str = "luca",
) -> NicheDiscoveryResult:
    """Run niche discovery pipeline.

    Args:
        proportions_path: Path to cell_type_proportions.parquet
        output_dir: Output directory
        n_archetypes: Number of archetypes
        reference: Reference atlas name (for labeling)

    Returns:
        NicheDiscoveryResult
    """
    proportions_path = Path(proportions_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {proportions_path}...")
    df = pd.read_parquet(proportions_path)
    print(f"  {len(df):,} spots")

    # Extract stage from sample name if not present
    if 'stage' not in df.columns and 'sample' in df.columns:
        df['stage'] = df['sample'].str.extract(r'_([A-Za-z]+)$')[0]

    print(f"Running NMF with k={n_archetypes}...")
    result = discover_niches(df, n_archetypes=n_archetypes)
    print(result.summarize())

    # Save outputs
    result.spot_assignments.to_parquet(output_dir / f'{reference}_niche_assignments.parquet')
    print(f"  Saved {reference}_niche_assignments.parquet")

    if result.stage_distribution is not None:
        result.stage_distribution.to_parquet(output_dir / f'{reference}_niche_by_stage.parquet')
        print(f"  Saved {reference}_niche_by_stage.parquet")

    # Save archetype definitions
    arch_defs = []
    for arch in result.archetypes:
        arch_defs.append({
            'index': arch.index,
            'name': arch.name,
            'n_spots_dominant': arch.n_spots_dominant,
            'top_cell_types': arch.top_cell_types,
            'stage_enrichment': arch.stage_enrichment,
        })
    pd.DataFrame(arch_defs).to_parquet(output_dir / f'{reference}_archetype_definitions.parquet')

    # Generate figure
    plot_niche_discovery(result, output_dir / f'{reference}_niche_archetypes')

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover niche archetypes via NMF")
    parser.add_argument("--proportions", required=True, help="Path to cell_type_proportions.parquet")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--k", type=int, default=8, help="Number of archetypes")
    parser.add_argument("--reference", default="luca", help="Reference name (luca/hlca)")
    args = parser.parse_args()

    run_niche_discovery(args.proportions, args.output, args.k, args.reference)
