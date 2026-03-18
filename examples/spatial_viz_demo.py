"""
Demo: Advanced spatial mapping visualizations with Tangram and DestVI.

Shows how to use the scvi-tools integrated backends with publication-quality
visualizations including:
- Tangram: Gene projection, spatial statistics
- DestVI: Gamma space exploration, cell-type-specific expression
"""

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

from stagebridge.spatial_backends import (
    TangramBackend,
    DestVIBackend,
    create_comprehensive_report,
)


def demo_tangram_visualizations(
    snrna: ad.AnnData,
    spatial: ad.AnnData,
    output_dir: Path,
):
    """Demo Tangram with advanced visualizations."""
    print("\n" + "=" * 60)
    print("Tangram (scvi-tools) Demo")
    print("=" * 60)

    # Initialize backend
    backend = TangramBackend(
        constrained=True,
        n_epochs=100,  # Reduced for demo
        marker_genes="auto",
    )

    # Run mapping
    print("\n1. Running Tangram mapping...")
    result = backend.map(snrna, spatial, output_dir=output_dir / "tangram")

    print(f"\n   Cell type proportions shape: {result.cell_type_proportions.shape}")
    print(f"   Confidence range: [{result.confidence.min():.3f}, {result.confidence.max():.3f}]")

    # Generate comprehensive report
    print("\n2. Generating comprehensive visualization report...")
    spatial_annotated = ad.read_h5ad(output_dir / "tangram" / "tangram_spatial_annotated.h5ad")
    create_comprehensive_report(
        result,
        spatial_annotated,
        output_dir / "tangram" / "visualizations",
        n_genes_to_plot=6,
    )

    # Advanced Tangram features
    print("\n3. Advanced Tangram features:")

    # Project specific genes
    print("   - Projecting marker genes...")
    marker_genes = ["CD3D", "CD79A", "MS4A1", "CD14", "FCGR3A"]
    available_markers = [g for g in marker_genes if g in snrna.var_names]

    if available_markers:
        projected = backend.project_genes(available_markers, aggregate=False)
        print(f"     Projected expression shape: {projected.shape}")

        # Plot projected genes
        backend.plot_projected_genes(
            available_markers[:3],
            aggregate=False,
            save_path=output_dir / "tangram" / "visualizations" / "projected_genes.png",
        )

    # Cell type spatial plots
    print("   - Plotting cell type spatial distributions...")
    cell_types = result.cell_type_proportions.columns.tolist()
    for ct in cell_types[:3]:  # First 3 cell types
        backend.plot_cell_type_spatial(
            ct,
            save_path=output_dir / "tangram" / "visualizations" / f"spatial_{ct.replace(' ', '_')}.png",
        )

    # Spatial statistics (requires squidpy)
    try:
        print("   - Computing spatial statistics (Moran's I)...")
        spatial_stats = backend.compute_spatial_statistics(cell_types=cell_types[:5])
        print(f"     Moran's I computed for {len(spatial_stats['morans_i'])} cell types")
        for ct, mi in list(spatial_stats["morans_i"].items())[:3]:
            print(f"       {ct}: {mi:.3f}")
    except ImportError:
        print("     Skipping (squidpy not installed)")

    print("\n✓ Tangram demo complete!")


def demo_destvi_visualizations(
    snrna: ad.AnnData,
    spatial: ad.AnnData,
    output_dir: Path,
):
    """Demo DestVI with multi-resolution analysis."""
    print("\n" + "=" * 60)
    print("DestVI (scvi-tools) Demo")
    print("=" * 60)

    # Initialize backend
    backend = DestVIBackend(
        n_latent=10,
        n_epochs_condsc=50,  # Reduced for demo
        n_epochs_destvi=500,  # Reduced for demo
        vamp_prior_p=50,
    )

    # Run mapping
    print("\n1. Running DestVI mapping...")
    result = backend.map(snrna, spatial, output_dir=output_dir / "destvi")

    print(f"\n   Cell type proportions shape: {result.cell_type_proportions.shape}")
    print(f"   Confidence range: [{result.confidence.min():.3f}, {result.confidence.max():.3f}]")

    # Generate comprehensive report
    print("\n2. Generating comprehensive visualization report...")
    spatial_annotated = ad.read_h5ad(output_dir / "destvi" / "destvi_spatial_annotated.h5ad")
    create_comprehensive_report(
        result,
        spatial_annotated,
        output_dir / "destvi" / "visualizations",
        n_genes_to_plot=6,
    )

    # Advanced DestVI features
    print("\n3. Advanced DestVI features (multi-resolution):")

    # Gamma space (intra-cell-type variation)
    print("   - Exploring gamma latent space...")
    cell_types = result.cell_type_proportions.columns.tolist()
    gamma_dict = backend.get_gamma(cell_types=cell_types[:3])

    for ct, gamma_df in gamma_dict.items():
        print(f"     {ct}: gamma shape = {gamma_df.shape}")

    # Automatic thresholding
    print("   - Computing automatic proportion thresholds...")
    thresholds = backend.automatic_proportion_threshold(cell_types=cell_types[:3])
    for ct, thresh in thresholds.items():
        print(f"     {ct}: threshold = {thresh:.3f}")

    # Cell-type-specific gene imputation
    print("   - Imputing cell-type-specific gene expression...")
    for ct in cell_types[:2]:
        indices = backend.filter_spots_by_celltype(ct, auto_threshold=True)
        if len(indices) > 10:
            print(f"     {ct}: {len(indices)} spots above threshold")

            # Impute marker genes for this cell type
            marker_genes = ["CD3D", "CD79A", "CD14"]
            available = [g for g in marker_genes if g in snrna.var_names]

            if available:
                ct_expr = backend.get_cell_type_specific_expression(
                    ct,
                    gene_names=available,
                    indices=indices,
                    aggregate=True,
                )
                print(f"       Imputed expression shape: {ct_expr.shape}")

                # Plot cell-type-specific expression
                backend.plot_cell_type_spatial(
                    ct,
                    gene_names=available,
                    save_path=output_dir / "destvi" / "visualizations" / f"ct_specific_{ct.replace(' ', '_')}.png",
                )

    # Gamma space exploration with spatially-weighted PCA
    print("   - Gamma space exploration (spatially-weighted PCA)...")
    for ct in cell_types[:2]:
        try:
            gamma_result = backend.explore_gamma_space(
                ct,
                save_dir=output_dir / "destvi" / "visualizations" / "gamma_pca",
            )
            print(f"     {ct}: explained variance = {gamma_result['explained_variance'][:2]}")
        except ValueError as e:
            print(f"     {ct}: skipped ({e})")

    print("\n✓ DestVI demo complete!")


def main():
    """Run full demo with synthetic data."""
    print("\n" + "=" * 60)
    print("Spatial Mapping Visualization Demo")
    print("=" * 60)

    # Create synthetic data
    print("\nGenerating synthetic data...")
    n_cells = 500
    n_spots = 200
    n_genes = 100

    snrna = ad.AnnData(
        X=np.random.negative_binomial(5, 0.3, (n_cells, n_genes)),
        obs=pd.DataFrame({
            "cell_type": np.random.choice(["T cells", "B cells", "Monocytes"], n_cells)
        }),
        var=pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)]),
    )
    snrna.obs["cell_type"] = snrna.obs["cell_type"].astype("category")

    spatial = ad.AnnData(
        X=np.random.negative_binomial(5, 0.3, (n_spots, n_genes)),
        obs=pd.DataFrame(index=[f"spot_{i}" for i in range(n_spots)]),
        var=pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)]),
        obsm={"spatial": np.random.rand(n_spots, 2) * 100},
    )

    print(f"  snRNA: {snrna.shape[0]} cells × {snrna.shape[1]} genes")
    print(f"  Spatial: {spatial.shape[0]} spots × {spatial.shape[1]} genes")

    # Output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Run demos
    demo_tangram_visualizations(snrna, spatial, output_dir)
    demo_destvi_visualizations(snrna, spatial, output_dir)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
