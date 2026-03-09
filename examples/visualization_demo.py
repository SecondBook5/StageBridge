"""Example script demonstrating the enhanced StageBridge visualizations.

This script shows how to use the new advanced plotting features.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project root to Python path to import viz module directly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Example 1: Enhanced UMAP with all features
def example_enhanced_umap():
    """Demonstrate enhanced UMAP plotting with density contours and hulls."""
    # from stagebridge.viz.embeddings import plot_umap_by_stage
    
    # Load your AnnData object (example)
    # adata = sc.read_h5ad("path/to/your/data.h5ad")
    
    # Create enhanced UMAP plot
    # plot_umap_by_stage(
    #     adata,
    #     output_path=Path("figures/umap_enhanced.png"),
    #     title="Enhanced UMAP with All Features",
    #     show_density=True,      # Show KDE density contours
    #     show_hulls=True,        # Show convex hulls around stages
    #     show_ellipses=True,     # Show 95% confidence ellipses
    #     point_size=3.0,
    #     alpha=0.6
    # )
    print("Example: Enhanced UMAP plotting")
    print("- Density contours for overall distribution")
    print("- Convex hulls around stage clusters")
    print("- Confidence ellipses (95% CI)")
    print("- Color-blind friendly palette")
    print()


# Example 2: Radar chart for multi-metric comparison
def example_radar_chart():
    """Demonstrate radar chart for comparing models across metrics."""
    from stagebridge.viz.advanced_plots import plot_radar_chart
    
    # Create example data
    metrics_df = pd.DataFrame({
        'label': ['StageBridge', 'Baseline_A', 'Baseline_B', 'Ablation_1'],
        'sinkhorn_mean': [0.12, 0.18, 0.15, 0.14],
        'mmd_rbf_mean': [0.08, 0.14, 0.11, 0.10],
        'classifier_auc_mean': [0.85, 0.78, 0.80, 0.82],
        'jsd_composition_mean': [0.06, 0.11, 0.09, 0.08]
    })
    
    # Create radar chart
    fig = plot_radar_chart(
        df=metrics_df,
        metrics=['sinkhorn_mean', 'mmd_rbf_mean', 'classifier_auc_mean', 'jsd_composition_mean'],
        labels_col='label',
        output_path=Path("figures/example_radar.png"),
        title="Multi-Metric Model Comparison",
        normalize=True
    )
    
    print("✓ Created radar chart: figures/example_radar.png")
    print("  Shows normalized metrics across 4 models")
    print()


# Example 3: Parallel coordinates plot
def example_parallel_coordinates():
    """Demonstrate parallel coordinates for high-dimensional comparison."""
    from stagebridge.viz.advanced_plots import plot_parallel_coordinates
    
    # Create example data
    metrics_df = pd.DataFrame({
        'label': ['StageBridge', 'Baseline_A', 'Baseline_B'],
        'metric_1': [0.85, 0.75, 0.80],
        'metric_2': [0.12, 0.18, 0.15],
        'metric_3': [0.90, 0.82, 0.87],
        'metric_4': [0.08, 0.14, 0.11],
        'metric_5': [0.78, 0.70, 0.75]
    })
    
    # Create parallel coordinates plot
    fig = plot_parallel_coordinates(
        df=metrics_df,
        metrics=['metric_1', 'metric_2', 'metric_3', 'metric_4', 'metric_5'],
        labels_col='label',
        output_path=Path("figures/example_parallel.png"),
        title="High-Dimensional Metric Comparison",
        normalize=True
    )
    
    print("✓ Created parallel coordinates plot: figures/example_parallel.png")
    print("  Shows 5 metrics across 3 models")
    print()


# Example 4: Correlation matrix
def example_correlation_matrix():
    """Demonstrate correlation matrix visualization."""
    from stagebridge.viz.advanced_plots import plot_correlation_matrix
    
    # Create example data
    np.random.seed(42)
    n_samples = 100
    metrics_df = pd.DataFrame({
        'sinkhorn': np.random.randn(n_samples) * 0.05 + 0.15,
        'mmd': np.random.randn(n_samples) * 0.03 + 0.10,
        'auc': np.random.randn(n_samples) * 0.05 + 0.80,
        'jsd': np.random.randn(n_samples) * 0.02 + 0.08,
        'fid': np.random.randn(n_samples) * 0.08 + 0.25
    })
    
    # Add some correlation structure
    metrics_df['mmd'] = metrics_df['mmd'] + 0.7 * metrics_df['sinkhorn']
    metrics_df['jsd'] = metrics_df['jsd'] + 0.5 * metrics_df['sinkhorn']
    
    # Create correlation matrix
    fig = plot_correlation_matrix(
        df=metrics_df,
        output_path=Path("figures/example_correlation.png"),
        title="Metric Correlation Matrix",
        method="pearson",
        show_values=True
    )
    
    print("✓ Created correlation matrix: figures/example_correlation.png")
    print("  Shows Pearson correlations between metrics")
    print()


# Example 5: 3D embedding
def example_3d_embedding():
    """Demonstrate 3D scatter plot for embeddings."""
    from stagebridge.viz.advanced_plots import plot_3d_embedding
    
    # Create example 3D embedding data
    np.random.seed(42)
    n_points = 500
    
    # Generate 3 clusters
    cluster_1 = np.random.randn(n_points // 3, 3) + [0, 0, 0]
    cluster_2 = np.random.randn(n_points // 3, 3) + [3, 3, 0]
    cluster_3 = np.random.randn(n_points // 3, 3) + [0, 3, 3]
    
    coords = np.vstack([cluster_1, cluster_2, cluster_3])
    labels = np.array(['Normal'] * (n_points // 3) + 
                     ['AAH'] * (n_points // 3) + 
                     ['AIS'] * (n_points // 3))
    
    # Create 3D embedding plot
    fig = plot_3d_embedding(
        coords=coords,
        labels=labels,
        output_path=Path("figures/example_3d_embedding.png"),
        title="3D Embedding Visualization",
        point_size=5.0,
        alpha=0.7
    )
    
    print("✓ Created 3D embedding plot: figures/example_3d_embedding.png")
    print("  Shows 3D scatter with color-coded labels")
    print()


# Example 6: Ridge plot (joyplot)
def example_ridge_plot():
    """Demonstrate ridge plot for distribution comparison."""
    from stagebridge.viz.advanced_plots import plot_ridge_distributions
    
    # Create example distribution data
    np.random.seed(42)
    data_dict = {
        'StageBridge': np.random.normal(0.85, 0.05, 200),
        'Baseline A': np.random.normal(0.78, 0.07, 200),
        'Baseline B': np.random.normal(0.80, 0.06, 200),
        'Ablation 1': np.random.normal(0.82, 0.05, 200)
    }
    
    # Create ridge plot
    fig = plot_ridge_distributions(
        data_dict=data_dict,
        output_path=Path("figures/example_ridge.png"),
        title="Metric Distribution Comparison"
    )
    
    print("✓ Created ridge plot: figures/example_ridge.png")
    print("  Shows distribution comparisons across models")
    print()


# Example 7: Enhanced benchmark bars
def example_enhanced_benchmark():
    """Demonstrate enhanced benchmark bar plots."""
    from stagebridge.viz.curves import plot_benchmark_bars
    
    # Create example benchmark data
    metrics_df = pd.DataFrame({
        'label': ['StageBridge', 'Baseline_A', 'Baseline_B', 'Ablation_1', 'Ablation_2'],
        'sinkhorn_mean': [0.12, 0.18, 0.15, 0.14, 0.16],
        'sinkhorn_std': [0.01, 0.02, 0.015, 0.012, 0.018]
    })
    
    # Create enhanced bar plot
    from pathlib import Path
    Path("figures").mkdir(exist_ok=True)
    
    plot_benchmark_bars(
        df=metrics_df,
        output_path=Path("figures/example_benchmark.png"),
        metric_col='sinkhorn_mean',
        metric_std_col='sinkhorn_std',
        title="Enhanced Benchmark Comparison",
        show_values=True,
        highlight_best=True
    )
    
    print("✓ Created enhanced benchmark plot: figures/example_benchmark.png")
    print("  Features:")
    print("    - Value annotations")
    print("    - Best model highlighting")
    print("    - Reference line")
    print("    - Enhanced error bars")
    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("StageBridge Enhanced Visualization Examples")
    print("=" * 60)
    print()
    
    # Create output directory
    Path("figures").mkdir(exist_ok=True)
    
    # Run examples that don't require real data
    print("Creating example visualizations...\n")
    
    example_enhanced_umap()
    example_radar_chart()
    example_parallel_coordinates()
    example_correlation_matrix()
    example_3d_embedding()
    example_ridge_plot()
    example_enhanced_benchmark()
    
    print("=" * 60)
    print("All examples completed!")
    print("Check the 'figures/' directory for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
