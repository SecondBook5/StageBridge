#!/usr/bin/env python3
"""Master script to generate all publication-quality poster panels.

Consolidates panel generation from:
- Fig 1: Embedding overview (9 panels)
- Fig 2: Biological features (7 panels)
- Fig 3: OT dynamics (11 panels)
- Fig 4: Signatures (violin plots, UMAPs, spatial)
- Fig 5: Trajectory/pseudotime (9 panels)

Usage:
    python generate_all_panels.py --data_dir $CANONICAL --output_dir $FIGS/panels

Output structure:
    output_dir/
        embedding/      # Fig 1 panels
        biological/     # Fig 2 panels
        ot_dynamics/    # Fig 3 panels
        signatures/     # Fig 4 panels
        trajectory/     # Fig 5 panels
        summary/        # Combined summary figures
"""

import argparse
import sys
from pathlib import Path
from importlib import import_module


def run_embedding_panels(data_dir: Path, output_dir: Path):
    """Generate Fig 1: Embedding overview panels."""
    from plot_embedding_panels import load_data, \
        plot_panel_a_stage_umap, plot_panel_b_density_contours, \
        plot_panel_c_cell_types, plot_panel_d_cell_cycle, \
        plot_panel_e_proliferation, plot_panel_f_stage_distribution, \
        plot_panel_g_celltype_by_stage, plot_panel_h_mean_embedding, \
        plot_panel_i_discriminative_dims

    out = output_dir / 'embedding'
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("FIG 1: EMBEDDING OVERVIEW")
    print("="*60)

    data = load_data(data_dir)

    plot_panel_a_stage_umap(data, out)
    plot_panel_b_density_contours(data, out)
    plot_panel_c_cell_types(data, out)
    plot_panel_d_cell_cycle(data, out)
    plot_panel_e_proliferation(data, out)
    plot_panel_f_stage_distribution(data, out)
    plot_panel_g_celltype_by_stage(data, out)
    plot_panel_h_mean_embedding(data, out)
    plot_panel_i_discriminative_dims(data, out)


def run_biological_panels(data_dir: Path, output_dir: Path):
    """Generate Fig 2: Biological feature panels."""
    from plot_biological_panels import load_data, plot_score_umap, \
        plot_panel_e_pathway_violins, plot_panel_f_mutation_heatmap, \
        plot_panel_g_clonal_patterns

    out = output_dir / 'biological'
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("FIG 2: BIOLOGICAL FEATURES")
    print("="*60)

    data = load_data(data_dir)

    plot_score_umap(data, 'EMT', ['EMT_score', 'emt_score'], 'A', out)
    plot_score_umap(data, 'Hypoxia', ['hypoxia_score', 'Hypoxia_score'], 'B', out)
    plot_score_umap(data, 'Inflammation', ['IL1_axis_score', 'inflammation_score', 'NFkB_score'], 'C', out)
    plot_score_umap(data, 'Proliferation', ['entropic_score', 'proliferation_score'], 'D', out)
    plot_panel_e_pathway_violins(data, out)
    plot_panel_f_mutation_heatmap(data, out)
    plot_panel_g_clonal_patterns(data, out)


def run_ot_panels(data_dir: Path, output_dir: Path):
    """Generate Fig 3: OT dynamics panels."""
    from plot_ot_panels import load_data, \
        panel_a_stages, panel_b_velocity_field, panel_c_ot_distance, \
        panel_d_divergence, panel_e_curl, panel_f_irreversibility_map, \
        panel_g_irreversibility_violin, panel_h_flow_speed, \
        panel_i_progression_cost, panel_j_transition_matrix, panel_k_key_metrics

    out = output_dir / 'ot_dynamics'
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("FIG 3: OPTIMAL TRANSPORT DYNAMICS")
    print("="*60)

    data = load_data(data_dir)

    if 'umap' in data and 'stages' in data:
        panel_a_stages(data['umap'], data['stages'], out)

    if 'umap' in data and 'velocity' in data:
        panel_b_velocity_field(data['umap'], data['velocity'], data.get('stages'), out)

    if 'w_distances' in data:
        panel_c_ot_distance(data['w_distances'], out)
        panel_i_progression_cost(data['w_distances'], out)

    if 'umap' in data and 'divergence' in data:
        panel_d_divergence(data['umap'], data['divergence'], data.get('stages'), out)

    if 'umap' in data and 'curl' in data:
        panel_e_curl(data['umap'], data['curl'], data.get('stages'), out)

    if 'umap' in data and 'flux_ratio' in data:
        panel_f_irreversibility_map(data['umap'], data['flux_ratio'], data.get('stages'), out)

    if 'flux_ratio' in data and 'stages' in data:
        panel_g_irreversibility_violin(data['flux_ratio'], data['stages'], out)

    if 'umap' in data and 'speed' in data:
        panel_h_flow_speed(data['umap'], data['speed'], data.get('stages'), out)

    if 'transition_probs' in data:
        panel_j_transition_matrix(data['transition_probs'], out)

    if 'metrics' in data:
        panel_k_key_metrics(data['metrics'], out)


def run_signature_panels(data_dir: Path, output_dir: Path):
    """Generate Fig 4: Signature panels."""
    from plot_signature_figures import plot_signature_by_stage

    out = output_dir / 'signatures'
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("FIG 4: BIOLOGICAL SIGNATURES")
    print("="*60)

    # Look for scores file
    scores_path = None
    for pattern in ['caf_kac_scores.parquet', 'signatures/caf_kac_scores.parquet']:
        path = data_dir / pattern
        if path.exists():
            scores_path = path
            break

    if scores_path:
        plot_signature_by_stage(scores_path, out)
    else:
        print("  Skipping: no scores file found")


def run_trajectory_panels(data_dir: Path, output_dir: Path):
    """Generate Fig 5: Trajectory panels."""
    from plot_trajectory_panels import load_data, \
        panel_a_velocity_field, panel_b_diffusion_pseudotime, \
        panel_c_diffusion_components, panel_d_pseudotime_ridgeline, \
        panel_e_transition_heterogeneity, panel_f_pathway_dynamics, \
        panel_g_density_landscape, panel_h_stage_entropy, panel_i_diffusion_spectrum

    out = output_dir / 'trajectory'
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("FIG 5: TRAJECTORY ANALYSIS")
    print("="*60)

    data = load_data(data_dir)

    panel_a_velocity_field(data, out)
    panel_b_diffusion_pseudotime(data, out)
    panel_c_diffusion_components(data, out)
    panel_d_pseudotime_ridgeline(data, out)
    panel_e_transition_heterogeneity(data, out)
    panel_f_pathway_dynamics(data, out)
    panel_g_density_landscape(data, out)
    panel_h_stage_entropy(data, out)
    panel_i_diffusion_spectrum(data, out)


def main():
    parser = argparse.ArgumentParser(
        description='Generate all publication-quality poster panels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all panels
    python generate_all_panels.py --data_dir $CANONICAL --output_dir $FIGS/panels

    # Generate only specific figure sets
    python generate_all_panels.py --data_dir $CANONICAL --output_dir $FIGS/panels --only embedding trajectory

    # Skip certain panels
    python generate_all_panels.py --data_dir $CANONICAL --output_dir $FIGS/panels --skip ot_dynamics
        """
    )
    parser.add_argument('--data_dir', type=Path, required=True,
                       help='Directory with processed data (cells.parquet, umap, etc.)')
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Output directory for panels')
    parser.add_argument('--only', nargs='+', choices=['embedding', 'biological', 'ot_dynamics', 'signatures', 'trajectory'],
                       help='Only generate these figure sets')
    parser.add_argument('--skip', nargs='+', choices=['embedding', 'biological', 'ot_dynamics', 'signatures', 'trajectory'],
                       default=[], help='Skip these figure sets')

    args = parser.parse_args()

    # Determine which panels to generate
    all_panels = ['embedding', 'biological', 'ot_dynamics', 'signatures', 'trajectory']

    if args.only:
        panels_to_run = args.only
    else:
        panels_to_run = [p for p in all_panels if p not in args.skip]

    print("="*60)
    print("STAGEBRIDGE POSTER PANEL GENERATOR")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Panels to generate: {', '.join(panels_to_run)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Add script directory to path for imports
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    # Run each panel set
    if 'embedding' in panels_to_run:
        try:
            run_embedding_panels(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  ERROR in embedding panels: {e}")

    if 'biological' in panels_to_run:
        try:
            run_biological_panels(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  ERROR in biological panels: {e}")

    if 'ot_dynamics' in panels_to_run:
        try:
            run_ot_panels(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  ERROR in OT dynamics panels: {e}")

    if 'signatures' in panels_to_run:
        try:
            run_signature_panels(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  ERROR in signature panels: {e}")

    if 'trajectory' in panels_to_run:
        try:
            run_trajectory_panels(args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  ERROR in trajectory panels: {e}")

    print("\n" + "="*60)
    print("PANEL GENERATION COMPLETE")
    print("="*60)
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
