import json
import subprocess
import sys



def test_poster_asset_script_outputs_expected_files(tmp_path):
    metrics = {
        "run_name": "test",
        "results": {
            "stagebridge": {
                "aggregate": {"sinkhorn_mean": 0.9, "sinkhorn_std": 0.1},
                "folds": [],
                "checkpoints": [],
            },
            "deepsets": {
                "aggregate": {"sinkhorn_mean": 1.2, "sinkhorn_std": 0.2},
                "folds": [],
                "checkpoints": [],
            },
        },
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    fig_dir = tmp_path / "figures"
    tab_dir = tmp_path / "tables"

    cmd = [
        sys.executable,
        "scripts/make_poster_assets.py",
        str(metrics_path),
        "--output-dir",
        str(fig_dir),
        "--table-dir",
        str(tab_dir),
    ]
    subprocess.run(cmd, check=True)

    expected = [
        fig_dir / "poster_panel_a_method_schematic.png",
        fig_dir / "poster_panel_b_benchmark.png",
        fig_dir / "poster_panel_c_trajectory.png",
        fig_dir / "poster_panel_d_metric_heatmap.png",
        tab_dir / "poster_table_1_metrics.csv",
        tab_dir / "poster_abstract_structure.txt",
    ]
    for path in expected:
        assert path.exists(), f"missing {path}"
