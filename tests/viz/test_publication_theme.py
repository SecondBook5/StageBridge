"""Tests for publication theme and styling utilities."""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

from stagebridge.viz.publication_theme import (
    configure_publication_style,
    save_publication_figure,
    get_stage_color,
    apply_clean_spines,
    add_clean_legend,
    create_figure,
    create_subplots,
    setup_publication_plotting,
    PUBLICATION_PALETTE,
)


def test_publication_palette():
    """Test that publication palette has all required stages."""
    required_stages = ["Normal", "AAH", "AIS", "MIA", "LUAD", "Unknown"]
    for stage in required_stages:
        assert stage in PUBLICATION_PALETTE
        assert PUBLICATION_PALETTE[stage].startswith("#")
        assert len(PUBLICATION_PALETTE[stage]) == 7  # #RRGGBB format


def test_configure_publication_style():
    """Test that publication style sets correct rcParams."""
    configure_publication_style()

    # Check key settings
    assert plt.rcParams["figure.facecolor"] == "#FFFFFF"
    assert plt.rcParams["axes.facecolor"] == "#FFFFFF"
    assert plt.rcParams["savefig.facecolor"] == "#FFFFFF"
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["figure.dpi"] == 150

    # Check font sizes
    assert plt.rcParams["font.size"] == 10
    assert plt.rcParams["axes.titlesize"] == 14
    assert plt.rcParams["axes.labelsize"] == 12

    # Check spines
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["axes.linewidth"] == 1.5


def test_get_stage_color():
    """Test stage color retrieval."""
    # Colors now match LungPCA paper palette
    assert get_stage_color("Normal") == "#33a02c"
    assert get_stage_color("AAH") == "#b2df8a"
    assert get_stage_color("LUAD") == "#ff7f00"
    assert get_stage_color("InvalidStage") == "#d9d9d9"  # Unknown fallback (LungPCA gray)


def test_apply_clean_spines():
    """Test spine removal."""
    fig, ax = plt.subplots()
    apply_clean_spines(ax)

    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_linewidth() == 1.5
    assert ax.spines["bottom"].get_linewidth() == 1.5
    plt.close(fig)


def test_add_clean_legend():
    """Test legend styling."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9], label="Data")
    legend = add_clean_legend(ax, title="Test")

    assert legend is not None
    assert legend.get_title().get_text() == "Test"
    assert legend.get_frame().get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # White
    plt.close(fig)


def test_create_figure():
    """Test figure creation with clean styling."""
    fig, ax = create_figure(figsize=(8, 6))

    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # White
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    plt.close(fig)


def test_create_subplots():
    """Test subplot creation with clean styling."""
    fig, axes = create_subplots(nrows=2, ncols=2)

    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # White
    assert axes.shape == (2, 2)

    # Check all subplots have clean spines
    for ax in axes.flat:
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

    plt.close(fig)


def test_save_publication_figure():
    """Test multi-format figure saving."""
    configure_publication_style()
    fig, ax = create_figure()
    ax.plot([1, 2, 3], [1, 4, 9])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_figure"
        saved_paths = save_publication_figure(fig, output_path, formats=["png", "pdf"])

        # Check that files were created
        assert "png" in saved_paths
        assert "pdf" in saved_paths
        assert saved_paths["png"].exists()
        assert saved_paths["pdf"].exists()

        # Check file extensions
        assert saved_paths["png"].suffix == ".png"
        assert saved_paths["pdf"].suffix == ".pdf"

    plt.close(fig)


def test_save_publication_figure_svg():
    """Test SVG format saving."""
    configure_publication_style()
    fig, ax = create_figure()
    ax.plot([1, 2, 3], [1, 4, 9])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_svg"
        saved_paths = save_publication_figure(fig, output_path, formats=["svg"])

        assert "svg" in saved_paths
        assert saved_paths["svg"].exists()
        assert saved_paths["svg"].suffix == ".svg"

    plt.close(fig)


def test_setup_publication_plotting():
    """Test one-line setup function."""
    setup_publication_plotting()

    # Should set publication style
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["figure.facecolor"] == "#FFFFFF"


def test_publication_figure_pipeline():
    """Test complete pipeline: setup -> create -> save."""
    setup_publication_plotting()

    # Create figure
    fig, ax = create_figure(figsize=(10, 8))

    # Add some data
    x = np.linspace(0, 10, 100)
    stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    for i, stage in enumerate(stages):
        y = np.sin(x + i) + i
        ax.plot(x, y, label=stage, color=get_stage_color(stage), linewidth=2)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Test Publication Figure")
    add_clean_legend(ax, title="Stage")
    ax.grid(True, alpha=0.3)

    # Save in all formats
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "complete_test"
        saved_paths = save_publication_figure(fig, output_path)

        assert len(saved_paths) == 3  # PNG, PDF, SVG by default
        for path in saved_paths.values():
            assert path.exists()

    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
