# ============================================================================
# NATURE PUBLICATION-QUALITY VISUALIZATION SETUP
# ============================================================================
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import json
import torch
import warnings
from IPython.display import display, clear_output
from datetime import datetime
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull

# Path setup
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

# ============================================================================
# NATURE-QUALITY COLOR PALETTE (Deeper, Richer Colors)
# ============================================================================
# Stage progression colors - saturated and distinct
STAGE_COLORS = {
    "Normal": "#00A650",   # Deep green (healthy)
    "AAH": "#E63946",      # Deep red (early lesion)
    "AIS": "#457B9D",      # Deep blue (intermediate)
    "MIA": "#F77F00",      # Deep orange (late precursor)
    "LUAD": "#9C6644",     # Deep brown (invasive carcinoma)
    "Unknown": "#6C757D",  # Slate gray
}
STAGE_ORDER = ["Normal", "AAH", "AIS", "MIA", "LUAD"]

# Extended color palette for Nature figures
NATURE_PALETTE = {
    "primary_blue": "#1E3A8A",    # Deep blue
    "secondary_red": "#B91C1C",    # Deep red
    "tertiary_green": "#047857",   # Deep emerald
    "accent_orange": "#EA580C",    # Deep orange
    "accent_purple": "#7C3AED",    # Deep purple
    "neutral_dark": "#1F2937",     # Charcoal
    "neutral_mid": "#6B7280",      # Slate
    "neutral_light": "#D1D5DB",    # Light gray
    "background_white": "#FFFFFF",  # Pure white
    "grid": "#E5E7EB",             # Very light gray for grids
}

# Colorblind-friendly sequential palettes
SEQUENTIAL_BLUE = ['#EFF6FF', '#DBEAFE', '#BFDBFE', '#93C5FD', '#60A5FA', '#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF', '#1E3A8A']
SEQUENTIAL_RED = ['#FEF2F2', '#FEE2E2', '#FECACA', '#FCA5A5', '#F87171', '#EF4444', '#DC2626', '#B91C1C', '#991B1B', '#7F1D1D']
SEQUENTIAL_GREEN = ['#ECFDF5', '#D1FAE5', '#A7F3D0', '#6EE7B7', '#34D399', '#10B981', '#059669', '#047857', '#065F46', '#064E3B']

# Diverging palette (for correlation matrices, heatmaps)
DIVERGING_PALETTE = sns.diverging_palette(240, 10, n=11, s=90, l=50, as_cmap=False)

# ============================================================================
# NATURE PUBLICATION MATPLOTLIB CONFIGURATION
# ============================================================================
mpl.rcParams.update({
    # Figure
    'figure.facecolor': '#FFFFFF',
    'figure.dpi': 150,                    # High-res display
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'figure.constrained_layout.use': True,

    # Axes
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#1F2937',          # Dark charcoal edges
    'axes.labelcolor': '#1F2937',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,                # Thicker axes
    'axes.grid': False,
    'axes.axisbelow': True,               # Grid behind data

    # Font (DejaVu Sans is Nature-acceptable, similar to Helvetica)
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 11,
    'font.weight': 'normal',

    # Lines
    'lines.linewidth': 2.0,               # Thicker lines
    'lines.markersize': 6,
    'lines.markeredgewidth': 0.5,

    # Patches (scatter points, bars)
    'patch.linewidth': 0.5,
    'patch.edgecolor': '#1F2937',

    # Ticks
    'xtick.color': '#1F2937',
    'ytick.color': '#1F2937',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    # Grid
    'grid.color': '#E5E7EB',
    'grid.alpha': 0.5,
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',

    # Legend
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.facecolor': '#FFFFFF',
    'legend.edgecolor': '#D1D5DB',
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'legend.borderpad': 0.5,
    'legend.labelspacing': 0.5,
    'legend.handlelength': 2.0,
    'legend.handleheight': 0.7,
    'legend.handletextpad': 0.8,
    'legend.borderaxespad': 0.5,
    'legend.columnspacing': 2.0,

    # Saving (300 DPI for publication)
    'savefig.dpi': 300,
    'savefig.facecolor': '#FFFFFF',
    'savefig.edgecolor': 'none',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.format': 'png',

    # Images
    'image.cmap': 'viridis',
    'image.interpolation': 'bilinear',
})

# ============================================================================
# SEABORN CONFIGURATION (for advanced statistical plots)
# ============================================================================
sns.set_palette([STAGE_COLORS[s] for s in STAGE_ORDER])
sns.set_context("paper", font_scale=1.2)

# ============================================================================
# UTILITY FUNCTIONS FOR NATURE-QUALITY FIGURES
# ============================================================================

def save_figure(fig, name, formats=['png', 'pdf']):
    """Save figure in multiple formats for publication."""
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=300 if fmt=='png' else None,
                   facecolor='white', edgecolor='none', bbox_inches='tight')
    print(f"Saved: {name} ({', '.join(formats)})")

def add_panel_label(ax, label, x=-0.15, y=1.05, fontsize=18, fontweight='bold'):
    """Add Nature-style panel labels (A, B, C, etc.)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va='top', ha='right')

def add_significance_bar(ax, x1, x2, y, h, text='***', fontsize=10):
    """Add significance bars for statistical comparisons."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=fontsize)

def style_axes(ax, xlabel=None, ylabel=None, title=None, grid=False):
    """Apply consistent Nature-style axis formatting."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='normal')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

def add_density_contours(ax, x, y, colors='black', alpha=0.3, levels=5):
    """Add density contours to scatter plots (Nature style)."""
    from scipy.stats import gaussian_kde
    try:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        # Contour plot
        from scipy.interpolate import griddata
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        ax.contour(xi, yi, zi, levels=levels, colors=colors,
                   alpha=alpha, linewidths=1.5)
    except Exception:
        pass  # Skip if contours fail

print("✓ Nature publication-quality visualization configured")
print(f"✓ Stage colors: {list(STAGE_COLORS.keys())}")
print(f"✓ DPI: {mpl.rcParams['savefig.dpi']} (publication quality)")
print(f"✓ Background: Pure white (#FFFFFF)")
