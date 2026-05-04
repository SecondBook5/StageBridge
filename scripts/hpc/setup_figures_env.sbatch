#!/bin/bash
#SBATCH --job-name=setup_figures_env
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/home/booka/StageBridge/logs/setup_figures_env_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/setup_figures_env_%j.err

# Create the stagebridge_figures conda environment for advanced visualizations
# Run with: sbatch scripts/hpc/setup_figures_env.sbatch

set -e

echo "=============================================="
echo "Setting up stagebridge_figures environment"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

module load miniforge3
eval "$(conda shell.bash hook)"

ENV_PATH="/scratch/chaunzt1/envs/stagebridge_figures"
ENV_YAML="/home/booka/StageBridge/envs/figures_env.yaml"

# Check if already exists
if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists at $ENV_PATH"
    echo "To recreate, first remove it:"
    echo "  rm -rf $ENV_PATH"
    echo ""
    echo "Updating existing environment instead..."
    conda env update -p "$ENV_PATH" -f "$ENV_YAML"
else
    echo "Creating new environment at $ENV_PATH"
    echo "This may take 10-20 minutes..."
    echo ""
    conda env create -f "$ENV_YAML" -p "$ENV_PATH"
fi

echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

conda activate "$ENV_PATH"

python << 'EOF'
import sys
print(f"Python: {sys.version}")

packages = [
    ('scanpy', 'sc'),
    ('phate', 'phate'),
    ('gseapy', 'gp'),
    ('networkx', 'nx'),
    ('seaborn', 'sns'),
    ('matplotlib', 'plt'),
]

print("\nPackage verification:")
for pkg_name, import_name in packages:
    try:
        mod = __import__(import_name if import_name else pkg_name)
        version = getattr(mod, '__version__', 'OK')
        print(f"  {pkg_name}: {version}")
    except ImportError as e:
        print(f"  {pkg_name}: MISSING - {e}")

# Check GraphRicciCurvature separately (different import)
try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    print("  GraphRicciCurvature: OK")
except ImportError as e:
    print(f"  GraphRicciCurvature: MISSING - {e}")

# Check optional packages
optional = ['scvelo', 'cellrank', 'squidpy']
print("\nOptional packages:")
for pkg in optional:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'OK')
        print(f"  {pkg}: {version}")
    except ImportError:
        print(f"  {pkg}: not installed (optional)")

print("\nEnvironment ready!")
EOF

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo "Environment: $ENV_PATH"
echo ""
echo "To activate manually:"
echo "  conda activate $ENV_PATH"
echo ""
echo "The full pipeline will auto-switch to this env for figures."
