import anndata
import numpy as np
from scvi.model import SCANVI

model_path = "/scratch/chaunzt1/stagebridge/references/hlca/hub_cache/models--scvi-tools--human-lung-cell-atlas-scanvi/snapshots/6978d287b08ac777ca7c015e5220f2feec29ad0a"
model = SCANVI.load(model_path, adata=None)

# Check field_registries for var_names
print("Checking field_registries...")
field_reg = model.adata_manager.registry.get('field_registries', {})
print(f"Field registry keys: {list(field_reg.keys())}")

# Look for var_names in each field
for key, val in field_reg.items():
    if isinstance(val, dict) and 'var_names' in str(val):
        print(f"  {key}: {val}")

# Try to get var_names directly from model's internal state
print("\nChecking model module...")
if hasattr(model.module, 'var_names'):
    print(f"module.var_names: {model.module.var_names[:5]}")

# Check the model file directly for var_names
import torch
state = torch.load(f"{model_path}/model.pt", map_location='cpu', weights_only=False)
print(f"\nModel state keys: {list(state.keys())}")
if 'var_names' in state:
    ref_genes = state['var_names']
    print(f"Found var_names in state: {len(ref_genes)} genes")
    print(f"First 5: {list(ref_genes)[:5]}")

# Also check the HLCA reference h5ad directly
print("\n--- Checking HLCA reference h5ad ---")
hlca = anndata.read_h5ad('/scratch/chaunzt1/stagebridge/references/hlca/hlca_reference.h5ad', backed='r')
print(f"HLCA ref: {hlca.n_vars} genes")
print(f"First 5 var_names: {hlca.var_names.tolist()[:5]}")
print(f"HLCA var columns: {hlca.var.columns.tolist()}")
print(f"HLCA var head:\n{hlca.var.head(3)}")

query = anndata.read_h5ad('/scratch/chaunzt1/stagebridge/processed/luad_evo/snrna_qc_normalized.h5ad', backed='r')
print(f'Query: {query.n_vars} genes')
print(f'First 5: {query.var_names.tolist()[:5]}')

overlap = set(ref_genes) & set(query.var_names.tolist())
print(f'Overlap: {len(overlap)} genes')
