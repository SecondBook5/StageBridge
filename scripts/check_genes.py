import anndata
from scvi.model import SCANVI

model_path = "/scratch/chaunzt1/stagebridge/references/hlca/hub_cache/models--scvi-tools--human-lung-cell-atlas-scanvi/snapshots/6978d287b08ac777ca7c015e5220f2feec29ad0a"
model = SCANVI.load(model_path, adata=None)

# Try different ways to get model's expected genes
print("Checking model registry...")
print(f"Registry keys: {list(model.adata_manager.registry.keys())}")

# Try to get var_names from registry
if 'var_names' in model.adata_manager.registry:
    ref_genes = model.adata_manager.registry['var_names']
elif 'setup_args' in model.adata_manager.registry:
    print(f"Setup args: {model.adata_manager.registry['setup_args']}")
    ref_genes = None
else:
    # Try the summary method
    print(model.adata_manager)
    ref_genes = None

if ref_genes is not None:
    print(f'Model: {len(ref_genes)} genes')
    print(f'First 5: {ref_genes[:5]}')

query = anndata.read_h5ad('/scratch/chaunzt1/stagebridge/processed/luad_evo/snrna_qc_normalized.h5ad', backed='r')
print(f'Query: {query.n_vars} genes')
print(f'First 5: {query.var_names.tolist()[:5]}')

overlap = set(ref_genes) & set(query.var_names.tolist())
print(f'Overlap: {len(overlap)} genes')
