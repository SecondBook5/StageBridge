"""Add ENSG IDs to query using HLCA's own mapping (guaranteed correct for model)."""
import anndata

# Load HLCA reference to get its symbol -> ENSG mapping
print("Loading HLCA reference for gene mapping...")
hlca_path = '/scratch/chaunzt1/stagebridge/references/hlca/hlca_reference.h5ad'
hlca = anndata.read_h5ad(hlca_path, backed='r')

# Build symbol -> ENSG mapping from HLCA
symbol_to_ensg = dict(zip(hlca.var_names, hlca.var['ensembl_id']))
print(f"HLCA mapping: {len(symbol_to_ensg)} genes (symbol -> ENSG)")
print(f"Sample: {list(symbol_to_ensg.items())[:3]}")

# Load query
print("\nLoading query...")
query_path = '/scratch/chaunzt1/stagebridge/processed/luad_evo/snrna_qc_normalized.h5ad'
query = anndata.read_h5ad(query_path)
query_symbols = set(query.var_names)
print(f"Query has {len(query_symbols)} genes")

# Check overlap between query symbols and HLCA symbols
hlca_symbols = set(hlca.var_names)
symbol_overlap = query_symbols & hlca_symbols
print(f"Query symbols that match HLCA symbols: {len(symbol_overlap)}/2000")

# This is the key number - how many of the model's 2000 genes are in the query?
import torch
model_path = "/scratch/chaunzt1/stagebridge/references/hlca/hub_cache/models--scvi-tools--human-lung-cell-atlas-scanvi/snapshots/6978d287b08ac777ca7c015e5220f2feec29ad0a"
state = torch.load(f"{model_path}/model.pt", map_location='cpu', weights_only=False)
model_ensg = set(state['var_names'])
print(f"Model expects {len(model_ensg)} ENSG IDs")

# Map query symbols -> ENSG using HLCA mapping, check model coverage
query_ensg_mapped = {symbol_to_ensg[s] for s in symbol_overlap if s in symbol_to_ensg}
model_coverage = query_ensg_mapped & model_ensg
print(f"Query genes that map to model's ENSG IDs: {len(model_coverage)}/2000")

if len(model_coverage) < 1500:
    print("\nWARNING: Low coverage! Model-based mapping may not work well.")
    print("Consider using k-NN fallback instead.")
else:
    print(f"\nGood coverage ({len(model_coverage)/2000:.1%})! Model-based mapping should work.")

# Add ensembl_id column to query var
print("\nAdding ensembl_id column to query...")
query.var['ensembl_id'] = [symbol_to_ensg.get(s, '') for s in query.var_names]
n_mapped = sum(1 for e in query.var['ensembl_id'] if e)
print(f"Mapped {n_mapped}/{query.n_vars} query genes to ENSG IDs")

# Save updated query
output_path = '/scratch/chaunzt1/stagebridge/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad'
print(f"\nSaving to {output_path}...")
query.write_h5ad(output_path)
print("Done!")

print("\nNext: Update pipeline to use this file and convert var_names to ENSG before surgery")
