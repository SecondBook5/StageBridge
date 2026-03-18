"""Add ENSG IDs to query data for model-based mapping."""
import anndata
import numpy as np

# Check what's available for gene mapping
try:
    import mygene
    USE_MYGENE = True
    print("Using mygene for gene ID mapping")
except ImportError:
    USE_MYGENE = False
    print("mygene not installed, trying pybiomart...")

if not USE_MYGENE:
    try:
        from pybiomart import Dataset
        USE_BIOMART = True
        print("Using pybiomart for gene ID mapping")
    except ImportError:
        USE_BIOMART = False
        print("pybiomart not installed either")
        print("Install with: pip install mygene")
        exit(1)

# Load query
print("\nLoading query...")
query_path = '/scratch/chaunzt1/stagebridge/processed/luad_evo/snrna_qc_normalized.h5ad'
query = anndata.read_h5ad(query_path)
symbols = query.var_names.tolist()
print(f"Query has {len(symbols)} genes")

# Map symbols to ENSG IDs
print("\nMapping gene symbols to ENSG IDs...")

if USE_MYGENE:
    mg = mygene.MyGeneInfo()
    # Query in batches
    results = mg.querymany(symbols, scopes='symbol', fields='ensembl.gene', species='human', verbose=False)

    symbol_to_ensg = {}
    for r in results:
        symbol = r.get('query')
        if 'ensembl' in r:
            ensg = r['ensembl']
            if isinstance(ensg, list):
                ensg = ensg[0]['gene']  # Take first if multiple
            elif isinstance(ensg, dict):
                ensg = ensg['gene']
            symbol_to_ensg[symbol] = ensg

elif USE_BIOMART:
    dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
    results = dataset.query(attributes=['hgnc_symbol', 'ensembl_gene_id'])
    symbol_to_ensg = dict(zip(results['HGNC symbol'], results['Gene stable ID']))

print(f"Mapped {len(symbol_to_ensg)} symbols to ENSG IDs")

# Check overlap with model's expected genes
import torch
model_path = "/scratch/chaunzt1/stagebridge/references/hlca/hub_cache/models--scvi-tools--human-lung-cell-atlas-scanvi/snapshots/6978d287b08ac777ca7c015e5220f2feec29ad0a"
state = torch.load(f"{model_path}/model.pt", map_location='cpu', weights_only=False)
model_genes = set(state['var_names'])
print(f"Model expects {len(model_genes)} genes")

# How many query genes map to model genes?
mapped_ensg = [symbol_to_ensg.get(s) for s in symbols]
overlap = sum(1 for e in mapped_ensg if e in model_genes)
print(f"Query genes that map to model genes: {overlap}")

# Add ensembl_id column to query
query.var['ensembl_id'] = [symbol_to_ensg.get(s, '') for s in symbols]
n_mapped = sum(1 for e in query.var['ensembl_id'] if e)
print(f"Added ensembl_id column: {n_mapped}/{len(symbols)} genes mapped")

# Save updated query
output_path = '/scratch/chaunzt1/stagebridge/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad'
print(f"\nSaving to {output_path}...")
query.write_h5ad(output_path)
print("Done!")

# Show sample
print("\nSample mapping:")
for s in symbols[:5]:
    print(f"  {s} -> {symbol_to_ensg.get(s, 'NOT FOUND')}")
