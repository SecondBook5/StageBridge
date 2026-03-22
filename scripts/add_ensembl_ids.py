#!/usr/bin/env python
"""Add ENSG IDs to query using HLCA + LuCA mappings for maximum coverage.

Usage:
    python scripts/add_ensembl_ids.py \
        --query $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
        --hlca $DATA/references/hlca/hlca_reference.h5ad \
        --luca $DATA/references/luca/luca_core_atlas.h5ad \
        --output $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad

This script:
1. Loads HLCA reference to extract symbol -> ENSG mapping (~2000 genes)
2. Loads LuCA reference to extract symbol -> ENSG mapping (~17000 genes)
3. Merges mappings (LuCA has more genes, HLCA takes precedence for conflicts)
4. Adds ensembl_id column to query.var
5. Saves updated query

The pipeline's model-based mapping will then auto-convert var_names to ENSG IDs.
"""
import argparse
from pathlib import Path

import anndata


def main():
    parser = argparse.ArgumentParser(description="Add ENSG IDs to query using HLCA+LuCA mapping")
    parser.add_argument("--query", required=True, help="Path to query h5ad file")
    parser.add_argument("--hlca", required=True, help="Path to HLCA reference h5ad file")
    parser.add_argument("--luca", default=None, help="Path to LuCA reference h5ad file (optional but recommended)")
    parser.add_argument("--output", required=True, help="Output path for updated query")
    parser.add_argument("--model-path", default=None, help="Optional: path to scANVI model to check coverage")
    args = parser.parse_args()

    symbol_to_ensg = {}

    # Load LuCA first (more genes, but HLCA takes precedence for conflicts)
    # Use h5py directly to avoid loading full adata into memory
    if args.luca:
        import h5py
        print("Loading LuCA reference gene mapping (lightweight h5py)...")
        with h5py.File(args.luca, 'r') as f:
            # Get var_names (ENSG IDs) - stored as _index
            if '_index' in f['var']:
                luca_ensg = f['var']['_index'][:].astype(str)
            else:
                print("WARNING: LuCA missing var index, skipping")
                luca_ensg = None

            # Get feature_name (gene symbols) - categorical
            if luca_ensg is not None and 'feature_name' in f['var']:
                fn = f['var']['feature_name']
                if 'categories' in fn:
                    # Categorical encoding
                    categories = fn['categories'][:].astype(str)
                    codes = fn['codes'][:]
                    luca_symbols = categories[codes]
                else:
                    luca_symbols = fn[:].astype(str)

                luca_mapping = {str(s): str(e) for s, e in zip(luca_symbols, luca_ensg) if s and e}
                symbol_to_ensg.update(luca_mapping)
                print(f"LuCA mapping: {len(luca_mapping)} genes (symbol -> ENSG)")
            else:
                print("WARNING: LuCA missing 'feature_name' column, skipping")

    # Load HLCA (takes precedence for any conflicts)
    print("Loading HLCA reference for gene mapping...")
    hlca = anndata.read_h5ad(args.hlca, backed='r')

    # Build symbol -> ENSG mapping from HLCA
    if 'ensembl_id' not in hlca.var.columns:
        raise ValueError("HLCA reference missing 'ensembl_id' column in var")

    hlca_mapping = dict(zip(hlca.var_names, hlca.var['ensembl_id']))
    symbol_to_ensg.update(hlca_mapping)  # HLCA overwrites LuCA for conflicts
    print(f"HLCA mapping: {len(hlca_mapping)} genes (symbol -> ENSG)")
    print(f"Combined mapping: {len(symbol_to_ensg)} unique symbols")
    print(f"Sample: {list(symbol_to_ensg.items())[:3]}")

    # Load query
    print("\nLoading query...")
    query = anndata.read_h5ad(args.query)
    query_symbols = set(query.var_names)
    print(f"Query has {len(query_symbols)} genes")

    # Check overlap between query symbols and combined mapping
    mapping_symbols = set(symbol_to_ensg.keys())
    symbol_overlap = query_symbols & mapping_symbols
    print(f"Query symbols with ENSG mapping: {len(symbol_overlap)}/{len(query_symbols)}")

    # Optionally check model coverage
    if args.model_path:
        import torch
        state = torch.load(f"{args.model_path}/model.pt", map_location='cpu', weights_only=False)
        model_ensg = set(state['var_names'])
        print(f"Model expects {len(model_ensg)} ENSG IDs")

        # Map query symbols -> ENSG using combined mapping, check model coverage
        query_ensg_mapped = {symbol_to_ensg[s] for s in symbol_overlap if s in symbol_to_ensg}
        model_coverage = query_ensg_mapped & model_ensg
        print(f"Query genes that map to model's ENSG IDs: {len(model_coverage)}/{len(model_ensg)} ({100*len(model_coverage)/len(model_ensg):.1f}%)")

        if len(model_coverage) < len(model_ensg) * 0.5:
            print("\nWARNING: Low coverage (<50%)! Model-based mapping may not work well.")
        else:
            print(f"\nGood coverage! Model-based mapping should work.")

    # Add ensembl_id column to query var
    print("\nAdding ensembl_id column to query...")
    query.var['ensembl_id'] = [symbol_to_ensg.get(s, '') for s in query.var_names]
    n_mapped = sum(1 for e in query.var['ensembl_id'] if e)
    print(f"Mapped {n_mapped}/{query.n_vars} query genes to ENSG IDs")

    # Save updated query
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_path}...")
    query.write_h5ad(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
