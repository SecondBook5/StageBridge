#!/usr/bin/env python
"""Add ENSG IDs to query using HLCA's own mapping (guaranteed correct for model).

Usage:
    python scripts/add_ensembl_ids.py \
        --query $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
        --hlca $DATA/references/hlca/hlca_reference.h5ad \
        --output $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad

This script:
1. Loads HLCA reference to extract symbol -> ENSG mapping
2. Adds ensembl_id column to query.var
3. Saves updated query

The pipeline's model-based mapping will then auto-convert var_names to ENSG IDs.
"""
import argparse
from pathlib import Path

import anndata


def main():
    parser = argparse.ArgumentParser(description="Add ENSG IDs to query using HLCA mapping")
    parser.add_argument("--query", required=True, help="Path to query h5ad file")
    parser.add_argument("--hlca", required=True, help="Path to HLCA reference h5ad file")
    parser.add_argument("--output", required=True, help="Output path for updated query")
    parser.add_argument("--model-path", default=None, help="Optional: path to scANVI model to check coverage")
    args = parser.parse_args()

    # Load HLCA reference to get its symbol -> ENSG mapping
    print("Loading HLCA reference for gene mapping...")
    hlca = anndata.read_h5ad(args.hlca, backed='r')

    # Build symbol -> ENSG mapping from HLCA
    if 'ensembl_id' not in hlca.var.columns:
        raise ValueError("HLCA reference missing 'ensembl_id' column in var")

    symbol_to_ensg = dict(zip(hlca.var_names, hlca.var['ensembl_id']))
    print(f"HLCA mapping: {len(symbol_to_ensg)} genes (symbol -> ENSG)")
    print(f"Sample: {list(symbol_to_ensg.items())[:3]}")

    # Load query
    print("\nLoading query...")
    query = anndata.read_h5ad(args.query)
    query_symbols = set(query.var_names)
    print(f"Query has {len(query_symbols)} genes")

    # Check overlap between query symbols and HLCA symbols
    hlca_symbols = set(hlca.var_names)
    symbol_overlap = query_symbols & hlca_symbols
    print(f"Query symbols that match HLCA symbols: {len(symbol_overlap)}/{len(hlca_symbols)}")

    # Optionally check model coverage
    if args.model_path:
        import torch
        state = torch.load(f"{args.model_path}/model.pt", map_location='cpu', weights_only=False)
        model_ensg = set(state['var_names'])
        print(f"Model expects {len(model_ensg)} ENSG IDs")

        # Map query symbols -> ENSG using HLCA mapping, check model coverage
        query_ensg_mapped = {symbol_to_ensg[s] for s in symbol_overlap if s in symbol_to_ensg}
        model_coverage = query_ensg_mapped & model_ensg
        print(f"Query genes that map to model's ENSG IDs: {len(model_coverage)}/{len(model_ensg)}")

        if len(model_coverage) < 1500:
            print("\nWARNING: Low coverage! Model-based mapping may not work well.")
            print("Consider using k-NN fallback instead.")
        else:
            print(f"\nGood coverage ({len(model_coverage)/len(model_ensg):.1%})! Model-based mapping should work.")

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
