#!/usr/bin/env python3
"""
Assign cell types to 'mixed' cells using dominant DestVI gamma.
Saves updated cells.parquet with cell_type_assigned column.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('/home/booka/projects/StageBridge/data')

# Load only needed columns to save memory
print('Loading cells.parquet...')
cells = pd.read_parquet(DATA_DIR / 'cells.parquet')
print(f'  {len(cells):,} cells')

gamma_cols = [f'gamma_{i}' for i in range(10)]

# Build mapping from gamma index to cell type using non-mixed cells
print('\nBuilding gamma -> cell type mapping...')
assigned = cells[cells['cell_type'] != 'mixed']

# For each cell type, find which gamma is most commonly dominant
gamma_to_ct = {}
for ct in assigned['cell_type'].unique():
    if pd.isna(ct):
        continue
    mask = assigned['cell_type'] == ct
    gamma_vals = assigned.loc[mask, gamma_cols].values
    dominant = np.argmax(gamma_vals, axis=1)
    most_common = pd.Series(dominant).value_counts().index[0]
    gamma_to_ct[most_common] = ct
    print(f'  gamma_{most_common} -> {ct}')

# Assign types to mixed cells
print('\nAssigning types to mixed cells...')
gamma_matrix = cells[gamma_cols].values
dominant_idx = np.argmax(gamma_matrix, axis=1)

# Create new column
cells['cell_type_assigned'] = cells['cell_type'].copy()
mixed_mask = cells['cell_type'] == 'mixed'
print(f'  {mixed_mask.sum():,} mixed cells to assign')

for idx, ct in gamma_to_ct.items():
    assign_mask = mixed_mask & (dominant_idx == idx)
    cells.loc[assign_mask, 'cell_type_assigned'] = ct
    print(f'    gamma_{idx} -> {ct}: {assign_mask.sum():,} cells')

# Handle any remaining (gamma index not in mapping)
remaining = cells['cell_type_assigned'] == 'mixed'
if remaining.sum() > 0:
    print(f'  {remaining.sum():,} cells still unassigned')
    # Assign based on dominant gamma name
    for idx in range(10):
        if idx not in gamma_to_ct:
            assign_mask = remaining & (dominant_idx == idx)
            if assign_mask.sum() > 0:
                cells.loc[assign_mask, 'cell_type_assigned'] = f'Type_{idx}'
                print(f'    gamma_{idx} -> Type_{idx}: {assign_mask.sum():,} cells')

print('\nFinal distribution:')
print(cells['cell_type_assigned'].value_counts())

# Save
out_path = DATA_DIR / 'cells_typed.parquet'
cells.to_parquet(out_path)
print(f'\nSaved: {out_path}')
