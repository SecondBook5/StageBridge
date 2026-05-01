#!/usr/bin/env python3
"""Create WES features parquet from annotated variants.

Generates per-sample mutation features with OncoKB actionability levels
for integration into StageBridge evolution branch.

Usage:
    python scripts/create_wes_features.py \
        --variants /path/to/annotated_variants.parquet \
        --output /path/to/wes_features.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# OncoKB levels for lung cancer driver mutations
ONCOKB_LEVELS = {
    # Level 1: FDA-approved, same tumor type
    ("EGFR", "L858R"): ("1", "Osimertinib, Erlotinib, Gefitinib, Afatinib"),
    ("EGFR", "exon19del"): ("1", "Osimertinib, Erlotinib, Gefitinib, Afatinib"),
    ("EGFR", "T790M"): ("1", "Osimertinib"),
    ("KRAS", "G12C"): ("1", "Sotorasib, Adagrasib"),
    ("BRAF", "V600E"): ("1", "Dabrafenib + Trametinib"),
    ("ALK", "fusion"): ("1", "Alectinib, Brigatinib, Lorlatinib"),
    ("ROS1", "fusion"): ("1", "Entrectinib, Crizotinib"),
    ("MET", "exon14skip"): ("1", "Capmatinib, Tepotinib"),
    ("RET", "fusion"): ("1", "Selpercatinib, Pralsetinib"),
    ("NTRK", "fusion"): ("1", "Larotrectinib, Entrectinib"),

    # Level 2: Standard care
    ("MET", "amplification"): ("2", "Capmatinib, Tepotinib"),
    ("ERBB2", "amplification"): ("2", "Trastuzumab deruxtecan"),

    # Level 4: Biological evidence only
    ("KRAS", "G12D"): ("4", ""),
    ("KRAS", "G12V"): ("4", ""),
    ("KRAS", "G12S"): ("4", ""),
    ("KRAS", "G13D"): ("4", ""),

    # Resistance mutations
    ("EGFR", "C797S"): ("R1", ""),
}


def compute_tmb(variants_df: pd.DataFrame, exome_mb: float = 38.0) -> float:
    """Compute tumor mutation burden (mutations per Mb)."""
    # Count coding mutations (simplified - use consequence if available)
    if 'consequence' in variants_df.columns:
        coding = variants_df[variants_df['consequence'].str.contains(
            'missense|nonsense|frameshift|splice', case=False, na=False
        )]
        return len(coding) / exome_mb
    return len(variants_df) / exome_mb


def create_wes_features(
    variants_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Create per-sample WES features from annotated variants.

    Args:
        variants_path: Path to annotated_variants.parquet
        output_path: Path to save wes_features.parquet

    Returns:
        DataFrame with WES features per sample
    """
    variants_path = Path(variants_path)
    output_path = Path(output_path)

    print(f"Loading variants from {variants_path}...")
    df = pd.read_parquet(variants_path)
    print(f"  {len(df):,} variants from {df['donor_id'].nunique()} patients")

    # Create sample_id from donor + stage
    df['sample_id'] = df['donor_id'] + '_' + df['stage']
    samples = df['sample_id'].unique()
    print(f"  {len(samples)} unique samples")

    records = []

    for sample_id in samples:
        sample_df = df[df['sample_id'] == sample_id]
        donor_id = sample_df['donor_id'].iloc[0]
        stage = sample_df['stage'].iloc[0]

        record = {
            'sample_id': sample_id,
            'donor_id': donor_id,
            'stage': stage,
        }

        # TMB
        record['tmb'] = compute_tmb(sample_df)

        # Binary mutation flags for key genes
        genes_to_check = ['KRAS', 'EGFR', 'TP53', 'STK11', 'KEAP1', 'SMAD4', 'BRAF',
                         'ALK', 'ROS1', 'MET', 'RET', 'ERBB2', 'PIK3CA', 'PTEN', 'NF1']

        for gene in genes_to_check:
            gene_lower = gene.lower()
            record[f'{gene_lower}_mut'] = int((sample_df['gene'] == gene).any())

        # Specific hotspot variants
        hotspots = sample_df[sample_df['is_hotspot'] == True]

        # EGFR specific
        record['egfr_L858R'] = int(((hotspots['gene'] == 'EGFR') & (hotspots['hotspot_type'] == 'L858R')).any())
        record['egfr_exon19del'] = int(((hotspots['gene'] == 'EGFR') & (hotspots['hotspot_type'] == 'exon19del')).any())
        record['egfr_T790M'] = int(((hotspots['gene'] == 'EGFR') & (hotspots['hotspot_type'] == 'T790M')).any())

        # KRAS specific
        record['kras_G12C'] = int(((hotspots['gene'] == 'KRAS') & (hotspots['hotspot_type'] == 'G12C')).any())
        record['kras_G12D'] = int(((hotspots['gene'] == 'KRAS') & (hotspots['hotspot_type'] == 'G12D')).any())
        record['kras_G12V'] = int(((hotspots['gene'] == 'KRAS') & (hotspots['hotspot_type'] == 'G12V')).any())
        record['kras_G12S'] = int(((hotspots['gene'] == 'KRAS') & (hotspots['hotspot_type'] == 'G12S')).any())

        # BRAF specific
        record['braf_V600E'] = int(((hotspots['gene'] == 'BRAF') & (hotspots['hotspot_type'] == 'V600E')).any())

        # OncoKB actionability
        oncokb_level = None
        therapies = []

        for _, row in hotspots.iterrows():
            key = (row['gene'], row['hotspot_type'])
            if key in ONCOKB_LEVELS:
                level, drugs = ONCOKB_LEVELS[key]
                if oncokb_level is None or level < oncokb_level:
                    oncokb_level = level
                if drugs:
                    therapies.extend(drugs.split(', '))

        record['oncokb_highest_level'] = oncokb_level
        record['has_level1_mutation'] = int(oncokb_level == '1')
        record['has_actionable_mutation'] = int(oncokb_level in ('1', '2', '3A', '3B'))
        record['recommended_therapies'] = ', '.join(sorted(set(therapies))) if therapies else None

        # Mutation co-occurrence patterns
        record['egfr_kras_comut'] = int(record['egfr_mut'] and record['kras_mut'])
        record['tp53_comut'] = int(record['tp53_mut'] and (record['egfr_mut'] or record['kras_mut']))
        record['stk11_keap1_comut'] = int(record['stk11_mut'] and record['keap1_mut'])

        records.append(record)

    result = pd.DataFrame(records)

    # Summary stats
    print(f"\n=== WES Features Summary ===")
    print(f"Samples: {len(result)}")
    print(f"TMB: mean={result['tmb'].mean():.1f}, median={result['tmb'].median():.1f}")
    print(f"\nMutation frequencies:")
    for gene in ['kras', 'egfr', 'tp53', 'stk11']:
        col = f'{gene}_mut'
        pct = result[col].mean() * 100
        print(f"  {gene.upper()}: {pct:.1f}%")

    print(f"\nHotspot frequencies:")
    for col in ['egfr_L858R', 'egfr_exon19del', 'kras_G12C', 'kras_G12V']:
        pct = result[col].mean() * 100
        print(f"  {col}: {pct:.1f}%")

    print(f"\nOncoKB actionability:")
    print(f"  Level 1 mutations: {result['has_level1_mutation'].sum()} samples ({result['has_level1_mutation'].mean()*100:.1f}%)")
    print(f"  Any actionable: {result['has_actionable_mutation'].sum()} samples ({result['has_actionable_mutation'].mean()*100:.1f}%)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WES features from annotated variants")
    parser.add_argument("--variants", required=True, help="Path to annotated_variants.parquet")
    parser.add_argument("--output", required=True, help="Output path for wes_features.parquet")
    args = parser.parse_args()

    create_wes_features(args.variants, args.output)
