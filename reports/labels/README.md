# Label Repair Workflow

## Required inputs
- Active LUAD lesion metadata via the existing StageBridge data layer
- Existing lesion WES proxy features from `wes_features.parquet`
- Optional parse-only CNA, clonal, phylogeny, or pathology summaries

## Parse-only mode
- Set `labels.parse_only=true`
- Provide backend summary paths under `labels.inputs.*`

## External-tool mode
- Set `labels.parse_only=false`
- Provide executable names and command templates under `labels.external_tools.*`
- The wrappers will fail loudly if a requested executable is unavailable

## Refined labels
- `positive`, `negative`, `uncertain`, and `exclude` are derived by a transparent rule-based engine
- `progression_risk_score` is continuous and auditable from its component contributions

## Viability report
- Use `binary_classification` only when donor support exists for both classes
- Use `continuous_risk` when score diversity and donor coverage exist but binary support fails
- Use `descriptive_only` or `exclude` when support remains too weak
