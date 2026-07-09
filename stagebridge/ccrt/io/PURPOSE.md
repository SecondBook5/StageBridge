# `io/` — CCRT Safe Paths, Table IO, and Provenance

This is the design-lock description of `stagebridge/ccrt/io/`. No implementation
code exists yet; this document states the contract this package will satisfy,
not modules that are present today.

`io/` is the safe input/output plumbing of the framework. It provides safe path
handling, standardized table IO, and provenance manifests. Its purpose is to
prevent generated data from entering git, to prevent random pickle outputs, and
to prevent silent artifact drift. All reads and writes of standardized artifacts
are routed through `io/` so that outputs land only in allowed roots and every
artifact carries provenance. It is strictly system-agnostic and knows nothing
about disease biology.

## Owns

- Safe path resolution that confines writes to the allowed output roots.
- Standardized, contract-conformant table IO (reading/writing the eight adapter
  outputs and related artifacts).
- Provenance manifests recording how each artifact was produced.

## Does NOT

- MUST NOT know biology, and MUST NOT define model architecture or training.
- MUST NOT write to forbidden roots: `stagebridge/`, `tests/`, or
  `~/projects/StageBridge-ccrt/data/`.
- MUST NOT emit ad-hoc pickles or untracked, provenance-free artifacts.

```
ALLOWED output roots (outside git):
  $STAGEBRIDGE_DATA/raw/
  $STAGEBRIDGE_DATA/interim/
  $STAGEBRIDGE_DATA/processed/   (e.g. .../panin_ccrt/, .../luad_ccrt/)
  $STAGEBRIDGE_RESULTS/
```

See `docs/ccrt/TABLE_CONTRACT.md` and `docs/ccrt/DIRECTORY_PURPOSES.md`.
