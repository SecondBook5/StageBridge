# ADR 005 Git Is Rollback

Decision: destructive cleanup is archived in git via tag and branch rather than via `legacy/` folders.

## Context

During development, code is written, tested, and sometimes replaced. The question is how to handle replaced code:

1. **Active-tree archival** — Move old code to `legacy/`, `archive/`, or `deprecated/` directories. The old code remains visible in the file tree.
2. **Git-history archival** — Delete old code from the active tree. Tag the commit before deletion. The old code is recoverable via `git checkout <tag>`.

## Decision

Git history is the archive. Old code is deleted, not moved to legacy folders.

## Rationale

- Legacy folders accumulate and create confusion about what is active
- Developers waste time reading legacy code they think might be relevant
- Import paths pointing to legacy directories are a maintenance burden
- Git tags provide a clean rollback mechanism with no active-tree clutter
- Any commit can be checked out to recover any historical state

## Consequences

- Before major deletions, tag the commit (e.g., `archive/pre-v1-cleanup`)
- Do not create `legacy/`, `archive/`, `old/`, or `deprecated/` directories
- If you need old code, check the git history
- Important results must reference their git commit so the producing code is always recoverable
