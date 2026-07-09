"""StageBridge / CCRT — Context-Residual Transport.

A grammar-conditioned neural transport framework for estimating how typed
local sender-context signals modify receiver-cell drift, growth, and
regulatory state along biological transition edges.

CCRT is unified at the *grammar* level, not the cell-type level. Different
biological systems (LUAD, PanIN, future viral systems) express their dynamics
through the same transition grammar while using system-specific vocabularies.

This package is currently in the architecture-lock phase. See
``docs/ccrt/ARCHITECTURE_LOCK.md`` and the companion contract documents for the
locked design. Implementation modules are intentionally absent until the
contracts are stable.
"""

__all__: list[str] = []
