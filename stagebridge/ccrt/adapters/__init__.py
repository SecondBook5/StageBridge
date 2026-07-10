"""CCRT adapters — disease-specific source-to-CCRT translation.

Each subpackage (e.g. ``panin``) translates a verified external source project
into standardized CCRT tables + grammar ids. Adapters may know biology but must
never import model architecture, operators, transport, or training. Subpackages
are imported explicitly (``from stagebridge.ccrt.adapters.panin import ...``)
rather than eagerly here, so importing this namespace stays lightweight.
"""

from __future__ import annotations

__all__: list[str] = []
