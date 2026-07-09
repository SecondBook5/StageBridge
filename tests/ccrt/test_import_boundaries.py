"""AST-based import-boundary and forbidden-term guardrails.

These tests statically parse the Python sources under ``stagebridge/ccrt`` (no
imports executed, so the heavy legacy package is never loaded) and assert:

1. ``contracts``, ``grammar``, ``io``, ``data``, and ``sender_context`` never
   import forbidden downstream CCRT packages (adapters/operators/transport/
   training/plotting/deconvolution/cli/evaluation).
2. Those packages only depend cross-package on the allowed core set
   (contracts/grammar/io/data/sender_context) plus the standard library.
3. Forbidden mechanism terms (world_token/ring_id/radial_bin/radius_bin/
   neighborhood_bin) appear as identifiers only in ``contracts/naming.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locate the CCRT source tree relative to this test file.
#   tests/ccrt/test_import_boundaries.py  ->  <repo>/stagebridge/ccrt
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CCRT_ROOT = REPO_ROOT / "stagebridge" / "ccrt"

CCRT_PKG_PREFIX = "stagebridge.ccrt"

# Packages implemented so far (the system-agnostic core slice we own).
IMPLEMENTED_PACKAGES = ("contracts", "grammar", "io", "data", "sender_context")

# Downstream / disease packages that the implemented packages must NEVER import.
FORBIDDEN_TARGETS = (
    "adapters",
    "operators",
    "transport",
    "training",
    "plotting",
    "deconvolution",
    "evaluation",
    "cli",
)

# Cross-package CCRT imports that ARE allowed from the implemented packages.
ALLOWED_CCRT_SUBPACKAGES = set(IMPLEMENTED_PACKAGES)

FORBIDDEN_MECHANISM_TERMS = (
    "world_token",
    "ring_id",
    "radial_bin",
    "radius_bin",
    "neighborhood_bin",
)


def _iter_source_files(root: Path):
    """Yield all .py files under ``root``."""
    return sorted(root.rglob("*.py"))


def _module_name_for(path: Path) -> str:
    """Dotted module name for a file under the repo (stagebridge.ccrt.x.y)."""
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts)


def _imported_modules(path: Path) -> list[str]:
    """Return the fully-qualified module targets imported by a source file.

    ``from . import x`` / ``from ..contracts import y`` are resolved relative to
    the file's own package so we can classify them as intra-CCRT imports.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    own_module = _module_name_for(path)
    own_parts = own_module.split(".")
    # The package containing this module (drop the module name itself).
    own_pkg_parts = own_parts[:-1]

    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import: resolve against the owning package.
                base_parts = own_pkg_parts[: len(own_pkg_parts) - (node.level - 1)]
                if node.module:
                    resolved = ".".join([*base_parts, node.module])
                else:
                    resolved = ".".join(base_parts)
                targets.append(resolved)
            elif node.module:
                targets.append(node.module)
    return targets


def _ccrt_subpackage_of(module: str) -> str | None:
    """If ``module`` is under stagebridge.ccrt, return its first subpackage."""
    if module == CCRT_PKG_PREFIX:
        return None
    prefix = CCRT_PKG_PREFIX + "."
    if module.startswith(prefix):
        remainder = module[len(prefix) :]
        return remainder.split(".", 1)[0]
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ccrt_root_exists():
    assert CCRT_ROOT.is_dir(), f"expected CCRT source at {CCRT_ROOT}"


def _implemented_source_files():
    files: list[Path] = []
    for pkg in IMPLEMENTED_PACKAGES:
        pkg_dir = CCRT_ROOT / pkg
        if pkg_dir.is_dir():
            files.extend(_iter_source_files(pkg_dir))
    return files


def test_implemented_packages_have_sources():
    files = _implemented_source_files()
    # At least the modules we authored should be present.
    assert len(files) >= 12, f"expected >=12 source files, found {len(files)}"


def test_no_forbidden_downstream_imports():
    """contracts/grammar/data must not import forbidden downstream packages."""
    violations: list[str] = []
    for path in _implemented_source_files():
        importer_pkg = _ccrt_subpackage_of(_module_name_for(path))
        for target in _imported_modules(path):
            sub = _ccrt_subpackage_of(target)
            if sub in FORBIDDEN_TARGETS:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} ({importer_pkg}) imports "
                    f"forbidden target '{sub}' via '{target}'"
                )
    assert not violations, "forbidden import(s) found:\n" + "\n".join(violations)


def test_cross_package_imports_are_in_allowed_set():
    """Any intra-CCRT import from the implemented packages must be allowed."""
    violations: list[str] = []
    for path in _implemented_source_files():
        for target in _imported_modules(path):
            sub = _ccrt_subpackage_of(target)
            if sub is None:
                continue  # not an intra-ccrt import (stdlib or the ccrt root)
            if sub not in ALLOWED_CCRT_SUBPACKAGES:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} imports disallowed "
                    f"ccrt subpackage '{sub}' via '{target}'"
                )
    assert not violations, "disallowed cross-package import(s):\n" + "\n".join(
        violations
    )


def test_forbidden_terms_only_in_naming_module():
    """Forbidden mechanism terms may appear as identifiers only in naming.py."""
    naming_path = (CCRT_ROOT / "contracts" / "naming.py").resolve()
    offenders: list[str] = []
    for path in _iter_source_files(CCRT_ROOT):
        if path.resolve() == naming_path:
            continue
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_MECHANISM_TERMS:
            if term in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains '{term}'")
    assert not offenders, (
        "forbidden mechanism term(s) outside contracts/naming.py:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("pkg", IMPLEMENTED_PACKAGES)
def test_each_implemented_package_is_importable_in_isolation(pkg):
    """Sanity: each package parses and has an __init__.py."""
    init = CCRT_ROOT / pkg / "__init__.py"
    assert init.is_file(), f"missing {init}"
    ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
