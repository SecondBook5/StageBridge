"""AST-based import-boundary and forbidden-term guardrails.

These tests statically parse the Python sources under ``stagebridge/ccrt`` (no
imports executed, so the heavy legacy package is never loaded) and assert:

1. ``contracts``, ``grammar``, ``io``, ``data``, ``sender_context``,
   ``operators``, ``representations``, and ``transport`` never import forbidden
   downstream CCRT packages (adapters/training/plotting/deconvolution/cli/
   evaluation).
2. Those packages only depend cross-package on the allowed core set plus the
   standard library.
3. Forbidden mechanism terms (world_token/ring_id/radial_bin/radius_bin/
   neighborhood_bin) appear as identifiers only in ``contracts/naming.py``.
4. Optional OT dependencies (geomloss/ot) are never imported at module top level
   in transport source — only lazily inside adapter operations.
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
IMPLEMENTED_PACKAGES = (
    "contracts",
    "grammar",
    "io",
    "data",
    "sender_context",
    "operators",
    "representations",
    "transport",
    "training",
    "synthetic",
)

# Downstream / disease packages that NO implemented package may import.
FORBIDDEN_TARGETS = (
    "adapters",
    "plotting",
    "deconvolution",
    "evaluation",
    "cli",
)

# Optional external OT dependencies that must never be imported at module top
# level anywhere in transport source (they are lazy-imported inside operations).
OPTIONAL_OT_MODULES = ("geomloss", "ot")

# Per-package allowed intra-CCRT dependency set. A package may only import the
# CCRT subpackages listed for it (plus stdlib/torch, which are not ccrt
# subpackages). This encodes the one-way dependency law:
#   * data may know contracts + grammar (for the index registry), nothing else;
#   * training sits at the top and may compose the whole core;
#   * nothing (outside training) may import training.
ALLOWED_INTRA_CCRT = {
    "contracts": set(),
    "grammar": {"contracts"},
    "io": {"contracts"},
    # data validates records through io (Milestone 2) and qualifies grammar ids
    # (Milestone 6); both are acyclic downstream-of-data-free dependencies.
    "data": {"contracts", "grammar", "io"},
    "representations": {"contracts"},
    "sender_context": {"contracts"},
    "operators": {"contracts", "grammar", "representations", "sender_context"},
    "transport": {"contracts", "representations"},
    "training": {
        "contracts",
        "grammar",
        "data",
        "sender_context",
        "operators",
        "representations",
        "transport",
    },
    # synthetic (Milestone 7) composes the whole core to teach + benchmark; it
    # sits above training and is imported by nothing.
    "synthetic": {
        "contracts",
        "grammar",
        "data",
        "sender_context",
        "operators",
        "representations",
        "transport",
        "training",
    },
}

# Teacher-independence: these synthetic modules must NOT import student model
# packages (the teacher is coded from explicit math, not from the CCRT model).
TEACHER_INDEPENDENT_FILES = (
    "synthetic/mechanisms.py",
    "synthetic/ground_truth.py",
)
STUDENT_ONLY_PACKAGES = ("sender_context", "operators", "transport", "training")
PROHIBITED_STUDENT_CLASS_NAMES = (
    "ContextResidualTransportOperator",
    "TypedSenderContextAttention",
    "RegulatoryBottleneck",
    "DriftHead",
    "GrowthHead",
    "SemanticTransportLoss",
    "CCRTTrainer",
)

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


def test_cross_package_imports_obey_per_package_rules():
    """Each package may import only its allowed intra-CCRT subpackages."""
    violations: list[str] = []
    for path in _implemented_source_files():
        importer = _ccrt_subpackage_of(_module_name_for(path))
        if importer is None:
            continue
        allowed = ALLOWED_INTRA_CCRT.get(importer, set())
        for target in _imported_modules(path):
            sub = _ccrt_subpackage_of(target)
            if sub is None or sub == importer:
                continue  # stdlib/torch/ccrt-root, or intra-package self-import
            if sub not in allowed:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} ({importer}) imports "
                    f"disallowed ccrt subpackage '{sub}' via '{target}'"
                )
    assert not violations, "disallowed cross-package import(s):\n" + "\n".join(
        violations
    )


def test_only_synthetic_imports_training():
    """Only the top layers (training itself, synthetic) may import training.

    training composes the model core; synthetic composes training to benchmark.
    No other package may reach into training.
    """
    allowed_importers = {"training", "synthetic"}
    violations: list[str] = []
    for path in _implemented_source_files():
        importer = _ccrt_subpackage_of(_module_name_for(path))
        if importer in allowed_importers:
            continue
        for target in _imported_modules(path):
            if _ccrt_subpackage_of(target) == "training":
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} ({importer}) imports training "
                    f"via '{target}'"
                )
    assert not violations, "disallowed import of training:\n" + "\n".join(violations)


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


def _top_level_import_names(path: Path) -> list[str]:
    """Return module names imported at the *top level* of a file (module scope).

    Imports nested inside function/class bodies are excluded, so lazy imports do
    not count.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in tree.body:  # only module-level statements
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def test_transport_does_not_top_level_import_optional_ot():
    """Optional OT backends must be lazy — never module-top-level imports.

    This is environment-independent: it inspects the source, not sys.modules
    (which the legacy stagebridge package may pollute).
    """
    transport_dir = CCRT_ROOT / "transport"
    offenders: list[str] = []
    for path in _iter_source_files(transport_dir):
        for name in _top_level_import_names(path):
            root = name.split(".", 1)[0]
            if root in OPTIONAL_OT_MODULES:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)} top-level imports '{name}'"
                )
    assert not offenders, (
        "optional OT dependency imported at module top level:\n"
        + "\n".join(offenders)
    )


def test_teacher_modules_do_not_import_student_packages():
    """The synthetic teacher must be independent from the CCRT student model.

    mechanisms.py and ground_truth.py must not import sender_context / operators
    / transport / training (anywhere, top-level or nested).
    """
    offenders: list[str] = []
    for rel in TEACHER_INDEPENDENT_FILES:
        path = CCRT_ROOT / rel
        if not path.is_file():
            continue
        for target in _imported_modules(path):
            sub = _ccrt_subpackage_of(target)
            if sub in STUDENT_ONLY_PACKAGES:
                offenders.append(f"{rel} imports student package '{sub}' via '{target}'")
    assert not offenders, "teacher independence violated:\n" + "\n".join(offenders)


def test_teacher_ground_truth_has_no_student_class_names():
    """ground_truth.py / mechanisms.py must not reference student class names."""
    offenders: list[str] = []
    for rel in TEACHER_INDEPENDENT_FILES:
        path = CCRT_ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for cls in PROHIBITED_STUDENT_CLASS_NAMES:
            if cls in text:
                offenders.append(f"{rel} references student class '{cls}'")
    assert not offenders, "student class name in teacher source:\n" + "\n".join(offenders)


def test_nothing_imports_synthetic():
    """No non-synthetic package may import the synthetic benchmark package."""
    violations: list[str] = []
    for path in _implemented_source_files():
        importer = _ccrt_subpackage_of(_module_name_for(path))
        if importer == "synthetic":
            continue
        for target in _imported_modules(path):
            if _ccrt_subpackage_of(target) == "synthetic":
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} ({importer}) imports synthetic "
                    f"via '{target}'"
                )
    assert not violations, "upstream import of synthetic:\n" + "\n".join(violations)
