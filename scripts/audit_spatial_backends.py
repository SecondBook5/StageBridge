#!/usr/bin/env python3
"""
Audit script for spatial backend wiring and structure.

Checks:
1. Import structure is correct
2. No circular dependencies
3. All classes have required methods
4. Documentation is present
5. Test coverage exists
"""

import ast
import sys
from pathlib import Path
from collections import defaultdict


def check_imports(module_path: Path) -> dict:
    """Check imports in a Python module."""
    with open(module_path) as f:
        tree = ast.parse(f.read(), filename=str(module_path))

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")

    return {"path": module_path, "imports": imports}


def check_class_methods(module_path: Path, class_name: str) -> dict:
    """Check if a class has required methods."""
    with open(module_path) as f:
        tree = ast.parse(f.read(), filename=str(module_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            return {"class": class_name, "methods": methods, "path": module_path}

    return {"class": class_name, "methods": [], "path": module_path, "found": False}


def check_docstrings(module_path: Path) -> dict:
    """Check if module and classes have docstrings."""
    with open(module_path) as f:
        tree = ast.parse(f.read(), filename=str(module_path))

    module_doc = ast.get_docstring(tree)
    class_docs = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_docs[node.name] = ast.get_docstring(node) is not None

    return {
        "path": module_path,
        "module_docstring": module_doc is not None,
        "class_docstrings": class_docs,
    }


def audit_backend_structure():
    """Run full audit of spatial backends."""
    base_dir = Path(__file__).parent.parent / "stagebridge" / "spatial_mapping"
    test_dir = Path(__file__).parent.parent / "tests" / "spatial_mapping"

    print("=" * 70)
    print("Spatial Backend Structure Audit")
    print("=" * 70)

    # 1. Check backend wrappers exist
    print("\n1. Backend Wrappers:")
    backends = {
        "tangram_wrapper.py": "TangramBackend",
        "destvi_wrapper.py": "DestVIBackend",
        "tacco_wrapper.py": "TACCOBackend",
    }

    for module, class_name in backends.items():
        module_path = base_dir / module
        if module_path.exists():
            print(f"   ✓ {module} exists")
            methods = check_class_methods(module_path, class_name)
            if not methods.get("found", True):
                print(f"     ✗ Class {class_name} not found!")
            else:
                print(f"     Methods: {len(methods['methods'])} found")
        else:
            print(f"   ✗ {module} MISSING")

    # 2. Check new visualization methods in backends
    print("\n2. Advanced Visualization Methods:")

    # Tangram methods
    tangram_methods = check_class_methods(
        base_dir / "tangram_wrapper.py", "TangramBackend"
    )
    required_tangram = [
        "map",
        "plot_cell_type_spatial",
        "project_genes",
        "plot_projected_genes",
        "compute_spatial_statistics",
    ]
    for method in required_tangram:
        if method in tangram_methods["methods"]:
            print(f"   ✓ TangramBackend.{method}")
        else:
            print(f"   ✗ TangramBackend.{method} MISSING")

    # DestVI methods
    destvi_methods = check_class_methods(base_dir / "destvi_wrapper.py", "DestVIBackend")
    required_destvi = [
        "map",
        "get_gamma",
        "get_cell_type_specific_expression",
        "automatic_proportion_threshold",
        "filter_spots_by_celltype",
        "plot_cell_type_spatial",
        "explore_gamma_space",
    ]
    for method in required_destvi:
        if method in destvi_methods["methods"]:
            print(f"   ✓ DestVIBackend.{method}")
        else:
            print(f"   ✗ DestVIBackend.{method} MISSING")

    # 3. Check viz_utils module
    print("\n3. Visualization Utilities (viz_utils.py):")
    viz_utils_path = base_dir / "viz_utils.py"
    if viz_utils_path.exists():
        print(f"   ✓ viz_utils.py exists")
        with open(viz_utils_path) as f:
            content = f.read()
            viz_functions = [
                "plot_proportions_spatial",
                "plot_gamma_pca_spatial",
                "plot_projected_genes_spatial",
                "plot_proportion_distribution",
                "plot_proportion_heatmap",
                "plot_entropy_vs_sparsity",
                "plot_spatial_autocorrelation",
                "create_comprehensive_report",
            ]
            for func in viz_functions:
                if f"def {func}" in content:
                    print(f"     ✓ {func}")
                else:
                    print(f"     ✗ {func} MISSING")
    else:
        print(f"   ✗ viz_utils.py MISSING")

    # 4. Check __init__.py exports
    print("\n4. Package Exports (__init__.py):")
    init_path = base_dir / "__init__.py"
    if init_path.exists():
        with open(init_path) as f:
            content = f.read()

        exports_to_check = [
            "TangramBackend",
            "DestVIBackend",
            "TACCOBackend",
            "plot_proportions_spatial",
            "plot_gamma_pca_spatial",
            "create_comprehensive_report",
            "get_backend",
        ]

        for export in exports_to_check:
            if f'"{export}"' in content or f"'{export}'" in content:
                print(f"   ✓ {export} in __all__")
            else:
                print(f"   ? {export} (may not be in __all__)")

    # 5. Check docstrings
    print("\n5. Documentation:")
    for module, class_name in backends.items():
        module_path = base_dir / module
        if module_path.exists():
            docs = check_docstrings(module_path)
            if docs["module_docstring"]:
                print(f"   ✓ {module} has module docstring")
            else:
                print(f"   ✗ {module} missing module docstring")

            if class_name in docs["class_docstrings"]:
                if docs["class_docstrings"][class_name]:
                    print(f"   ✓ {class_name} has docstring")
                else:
                    print(f"   ✗ {class_name} missing docstring")

    # 6. Check test coverage
    print("\n6. Test Coverage:")
    test_files = list(test_dir.glob("test_*.py"))
    print(f"   Found {len(test_files)} test files:")
    for test_file in sorted(test_files):
        print(f"     - {test_file.name}")

    # Check for specific test file
    wrapper_test = test_dir / "test_wrappers.py"
    if wrapper_test.exists():
        print(f"\n   ✓ test_wrappers.py exists")
        with open(wrapper_test) as f:
            content = f.read()
            test_functions = content.count("def test_")
            print(f"     Contains {test_functions} test functions")
    else:
        print(f"\n   ✗ test_wrappers.py MISSING")

    # 7. Check for circular imports
    print("\n7. Import Structure:")
    viz_imports = check_imports(base_dir / "viz_utils.py")
    circular = False
    for imp in viz_imports["imports"]:
        if "viz_utils" in imp and "spatial_backends" in imp:
            print(f"   ⚠ Potential circular import: {imp}")
            circular = True

    if not circular:
        print("   ✓ No obvious circular imports detected")

    # 8. Check example/demo
    print("\n8. Examples/Demos:")
    examples_dir = Path(__file__).parent.parent / "examples"
    if examples_dir.exists():
        demo_files = list(examples_dir.glob("*spatial*viz*.py"))
        if demo_files:
            for demo in demo_files:
                print(f"   ✓ {demo.name}")
        else:
            print("   ✗ No spatial visualization demo found")
    else:
        print("   ✗ examples/ directory not found")

    # 9. Check pipeline integration
    print("\n9. Pipeline Integration:")
    pipeline_path = (
        Path(__file__).parent.parent
        / "stagebridge"
        / "pipelines"
        / "run_spatial_benchmark.py"
    )
    if pipeline_path.exists():
        with open(pipeline_path) as f:
            content = f.read()
            if "TangramBackend" in content:
                print("   ✓ TangramBackend imported in pipeline")
            if "DestVIBackend" in content:
                print("   ✓ DestVIBackend imported in pipeline")
            if "TACCOBackend" in content:
                print("   ✓ TACCOBackend imported in pipeline")

    print("\n" + "=" * 70)
    print("Audit Complete")
    print("=" * 70)


if __name__ == "__main__":
    audit_backend_structure()
