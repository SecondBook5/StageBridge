"""
Dataset adapters for StageBridge.

Adapters provide dataset-specific implementations for:
- Raw data loading
- Metadata harmonization
- QC parameter defaults
- Export configuration

Usage:
    from stagebridge.data.adapters import LuadEvoAdapter, get_adapter

    adapter = get_adapter("luad_evo")
    adata = adapter.load_raw()
    adata = adapter.harmonize_metadata(adata)
"""

from stagebridge.data.adapters.base import DatasetAdapter

# Registry of available adapters
_ADAPTER_REGISTRY: dict[str, type["DatasetAdapter"]] = {}


def register_adapter(name: str, adapter_class: type["DatasetAdapter"]) -> None:
    """Register a dataset adapter.

    Parameters
    ----------
    name : str
        Adapter name.
    adapter_class : type
        Adapter class.
    """
    _ADAPTER_REGISTRY[name] = adapter_class


def get_adapter(name: str, **kwargs) -> "DatasetAdapter":
    """Get a dataset adapter by name.

    Parameters
    ----------
    name : str
        Adapter name.
    **kwargs
        Additional arguments for adapter initialization.

    Returns
    -------
    DatasetAdapter
        Instantiated adapter.
    """
    if name not in _ADAPTER_REGISTRY:
        raise KeyError(f"Unknown adapter: {name}. Available: {list(_ADAPTER_REGISTRY.keys())}")
    return _ADAPTER_REGISTRY[name](**kwargs)


def list_adapters() -> list[str]:
    """List available adapter names.

    Returns
    -------
    list[str]
        Adapter names.
    """
    return sorted(_ADAPTER_REGISTRY.keys())


__all__ = [
    "DatasetAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
