"""Search engine and indexing."""

__all__ = ["FaceSearchEngine", "build_gallery"]


def __getattr__(name: str):
    """Lazy import to avoid RuntimeWarning when running as __main__."""
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
