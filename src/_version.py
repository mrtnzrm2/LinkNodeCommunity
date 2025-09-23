from __future__ import annotations

def get_version() -> str:
    # 1) Prefer the version baked into package metadata (sdist/wheel)
    try:
        from importlib.metadata import version, PackageNotFoundError  # Py>=3.8
    except Exception:  # pragma: no cover
        try:
            # Py<3.8 backport
            from importlib_metadata import version, PackageNotFoundError  # type: ignore
        except Exception:  # last resort
            version = None  # type: ignore
            PackageNotFoundError = Exception  # type: ignore

    if version is not None:
        try:
            return version("LinkNodeCommunity")
        except PackageNotFoundError:
            pass

    # 2) Fallback to a dev marker (useful in editable installs without SCM metadata)
    return "0+unknown"