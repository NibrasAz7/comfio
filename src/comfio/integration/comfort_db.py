"""ASHRAE Global Thermal Comfort Database II loader.

Downloads the pinned v2.1.0 release from the CenterForTheBuiltEnvironment
``ashrae-db-II`` repository, verifies SHA-256 checksums, caches the files
locally, and returns measurements left-joined with building metadata on
``building_id``.

Caching uses ``$COMFIO_CACHE_DIR/ashrae-db2`` when the environment
variable is set, else ``~/.cache/comfio/ashrae-db2`` (created on Windows
too). Every file is sha256-verified before it is trusted — cached or
freshly downloaded. A corrupt cache entry is deleted and re-fetched once;
if the replacement also fails (or there is no network),
:class:`ComfortDBUnavailable` is raised and no partial ``.tmp`` file is
left behind.

Data licence: CC0-1.0 (Dryad doi:10.6078/D1F671).
Citation: Földváry Ličina et al. 2018, doi:10.1016/j.buildenv.2018.06.022.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

DB2_COMMIT = "c527d6115ebadb4725a4e1406b2439a7a1ffd214"
DB2_BASE = (
    f"https://raw.githubusercontent.com/CenterForTheBuiltEnvironment/ashrae-db-II/{DB2_COMMIT}/"
)
DB2_FILES = {
    "measurements": (
        "v2.1.0/db_measurements_v2.1.0.csv.gz",
        "9722be588ec3e779de21679513711e2d980138359a1b44bb9775e9499bc68380",
    ),
    "metadata": (
        "v2.1.0/db_metadata.csv",
        "af6770a2b469f079cd172bd6632d1ae726a3a663515126816ff6d2d8f51bfc10",
    ),
}
DB2_CACHE_ENV = "COMFIO_CACHE_DIR"
DB2_CITATION = (
    "Földváry Ličina et al. (2018), Development of the ASHRAE Global "
    "Thermal Comfort Database II, Building and Environment 142:502–512, "
    "doi:10.1016/j.buildenv.2018.06.022"
)

__all__ = [
    "DB2_BASE",
    "DB2_CACHE_ENV",
    "DB2_CITATION",
    "DB2_FILES",
    "ComfortDBUnavailable",
    "load_ashrae_db2",
]


class ComfortDBUnavailable(RuntimeError):  # noqa: N818 — public API name
    """The comfort database could not be loaded from cache or downloaded."""


def _cache_root(cache_dir: str | Path | None) -> Path:
    """Resolve the directory holding the DB2 files."""
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get(DB2_CACHE_ENV)
    root = Path(env) if env else Path.home() / ".cache" / "comfio"
    return root / "ashrae-db2"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str) -> None:
    """Fetch ``url`` to ``<dest>.tmp``, verify, then atomically rename."""
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise ComfortDBUnavailable(f"Could not download {url}: {exc}") from exc
    digest = _sha256(tmp)
    if digest != sha256:
        tmp.unlink(missing_ok=True)
        raise ComfortDBUnavailable(
            f"Downloaded {url} failed sha256 verification "
            f"(expected {sha256[:12]}…, got {digest[:12]}…); file discarded."
        )
    os.replace(tmp, dest)


def _ensure(url_path: str, sha256: str, cache: Path, download: bool) -> Path:
    """Return a sha256-verified local copy of one DB2 file."""
    dest = cache / Path(url_path).name
    if dest.exists():
        if _sha256(dest) == sha256:
            return dest
        dest.unlink()  # corrupt cache entry — refetch once below
    if not download:
        raise ComfortDBUnavailable(
            f"{dest.name} is not in the cache at {cache} and download=False was requested."
        )
    cache.mkdir(parents=True, exist_ok=True)
    _download(DB2_BASE + url_path, dest, sha256)
    return dest


def load_ashrae_db2(cache_dir: str | Path | None = None, *, download: bool = True) -> pd.DataFrame:
    """Load ASHRAE Global Thermal Comfort DB II as one merged frame.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory holding the downloaded files. Defaults to
        ``$COMFIO_CACHE_DIR/ashrae-db2`` or ``~/.cache/comfio/ashrae-db2``.
    download : bool, default True
        When False, only the verified cache is used — never the network.

    Returns
    -------
    pd.DataFrame
        Measurements left-joined with building metadata on ``building_id``
        (~109k rows, v2.1.0).

    Raises
    ------
    ComfortDBUnavailable
        No valid cached copy and the download failed (offline, checksum
        mismatch, or write error).
    """
    cache = _cache_root(cache_dir)
    paths = {key: _ensure(rel, sha, cache, download) for key, (rel, sha) in DB2_FILES.items()}
    measurements = pd.read_csv(paths["measurements"])
    metadata = pd.read_csv(paths["metadata"])
    return measurements.merge(metadata, on="building_id", how="left")
