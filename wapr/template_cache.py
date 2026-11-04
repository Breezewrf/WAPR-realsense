"""Select validated CAD template caches without replacing incompatible artifacts."""

import hashlib
import json
from pathlib import Path


def _bank_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path, identity):
    try:
        meshes = json.loads(Path(str(path) + ".meshes.json").read_text())
        bank = json.loads(Path(str(path) + ".json").read_text())
        return (meshes == identity and isinstance(bank, dict)
                and bank.get("bank_sha256") == _bank_digest(path)
                and bank.get("weight_sha256") == identity["weight_sha256"]
                and bank.get("source_sha256") == identity["source_sha256"])
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _occupied(path):
    return any(Path(str(path) + suffix).exists() for suffix in ("", ".json", ".meshes.json"))


def select_cad_cache(cache_path, identity, policy="error"):
    """Return (path, cache_hit); versioned paths preserve old or corrupt caches.

    The identity includes mesh/texture, weights, code, shaders and render recipe.
    It intentionally excludes detection options such as SAM2 and score thresholds.
    """
    if policy not in ("error", "version"):
        raise ValueError("cache_policy must be error or version")
    requested = Path(cache_path).absolute()
    if _matches(requested, identity):
        return str(requested), True
    if not _occupied(requested):
        return str(requested), False
    if policy == "error":
        raise ValueError("Existing CAD cache has different inputs/code; choose a new cache path")
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    stem = requested.stem + "." + fingerprint
    candidate = requested.with_name(stem + requested.suffix)
    repair = 0
    hit = _matches(candidate, identity)
    while not hit and _occupied(candidate):
        # Preserve corrupt versions and find an existing repaired copy on later runs.
        repair += 1
        candidate = requested.with_name(stem + "." + str(repair) + requested.suffix)
        hit = _matches(candidate, identity)
    print("DET2D_CAD_CACHE_CHANGED", {"requested": str(requested), "selected": str(candidate),
                                      "cache_hit": hit}, flush=True)
    return str(candidate), hit
