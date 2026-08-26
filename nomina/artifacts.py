"""
NOMINA artifact store — build once, reuse everywhere, publish to the repository.

What gets persisted and why
---------------------------
Not everything slow is worth caching, and not everything worth caching is slow. Timing
the cold path against the committed corpus:

    screening corpus      0.03s      training corpus     0.05s
    n-gram fit            0.01s      substring index     0.02s
    Verifier build        0.08s      LIVE CORPUS FETCH   20-60s

So the honest answer is that the expensive thing is the *network*, not the arithmetic.
The artifact store therefore persists three classes of object, each for a different
reason:

1. **The assembled corpus snapshot.** Purely a speed decision: it removes a 20-60 second
   multi-regulator fetch from every run, and it means a rate-limited openFDA or an EMA
   outage cannot change your results mid-experiment.

2. **Trained models** (character n-grams, shape references). Cheap to refit, but
   persisting them is a *reproducibility* decision rather than a speed one. A published
   result should be re-derivable from a committed artifact, not from whatever the model
   happened to fit that afternoon.

3. **The manifest.** Fingerprint, source records, config, package version, git SHA.
   This is what lets a reviewer establish that two runs are comparable.

Cache invalidation is by content, not by time. Every artifact is keyed on the corpus
fingerprint plus the relevant config hash, so changing the stem table or the n-gram
order produces a different key and the stale model simply is not found. There is no
manual "remember to clear the cache" step, because that step is always forgotten.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ARTIFACT_FORMAT_VERSION = "1.0"

_HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = _HERE.parent
ARTIFACT_DIR = Path(os.environ.get("NOMINA_ARTIFACT_DIR", PACKAGE_ROOT / "artifacts"))

REMOTE_BASE = os.environ.get(
    "NOMINA_ARTIFACT_REMOTE",
    "https://raw.githubusercontent.com/vedanshshetty/Pharmaceutical-Name-Generation/production/artifacts",
)


def _hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def key_for(kind: str, fingerprint: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Content-addressed artifact key: kind + corpus fingerprint + config hash."""
    return f"{kind}__{fingerprint}__{_hash(config or {})}"


# ===========================================================================
# Store
# ===========================================================================

class ArtifactStore:
    """Local-first artifact cache with an optional published remote.

    Resolution order on read is local, then remote, then miss. Writes are always local;
    publishing to the repository is an explicit, human-run step, because silently
    committing generated binaries on every run is how repositories rot.
    """

    def __init__(self, directory: Path = ARTIFACT_DIR, remote_base: str = REMOTE_BASE,
                 allow_remote: bool = True):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.remote_base = remote_base
        self.allow_remote = allow_remote
        self.log: List[str] = []

    # -- low level ---------------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json.gz"

    def _write(self, key: str, payload: Dict[str, Any]) -> Path:
        path = self._path(key)
        payload = {"_format": ARTIFACT_FORMAT_VERSION,
                   "_written_at": datetime.now(timezone.utc).isoformat(),
                   **payload}
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    def _read_local(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:                                   # noqa: BLE001
            path.unlink(missing_ok=True)                    # corrupt cache self-heals
            return None

    def _read_remote(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.allow_remote:
            return None
        try:
            from .data_layer import _http_get
            raw = _http_get(f"{self.remote_base}/{key}.json.gz", timeout=20)
            payload = json.loads(gzip.decompress(raw).decode())
            with open(self._path(key), "wb") as fh:         # promote into the local cache
                fh.write(raw)
            return payload
        except Exception:                                   # noqa: BLE001
            return None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        hit = self._read_local(key)
        if hit is not None:
            self.log.append(f"local hit   {key}")
            return hit
        hit = self._read_remote(key)
        self.log.append(("remote hit  " if hit else "miss        ") + key)
        return hit

    def put(self, key: str, payload: Dict[str, Any]) -> Path:
        self.log.append(f"write       {key}")
        return self._write(key, payload)

    # -- the load-or-build pattern every caller uses -----------------------
    def load_or_build(self, key: str, build: Callable[[], Any],
                      dump: Callable[[Any], Dict[str, Any]],
                      load: Callable[[Dict[str, Any]], Any]) -> Any:
        """Return a deserialised artifact, building and caching it on a miss.

        Deserialisation failures fall through to a rebuild rather than raising: an
        artifact written by an older format version must never be able to break a run,
        it should just cost one refit.
        """
        payload = self.get(key)
        if payload is not None:
            try:
                return load(payload)
            except Exception:                               # noqa: BLE001
                self.log.append(f"stale       {key}")
        obj = build()
        try:
            self.put(key, dump(obj))
        except Exception as exc:                            # noqa: BLE001
            self.log.append(f"write-fail  {key}: {exc}")
        return obj

    # -- publishing --------------------------------------------------------
    def publish(self, keys: Optional[List[str]] = None,
                dest: Path = PACKAGE_ROOT / "artifacts",
                git_add: bool = True) -> List[Path]:
        """Stage artifacts for commit so a fresh clone starts warm.

        This deliberately stops at `git add`. It does not commit and it does not push:
        a tool that pushes to a user's repository as a side effect of a training run is
        a tool that will eventually push something unwanted.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        keys = keys or [p.name.replace(".json.gz", "") for p in self.dir.glob("*.json.gz")]
        staged: List[Path] = []
        for k in keys:
            src = self._path(k)
            if not src.exists():
                continue
            tgt = dest / src.name
            if src.resolve() != tgt.resolve():
                shutil.copy2(src, tgt)
            staged.append(tgt)
        if git_add and staged:
            try:
                subprocess.run(["git", "add", *[str(p) for p in staged]],
                               cwd=PACKAGE_ROOT, check=False,
                               capture_output=True, timeout=30)
            except Exception:                               # noqa: BLE001
                pass
        return staged

    def summary(self) -> str:
        files = sorted(self.dir.glob("*.json.gz"))
        total = sum(f.stat().st_size for f in files)
        lines = [f"Artifact store: {self.dir}  ({len(files)} artifacts, {total/1024:.0f} KB)"]
        lines += [f"  {f.name}  {f.stat().st_size/1024:.0f} KB" for f in files[:12]]
        lines += ["", "Access log:"] + [f"  {e}" for e in self.log[-12:]]
        return "\n".join(lines)


# ===========================================================================
# Serialisers for the specific objects we persist
# ===========================================================================

def dump_ngram(model) -> Dict[str, Any]:
    """Character n-gram model -> JSON.

    Contexts are tuples of characters, which JSON cannot key, so they are joined with a
    unit separator that cannot occur inside a folded name (letters and the two boundary
    markers only). The reference log-probability vector is kept because `typicality()`
    is a percentile against it, and recomputing that needs the training corpus.
    """
    return {
        "kind": "char_ngram",
        "order": model.order,
        "k": model.k,
        "n_training": model.n_training,
        "vocab": list(model.vocab),
        "char_freq": dict(model._char_freq),
        "start_contexts": ["\x1f".join(c) for c in model.start_contexts],
        "successors": {"\x1f".join(ctx): dict(cnt)
                       for ctx, cnt in model.successors.items()},
        "reference_logprobs": [round(x, 6) for x in model._reference_logprobs],
    }


def load_ngram(payload: Dict[str, Any], cls):
    """Rehydrate without refitting.

    `__new__` bypasses the constructor deliberately: the constructor's job is to count
    over a training corpus that, at load time, we no longer hold in memory. Rebuilding
    it just to throw it away would defeat the point of persisting the model.
    """
    from collections import Counter
    m = cls.__new__(cls)
    m.order = payload["order"]
    m.k = payload["k"]
    m.n_training = payload.get("n_training", 0)
    m.vocab = list(payload["vocab"])
    m._char_freq = Counter(payload.get("char_freq", {}))
    m.start_contexts = [tuple(c.split("\x1f")) if c else tuple()
                        for c in payload.get("start_contexts", [])]
    m.successors = {tuple(ctx.split("\x1f")) if ctx else tuple(): Counter(cnt)
                    for ctx, cnt in payload["successors"].items()}
    m._reference_logprobs = payload.get("reference_logprobs") or [0.0]
    return m


def dump_shape(ref) -> Dict[str, Any]:
    return {"kind": "shape_reference", **asdict(ref)}


def load_shape(payload: Dict[str, Any], cls):
    return cls(mean_len=payload["mean_len"], std_len=payload["std_len"],
               mean_syl=payload["mean_syl"], std_syl=payload["std_syl"],
               n=payload["n"])


def dump_snapshot(snap) -> Dict[str, Any]:
    return {
        "kind": "data_snapshot",
        "manifest": snap.manifest(),
        "names": snap.names.to_dict(orient="list"),
        "stems": snap.stems.to_dict(orient="list"),
    }


def load_snapshot(payload: Dict[str, Any], cls, source_cls):
    import pandas as pd
    man = payload["manifest"]
    snap = cls(names=pd.DataFrame(payload["names"]),
               stems=pd.DataFrame(payload["stems"]),
               sources=[source_cls(**s) for s in man.get("sources", [])],
               built_at=man.get("built_at", ""))
    return snap
