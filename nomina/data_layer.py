"""
NOMINA data layer — live-first, static-fallback reference data with provenance.

Design rules this module enforces:

1. **Every corpus carries provenance.** A screening result is only as defensible as the
   universe it was screened against, so `DataSnapshot` records source, endpoint, fetch
   timestamp and row counts, and that record travels into the run manifest. A number
   without a provenance record is not a result, it is an anecdote.

2. **Live is the primary path, static is the fallback, and the caller is always told
   which one ran.** Silent degradation from a 40,000-name live universe to a 2,000-name
   committed snapshot would quietly inflate every distinctiveness margin in the run.
   `DataSnapshot.mode` makes that visible; the notebook prints it.

3. **Only regulator-published or WHO-published sources.** openFDA (US FDA), the EMA
   medicines register (EU), and RxNorm (US NLM) are all primary-source, openly licensed
   and citable in a paper. Scraped aggregator sites are not used at any point.

4. **Network access is optional.** Every fetch is wrapped, bounded by a timeout, and
   degrades to the committed snapshot. A fresh clone with no internet still runs.
"""

from __future__ import annotations

import functools
import io
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

DATA_LAYER_VERSION = "2.0.0"

# --------------------------------------------------------------------------
# Where things live
# --------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = _HERE.parent
DATA_DIR = Path(os.environ.get("NOMINA_DATA_DIR", PACKAGE_ROOT / "data"))
CACHE_DIR = Path(os.environ.get("NOMINA_CACHE_DIR", PACKAGE_ROOT / ".nomina_cache"))

GITHUB_RAW_BASE = os.environ.get(
    "NOMINA_RAW_BASE",
    "https://raw.githubusercontent.com/vedanshshetty/Pharmaceutical-Name-Generation/production",
)

# Primary sources. Each is a regulator or WHO publication, not an aggregator.
OPENFDA_NDC = "https://api.fda.gov/drug/ndc.json"
OPENFDA_DRUGSFDA = "https://api.fda.gov/drug/drugsfda.json"
EMA_MEDICINES = (
    "https://www.ema.europa.eu/en/documents/other/"
    "medicines-output-medicines-report_en.xlsx"
)
RXNORM_ALLCONCEPTS = (
    "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=IN+BN"
)

DEFAULT_TIMEOUT = float(os.environ.get("NOMINA_HTTP_TIMEOUT", "30"))


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

@dataclass
class SourceRecord:
    """One upstream source that contributed rows to a snapshot."""
    name: str
    endpoint: str
    mode: str                      # 'live' | 'static' | 'cache' | 'failed'
    rows: int = 0
    fetched_at: Optional[str] = None
    note: str = ""


@dataclass
class DataSnapshot:
    """A reference corpus plus everything needed to cite it."""
    names: pd.DataFrame
    stems: pd.DataFrame
    sources: List[SourceRecord] = field(default_factory=list)
    built_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def mode(self) -> str:
        """'live' if any upstream regulator source was reached, else 'static'."""
        return "live" if any(s.mode == "live" for s in self.sources) else "static"

    @property
    def fingerprint(self) -> str:
        """Stable hash of the corpus contents. Keys the trained-artifact cache, so a
        corpus change automatically invalidates any model trained on the old one."""
        import hashlib
        h = hashlib.sha256()
        for col in ("generic_name", "brand_name"):
            if col in self.names.columns:
                for v in sorted(str(x) for x in self.names[col].dropna().unique()):
                    h.update(v.encode())
        for v in sorted(str(x) for x in self.stems["stem"]):
            h.update(v.encode())
        return h.hexdigest()[:16]

    def manifest(self) -> Dict[str, Any]:
        return {
            "data_layer_version": DATA_LAYER_VERSION,
            "built_at": self.built_at,
            "mode": self.mode,
            "fingerprint": self.fingerprint,
            "name_rows": int(len(self.names)),
            "unique_generic": int(self.names["generic_name"].nunique()) if "generic_name" in self.names else 0,
            "unique_brand": int(self.names["brand_name"].nunique()) if "brand_name" in self.names else 0,
            "stem_rows": int(len(self.stems)),
            "sources": [asdict(s) for s in self.sources],
        }

    def summary(self) -> str:
        lines = [f"NOMINA data snapshot  [{self.mode.upper()}]  fingerprint={self.fingerprint}"]
        for s in self.sources:
            lines.append(f"  {s.mode:7s} {s.name:18s} rows={s.rows:<7d} {s.note}")
        lines.append(f"  total unique names: {self.manifest()['unique_generic']} generic / "
                     f"{self.manifest()['unique_brand']} brand;  stems: {len(self.stems)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Normalisation (shared by both halves of the system)
# --------------------------------------------------------------------------

def normalize_name(name: Optional[str]) -> Optional[str]:
    """Public normaliser kept name-compatible with data_layer v1 so the original
    notebooks keep working against this module."""
    if name is None or isinstance(name, float):
        return None
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\- ]", "", s)
    return s or None


# --------------------------------------------------------------------------
# HTTP with cache
# --------------------------------------------------------------------------

def _cache_path(key: str, ext: str = "json") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:120]
    return CACHE_DIR / f"{safe}.{ext}"


def _http_get(url: str, params: Optional[Dict[str, Any]] = None,
              timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """Bare GET. Uses `requests` when present, urllib otherwise, so the package has one
    fewer hard dependency in a locked-down environment."""
    try:
        import requests
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": f"NOMINA/{DATA_LAYER_VERSION}"})
        r.raise_for_status()
        return r.content
    except ImportError:
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen
        full = url + ("?" + urlencode(params) if params else "")
        req = Request(full, headers={"User-Agent": f"NOMINA/{DATA_LAYER_VERSION}"})
        with urlopen(req, timeout=timeout) as fh:
            return fh.read()


# --------------------------------------------------------------------------
# Live fetchers — one per regulator
# --------------------------------------------------------------------------

def fetch_openfda_ndc(limit: int = 20000, page_size: int = 1000,
                      timeout: float = DEFAULT_TIMEOUT) -> pd.DataFrame:
    """US FDA National Drug Code directory via openFDA.

    Paginated with `skip`, which openFDA caps at 25000 without an API key; the default
    limit stays inside that ceiling so an unauthenticated run never 400s halfway through
    and leaves a half-built corpus behind.
    """
    rows: List[Dict[str, Any]] = []
    skip = 0
    while skip < limit:
        want = min(page_size, limit - skip)
        payload = json.loads(_http_get(OPENFDA_NDC,
                                       {"limit": want, "skip": skip},
                                       timeout=timeout))
        batch = payload.get("results", [])
        if not batch:
            break
        for r in batch:
            rows.append({
                "generic_name_raw": r.get("generic_name"),
                "brand_name_raw": r.get("brand_name"),
                "product_type": r.get("product_type"),
                "route": ", ".join(r.get("route", []) or []),
                "source": "openFDA:ndc",
            })
        skip += len(batch)
        if len(batch) < want:
            break
        time.sleep(0.15)          # courtesy rate limit against a public endpoint
    return pd.DataFrame(rows)


def fetch_rxnorm_names(timeout: float = DEFAULT_TIMEOUT) -> pd.DataFrame:
    """US NLM RxNorm ingredient (IN) and brand (BN) concepts.

    Included because RxNorm carries the *normalised* ingredient vocabulary rather than
    the packaging-level strings NDC carries, so it contributes clean single-token
    generic names that the NDC feed buries inside product descriptions.
    """
    payload = json.loads(_http_get(RXNORM_ALLCONCEPTS, timeout=timeout))
    concepts = (payload.get("minConceptGroup", {}) or {}).get("minConcept", []) or []
    rows = []
    for c in concepts:
        nm, tty = c.get("name"), c.get("tty")
        if not nm:
            continue
        rows.append({
            "generic_name_raw": nm if tty == "IN" else None,
            "brand_name_raw": nm if tty == "BN" else None,
            "product_type": "RXNORM_" + str(tty),
            "route": None,
            "source": "RxNorm:" + str(tty),
        })
    return pd.DataFrame(rows)


def fetch_ema_medicines(timeout: float = DEFAULT_TIMEOUT) -> pd.DataFrame:
    """European Medicines Agency medicines register.

    This is what makes the screen non-US. A name that is free in the US NDC universe can
    still collide with a centrally authorised EU product, and a tool that only screens
    FDA data would report a clean margin and be wrong for every market outside one.
    """
    raw = _http_get(EMA_MEDICINES, timeout=timeout)
    # The EMA export carries several banner rows above the real header; find it by
    # looking for the column the file has always exposed rather than hardcoding an
    # offset that breaks the next time they add a line to the banner.
    for header_row in range(0, 30):
        try:
            df = pd.read_excel(io.BytesIO(raw), header=header_row)
        except Exception:
            continue
        cols = {str(c).strip().lower(): c for c in df.columns}
        name_col = next((cols[k] for k in cols if "name of medicine" in k or k == "medicine name"), None)
        inn_col = next((cols[k] for k in cols
                        if "international non-proprietary" in k or "inn" == k.strip()
                        or "active substance" in k), None)
        if name_col is not None:
            out = pd.DataFrame({
                "generic_name_raw": df[inn_col] if inn_col is not None else None,
                "brand_name_raw": df[name_col],
            })
            out["product_type"] = "EMA_AUTHORISED"
            out["route"] = None
            out["source"] = "EMA:medicines"
            return out.dropna(how="all", subset=["generic_name_raw", "brand_name_raw"])
    raise ValueError("Could not locate the header row in the EMA medicines export.")


# --------------------------------------------------------------------------
# Static fallbacks
# --------------------------------------------------------------------------

def _read_local_or_raw(filename: str) -> tuple[pd.DataFrame, str]:
    """Local file first (fast, offline, deterministic), GitHub raw second (so a bare
    notebook with no clone still works)."""
    local = DATA_DIR / filename
    if local.exists():
        return pd.read_csv(local), f"local:{local}"
    return pd.read_csv(f"{GITHUB_RAW_BASE}/data/{filename}"), f"raw:{filename}"


def load_static_names() -> tuple[pd.DataFrame, str]:
    return _read_local_or_raw("existing_drug_names.csv")


def load_static_stems() -> tuple[pd.DataFrame, str]:
    """Expanded INN/USAN stem table, with the original 34-row seed as a last resort."""
    try:
        return _read_local_or_raw("inn_usan_stems.csv")
    except Exception:
        return _read_local_or_raw("usan_stems_seed.csv")


# --------------------------------------------------------------------------
# Snapshot assembly
# --------------------------------------------------------------------------

def _harmonise(df: pd.DataFrame) -> pd.DataFrame:
    """Every source lands in the same four-column shape the rest of the system expects."""
    out = pd.DataFrame({
        "generic_name_raw": df.get("generic_name_raw"),
        "brand_name_raw": df.get("brand_name_raw"),
        "product_type": df.get("product_type"),
        "route": df.get("route"),
        "source": df.get("source", "static"),
    })
    out["generic_name"] = out["generic_name_raw"].map(normalize_name)
    out["brand_name"] = out["brand_name_raw"].map(normalize_name)
    return out


OFFLINE = os.environ.get("NOMINA_OFFLINE", "").strip() not in ("", "0", "false", "False")


def build_snapshot(live: bool = True,
                   sources: Sequence[str] = ("openfda", "rxnorm", "ema"),
                   ndc_limit: int = 20000,
                   timeout: float = DEFAULT_TIMEOUT,
                   verbose: bool = True) -> DataSnapshot:
    """Assemble the reference universe.

    `live=True` attempts each regulator source in turn and unions whatever succeeds with
    the committed snapshot. Failures are recorded, never raised: a rate-limited openFDA
    on demo day must degrade the corpus, not kill the run. `live=False` forces the
    committed snapshot, which is what CI and every reproducibility test use.
    """
    if OFFLINE:
        # Hard offline switch. CI sets this so a red build always means our code broke,
        # never that a regulator's API had a bad afternoon.
        live = False
    records: List[SourceRecord] = []
    frames: List[pd.DataFrame] = []

    static_df, static_origin = load_static_names()
    frames.append(_harmonise(static_df.assign(source="snapshot")))
    records.append(SourceRecord("committed-snapshot", static_origin, "static",
                                rows=len(static_df),
                                note="baseline corpus committed to the repository"))

    if live:
        fetchers = {
            "openfda": ("openFDA:ndc", OPENFDA_NDC,
                        lambda: fetch_openfda_ndc(limit=ndc_limit, timeout=timeout)),
            "rxnorm": ("RxNorm:IN+BN", RXNORM_ALLCONCEPTS,
                       lambda: fetch_rxnorm_names(timeout=timeout)),
            "ema": ("EMA:medicines", EMA_MEDICINES,
                    lambda: fetch_ema_medicines(timeout=timeout)),
        }
        for key in sources:
            if key not in fetchers:
                continue
            label, endpoint, fn = fetchers[key]
            t0 = time.perf_counter()
            try:
                df = fn()
                frames.append(_harmonise(df))
                records.append(SourceRecord(
                    label, endpoint, "live", rows=len(df),
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    note=f"{time.perf_counter() - t0:.1f}s"))
                if verbose:
                    print(f"  [live]   {label}: {len(df)} rows")
            except Exception as exc:                      # noqa: BLE001 - degrade, never crash
                records.append(SourceRecord(label, endpoint, "failed", 0,
                                            note=f"{type(exc).__name__}: {exc}"[:160]))
                if verbose:
                    print(f"  [failed] {label}: {type(exc).__name__} — falling back")

    names = pd.concat(frames, ignore_index=True)
    names = names.drop_duplicates(subset=["generic_name", "brand_name"], keep="first")

    stems_df, stem_origin = load_static_stems()
    records.append(SourceRecord("INN/USAN stems", stem_origin, "static",
                                rows=len(stems_df),
                                note="curated from the WHO INN stem book and USAN list"))

    snap = DataSnapshot(names=names, stems=stems_df, sources=records)
    if verbose:
        print(snap.summary())
    return snap


@functools.lru_cache(maxsize=4)
def get_snapshot(live: bool = True, ndc_limit: int = 20000) -> DataSnapshot:
    """Process-wide memoised snapshot. Without this, `Verifier.from_data_layer` and
    `Generator.from_data_layer` each triggered an independent full corpus fetch."""
    return build_snapshot(live=live, ndc_limit=ndc_limit, verbose=False)


# --------------------------------------------------------------------------
# v1-compatible surface
# --------------------------------------------------------------------------
# The original notebooks call these four functions. They keep working, they just now
# resolve through the snapshot layer, so nothing downstream had to change to gain
# caching, live sourcing and provenance.

def load_existing_names(live: bool = False) -> pd.DataFrame:
    return get_snapshot(live=live).names


def load_usan_stems(live: bool = False) -> pd.DataFrame:
    return get_snapshot(live=live).stems


def all_existing_name_strings(live: bool = False) -> List[str]:
    df = get_snapshot(live=live).names
    return pd.concat([df["generic_name"], df["brand_name"]]).dropna().unique().tolist()


def stems_for_class(class_keyword: str, live: bool = False) -> List[tuple]:
    df = get_snapshot(live=live).stems
    m = df[df["meaning"].str.contains(class_keyword, case=False, na=False, regex=False)]
    return list(m.itertuples(index=False, name=None))


def search_classes(query: str = "", live: bool = False) -> pd.DataFrame:
    """Free-text search over the stem table — what the notebook's class picker reads."""
    df = get_snapshot(live=live).stems
    if not query:
        return df
    q = query.lower()
    mask = (df["meaning"].str.lower().str.contains(q, na=False, regex=False)
            | df["stem"].str.lower().str.contains(q, na=False, regex=False))
    return df[mask]
