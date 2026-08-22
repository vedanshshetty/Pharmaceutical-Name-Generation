"""
NOMINA shared data layer.
Both the generator notebook and the verifier notebook import this
(see Step 7 below) instead of loading CSVs directly.
This version reads from the public GitHub repo raw URLs, not local files,
so it works from a fresh Colab runtime with no setup beyond the URL below.
"""
import re
import pandas as pd

# EDIT THIS to your repo before sharing with your teammate:
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vedanshshetty/Pharmaceutical-Name-Generation/main"


def normalize_name(name):
    if name is None or (isinstance(name, float)):
        return None
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^a-z0-9\- ]", "", name)
    return name if name else None


def load_existing_names():
    """Cleaned corpus of real generic/brand drug names: generic_name, brand_name, product_type, route."""
    return pd.read_csv(f"{GITHUB_RAW_BASE}/existing_drug_names.csv")


def load_usan_stems():
    """USAN stem reference table: stem, meaning."""
    return pd.read_csv(f"{GITHUB_RAW_BASE}/usan_stems.csv")


def all_existing_name_strings():
    """Flat list of every known generic + brand name string — for the verifier's similarity check (Person B)."""
    df = load_existing_names()
    return pd.concat([df["generic_name"], df["brand_name"]]).dropna().unique().tolist()


def stems_for_class(class_keyword):
    """Look up stem(s) by class keyword, e.g. stems_for_class('beta-blocker'). Useful for Person A picking a stem."""
    df = load_usan_stems()
    matches = df[df["meaning"].str.contains(class_keyword, case=False, na=False)]
    return list(matches.itertuples(index=False, name=None))
