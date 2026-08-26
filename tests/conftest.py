"""Shared fixtures. Everything is offline and deterministic by construction."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("NOMINA_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def system():
    from nomina import build_system
    from nomina.verifier import VerifierConfig
    return build_system(live=False, use_llm=False, use_artifacts=False,
                        verifier_config=VerifierConfig(stem_aware_similarity=True))
