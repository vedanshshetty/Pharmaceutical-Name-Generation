"""Shared fixtures. Everything is offline and deterministic by construction."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("PHARMA_NAME_GEN_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def system():
    from pharma_name_gen import build_system
    from pharma_name_gen.verifier import VerifierConfig
    return build_system(live=False, use_llm=False, use_artifacts=False,
                        verifier_config=VerifierConfig(stem_aware_similarity=True))
