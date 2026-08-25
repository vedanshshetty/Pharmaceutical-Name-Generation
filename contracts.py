"""
NOMINA — Interface contract between the Generator (Person A) and the Verifier (Person B).

This module is the *frozen schema* both halves of the project build against. It is
deliberately dependency-light (pydantic v2 only) so either side can import it without
pulling in the other side's stack.

Design rule that matters most: failure reasons are ENUMERATED CODES with a machine-usable
payload, never free prose. The generator must be able to act on a rejection without
making an extra LLM call to interpret the verifier's own output.

Usage
-----
    from contracts import CandidateRequest, VerifierResponse, FailureCode, MockVerifier

    req = CandidateRequest(candidate_name="metoprolol", target_type="generic",
                           target_class="beta-blocker", target_stem="-olol")
    resp = MockVerifier().verify(req)      # schema-valid stub, for Person A's dev loop
    print(resp.model_dump_json(by_alias=True, indent=2))
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TargetType(str, Enum):
    """Which naming system the candidate is competing in."""
    GENERIC = "generic"   # INN / USAN nonproprietary name: stem-governed
    BRAND = "brand"       # proprietary name: stem-avoiding, trademark-exposed


class Severity(str, Enum):
    FAIL = "fail"      # hard constraint violated -> candidate rejected
    WARN = "warn"      # soft constraint / advisory -> candidate survives, flagged
    INFO = "info"


class CheckName(str, Enum):
    SIMILARITY = "similarity"                    # V1  POCA-style orthographic + phonetic
    STEM_CONFLICT = "stem_conflict"              # V2  USAN/INN stem rules
    TRADEMARK_COLLISION = "trademark_collision"  # V3  registered-mark screening proxy
    PRONOUNCEABILITY = "pronounceability"        # V4  phonotactic well-formedness
    CROSSLINGUAL = "crosslingual"                # V5  adverse meaning / implied claim
    WELL_FORMEDNESS = "well_formedness"          # V0  basic input sanity


class FailureCode(str, Enum):
    """The complete, closed set of machine-actionable rejection reasons.

    The generator is expected to branch on these. Adding a member is a contract
    change and requires agreement from both sides.
    """
    # V0 — input sanity
    MALFORMED_CANDIDATE = "MALFORMED_CANDIDATE"
    LENGTH_OUT_OF_RANGE = "LENGTH_OUT_OF_RANGE"
    NON_ALPHABETIC = "NON_ALPHABETIC"

    # V1 — similarity to existing marketed names
    SIMILARITY_TOO_HIGH = "SIMILARITY_TOO_HIGH"          # composite >= high cutoff
    SIMILARITY_MODERATE = "SIMILARITY_MODERATE"          # composite in the grey band
    EXACT_NAME_COLLISION = "EXACT_NAME_COLLISION"        # candidate already exists

    # V2 — USAN/INN stem rules
    STEM_MISSING = "STEM_MISSING"                        # generic name lacks required stem
    STEM_MISMATCH = "STEM_MISMATCH"                      # carries a *different* class's stem
    STEM_MISUSE_IN_BRAND = "STEM_MISUSE_IN_BRAND"        # brand name ends in a real stem
    STEM_EMBEDDED_IN_BRAND = "STEM_EMBEDDED_IN_BRAND"    # brand name contains a stem (advisory)
    STEM_PREFIX_TOO_SHORT = "STEM_PREFIX_TOO_SHORT"      # nothing distinctive before the stem
    INTRA_STEM_TOO_CLOSE = "INTRA_STEM_TOO_CLOSE"        # too close to a same-stem sibling

    # V3 — trademark
    TRADEMARK_HIT = "TRADEMARK_HIT"

    # V4 — pronounceability
    UNPRONOUNCEABLE = "UNPRONOUNCEABLE"
    ILLEGAL_ONSET_CLUSTER = "ILLEGAL_ONSET_CLUSTER"
    NO_VOWEL_NUCLEUS = "NO_VOWEL_NUCLEUS"

    # V5 — cross-lingual / promotional
    CROSSLINGUAL_ADVERSE_MEANING = "CROSSLINGUAL_ADVERSE_MEANING"
    IMPLIED_CLAIM = "IMPLIED_CLAIM"


# ---------------------------------------------------------------------------
# Generator -> Verifier
# ---------------------------------------------------------------------------

class CandidateRequest(BaseModel):
    """One candidate name submitted for screening."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    candidate_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    candidate_name: str
    target_type: TargetType = TargetType.GENERIC
    target_class: Optional[str] = Field(
        default=None, description="Pharmacological class, e.g. 'beta-blocker'.")
    target_stem: Optional[str] = Field(
        default=None, description="Required USAN/INN stem for generic names, e.g. '-olol'.")
    generation_strategy: Optional[str] = Field(
        default=None, description="'llm_baseline' | 'rejection_sampling' | "
                                  "'constrained_decoding' | 'rl_refined' | ...")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateBatch(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    schema_version: str = SCHEMA_VERSION
    candidates: List[CandidateRequest]


# ---------------------------------------------------------------------------
# Verifier -> Generator
# ---------------------------------------------------------------------------

class NearestMatch(BaseModel):
    """One existing name the candidate resembles, with the score decomposition."""
    model_config = ConfigDict(populate_by_name=True)
    name: str
    composite: float = Field(description="POCA-style composite score, 0-100.")
    orthographic: float = Field(default=0.0, description="0-100.")
    phonetic: float = Field(default=0.0, description="0-100.")
    levenshtein: float = Field(default=0.0, description="0-100.")
    bi_sim: float = Field(default=0.0, description="0-100.")
    source: str = Field(default="corpus", description="'generic' | 'brand' | 'trademark' | ...")


class CheckOutcome(BaseModel):
    """Base result shape shared by every check. `passed` serialises as "pass"."""
    model_config = ConfigDict(populate_by_name=True)
    passed: bool = Field(default=True, alias="pass")
    score: Optional[float] = None
    threshold: Optional[float] = None
    codes: List[FailureCode] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class SimilarityCheck(CheckOutcome):
    nearest_match: Optional[str] = None
    nearest_match_score: Optional[float] = None
    top_matches: List[NearestMatch] = Field(default_factory=list)
    distinctiveness_margin: Optional[float] = Field(
        default=None,
        description="high_cutoff - nearest_match_score. Small positive = passed, but barely.")


class StemCheck(CheckOutcome):
    reason: Optional[str] = None
    detected_stem: Optional[str] = None
    expected_stem: Optional[str] = None
    same_stem_siblings: List[str] = Field(default_factory=list)


class TrademarkCheck(CheckOutcome):
    conflicts: List[NearestMatch] = Field(default_factory=list)
    source: str = Field(default="offline_proxy", description="'offline_proxy' | 'live_uspto' | ...")


class PronounceabilityCheck(CheckOutcome):
    phonemes: List[str] = Field(default_factory=list)
    syllables: List[str] = Field(default_factory=list)
    syllable_count: int = 0


class CrossLingualHit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    language: str
    term: str
    gloss: str
    match_type: str = Field(description="'substring' | 'phonetic'")
    similarity: float = 0.0


class CrossLingualCheck(CheckOutcome):
    hits: List[CrossLingualHit] = Field(default_factory=list)


class CheckBundle(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    well_formedness: CheckOutcome = Field(default_factory=CheckOutcome)
    similarity: SimilarityCheck = Field(default_factory=SimilarityCheck)
    stem_conflict: StemCheck = Field(default_factory=StemCheck)
    trademark_collision: TrademarkCheck = Field(default_factory=TrademarkCheck)
    pronounceability: PronounceabilityCheck = Field(default_factory=PronounceabilityCheck)
    crosslingual: CrossLingualCheck = Field(default_factory=CrossLingualCheck)


class RefinementSignal(BaseModel):
    """One actionable instruction back to the generator.

    `payload` is deliberately structured: e.g. for SIMILARITY_TOO_HIGH it carries the
    colliding name and by how much the score exceeded the cutoff, so the generator can
    steer away from that region instead of blindly resampling.
    """
    model_config = ConfigDict(populate_by_name=True)
    code: FailureCode
    severity: Severity = Severity.FAIL
    check: CheckName
    payload: Dict[str, Any] = Field(default_factory=dict)
    human_readable: str = ""


class VerifierResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    candidate_name: str
    target_type: TargetType = TargetType.GENERIC
    overall_pass: bool = True
    composite_risk_score: float = Field(
        default=0.0,
        description="0-100 headline risk. Currently the nearest-match composite similarity.")
    checks: CheckBundle = Field(default_factory=CheckBundle)
    refinement_feedback: List[RefinementSignal] = Field(default_factory=list)
    verifier_version: str = SCHEMA_VERSION
    timing_ms: Optional[float] = None

    @property
    def failure_codes(self) -> List[FailureCode]:
        return [s.code for s in self.refinement_feedback if s.severity == Severity.FAIL]

    @property
    def warning_codes(self) -> List[FailureCode]:
        return [s.code for s in self.refinement_feedback if s.severity == Severity.WARN]


class VerifierBatchResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    schema_version: str = SCHEMA_VERSION
    results: List[VerifierResponse]

    @property
    def pass_rate(self) -> float:
        return (sum(r.overall_pass for r in self.results) / len(self.results)) if self.results else 0.0


# ---------------------------------------------------------------------------
# Mock verifier — for Person A to develop against before the real one lands
# ---------------------------------------------------------------------------

class MockVerifier:
    """Schema-valid stub. Deterministic, no data files, no network.

    Crude heuristics only (length, vowel presence, stem suffix). Its ONLY job is to let
    the generator side exercise the full request/response/refinement loop before the real
    verifier exists. Swapping this for the real Verifier must be a one-line change.
    """

    def __init__(self, reject_probability: float = 0.0, seed: int = 0):
        self.reject_probability = reject_probability
        self._seed = seed

    def verify(self, request: CandidateRequest) -> VerifierResponse:
        name = (request.candidate_name or "").strip().lower()
        feedback: List[RefinementSignal] = []
        checks = CheckBundle()

        if not name or not name.replace("-", "").isalpha():
            checks.well_formedness.passed = False
            checks.well_formedness.codes = [FailureCode.NON_ALPHABETIC]
            feedback.append(RefinementSignal(
                code=FailureCode.NON_ALPHABETIC, check=CheckName.WELL_FORMEDNESS,
                payload={"candidate_name": request.candidate_name},
                human_readable="Candidate contains non-alphabetic characters."))
        elif not (4 <= len(name) <= 20):
            checks.well_formedness.passed = False
            checks.well_formedness.codes = [FailureCode.LENGTH_OUT_OF_RANGE]
            feedback.append(RefinementSignal(
                code=FailureCode.LENGTH_OUT_OF_RANGE, check=CheckName.WELL_FORMEDNESS,
                payload={"length": len(name), "min": 4, "max": 20},
                human_readable=f"Length {len(name)} outside the 4-20 character range."))

        # Fake similarity: hash-derived but deterministic, so dev runs are reproducible.
        pseudo = (sum(ord(c) for c in name) * 37 + self._seed) % 100
        checks.similarity.score = float(pseudo)
        checks.similarity.threshold = 70.0
        checks.similarity.nearest_match = "mock-existing-name"
        checks.similarity.nearest_match_score = float(pseudo)
        checks.similarity.distinctiveness_margin = 70.0 - pseudo
        if pseudo >= 70:
            checks.similarity.passed = False
            checks.similarity.codes = [FailureCode.SIMILARITY_TOO_HIGH]
            feedback.append(RefinementSignal(
                code=FailureCode.SIMILARITY_TOO_HIGH, check=CheckName.SIMILARITY,
                payload={"nearest_match": "mock-existing-name", "score": float(pseudo),
                         "cutoff": 70.0, "excess": float(pseudo) - 70.0},
                human_readable="Too similar to an existing name (mock)."))

        stem = request.target_stem
        if request.target_type == TargetType.GENERIC and stem:
            bare = stem.lstrip("-")
            checks.stem_conflict.expected_stem = stem
            if not name.endswith(bare):
                checks.stem_conflict.passed = False
                checks.stem_conflict.codes = [FailureCode.STEM_MISSING]
                checks.stem_conflict.reason = f"does not end in {stem}"
                feedback.append(RefinementSignal(
                    code=FailureCode.STEM_MISSING, check=CheckName.STEM_CONFLICT,
                    payload={"expected_stem": stem, "candidate_name": name},
                    human_readable=f"Generic name must end in {stem}."))
            else:
                checks.stem_conflict.detected_stem = stem

        checks.pronounceability.score = 1.0 if any(v in name for v in "aeiou") else 0.0
        if checks.pronounceability.score == 0.0:
            checks.pronounceability.passed = False
            checks.pronounceability.codes = [FailureCode.NO_VOWEL_NUCLEUS]
            feedback.append(RefinementSignal(
                code=FailureCode.NO_VOWEL_NUCLEUS, check=CheckName.PRONOUNCEABILITY,
                payload={"candidate_name": name},
                human_readable="No vowel present."))

        overall = all([checks.well_formedness.passed, checks.similarity.passed,
                       checks.stem_conflict.passed, checks.trademark_collision.passed,
                       checks.pronounceability.passed, checks.crosslingual.passed])
        return VerifierResponse(
            candidate_id=request.candidate_id, candidate_name=request.candidate_name,
            target_type=request.target_type, overall_pass=overall,
            composite_risk_score=float(pseudo), checks=checks,
            refinement_feedback=feedback, verifier_version="mock-" + SCHEMA_VERSION)

    def verify_batch(self, batch: CandidateBatch) -> VerifierBatchResponse:
        return VerifierBatchResponse(results=[self.verify(c) for c in batch.candidates])


def export_json_schema(path: str = "nomina_contract_schema.json") -> str:
    """Dump both directions of the contract as JSON Schema, for the report appendix."""
    import json
    blob = {
        "schema_version": SCHEMA_VERSION,
        "generator_to_verifier": CandidateBatch.model_json_schema(by_alias=True),
        "verifier_to_generator": VerifierBatchResponse.model_json_schema(by_alias=True),
        "failure_codes": [c.value for c in FailureCode],
    }
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)
    return path