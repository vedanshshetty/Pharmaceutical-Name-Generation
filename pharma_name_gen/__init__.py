"""Generative-verifier architecture for regulation-compliant pharmaceutical name generation."""
from .system import VERSION, NominaSystem, build_system
from .orchestrator import PipelineConfig, RunReport, Candidate
from .contracts import TargetType, RiskBand, FailureCode

__version__ = VERSION
__all__ = ["build_system", "NominaSystem", "PipelineConfig", "RunReport",
           "Candidate", "TargetType", "RiskBand", "FailureCode", "__version__"]
