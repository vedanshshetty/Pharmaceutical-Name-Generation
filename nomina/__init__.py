"""NOMINA — regulation-aware pharmaceutical name generation and screening."""
from .system import NOMINA_VERSION, NominaSystem, build_system
from .orchestrator import PipelineConfig, RunReport, Candidate
from .contracts import TargetType, RiskBand, FailureCode

__version__ = NOMINA_VERSION
__all__ = ["build_system", "NominaSystem", "PipelineConfig", "RunReport",
           "Candidate", "TargetType", "RiskBand", "FailureCode", "__version__"]
