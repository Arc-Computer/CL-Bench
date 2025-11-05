"""Compatibility shim for legacy validator imports."""

from src.evaluation.verification import CrmStateSnapshot, ValidationResult

__all__ = ["CrmStateSnapshot", "ValidationResult"]
