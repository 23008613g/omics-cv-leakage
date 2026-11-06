"""Audit where a supervised feature step sits relative to cross-validation."""
from .core import AuditResult, evaluate_protocols, logistic_l2, pooled_auc
from .permutation import audit, empirical_p, permute_labels
from .preprocess import StandardPreprocessor
from .risk import expected_presplit_null
from .splits import OuterFold, default_splitter, make_splits, validate_splits
from .steps import HTSPLS, OPLSDA, PLSDA, RFE, CustomStep, TopQ, VIPSelect

__version__ = "0.2.0"
__all__ = ["AuditResult", "CustomStep", "HTSPLS", "OPLSDA", "OuterFold", "PLSDA", "RFE", "StandardPreprocessor",
           "TopQ", "VIPSelect",
           "audit", "default_splitter", "empirical_p", "evaluate_protocols", "expected_presplit_null", "logistic_l2", "make_splits",
           "permute_labels", "pooled_auc", "validate_splits"]
