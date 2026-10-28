"""Interpretability: permutation importance and optional SHAP availability."""

from src.interpretability.importance import permutation_importance_cv, shap_available

__all__ = ["permutation_importance_cv", "shap_available"]
