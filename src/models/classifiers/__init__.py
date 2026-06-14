# SPDX-License-Identifier: CC-BY-NC-4.0
"""Per-domain time-series classifiers used as ARTS tools.

Each domain has a self-contained subpackage holding its model definition
(``model.py`` / dedicated modules) and its training entrypoint (``train.py``).
These classifiers are the ARTS "g_phi" tools: trained here, then exposed to
the agentic orchestrator via ``src.models.ts_llm.arts.tools``.

Layout::

    classifiers/
      encoders.py        shared frozen encoders (Chronos-2, OxWearables)
      capture24/         dual-encoder + MLP head        (activity recognition)
      sleep/             Chronos-2 multivariate + head  (stages / arousals)
      ecg/               HTF beat classifier + rhythm   (LTAF)
      uk_dale/           dilated TCN                     (appliance power)

``get_classifier_class`` resolves a domain key to its primary classifier
class. Imports are deferred so that reading the registry does not pull heavy
encoder dependencies (torch, chronos) unless a classifier is actually built.
"""

from __future__ import annotations

from typing import Any

# Domain key -> "module:attr" of the primary classifier class. Kept as strings
# so importing this package stays cheap; resolved lazily on demand.
CLASSIFIER_REGISTRY: dict[str, str] = {
    "capture24": "src.models.classifiers.capture24.model:HARClassifier",
    "sleep": "src.models.classifiers.sleep.model:SleepClassifier",
    "ecg_beat": "src.models.classifiers.ecg.beat_htf:EcgBeatHTFClassifier",
    "ecg_rhythm": "src.models.classifiers.ecg.rhythm_from_beats:RhythmFromBeats",
    "uk_dale": "src.models.classifiers.uk_dale.model:UKDaleClassifier",
}


def get_classifier_class(domain: str) -> type:
    """Resolve a domain key to its primary classifier class.

    Args:
        domain: Registry key (e.g. ``"capture24"``).

    Returns:
        The classifier class.

    Raises:
        KeyError: If the domain is not registered.
    """
    if domain not in CLASSIFIER_REGISTRY:
        raise KeyError(
            f"Unknown classifier domain '{domain}'. "
            f"Available: {list(CLASSIFIER_REGISTRY.keys())}"
        )
    import importlib

    module_path, attr = CLASSIFIER_REGISTRY[domain].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def list_classifiers() -> list[str]:
    """Return all registered classifier domain keys."""
    return list(CLASSIFIER_REGISTRY.keys())
