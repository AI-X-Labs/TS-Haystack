"""
v1 activity vocabulary for UK-DALE-Haystack.

Defines:
  - RAW_TO_CANONICAL: NILMTK appliance type -> canonical haystack name
  - REGIMES: 4 regime taxonomy used for distractor sampling
  - BOUT_DEFAULTS: per-appliance (on_w, off_w, min_on_s, min_off_s) for the
    contextual-ON hysteresis extractor

NILMTK encodes appliance types with spaces ("dish washer", "hair dryer") rather
than the underscore form used elsewhere in the haystack codebase. The canonical
mapping handles both spellings so downstream code can speak in canonical names
only.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Maps every accepted raw NILMTK appliance.type (or original_name) to the
# canonical v1 vocab. Anything not present here is excluded from v1.
RAW_TO_CANONICAL: dict[str, str] = {
    # impulse
    "kettle": "kettle",
    "microwave": "microwave",
    "toaster": "toaster",
    "hair dryer": "hair_dryer",
    "hair_dryer": "hair_dryer",
    "hairdryer": "hair_dryer",
    # long_cycle
    "washing machine": "washing_machine",
    "washing_machine": "washing_machine",
    "dish washer": "dishwasher",
    "dish_washer": "dishwasher",
    "dishwasher": "dishwasher",
    "washer dryer": "washer_dryer",
    "washer_dryer": "washer_dryer",
    # cooking
    "oven": "oven",
    "electric oven": "oven",
    "electric_oven": "oven",
    # refrig
    "fridge": "fridge",
    "fridge freezer": "fridge_freezer",
    "fridge_freezer": "fridge_freezer",
    "freezer": "freezer",
}


# freezer is in RAW_TO_CANONICAL for completeness but not in REGIMES: houses
# {1, 2, 5} (the v1 inventory) contain only fridge / fridge_freezer, no
# standalone freezer. v1 ships 10 appliances across 4 regimes.
REGIMES: dict[str, list[str]] = {
    "impulse":    ["kettle", "microwave", "toaster", "hair_dryer"],
    "long_cycle": ["washing_machine", "dishwasher", "washer_dryer"],
    "cooking":    ["oven"],
    "refrig":     ["fridge", "fridge_freezer"],
}

# Inverse lookup: canonical -> regime
ACTIVITY_TO_REGIME: dict[str, str] = {
    a: regime for regime, acts in REGIMES.items() for a in acts
}

V1_VOCAB: list[str] = sorted(ACTIVITY_TO_REGIME.keys())


# ---------------------------------------------------------------------------
# Per-appliance bout-extraction defaults
# ---------------------------------------------------------------------------
# Tuple = (on_w, off_w, min_on_s, min_off_s). off_w < on_w gives hysteresis on
# the trailing edge; min_off_s is the cycle-absorption parameter (gaps below
# this are merged into the surrounding ON run -- this is what makes a washing
# machine cycle a single bout instead of dozens of motor bursts).
BOUT_DEFAULTS: dict[str, tuple[float, float, float, float]] = {
    "kettle":          (2000.0, 100.0,   12.0,   12.0),
    "microwave":       ( 200.0,  50.0,   12.0,   12.0),
    "toaster":         (1000.0, 100.0,   18.0,   12.0),
    "hair_dryer":      ( 300.0,  50.0,   12.0,   12.0),
    "washing_machine": (  20.0,  10.0, 1200.0,  300.0),
    "dishwasher":      (  50.0,  20.0, 1200.0,  300.0),
    "washer_dryer":    (  20.0,  10.0, 1500.0,  300.0),
    "oven":            ( 100.0,  30.0,   60.0,   60.0),
    "fridge":          (  50.0,  30.0,   30.0,   30.0),
    "fridge_freezer":  (  50.0,  30.0,   30.0,   30.0),
    "freezer":         (  50.0,  30.0,   30.0,   30.0),
}

# Hard ceilings (for anomaly synthesis) -- a UK domestic ring main cannot
# physically deliver more than ~3.5 kW from a single socket without tripping.
MAX_POWER_W: dict[str, float] = {
    "kettle":          3300.0,
    "microwave":       1800.0,
    "toaster":         2400.0,
    "hair_dryer":      2200.0,
    "washing_machine": 2500.0,
    "dishwasher":      2400.0,
    "washer_dryer":    2500.0,
    "oven":            3300.0,
    "fridge":           300.0,
    "fridge_freezer":   300.0,
    "freezer":          300.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canonicalize(raw_name: str) -> str | None:
    """Map a raw NILMTK appliance name to canonical v1 vocab, or None if not in v1."""
    if raw_name is None:
        return None
    key = raw_name.strip().lower()
    return RAW_TO_CANONICAL.get(key)


def regime_of(activity: str) -> str:
    """Regime label for a canonical activity. Raises KeyError if not in v1."""
    return ACTIVITY_TO_REGIME[activity]


def same_regime_activities(activity: str, exclude_self: bool = True) -> list[str]:
    """All v1 activities sharing a regime with `activity`."""
    regime = regime_of(activity)
    peers = list(REGIMES[regime])
    if exclude_self and activity in peers:
        peers.remove(activity)
    return peers


def bout_defaults(activity: str) -> tuple[float, float, float, float]:
    """(on_w, off_w, min_on_s, min_off_s) for canonical activity."""
    return BOUT_DEFAULTS[activity]
