"""Subject-to-treatment-group mapping for the VML study.

Derived from AFIRM_III_and_BAA_Gait_Animals.xlsx.
"""
from pathlib import Path

# =========================================================================
# AFIRM Study: 48 animals, 6 treatment groups, all VML to right GN
# Timepoints: Baseline, 4 wk, 8 wk, 12 wk, 16 wk, 20 wk, 24 wk
# =========================================================================

AFIRM_GROUPS: dict[str, list[str]] = {
    "No Repair": [
        "LGS09", "LGS10", "LGS13", "LGS14",
        "E17", "E20", "E21",
        "LGS11", "LGS12", "LGS15", "LGS16",
        "T05", "T06", "E16", "E18", "E19",
        "LGS08", "T08", "E08", "E12",
        "E13", "E14", "E15", "E26",
        "LGS05", "LGS06", "LGS07", "T07",
    ],
    "TEMR": [
        "T01", "T02", "T03", "T04",
        "T09", "T10", "T11", "T12",
        "G14_E23", "E09", "E10",
        "A04", "A05", "A06", "A07", "A08",
    ],
    "Healy Hydrogel + TEMR": [
        "H01", "H02", "H03", "H04",
        "H05", "H06", "H07", "H08",
        "G16_E24", "E02", "E06", "E11",
        "A09", "A10", "A11", "A12",
    ],
    "Keratin Gel + TEMR": [
        "K01", "K02", "K03", "K04",
        "K05", "K06", "K07", "K08",
        "G13_E22", "E01", "E03", "E04", "E05",
        "A01", "A02", "A03",
    ],
    "Healy Hydrogel": [
        "G01", "G02", "G03", "G04",
        "G05", "G06", "G07", "G08",
        "LGSH01", "LGSH02", "LGSH03", "LGSH04",
        "G13", "G14", "G15", "G16", "G17", "G18",
    ],
    "Healy Sponge": [
        "S01", "S03", "S04", "S05",
        "S06", "S07", "S08", "S13",
        "S09", "S10", "S11", "S12",
        "S14", "S15", "S16", "S17",
    ],
}

# =========================================================================
# BAA Study: previously collected dataset
# Conditions, not treatment groups — these are the "Control" reference
# =========================================================================
BAA_CONDITIONS: dict[str, list[str]] = {
    "Control":           [f"BAA{i:02d}" for i in range(1, 33)],
}

# Build a flat lookup: subject_id -> group name
SUBJECT_TO_GROUP: dict[str, str] = {}
for group, subjects in AFIRM_GROUPS.items():
    for s in subjects:
        SUBJECT_TO_GROUP[s] = group
for group, subjects in BAA_CONDITIONS.items():
    for s in subjects:
        SUBJECT_TO_GROUP[s] = group


def get_group(subject_id: str) -> str:
    """Return treatment group for a subject ID.  Empty string if unknown."""
    return SUBJECT_TO_GROUP.get(subject_id, "")


def build_group_map() -> dict[str, str]:
    """Return a copy of the full subject->group mapping."""
    return dict(SUBJECT_TO_GROUP)


# Session names from the data
VML_SESSIONS = [
    "Baseline", "Week04", "Week08", "Week12", "Week16", "Week20", "Week24",
]
THESIS_SESSION = "Week24"  # The 24-week endpoint used in the thesis
