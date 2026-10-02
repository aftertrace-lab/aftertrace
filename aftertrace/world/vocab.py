"""The fixed symbolic vocabulary used throughout Aftertrace."""

from __future__ import annotations

SPECIAL = (
    "PAD", "BOS", "EOS", "SEP", "EP", "EPP", "RES", "SUM", "Q", "A",
    "ABSTAIN", "MASK", "PRED", "GIST",
)
RELATIONS = (
    "WORKS", "MOVED", "JOINED_YEAR", "TOLD", "AT", "MET", "OWNS",
    "COLOR", "LIVES", "STUDIED", "MANAGES", "SIBLING",
)
OPERATORS = ("CURRENT", "PREV", "HOP2", "CTX", "TIME")

CITIES = tuple(f"C_{i}" for i in range(60))
COMPANIES = tuple(f"O_{i}" for i in range(40))
YEARS = tuple(f"Y_{i}" for i in range(30))
COLORS = tuple(f"K_{i}" for i in range(12))
PLACES = tuple(f"P_{i}" for i in range(20))
OBJECTS = tuple(f"J_{i}" for i in range(16))
ENTITIES = tuple(f"E_{i}" for i in range(512))
TIMES = tuple(f"T_{i}" for i in range(64))
FILLER = tuple(f"F_{i}" for i in range(64))

TOKENS = (
    SPECIAL + RELATIONS + OPERATORS + CITIES + COMPANIES + YEARS + COLORS
    + PLACES + OBJECTS + ENTITIES + TIMES + FILLER
)
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}

assert len(TOKENS) == len(TOKEN_TO_ID), "vocabulary tokens must be unique"
assert len(TOKENS) < 1000, "SPEC requires a vocabulary smaller than 1,000 tokens"
