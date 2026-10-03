"""Per-query resource accounting and unit aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class AccountingRecord:
    context_tokens: int
    retrieval_ops: int
    reconstruction_passes: int
    store_tokens: int
    store_bits: float
    maintenance_ops_per_step: float
    policy_params: int
    wallclock_ms: float

    def __post_init__(self) -> None:
        if any(value is None for value in asdict(self).values()):
            raise ValueError("all SPEC §10 accounting fields must be populated")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def storage_bits(
    token_count: int,
    item_count: int,
    *,
    vocab_size: int,
    metadata_fields: int,
) -> float:
    return token_count * log2(vocab_size) + item_count * metadata_fields * 32


def aggregate_accounting(
    records: Iterable[AccountingRecord],
) -> dict[str, float]:
    rows = [record.as_dict() for record in records]
    if not rows:
        raise ValueError("cannot aggregate an empty set of accounting records")
    return {key: float(mean(row[key] for row in rows)) for key in rows[0]}


def accounting_from_values(**values: Any) -> AccountingRecord:
    """Construct a record while rejecting any omitted SPEC field."""

    required = set(AccountingRecord.__dataclass_fields__)
    missing = required - values.keys()
    extra = values.keys() - required
    if missing or extra:
        raise ValueError(f"accounting fields missing={sorted(missing)} extra={sorted(extra)}")
    return AccountingRecord(**values)
