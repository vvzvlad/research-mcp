"""Persistent per-instance quotas with atomic slot reservation.

A ``Quota`` says how many calls an instance may make per period and when that
budget resets. The store keeps one row per ``(instance, period)``, and the
period key is computed from the clock on every call — ``"2026-10"`` for a
monthly quota, ``"2026-10-11"`` for a daily one, ``"lifetime"`` for a one-off
grant. A new month is simply a new row: no timer, no reset task.

``reserve`` books one slot BEFORE the network call with a single conditional
upsert, so the search instances that ``Pipeline.search`` runs concurrently
cannot overshoot the limit. ``refund`` gives the slot back when the call
failed; ``mark_exhausted`` blocks the rest of the period when the vendor's own
API said the balance is spent. An instance without a quota is a no-op here.

The same file also holds the cumulative billed counters behind the
``cum_paid=`` / ``cum_calls=`` log fields, so they survive a restart.

Synchronous stdlib sqlite3 on purpose: every statement is tiny, and with no
await between the check and the write a reservation is atomic inside the event
loop as well as across processes — the upsert itself is the check.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from typing import Literal

from loguru import logger

Reset = Literal["monthly", "daily", "once"]


@dataclass(frozen=True, slots=True)
class Quota:
    """How many calls an instance may make per period, and when it resets."""

    limit: int
    reset: Reset = "monthly"

    def __post_init__(self) -> None:
        # The INSERT branch of the reservation upsert does not check the limit,
        # so a limit below 1 would still let the first call of a period through.
        if self.limit < 1:
            raise ValueError(f"quota limit must be >= 1, got {self.limit}")


def period_key(reset: Reset, now: datetime) -> str:
    """The period a call made at ``now`` (UTC) is counted against."""
    if reset == "monthly":
        return now.strftime("%Y-%m")
    if reset == "daily":
        return now.strftime("%Y-%m-%d")
    return "lifetime"


@dataclass(frozen=True, slots=True)
class Slot:
    """One reserved call. ``period`` is None for an instance without a quota.

    The slot remembers its period so a refund lands in the period it was taken
    from, even if the call crossed a period boundary.
    """

    instance: str
    period: str | None


class QuotaStore:
    """SQLite-backed quota usage per ``(instance, period)`` plus billed counters."""

    def __init__(
        self,
        path: str,
        quotas: Mapping[str, Quota],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._quotas = dict(quotas)
        self._now = now
        # isolation_level=None: autocommit, each statement is its own
        # transaction. check_same_thread=False: the store is opened in
        # Pipeline.build, before the event loop starts.
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS usage ("
            "instance TEXT NOT NULL, period TEXT NOT NULL, "
            "used INTEGER NOT NULL DEFAULT 0, exhausted INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY (instance, period))"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL)"
        )

    def reserve(self, instance: str) -> Slot | None:
        """Book one call for ``instance``; ``None`` when its budget is spent.

        One conditional upsert is both the check and the write: the UPDATE
        branch fires only while the row is not exhausted and stays within the
        limit, so ``rowcount`` is 1 exactly when the slot was taken.
        """
        quota = self._quotas.get(instance)
        if quota is None:
            return Slot(instance, None)
        period = period_key(quota.reset, self._now())
        cur = self._db.execute(
            "INSERT INTO usage (instance, period, used) VALUES (?, ?, 1) "
            "ON CONFLICT (instance, period) DO UPDATE SET used = used + 1 "
            "WHERE exhausted = 0 AND used + 1 <= ?",
            (instance, period, quota.limit),
        )
        return Slot(instance, period) if cur.rowcount == 1 else None

    def refund(self, slot: Slot) -> None:
        """Give back a slot whose call failed before the vendor counted it."""
        if slot.period is None:
            return
        self._db.execute(
            "UPDATE usage SET used = MAX(used - 1, 0) WHERE instance = ? AND period = ?",
            (slot.instance, slot.period),
        )

    def mark_exhausted(self, slot: Slot) -> None:
        """Block the slot's instance until its period ends (the vendor said 402/429)."""
        if slot.period is None:
            return
        self._db.execute(
            "UPDATE usage SET exhausted = 1 WHERE instance = ? AND period = ?",
            (slot.instance, slot.period),
        )
        logger.warning(
            "quota: '{}' exhausted until the end of period {}", slot.instance, slot.period
        )

    def _current_row(self, instance: str) -> tuple[int, int] | None:
        quota = self._quotas.get(instance)
        if quota is None:
            return None
        return self._db.execute(
            "SELECT used, exhausted FROM usage WHERE instance = ? AND period = ?",
            (instance, period_key(quota.reset, self._now())),
        ).fetchone()

    def used(self, instance: str) -> int:
        """Calls booked for ``instance`` in the current period."""
        row = self._current_row(instance)
        return row[0] if row else 0

    def is_exhausted(self, instance: str) -> bool:
        """True if ``instance`` is blocked for the rest of the current period."""
        row = self._current_row(instance)
        return bool(row and row[1])

    def add_billed(self, paid: int, calls: int) -> tuple[int, int]:
        """Add one request's billed calls to the cumulative counters.

        Returns the new ``(cum_paid, cum_calls)``; ``add_billed(0, 0)`` reads
        the saved totals.
        """
        for name, value in (("paid", paid), ("calls", calls)):
            self._db.execute(
                "INSERT INTO counters (name, value) VALUES (?, ?) "
                "ON CONFLICT (name) DO UPDATE SET value = value + excluded.value",
                (name, value),
            )
        totals = dict(self._db.execute("SELECT name, value FROM counters").fetchall())
        return totals["paid"], totals["calls"]

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()
