"""Quota store: persistence, period roll-over, exhaustion, refund and the limit."""

from contextlib import closing
from datetime import UTC, datetime

from src.quota import Quota, QuotaStore


class _Clock:
    """An injectable ``now`` that the test moves by hand."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _open(tmp_path, quotas, clock):
    """A store on the test's own file, closed when the ``with`` block ends."""
    return closing(QuotaStore(str(tmp_path / "quota.sqlite3"), quotas, now=clock))


def test_counters_survive_a_restart(tmp_path):
    clock = _Clock(datetime(2026, 10, 11, tzinfo=UTC))
    with _open(tmp_path, {"brave": Quota(10)}, clock) as store:
        assert store.reserve("brave") is not None
        assert store.add_billed(1, 3) == (1, 3)

    # A new store on the same file is what a process restart amounts to.
    with _open(tmp_path, {"brave": Quota(10)}, clock) as store:
        assert store.used("brave") == 1
        assert store.add_billed(0, 0) == (1, 3)


def test_a_new_calendar_month_starts_from_zero(tmp_path):
    clock = _Clock(datetime(2026, 10, 31, 23, 59, tzinfo=UTC))
    with _open(tmp_path, {"brave": Quota(1)}, clock) as store:
        assert store.reserve("brave") is not None
        assert store.reserve("brave") is None  # October is spent

        # Nothing runs in between: the clock alone moves the count to a new row.
        clock.now = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
        assert store.used("brave") == 0
        assert store.reserve("brave") is not None


def test_an_exhausted_instance_stays_blocked_until_the_period_ends(tmp_path):
    clock = _Clock(datetime(2026, 10, 11, tzinfo=UTC))
    with _open(tmp_path, {"serper": Quota(100)}, clock) as store:
        slot = store.reserve("serper")
        assert slot is not None
        store.mark_exhausted(slot)
        assert store.is_exhausted("serper")
        assert store.reserve("serper") is None

        clock.now = datetime(2026, 10, 31, 23, 59, tzinfo=UTC)
        assert store.reserve("serper") is None  # still the same period

        clock.now = datetime(2026, 11, 1, tzinfo=UTC)
        assert not store.is_exhausted("serper")
        assert store.reserve("serper") is not None


def test_refund_gives_the_slot_back(tmp_path):
    clock = _Clock(datetime(2026, 10, 11, tzinfo=UTC))
    with _open(tmp_path, {"serper": Quota(1)}, clock) as store:
        slot = store.reserve("serper")
        assert slot is not None
        assert store.reserve("serper") is None

        store.refund(slot)
        assert store.used("serper") == 0
        assert store.reserve("serper") is not None


def test_reservations_stop_at_the_limit(tmp_path):
    clock = _Clock(datetime(2026, 10, 11, tzinfo=UTC))
    with _open(tmp_path, {"serper": Quota(3)}, clock) as store:
        taken = [store.reserve("serper") is not None for _ in range(5)]
        assert taken == [True, True, True, False, False]
        assert store.used("serper") == 3
