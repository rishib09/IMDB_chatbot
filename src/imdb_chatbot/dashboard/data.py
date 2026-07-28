"""Plain, unit-testable data-access helpers for the dashboard.

Kept free of any Streamlit imports so the read logic can be exercised in
tests without spinning up a Streamlit server. The Streamlit rendering code
in ``app.py`` calls into these functions.
"""

from __future__ import annotations

from ..store import TraceStore

# Columns selected from the change_ledger table, in display order.
LEDGER_COLUMNS = ["change_id", "ts", "artifact_type"]


def load_change_ledger(store: TraceStore) -> list[dict]:
    """Return every row of the change ledger, newest first.

    Reads directly from the ``change_ledger`` table via the store's read
    connection. Against a fresh (empty) store this returns an empty list.

    Each returned dict has the keys in ``LEDGER_COLUMNS``.
    """
    with store._read_lock:
        rows = store._read_conn.execute(
            "SELECT change_id, ts, artifact_type FROM change_ledger ORDER BY ts DESC"
        ).fetchall()
    return [
        {"change_id": r["change_id"], "ts": r["ts"], "artifact_type": r["artifact_type"]}
        for r in rows
    ]
