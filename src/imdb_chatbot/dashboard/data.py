"""Plain, unit-testable data-access helpers for the dashboard.

Kept free of any Streamlit imports so the read logic can be exercised in
tests without spinning up a Streamlit server. The Streamlit rendering code
in ``app.py`` calls into these functions.

The Change Ledger page is built from three PURE render-data helpers here - the
topline metric strip, the metric timeline (a series plus one marker per promoted
change), and the ledger table - so all the shaping logic is unit-testable and
``app.py`` only does the drawing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..eval.detect import detect
from ..schemas import ChangeRecord
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


# -- topline strip ------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """One topline metric and its traffic-light threshold ladder.

    ``higher_better`` sets the comparison direction; a value at/beyond ``green``
    (above it for higher-is-better, below it for lower-is-better) is green, past
    ``amber`` it is red, in between it is amber. These cutoffs are the section 8.4
    gated knobs, kept editable in one place.
    """

    key: str
    label: str
    higher_better: bool
    green: float
    amber: float
    fmt: str


TOPLINE_SPECS: list[MetricSpec] = [
    MetricSpec("recall_at_5", "Recall@5", True, green=0.80, amber=0.70, fmt="{:.2f}"),
    MetricSpec(
        "exclusion_precision", "Exclusion precision", True, green=1.00, amber=0.95, fmt="{:.2f}"
    ),
    MetricSpec("adherence", "Adherence", True, green=0.95, amber=0.90, fmt="{:.2f}"),
    MetricSpec("p95_latency_ms", "p95 latency (ms)", False, green=3000, amber=5000, fmt="{:.0f}"),
    MetricSpec("cost_per_conv_usd", "$/conv", False, green=0.05, amber=0.10, fmt="${:.3f}"),
]


def _status_color(spec: MetricSpec, value: float | None) -> str:
    """Map a metric value to a traffic-light color against its threshold ladder."""
    if value is None:
        return "gray"
    if spec.higher_better:
        if value >= spec.green:
            return "green"
        if value >= spec.amber:
            return "amber"
        return "red"
    # lower-is-better
    if value <= spec.green:
        return "green"
    if value <= spec.amber:
        return "amber"
    return "red"


def _latest_metrics(store: TraceStore) -> dict[str, float]:
    """The most recent promoted metric snapshot (``metric_after`` of the newest change)."""
    records = read_change_records(store, newest_first=True)
    return dict(records[0].metric_after) if records else {}


def topline_strip(store_or_metrics: TraceStore | Mapping[str, float]) -> list[dict]:
    """Return the current topline metrics, each with a threshold status color.

    Accepts either a metrics mapping directly, or a ``TraceStore`` (from which the
    newest promoted ``metric_after`` snapshot is read). Every configured metric is
    returned in a fixed order; a metric absent from the source shows ``value=None``
    and a gray status.
    """
    if isinstance(store_or_metrics, TraceStore):
        metrics: Mapping[str, float] = _latest_metrics(store_or_metrics)
    else:
        metrics = store_or_metrics

    strip: list[dict] = []
    for spec in TOPLINE_SPECS:
        value = metrics.get(spec.key)
        strip.append(
            {
                "key": spec.key,
                "label": spec.label,
                "value": value,
                "display": spec.fmt.format(value) if value is not None else "n/a",
                "status": _status_color(spec, value),
            }
        )
    return strip


# -- change records + timeline ------------------------------------------------


def read_change_records(store: TraceStore, *, newest_first: bool = False) -> list[ChangeRecord]:
    """Read the full ``ChangeRecord`` objects from the ledger.

    Chronological (oldest-first) by default so callers can build a timeline;
    pass ``newest_first=True`` for table order.
    """
    order = "DESC" if newest_first else "ASC"
    with store._read_lock:
        rows = store._read_conn.execute(
            f"SELECT data FROM change_ledger ORDER BY ts {order}, change_id {order}"
        ).fetchall()
    return [ChangeRecord.model_validate_json(r["data"]) for r in rows]


def metric_timeline(store: TraceStore, metric_key: str) -> dict[str, list[dict]]:
    """Build a time series for ``metric_key`` with one marker per promoted change.

    Returns ``{"points": [...], "markers": [...]}``:

    - ``points`` - chronological ``{ts, value, change_id}`` samples taken from each
      change's ``metric_after[metric_key]`` (changes lacking the metric are
      skipped).
    - ``markers`` - one vertical marker per promoted ``ChangeRecord`` (every change
      in the ledger is a promotion), each carrying its ``change_id`` and enough
      detail to render a clickable annotation (artifact type, version move, and
      the metric delta for the selected key).
    """
    records = read_change_records(store)
    points: list[dict] = []
    markers: list[dict] = []
    for rec in records:
        after = rec.metric_after.get(metric_key)
        before = rec.metric_before.get(metric_key)
        if after is not None:
            points.append({"ts": rec.ts.isoformat(), "value": after, "change_id": rec.change_id})
        delta = (after - before) if (after is not None and before is not None) else None
        markers.append(
            {
                "change_id": rec.change_id,
                "ts": rec.ts.isoformat(),
                "artifact_type": rec.artifact_type,
                "label": _version_move(rec),
                "value": after,
                "delta": delta,
            }
        )
    return {"points": points, "markers": markers}


# -- ledger table -------------------------------------------------------------


def _version_move(rec: ChangeRecord) -> str:
    """A compact 'what changed' string, e.g. ``prompt: p-v1 -> p-v2``."""
    before = rec.version_before if rec.version_before is not None else "(none)"
    return f"{rec.artifact_type}: {before} -> {rec.version_after}"


def _metric_deltas(rec: ChangeRecord) -> dict[str, float]:
    """Per-metric ``after - before`` for every metric present in both snapshots."""
    deltas: dict[str, float] = {}
    for key, after in rec.metric_after.items():
        before = rec.metric_before.get(key)
        if before is not None:
            deltas[key] = after - before
    return deltas


def _why_codes(store: TraceStore, rec: ChangeRecord) -> dict[str, int]:
    """Taxonomy codes x counts for a change, derived from its motivating traces.

    Each motivating trace id is resolved against the store and run through the
    (pure, network-free) detection pass; the fired codes are tallied. Trace ids
    that are not in the store are simply skipped, so a record whose motivating
    traces were pruned still renders (with an empty 'why').
    """
    counts: dict[str, int] = {}
    for trace_id in rec.motivating_trace_ids:
        trace = store.read_trace(trace_id)
        if trace is None:
            continue
        for code in detect(trace):
            counts[code] = counts.get(code, 0) + 1
    return counts


def ledger_table(store: TraceStore) -> list[dict]:
    """Return newest-first ledger rows shaped for the dashboard table.

    Each row: ``change_id``, ``date`` (ISO ts), ``artifact_type``, ``what_changed``
    (version move), ``why`` (taxonomy codes x counts from the motivating traces),
    and ``metric_deltas`` (per-metric after-before).
    """
    records = read_change_records(store, newest_first=True)
    rows: list[dict] = []
    for rec in records:
        rows.append(
            {
                "change_id": rec.change_id,
                "date": rec.ts.isoformat(),
                "artifact_type": rec.artifact_type,
                "what_changed": _version_move(rec),
                "why": _why_codes(store, rec),
                "metric_deltas": _metric_deltas(rec),
            }
        )
    return rows


def _format_why(why: Mapping[str, int]) -> str:
    """Render a 'why' code->count map as a compact string, e.g. ``R1 x2, O1 x1``."""
    if not why:
        return "-"
    return ", ".join(f"{code} x{count}" for code, count in sorted(why.items()))


def _format_deltas(deltas: Mapping[str, float]) -> str:
    """Render metric deltas compactly with explicit signs, e.g. ``recall_at_5 +0.05``."""
    if not deltas:
        return "-"
    return ", ".join(f"{key} {value:+.3g}" for key, value in sorted(deltas.items()))


def ledger_table_display(store: TraceStore) -> list[dict]:
    """Like :func:`ledger_table`, but with 'why' and deltas pre-rendered to strings.

    A convenience for the Streamlit table view, which wants flat string cells.
    """
    rows: list[dict[str, Any]] = []
    for row in ledger_table(store):
        rows.append(
            {
                "change_id": row["change_id"],
                "date": row["date"],
                "artifact_type": row["artifact_type"],
                "what_changed": row["what_changed"],
                "why": _format_why(row["why"]),
                "metric_deltas": _format_deltas(row["metric_deltas"]),
            }
        )
    return rows
