"""Streamlit dashboard for the IMDb RAG harness.

Two pages:

- A chat placeholder (real chat arrives in a later ticket).
- A Change Ledger view backed by ``TraceStore``.

The data-access logic lives in ``data.py`` (plain, unit-testable functions)
so it stays decoupled from the hard-to-test Streamlit rendering code.
"""
