# IMDb RAG Chatbot - Eval-Driven Reliability Harness

An **eval-driven reliability harness** with a movie-recommendation chatbot inside it: deterministic orchestration (LangGraph), typed data contracts (Pydantic v2), hybrid retrieval (FAISS + BM25 + RRF), and a self-hardening evaluation loop.

- **Spec:** [docs/PRD_IMDb_Chatbot.md](docs/PRD_IMDb_Chatbot.md)
- **Decision map:** [issue #2](https://github.com/rishib09/IMDB_chatbot/issues/2)
- **MVP build tickets:** #11-#26 (`MVP` milestone)

## Quickstart (development)

```bash
py -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest -q
```

## Running the tests

Three tiers, one command each:

| Tier | Command |
|------|---------|
| Default suite (no network, no key; includes the recorded-cassette LLM replays) | `pytest -q` |
| Live tier: real model + index + corpus, real spend | `npx @dotenvx/dotenvx run -f .env -- pytest -m live -q` |
| Re-record the LLM cassettes after a prompt/model change | `npx @dotenvx/dotenvx run -f .env -- pytest tests/test_recorded_llm.py --record-mode=rewrite` |

The live tier is deselected by default (`addopts = ["-m", "not live"]` in
`pyproject.toml`); a CLI `-m live` overrides it. Live tests discover the corpus
and index via `tests/conftest.py` - defaults are `data/corpus.sqlite` and the
`config/live_index.json` pointer, overridable with `IMDB_TEST_CORPUS` and
`IMDB_TEST_INDEX` - and skip with a named reason when a resource (or the
OpenRouter key) is missing.

## Configuration

Versioned artifacts live under [`config/`](config/):

| File | Purpose |
|------|---------|
| `models.yaml` | OpenRouter model slots (rewriter / extractor / generator / judge) |
| `limits.yaml` | Security and cost limit ladder (PRD section 5.6) |
| `live_index.json` | Pointer to the live index artifact (hot-swap) |

## Secrets

Never commit plaintext secrets. For local development, copy the template and fill your own values:

```bash
cp .env.example .env          # .env is git-ignored
```

Required keys are documented in [`.env.example`](.env.example). In CI and on Hugging Face Spaces these arrive as platform secrets. An encrypted-in-repo option is being set up - see `docs/secrets.md`.
