# PRD — IMDb RAG Chatbot: An Eval-Driven Reliability Harness

**Version:** 1.0
**Owner:** Rishi Bhatt
**Executor:** Claude Code (milestone-by-milestone)
**Stack:** Python 3.12 · LangGraph · LangChain components · Pydantic v2 · FAISS · rank_bm25 · SQLite/DuckDB · Streamlit · OpenRouter · Hugging Face Spaces

---

## 1. Overview, Goals, Non-Goals

### 1.1 Thesis

This project is **not** "a chatbot built cheaply." It is an **eval-driven reliability harness that happens to have a chatbot inside it**. The demonstration target is the problem every applied-AI team faces: making a stochastic component (an LLM) behave like dependable infrastructure through systems engineering — deterministic orchestration, typed data contracts, layered guards, and a self-hardening evaluation loop.

Cheap open-weight models (Gemma 3, DeepSeek via OpenRouter) are chosen **deliberately as failure-mode generators**: a 9B model surfaces in week one the failure modes a frontier model would hide until month six. Every failure the model produces becomes a permanent regression test, so the harness only ever gets stronger.

**Architecture thesis (the through-line of every section):**
> Deterministic spine, stochastic leaves, and a single promotion gate that every behavioral artifact — prompt, model, index, chunking policy, threshold — must pass through.

### 1.2 Goals

1. Conversational movie discovery over the IMDb dataset with multi-turn context, exclusion handling, and poster rendering.
2. Hybrid retrieval (FAISS dense + BM25 sparse + RRF fusion + deterministic metadata filters).
3. Multi-region support (US + India initially) via one index with region metadata filtering.
4. Model-agnostic: any OpenRouter model swappable per call-site via config.
5. **Eval-driven improvement loop**: Observe → Detect → Triage → Harden → Fix → Gate, with a one-page Change Ledger dashboard as historical evidence of quantified improvement.
6. Deployment on Hugging Face Spaces (Streamlit SDK), CI-gated by the regression suite.

### 1.3 Non-Goals (v1)

- No fine-tuning of any model. RAG + prompting + system guards only.
- No semantic response caching (evaluated and deferred — conflicts with per-user personalization; documented in §12).
- No agentic multi-hop retrieval **behavior** in v1 (phase 2). To be explicit: **LangGraph ships in v1** (full §7.3 topology, M4); what is deferred is the ReAct-style pattern where the LLM itself decides when/what to retrieve and whether to retrieve again (multi-hop). Rationale: (a) it moves a control-flow decision into the stochastic component — a determinism trade that must be *measured against the v1 baseline*, not assumed; (b) the v1 query taxonomy (§8.7) is fully single-hop; (c) the topology accepts it as one added conditional cycle, promoted through the same gate (§8.4).
- No DSPy in v1 (phase 2, scoped to rewriter/extractor optimization, gated by the same regression protocol).
- No non-English datasets.

---

## 2. Canonical Data Schema & Multi-Region Normalization

### 2.1 Canonical `MovieRecord` (Pydantic v2)

Every ingested row from every regional CSV is validated into this schema. Validation failure → row quarantined to a rejects table with the `ValidationError`, never silently dropped.

```python
from pydantic import BaseModel, Field, HttpUrl, field_validator

class MovieRecord(BaseModel):
    movie_id: str                    # deterministic: sha1(title|year|region)[:12]
    title: str
    year: int = Field(ge=1888, le=2030)
    genres: list[str]                # normalized lowercase, split on '/', ','
    director: str | None = None
    cast: list[str]                  # normalized "First Last" strings
    plot: str | None = None
    rating_raw: float | None = Field(default=None, ge=0, le=10)
    rating_z: float | None = None    # z-score WITHIN region (see 2.2)
    metascore: float | None = None
    certificate_raw: str | None = None   # "PG-13", "U/A", "R"...
    certificate_norm: str | None = None  # mapped to {ALL, TEEN, MATURE, ADULT}
    region: str                      # "US", "IN", ...
    duration_min: float | None = None
    poster_url: HttpUrl | None = None

    @field_validator("cast", "genres", mode="before")
    @classmethod
    def _split_strings(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.replace("/", ",").split(",") if s.strip()]
        return v
```

### 2.2 Multi-region normalization rules

- **One index, not one per region.** Regional differences are metadata concerns, not embedding concerns (all text is English). `region` is one more deterministic filter.
- **Certificates:** per-region mapping tables (`US: PG-13 → TEEN`, `IN: U/A → TEEN`, etc.) stored as versioned YAML in `config/certificates/`. Unknown certificate → `certificate_norm=None` + warning logged, never a hard failure.
- **Ratings:** `rating_z = (rating_raw − mean_region) / std_region` computed at ingestion. Cross-region comparisons and `min_rating` filters operate on `rating_z` when the query spans regions, `rating_raw` when single-region.
- **Adding a country = new CSV + normalization adapter (one YAML + optional column-mapping dict). Zero architecture change.** This is an acceptance criterion, not an aspiration: milestone M2 includes adding IN after US to prove it.

---

## 3. Ingestion & Indexing Pipeline (Offline Plane)

### 3.1 Chunking policy (versioned artifact: `chunk_policy`)

- **v1 policy:** 1 movie = 1 chunk. Composed text: `"{title} ({year}). Genre: {genres}. Director: {director}. Cast: {cast}. Plot: {plot}"`.
- **Long-field rule:** if composed text > 500 tokens, split `plot` into overlapping sub-chunks (50-token overlap) sharing the parent `movie_id`; de-duplicate to parent at retrieval time.
- **Re-chunking triggers:** (a) dataset version bump, (b) schema change, (c) chunk policy change, (d) embedder swap (forces full re-embed — vector spaces are not comparable across models).

### 3.2 Embedding pipeline

- **Default embedder:** local `sentence-transformers/all-MiniLM-L6-v2` (384-dim, free, CPU-friendly on HF Spaces). Alternatives behind the same `Embedder` protocol: `bge-small-en-v1.5`, OpenAI `text-embedding-3-small`.
- **Embedding cache (mandatory):** SQLite table keyed `sha256(f"{model_name}::{text}")` → vector blob. Infinite TTL (pure function). Cache ships with the repo so eval re-runs are free and Spaces restarts are cheap.
- Batched encoding; L2-normalize all vectors at index build AND at query time (cosine-via-inner-product; the classic forget-to-normalize-the-query bug is a named unit test).

### 3.3 Index artifacts (versioned)

Every index is stamped **`{dataset_version}_{embedder}_{chunk_policy}`**:

| Artifact | Tech | Notes |
|---|---|---|
| Dense index | FAISS `IndexFlatIP` on normalized vectors | Exact search; corpus is small |
| Sparse index | `rank_bm25` (BM25Okapi) pickled | Tokenized composed text, lowercase |
| Metadata store | SQLite (DuckDB attached for analytics) | `movies`, `rejects`, caches, traces, change ledger |

**Zero-downtime reindex:** new versions build offline; the app resolves the live index via a pointer file (`config/live_index.json`). Hot-swap = pointer flip; rollback = pointer revert. (Same zero-disruption principle as a dual-build system migration: change the system underneath without disrupting consumers on top.)

---

## 4. Retrieval Spec (Hybrid + RRF + Filters)

### 4.1 Pipeline

1. **Dense:** embed rewritten query (cache check) → FAISS top-K (K=20) by inner product on normalized vectors.
2. **Sparse:** BM25 scores on tokenized rewritten query → top-K (K=20).
3. **Fusion:** Reciprocal Rank Fusion, `RRF(d) = Σ_r 1/(k + rank_r(d))`, k=60 (hand-rolled ~15 lines; no dependency).
4. **Deterministic metadata filters** (pure Python, order: cheap → expensive): `region`, `exclude_actors`, `exclude_genres`, `min_year`/`max_year`, `min_rating` (raw or z per §2.2), `shown_movies` (no repeats within session).
5. Emit top `final_k=5` `ScoredMovie` candidates.

### 4.2 Retrieval cache

In-process LRU keyed `(hash(rewritten_query), index_version, hash(filters))`. Index versioning provides natural invalidation — no TTL logic.

### 4.3 Acceptance thresholds (gate inputs)

| Metric | Threshold | Notes |
|---|---|---|
| Recall@5 (labeled set) | ≥ 0.85 | headline retrieval metric |
| MRR | ≥ 0.70 | |
| **Exclusion precision (post-filter)** | **= 1.00** | deterministic code; ANY violation is a bug (F1), not variance |
| Fallback rate | 5–15% | outside band → threshold recalibration task |

---

## 5. Orchestration & Generation Spec

### 5.1 Model strategy: task-tiered via OpenRouter (versioned artifact: `model_config`)

All models resolved from `config/models.yaml` — swapping any slot is a config edit gated by the regression protocol (§8.4), never a code change.

| Slot | Task | Default | Fallback | Temp |
|---|---|---|---|---|
| `rewriter` | history-aware standalone query | `google/gemma-3-9b-it` | `meta-llama/llama-3.1-8b-instruct` | 0 |
| `extractor` | `ParsedQuery` JSON (JSON mode, NOT function-calling — more portable across OpenRouter models) | `google/gemma-3-9b-it` | regex fallback after 2 retries | 0 |
| `generator` | conversational recommendation | `deepseek/deepseek-chat` | `google/gemma-3-27b-it` | 0.7 |
| `judge` (eval only) | faithfulness/adherence scoring | DeepSeek-V3 or mid-tier frontier | — | 0 |

Adapter: LangChain `init_chat_model(..., base_url="https://openrouter.ai/api/v1")`. Single secret: `OPENROUTER_API_KEY`.

### 5.2 Prompt management (versioned artifact: `prompt`)

- `prompts/` directory, numbered files (`generator_v7.md`), header block: date, motivating failure codes + trace IDs, expected metric movement.
- **Static-first segment ordering** (provider prefix-caching friendly, and correct habit regardless): system prompt → rules → few-shot examples → ⟨cache boundary⟩ → user state → retrieved context → history window → current query.
- No drive-by edits: a prompt change that cannot cite motivating failures does not get a gate run (§8.4).

### 5.3 Structured outputs (Pydantic enforcement points 2 & 3)

- `ParsedQuery`: genres, similar_to, exclude_actors, exclude_genres, min_year, max_year, min_rating, region.
- `RecommendationSet(picks: list[MovieRecommendation], prose: str)`; `MovieRecommendation(title, year, reason, poster_url)` — structured fields drive poster rendering in the UI.
- Validation failure → retry with the `ValidationError` text appended to the prompt (max 2), then deterministic fallback.

### 5.4 Output validation guards (defense in depth — non-overlapping gates)

1. **Gate 1** — extractor parses exclusions from natural language ("without X", "nothing scary").
2. **Gate 2** — deterministic metadata filter post-retrieval (must be perfect; F1 bugs).
3. **Gate 3** — system-prompt hard-exclusion instruction (belt and suspenders).
4. **Gate 4** — output validation (deterministic): (a) no excluded actor in any pick/prose; (b) every recommended title must exist in `state.candidates` (anti-hallucination / G1); (c) **prose fact-grounding** — years, director names, and cast names mentioned in prose must match the candidate's DB record (regex + string match against metadata; catches training-knowledge leakage past the context boundary). Violation → regenerate-with-violation-message cycle (max 2) → fallback. System prompt hardened accordingly: *only facts present in provided records; no external trivia*. Recorded caveat: free-prose grounding is not fully deterministically verifiable — everything checkable is pinned in code; the async judge (faithfulness, §8.5.E) covers the remainder.

### 5.5 Graceful degradation ladder

**Principle: degrade toward determinism, never toward silence or a stack trace.** Ranking rule: a violating answer < no answer < a degraded-but-honest answer. Every degradation event writes a trace flag (`degradation: L1_rewriter_skipped`) → dashboard degradation-rate metric per level; a rising L1 rate is a triage item (O-code) before users feel L2.

| Level | Trigger | Behavior |
|---|---|---|
| **L0** | nominal | full service |
| **L1** component fallback (invisible to user) | generator fail/timeout → fallback model, one attempt · extractor JSON fails ×2 → regex extraction · **rewriter fails → skip; retrieve on raw query** (rewriter is an optimization, not a dependency) · judge down → `judge_scores=null`, production unaffected (evals async) | |
| **L2** LLM-free mode | OpenRouter fully down / budget exhausted / all slots failing | local embedder + hybrid retrieval + RRF + filters (all deterministic, all still working) → render top-5 as structured metadata cards (title/year/genre/rating/poster), banner: "Chat is temporarily limited — here's what matched your search." No generated prose, no fake conversation. *This mode exists because generation is a leaf, not the spine.* |
| **L3** honest exit | index pointer unresolvable / FAISS artifact missing/corrupt / DB won't open | plain message ("The movie index is unavailable; nothing you typed was lost."), loud logging. Nearly unreachable: **startup health check** (pointer resolves → index loads → checksum → DB opens WAL) refuses to serve rather than serving broken. |

**Within-conversation graceful exits** (dead ends, not outages): empty candidates → relax-a-constraint offer with buttons (existing `fallback` node); validation retries exhausted → "couldn't produce a clean recommendation — rephrase?" rather than shipping a Gate-4-violating response.

### 5.6 Security & cost controls (MVP scope — ships with v1)

All limits are **configurable knobs** in `config/limits.yaml` (tunable without code changes; changes are gated artifacts per §8.4):

| Layer | Default cap (tunable) | Enforced where |
|---|---|---|
| Per query | input ≤ 500 chars (UI) / ≤ 150 tokens (server-side, authoritative); context ≤ 2K tokens; `max_tokens=400` on every LLM call | `st.chat_input` / graph entry / API params |
| Per turn | max 2 retries per LLM slot | conditional edges |
| Per session | ~30 turns → polite "start a new session" | session state |
| Per day (global) | token/dollar budget → **degrade to L2 LLM-free mode** (§5.5), never an outage | budget counter |
| Hard backstop | OpenRouter spend limit | provider dashboard |

Oversize input → friendly truncation notice (not an error) + `O1_input_cap` trace flag. `max_tokens` on every call is mandatory: an injected "write 10 pages" dies at 400 tokens regardless of other gates. Budget exhaustion reuses the degradation ladder — cost defense and graceful degradation are the same mechanism; abuse can at worst downgrade the app to free deterministic search, never run up spend.

**Threat controls:**
- **Prompt injection (direct):** structurally blunted — user text reaches LLMs only via rewriter/extractor whose outputs are Pydantic-validated structures; injected instructions have nowhere to land. Gate 4 catches off-format generation.
- **Scope escape:** deterministic + cheap-LLM **topic gate** before `rewrite` (off-topic → fixed refusal: movies only); OOD centroid check (§8.5.C) doubles as scope enforcement.
- **Indirect injection via data:** context is data, never instructions (system-prompt rule); low risk with curated CSVs, re-assess if user-generated text (reviews) is ever ingested.
- **Identity spoofing:** self-asserted identity, recorded caveat (§6.5); no sensitive data in triples.
- **Poster URL safety:** `HttpUrl` validation + domain allowlist at ingestion.

---

## 6. Memory Spec (Three Tiers)

| Tier | Storage | Contents | Lifetime |
|---|---|---|---|
| Short-term window | `st.session_state` | last 6 turns verbatim + running summary of older turns | session |
| Constraint state | `st.session_state` (Pydantic `ConversationState`) | exclusions, preferences, `shown_movies`, feedback | session |
| Long-term KG | SQLite triples per user | `(user)-[LIKES]->(genre)`, `(user)-[DISLIKED]->(movie)`, `(user)-[EXCLUDED]->(actor)` | persistent (best-effort on HF free tier; see §10) |

- Two-call architecture per turn: cheap rewriter (sees history) + generator. Rewritten query used for retrieval; **raw** query stored in history.
- KG extraction: every N=5 turns (and at session end), an extractor call distills durable preferences into triples. At session start, user triples are loaded and injected into the system prompt as "known preferences."
- Token budget per turn enforced in code (hard cap; O1 failure if exceeded).

### 6.1 Memory write triggers & feedback mechanisms (human in the loop)

Principle: **write-weight ∝ explicitness.** Three signal tiers:

| Tier | Signal | Write behavior |
|---|---|---|
| 1 — Explicit widgets | 👍/👎, "Seen it", "Not interested" chips on every recommendation card; optional session-end micro-prompt | Durable triple immediately: `LIKED` / `DISLIKED` / `WATCHED` (permanent no-repeat) / `REJECTED` (repeated → candidate exclusion) |
| 2 — Explicit statements | "loved it", "never show horror", "already watched that" in text | Every-N-turns extraction call distills into durable triples (same confidence as widgets) |
| 3 — Implicit behavior | poster click, follow-up about a pick, **terminal accept** (session ends right after a recommendation), rephrase (= implicit negative) | **Candidate** triples only; promote to durable on repetition (e.g., 2+ sessions). Prevents one accidental click from distorting the profile |

Every triple carries provenance (`widget | stated | inferred`), timestamp, and decay (inferred facts fade; stated facts persist until contradicted).

**Feedback feeds both loops:** a 👎 also flags the `TurnTrace` → triage queue → taxonomy code → potential regression case. Users label production traffic as a side effect of using the product; the monthly judge-vs-human audit (§8.5.E) is the second human loop.

### 6.2 Recall policy (collision-triggered active recall)

- **Passive layer (always on, silent):** *resolved* facts (`WATCHED`, `REJECTED`, exclusions) feed deterministic filters and preference-weighted generation on every turn. Settled past is never asked about (asking about resolved facts = nagging).
- **Active recall (primary mechanism for *unresolved* facts) — fires on a memory–retrieval collision, not a schedule.** Rule: ask about a remembered movie only when **(a)** its memory fact is an open loop (e.g., `TERMINAL_ACCEPT` with no `WATCHED`/`DISLIKED` resolution) **AND (b)** current-turn retrieval ranks it within top `RECALL_RANK_K` (default 3) — i.e., the ambiguity blocks the best answer to the *current query*. The question then services the query, resolves the loop into a durable triple, and harvests Tier-1 feedback in one turn.
- Anti-overuse guards (all deterministic): max 1 callback per session; never re-ask the same movie (skip writes a `NO_RESPONSE` resolution — one shot per fact, ever); cooldown after any skip; open loops expire with the session window (§6.4), so stale callbacks are structurally impossible.
- Implementation: a deterministic `recall_check` node between `filter` and `generate` with a conditional edge to a `recall_prompt` path; every firing visible in `path_taken`.
- Reference cases: memory-movie at rank 1 + open loop → ASK ("Last time we landed on John Wick — did you watch it? If so I'll line up something new; if not, it's still my top pick"). Rank 8 → silently exclude, recommend others. Fact resolved → filter silently, never ask. Irrelevant query (comedy request, movie not in top-K) → silence.
- Tuning via evals, not vibes: `RECALL_RANK_K` is a gated behavioral threshold (§8.4). Metrics: **recall precision** (callbacks engaged / fired, target ≥ 0.7) and **recall yield** (callbacks resolving an open loop into a durable triple) — both in §8.5.F.

**Constraint collision (second collision type).** Signal precedence is strict: **current turn > session state > durable stated > durable inferred** — the live request always outranks memory; memory is context for interpreting requests, never a veto over them. When the parsed query contradicts a durable stated fact (e.g., stored `EXCLUDES horror`, user asks "good horror movies?"): the `filter` node detects it, **suspends that exclusion for the session, and complies immediately** — the bot never refuses a live request on memory grounds. The contradiction then gets one load-bearing question appended to the (complied-with) response: drop the rule going forward, or keep it as a one-off? "Drop" deletes the triple; "one-off" keeps it with session-scoped suspension; skip defaults to **keep** (a stated instruction survives until explicitly revoked). Asked once per collision, resolution recorded, never re-asked. Anti-patterns this prevents: silent-keep (user thinks they were ignored next session) and silent-delete (a one-night mood permanently erases a deliberate instruction).

### 6.4 Retention policy (rolling sessions, aggregated facts)

**Raw sessions roll; triples aggregate.** Two layers:

| Layer | Contents | Lifetime |
|---|---|---|
| Session log | full per-session record: queries, shown movies, resolutions, open loops | **rolling window of 5 sessions** (POC setting); on window exit, session is distilled into triples and dropped |
| Aggregated triples | counters + weights, not history | per-type rules below |

Per-relationship lifetimes — **three memory classes, three distinct end-of-life mechanisms** (intuition: class 1 = *what happened*, class 2 = *what we guessed*, class 3 = *what we were told*):

| # | Class | Example contents | Ends by |
|---|---|---|---|
| 1 | Session log (raw) | "Session 7: asked 'action movie'; shown John Wick, Mad Max, The Raid; open loop: `TERMINAL_ACCEPT(John Wick)` unresolved" | **time** — rolls off the 5-session window; distilled into classes 2/3 on exit |
| 2 | Inferred preferences | `PREFERS(action, weight=5)` · `PREFERS(90s thrillers, weight=2)` · `SHOWN(Mad Max, count=2, last=s4)` | **decay** — weight fades without refresh; `SHOWN` expires with window |
| 3 | Stated facts | `EXCLUDES(horror)` · `WATCHED(John Wick)` · `DISLIKED(Atomic Blonde)` · `REJECTED(Tom Cruise films)` | **contradiction only** — the §6.2 collision question settles it |

Stated facts never expire on a timer: silent expiry of "never show horror" after 5 sessions would be the system forgetting a direct instruction the user still trusts. Explicit in, explicit out. (`WATCHED` is class 3, not class 1: recommending an already-watched movie is a failure regardless of how long ago the watching happened.) Cap ~200 triples/user; LRU eviction applies to inferred triples only. Repeated patterns across sessions increment weight rather than duplicating rows — triples are the compression that lets the raw window stay at 5.

### 6.5 User identity (POC scope: no login)

No authentication. Sidebar name field defaulting to **"Rishi"** → lowercased `user_id` keys the KG, session log, and traces. Eval/dashboard pages display the active user ("Memory: Rishi · last 5 sessions"). Typing a different name yields a separate profile — real multi-user structure, zero auth. Recorded caveat: identity is self-asserted (anyone can type any name); acceptable for POC, out of scope to fix in v1.

### 6.3 Memory vs semantic cache (scope clarification)

A semantic cache stores *answers keyed on questions* (user-agnostic replay); memory stores *facts keyed on users* (inputs to fresh generation). The "user accepted John Wick" scenario is a memory write, not a cache hit — replaying the cached response next session would recommend the movie the user already took (S2 at cache speed), while memory keeps generation live and personalized by the fact. Deferring the cache (§12) does not defer memory; memory is core v1.

---

## 7. Framework Decision Record: LangGraph vs LangChain, and Where Pydantic Sits

> **Purpose of this section:** the canonical explanation (and interview artifact) for why this project uses LangGraph for control flow and Pydantic for data contracts, why plain LangChain/LCEL was rejected, and how the two interlock. Includes the worked example.

### 7.1 The one-sentence division of labor

**LangGraph owns control flow — *what happens next*. Pydantic owns data contracts — *what shape everything must have*.** They are not two solutions to one problem; they solve two orthogonal problems that meet in the graph state.

```
┌──────────────────────────────────────────────┐
│  LangGraph   ← control plane (the graph)     │
├──────────────────────────────────────────────┤
│  LangChain   ← components (init_chat_model,  │
│                embedder ifaces, LangSmith)   │
├──────────────────────────────────────────────┤
│  Pydantic    ← data plane (state, inputs,    │
│                outputs, traces)              │
└──────────────────────────────────────────────┘
```

LangGraph is built by the LangChain team **on top of** LangChain. This is not a framework migration — we keep LangChain's *components* (model-agnostic adapter, LangSmith tracing) and replace only its *composition style* (LCEL chains → state graph).

### 7.2 Why LCEL was the initial prescription, and why it was outgrown

| Decision point | Need | Choice | Rationale |
|---|---|---|---|
| Initial spec | Model-agnostic adapter | LangChain (LCEL) | One interface over every provider — cheapest path to model-agnosticism |
| Eval-driven reframing | Retry loops, fallback branches, per-turn path tracing, phase-2 agentic cycles | **LangGraph** | The pipeline stopped being a straight line; it is a state machine |

**Why LCEL fights this design.** An LCEL chain is function composition: `rewrite | extract | retrieve | filter | generate` — a DAG, executed left to right. This pipeline has:
- **Two cycles:** extract→(ValidationError)→extract retry; generate→validate→(violation)→generate retry. LCEL's `with_retry()` retries a step blindly and cannot feed the validation error back into the prompt. Hand-rolled while-loops inside a node bury control flow where no trace can see it.
- **Two branches:** empty-candidates → fallback; retries-exhausted → fallback. `RunnableBranch` works once, becomes unreadable nested.
- **No `path_taken`:** LCEL has no first-class record of which route an execution took — and the entire eval loop (§8) pivots on exactly that.

One-liner: **LCEL composes functions; LangGraph composes decisions.** Phase-2 agentic retrieval (reason→retrieve→observe→repeat) is *only* expressible as a cycle.

Honest caveat (state it in interviews): this could be built without LangGraph — a while-loop and a match statement over a state enum. LangGraph buys visualization, LangSmith integration, checkpointing, and shared vocabulary, at the cost of a dependency. The framework is convenience, not magic.

### 7.3 Graph topology

```python
builder = StateGraph(TurnState)
builder.add_node("rewrite",  rewrite_query)          # cheap LLM, T=0
builder.add_node("extract",  extract_filters)        # cheap LLM, JSON mode, T=0
builder.add_node("retrieve", hybrid_retrieve)        # deterministic
builder.add_node("filter",   apply_metadata_filters) # deterministic
builder.add_node("generate", generate_response)      # generator LLM, T=0.7
builder.add_node("validate", validate_output)        # deterministic (Gate 4)
builder.add_node("fallback", no_match_response)      # deterministic

builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "extract")
builder.add_conditional_edges("extract",             # cycle 1: JSON retry
    lambda s: "extract" if s.extract_failed and s.extract_retries < 2
              else "retrieve")
builder.add_edge("retrieve", "filter")
builder.add_conditional_edges("filter",              # branch 1
    lambda s: "fallback" if len(s.candidates) == 0 else "generate")
builder.add_conditional_edges("validate",            # cycle 2 + branch 2
    lambda s: "generate" if s.validation_failed and s.retries < 2
              else "fallback" if s.validation_failed
              else END)
builder.add_edge("generate", "validate")
builder.add_edge("fallback", END)
```

### 7.4 The five Pydantic enforcement points

| # | Where | Model | Failure prevented |
|---|---|---|---|
| 1 | Ingestion | `MovieRecord` | garbage silently entering the index |
| 2 | LLM extraction | `ParsedQuery` | X2 (malformed JSON); `ValidationError` *triggers* the retry edge |
| 3 | LLM generation | `RecommendationSet` / `MovieRecommendation` | G1/G2 leaking to UI; enables poster rendering |
| 4 | Graph state | `TurnState` | a node writing malformed state corrupting downstream nodes |
| 5 | Observability | `TurnTrace`, `ChangeRecord` | unqueryable, schema-drifting logs |

**The join point is #4: LangGraph's state object is *defined as* a Pydantic model.** Every edge in the graph is implicitly a validation boundary. The orchestrator and the validator jointly implement the architecture thesis: LangGraph makes control flow deterministic and inspectable; Pydantic makes data deterministic and inspectable; the only stochastic tissue left is inside the two LLM nodes — each immediately followed by a Pydantic gate.

> One-liner: *"LangGraph is my control plane, Pydantic is my data plane, and the graph state is itself a Pydantic model — so every hop is both an orchestration step and a validation gate."*

### 7.5 Worked example (embed in tests as the canonical trace)

Turn 2. Turn 1 was "recommend an action movie" → bot suggested *Top Gun: Maverick*. User types: **"something like that but without Tom Cruise."**

```
ENTRY   TurnState(raw_query="something like that but without Tom Cruise",
                  history=[turn 1], user_state={shown: [Top Gun: Maverick]})

rewrite   → rewritten_query = "action movie similar to Top Gun: Maverick
            without Tom Cruise"          # resolves the pronoun "that"

extract   → ParsedQuery(genres=[action], similar_to="Top Gun: Maverick",
            exclude_actors=["Tom Cruise"])
            # if 9B model emits trailing prose → ValidationError → retry edge
            # → 2nd attempt → regex fallback. The retry loop is an EDGE,
            # not a try/except buried in a function.

retrieve  → candidates: [Top Gun: Maverick (rank 1!), John Wick, Mad Max,
            The Raid, ...]
            # cosine loves Top Gun — maximally similar to itself.
            # Retrieval alone cannot save us. As designed.

filter    → exclude_actors drops Top Gun (Cruise in cast);
            shown_movies would drop it anyway (no repeats).
            candidates: [John Wick, Mad Max, The Raid]

(edge)    → len(candidates) > 0 → generate.
            # Had all candidates been filtered → fallback branch:
            # "couldn't find a match — relax which constraint?"

generate  → RecommendationSet(picks=[John Wick ...], prose="...")
            # suppose prose slips in "...unlike Tom Cruise's flying..."

validate  → Gate 4 fails (excluded actor in prose) → validation_failed=True,
            retries=1 → edge routes BACK to generate with the violation
            appended → clean second generation → END.

EXIT      path_taken = [rewrite, extract, retrieve, filter,
                        generate, validate, generate, validate]
          # The double generate→validate visible in the trace IS a logged
          # G2 near-miss caught by the gate. Observability falls out of
          # the orchestration choice for free.
```

Final `TurnState` serializes to `TurnTrace` → SQLite → dashboard.


---

## 8. Eval & Observability Spec (The Spine of the Product)

### 8.1 The central loop

```
OBSERVE → DETECT → TRIAGE → HARDEN → FIX → GATE → (repeat)
```

- **Observe:** every turn writes a full `TurnTrace`.
- **Detect:** automated checks flag failures (exclusion violated, fallback misfired, JSON retries exhausted, low judge score, user-rephrase signal).
- **Triage:** each failure gets exactly one primary taxonomy code + a layer assignment.
- **Harden:** the failure becomes a permanent regression test. **The suite only ever grows.**
- **Fix:** prompt edit / filter logic / threshold tune / model swap — whatever the layer demands.
- **Gate:** full regression suite; promotion iff the new case passes AND zero old cases break AND no topline metric regresses beyond noise bounds.

### 8.2 Failure taxonomy

| Code | Failure class | Layer | Detected by |
|---|---|---|---|
| R1 | relevant docs not retrieved | retrieval | Recall@5; user-rephrase signal |
| R2 | exclusion leaked into candidates | retrieval | pre-filter exclusion audit |
| X1 | filter extraction wrong/incomplete | extraction | ParsedQuery vs labeled parse |
| X2 | malformed JSON, retries exhausted | extraction | retry counter |
| G1 | hallucinated title/fact not in context | generation | Gate 4 + judge faithfulness |
| G2 | instruction violation (exclusion/format) | generation | Gate 4 + judge |
| S1 | constraint dropped across turns | state | multi-turn replay invariants |
| S2 | repetition / stale recommendation | state | shown-movie tracking |
| F1 | deterministic filter bug | filter | unit tests (this is just code) |
| O1 | latency/cost budget exceeded | ops | trace timings |

Fix-type routing: R→index/embedder/hybrid weights · X→extractor prompt or regex fallback · G→generator prompt or model tier · S→state logic (deterministic code) · F→bug fix · O→caching/model tier/budget.

### 8.3 TurnTrace schema (Pydantic point 5)

```python
class TurnTrace(BaseModel):
    trace_id: str; ts: datetime; session_id: str; user_id: str | None
    raw_query: str; rewritten_query: str
    parsed: ParsedQuery | None
    retrieved: list[ScoredMovie]         # pre-filter top-K with scores
    candidates: list[ScoredMovie]        # post-filter
    filters_applied: dict
    prompt_version: str; model_config_version: str; index_version: str
    response: RecommendationSet | None
    path_taken: list[str]                # the graph route — triage pivots on this
    extract_retries: int; gen_retries: int
    timings_ms: dict[str, float]; token_usage: dict[str, int]; cost_usd: float
    flags: list[str]                     # taxonomy codes fired by detection rules
    judge_scores: dict[str, float] | None  # filled async by eval jobs
```

**Capture mechanics (four layers):**
1. **Per-node decorator** — every graph node is wrapped by a `@traced(node_name)` decorator recording timing, `path_taken` entry (`"validate:error"`), and error capture into state. Uniform, unforgettable, free for new nodes. LLM nodes additionally record model, prompt version, token counts, cost, and the **raw pre-validation completion** (failed JSON preserved as X2 evidence, not lost in retry).
2. **Per-turn assembly** — no mid-flight logging; state accumulates everything and serializes **once at `END`**: final `TurnState` → `TurnTrace`, stamped with the lineage triple (`prompt_version`, `model_config_version`, `index_version`), written via the single WAL writer thread. One turn = one row = the complete story; any response traces back to the full set of inputs and artifact versions that produced it.
3. **Detection pass** — post-write deterministic rules append taxonomy `flags` (retries>0, fallback/degradation path, exclusion string present); async jobs fill `judge_scores` later. Production latency never waits on evaluation.
4. **Dev-time spans** — LangSmith (`LANGCHAIN_TRACING_V2`) for raw prompt/response microscopy during development. **SQLite `TurnTrace` is the system of record**; dashboard, taxonomy, and Change Ledger run on it exclusively.

### 8.4 Unified artifact promotion protocol (one gate for everything)

**Prompts, models, indexes, chunking policies, and thresholds are the same kind of thing — versioned behavioral artifacts — and every one passes the same gate.** No privileged artifacts.

1. **Motivate.** Change must cite failure(s): taxonomy codes + trace IDs. No drive-by edits.
2. **Version.** New artifact file with header: date, motivation, expected metric movement.
3. **Gate.** Full regression suite vs candidate. Promote iff: motivating cases pass, zero previously-passing cases fail, no topline metric regresses beyond noise bounds. Stochastic metrics: n≥25 runs, report CIs. *Known statistical power limit: n=25 cannot detect a ~1% adherence regression — state this honestly; do not claim the gate is airtight.*
4. **Canary, promote.** Config pointer flip. Old versions kept forever; rollback = one-line revert.
5. **Trace.** Every production trace records prompt/model/index versions → full lineage from any output to the exact artifacts that produced it.
6. **Emit `ChangeRecord`** automatically at promotion: artifact type, versions before/after, motivating trace IDs, full metric snapshot before/after with CIs, suite size before/after, timestamp. The Change Ledger is emitted by the process, never reconstructed manually.

Thresholds asymmetry (deterministic vs stochastic): exclusion precision post-filter = 1.00 (any violation is a bug); response adherence ≥ 0.98 statistical (LLM variance tolerated, measured).

### 8.5 Eval metric catalog (stage-by-stage; all selectable in the dashboard timeline)

Principle: every pipeline stage gets its own metrics so a topline regression can be attributed to a layer (mirrors the failure taxonomy). Ranking metrics computed at k=5 default, k∈{1,3,5,10} stored.

**A. Ingestion / data quality (offline, per dataset version)**
- Reject rate (rows failing `MovieRecord` validation), per region
- Field completeness (% non-null plot, cast, poster_url, certificate_norm)
- Duplicate rate (movie_id collisions), cast/genre normalization anomaly count

**B. Index / embedding-space health (offline, per index version)**
- Corpus centroid + score distribution baselines (feeds OOD detection)
- Intra-genre vs inter-genre mean cosine separation (embedding discriminative power)
- Embedding cache hit rate at build; index build time

**C. Retrieval (vs labeled set)**
- Recall@k, Precision@k, Hit Rate@k, MRR, **NDCG@k** (position-weighted; needed once graded relevance labels exist)
- **Context Precision / Context Recall** (RAGAS-style: fraction of retrieved chunks that are relevant / fraction of needed info retrieved)
- Exclusion precision post-filter (=1.00 hard), exclusion leak rate pre-filter (R2 monitor)
- Score diagnostics: top-1 score distribution per query category, score spread (top1−topK), fallback rate, OOD rate (distance-to-centroid gate)
- Paraphrase stability (Jaccard overlap of top-K across query paraphrase pairs)

**D. Extraction (vs hand-labeled parses)**
- Per-field exact match (genres, exclude_actors, year/rating bounds, region); slot F1 across fields
- Hallucinated-filter rate (extractor invents a constraint the user never stated)
- JSON validity rate, retry rate, regex-fallback rate (X2 monitors)

**E. Generation (judge + deterministic checks)**
- Faithfulness / groundedness (every claim traceable to context), answer relevance, **context utilization** (did the answer use the retrieved context or ignore it)
- Exclusion adherence (string + judge), hallucinated-title rate (G1: recommended title ∉ candidates — deterministic)
- Format compliance rate (RecommendationSet parses first try), poster_url validity
- Judge reliability: judge–human agreement rate on a monthly 30-sample audit (guards the judge itself)

**F. Multi-turn & memory (replay suite + production)**
- Constraint adherence across turns, reference-resolution accuracy, repetition rate, conversation success rate, turns-to-success
- **Recall precision** (active-recall callbacks engaged / fired; target ≥ 0.7 — below, the rank gate is too loose) and **recall yield** (callbacks that resolved an open loop into a durable triple)

**G. Ops / cost / limits**
- p50/p95/p99 latency per graph node, cost/conversation, tokens per call site, cache hit rates (embedding, retrieval), retry counts, budget-violation rate (O1)
- **Limit-hit metrics (all §5.6 knobs, selectable in the timeline):** input-cap hits per day (chars + tokens), `max_tokens` truncation rate per call site, per-turn retry-cap exhaustions, session-cap hits, daily-budget hits (→ L2 entries), **hard-backstop (provider spend limit) hits** — target 0; any hit is an incident
- Topic-gate refusal rate (scope-escape attempts per day)

**H. Drift monitors (production, no labels needed)**
- Rolling top-1 score P50, fallback rate, query-length and category mix shift, user-rephrase rate (implicit failure signal)

**Statistical testing:** property-based invariants over n≥25–50 runs (e.g., excluded actor never appears, ≥98%); CIs displayed on all stochastic metrics.

### 8.6 Dashboard: the one-page Change Ledger

Single Streamlit page (part of the deployed artifact), reading SQLite. Three elements, progressive disclosure:

1. **Topline strip** (small, one row): current Recall@5, exclusion precision, adherence, faithfulness, retry rate, p95, $/conv — each with threshold status color.
2. **Metric timeline** (centerpiece): one metric-selectable chart where **every promoted change is a vertical marker**, clickable. The historical evidence of improvement, made literal: Recall@5 climbing 0.70→0.87 with every step named and quantified.
3. **Change ledger table** (newest first): ID · date · artifact type (chunking/embedder/prompt/model/filter/threshold) · what changed · why (taxonomy codes × counts) · quantified metric deltas. Row click expands: motivating trace IDs, full before/after metric table with CIs, test cases that flipped status, suite size at promotion.

Drill-downs replace the previously considered multi-page design: change → detail; failure → trace view; cost/latency = more metrics in the timeline selector. LangSmith remains for deep dev-time trace inspection; the dashboard owns curated metrics (self-contained on HF Spaces, and building it is the showcased skill).

### 8.7 Eval dataset spec (proposal — react to this)

**Target: ~150 single-turn labeled queries + 30 multi-turn scripted conversations.**

Single-turn coverage matrix (authoring: Rishi labels ground-truth relevant `movie_id`s; Claude Code drafts candidate queries per cell for review):

| Category | Count | Notes |
|---|---|---|
| Standard semantic ("gritty revenge thriller") | 40 | R1 coverage |
| Exact title / proper noun | 15 | BM25 justification |
| Negation / exclusion ("without X", "nothing scary") | 25 | R2/X1/G2 coverage — the signature category |
| Short (1–2 words) | 15 | calibration / noisy embeddings |
| Region-conditioned ("Indian thrillers", certificate filters) | 20 | US/IN split; §2.2 |
| Numeric filters (year/rating ranges) | 15 | X1 coverage |
| Ambiguous / lexical-overlap traps | 10 | "bank heist" vs "Bank of America" class |
| OOD / unanswerable | 10 | fallback behavior; correct answer = fallback |

Multi-turn scripts (30): each 3–6 turns with code-verifiable invariants (constraint set in turn 1 must hold at turn N; no repeats; reference resolution). Origin discipline thereafter: **every production failure becomes a new labeled case** — the dataset, like the suite, only grows.

---

## 9. UI Spec (Streamlit)

- `st.chat_message` conversation; `st.chat_input`.
- Recommendation cards render from **structured** `MovieRecommendation` fields: `st.image(rec.poster_url)` + title/year/reason. No markdown parsing.
- Fallback turns render the relax-a-constraint prompt with buttons (deterministic quick replies).
- Second page: the Change Ledger dashboard (§8.6). Deployed together — the demo IS the harness.

## 10. Deployment Spec (Hugging Face Spaces)

- Streamlit SDK Space; single secret `OPENROUTER_API_KEY`.
- Constraints designed around: ~16GB disk, 2 vCPU, **ephemeral storage on restart** → prebuilt FAISS/BM25/SQLite artifacts ship in the repo (or HF Datasets); local embedder default; per-user KG persistence best-effort on free tier (durable option: HF Datasets sync or external libSQL).
- CI: GitHub Actions → regression suite (the gate) → HF Space auto-deploy on pass. A red suite blocks deploy — the gate is not advisory.

## 11. Milestones (sequenced for Claude Code sessions; each independently testable)

**Instrument first, build second — the instrument is the product.**

| M | Deliverable | Acceptance |
|---|---|---|
| M1 | Schemas (`MovieRecord`, `ParsedQuery`, `TurnState`, `TurnTrace`, `ChangeRecord`) + SQLite trace store + dashboard skeleton (empty ledger renders) | pytest on all schemas; dashboard loads; trace store opens with `journal_mode=WAL` and all trace writes go through a single writer thread (concurrent-session write test passes) |
| M2 | Ingestion + normalization (US, then IN to prove the adapter claim) + embedding cache + versioned FAISS/BM25 build | rejects table populated on bad rows; index pointer file works; re-run is cache-hot |
| M3 | Hybrid retrieval + RRF + filters + retrieval cache + labeled dataset v1 (§8.7) + retrieval eval harness | Recall@5 ≥ 0.85, exclusion precision = 1.00 on labeled set |
| M4 | LangGraph pipeline (full topology §7.3 + `recall_check` node §6.2 + **topic gate and input caps §5.6**) + prompts v1 + structured outputs + Gate 4 + memory tiers with 5-session retention (§6.4) + name-field identity (§6.5) + Streamlit chat with posters and feedback chips (§6.1) | worked example (§7.5) passes as an integration test; multi-turn replay suite green; collision-recall reference cases (§6.2) pass; chips write triples and flag traces; 6th session triggers distill-and-drop; oversize input truncates gracefully; off-topic query hits refusal template; `max_tokens` set on every LLM call |
| M5 | Full loop live: detection rules, triage tooling, promotion protocol emitting ChangeRecords, Change Ledger dashboard complete (incl. §8.5.G limit-hit metrics), CI gate wired to HF deploy, **log-growth hygiene**: tiered trace retention (full 30 days → compact slim rows; Change-Ledger/regression-referenced traces never deleted), nightly compact + `VACUUM` job, cold archive to compressed JSONL, daily budget → L2 degradation wired | one end-to-end demonstrated cycle: seeded failure → taxonomy → fix → gate → ledger row; retention job demonstrated on synthetic aged traces; budget exhaustion demonstrably enters L2, not an outage |
| M6 (phase 2) | DSPy on rewriter/extractor (automated fix-author, same gate) · agentic multi-hop (one added cycle) · additional regions · semantic cache revisit | each behind the same promotion protocol |

## 12. Deferred-with-Rationale Register

| Item | Status | Rationale |
|---|---|---|
| Semantic response cache | deferred | conflicts with per-user KG personalization and multi-turn statefulness; right for stateless FAQ bots, wrong here |
| Function-calling API for extraction | rejected | inconsistent across OpenRouter models; JSON mode + Pydantic is portable |
| Neo4j for KG | rejected | SQLite triples sufficient at this scale |
| Frontier models as defaults | rejected | they mask system-design flaws; cheap models are the failure-mode generators this project needs |
| Elasticsearch for sparse | rejected | rank_bm25 is zero-infra and sufficient |
| DSPy in v1 | deferred to M6 | earn the vocabulary: v1 hand-written prompts + evals establish the baseline DSPy must beat through the same gate |
