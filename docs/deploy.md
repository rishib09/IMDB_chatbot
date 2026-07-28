# Deploy guide - Hugging Face Spaces (ticket #26)

This is the OWNER go-live checklist. The deploy is CI-gated: the
`.github/workflows/deploy.yml` deploy job `needs:` the test job, so a RED
regression suite (ruff or pytest) BLOCKS the push to the Space. Nothing ships
unless the suite is green.

## Architecture recap

- `app.py` (repo root) is the HF Spaces Streamlit entry point. At startup it runs
  the L3 health check (`guards.degradation.health_check`): the live-index pointer
  must resolve, the index artifacts must exist, and the trace DB must open. If any
  check fails it renders the honest "index unavailable" message instead of
  crashing. When healthy, budget exhaustion degrades to L2 (LLM-free mode) rather
  than erroring.
- `requirements.txt` lists the runtime deps so the Space can build.
- `README_SPACE.md` holds the HF Spaces config YAML header (sdk, app_file,
  python_version) and the names of the secrets the Space expects.

## Go-live checklist

### 1. Create the HF Space

- Create a new Space at `https://huggingface.co/spaces/<owner>/<space>`.
- Choose the **Streamlit** SDK.
- Copy the YAML header from `README_SPACE.md` into the Space's own `README.md`
  (HF reads the Space config from the README front matter). Set `app_file: app.py`
  and `python_version: "3.12"`.

### 2. Set the Space secrets

Under Space -> Settings -> Variables and secrets, add these (values entered in
the HF UI, never committed to git):

- `OPENROUTER_API_KEY` - OpenRouter key for the LLM slots.
- `HF_TOKEN` - HF token used to pull the prebuilt index artifacts from an HF
  Dataset at startup (see step 3, option B).
- `DOTENV_PRIVATE_KEY` - dotenvx private key that decrypts the committed `.env`
  ciphertext at runtime.

No secret VALUE appears anywhere in this repo - only the names.

### 3. Ship the prebuilt index artifacts

The index artifacts (`index.faiss`, `bm25.pkl`, `sidecar.json`) are git-ignored
and large (about 70 MB), so they are NOT in this repo. The L3 health check will
(correctly) REFUSE to serve until they are present and `config/live_index.json`
points at them with a non-null `active` version. Pick one:

- **Option A - commit to the Space via git-lfs.** In the Space repo:
  `git lfs install`, `git lfs track "*.faiss" "*.pkl"`, add the artifacts under
  the path `config/live_index.json` points to, commit, and push. Simplest, but
  the artifacts live in the Space git history.
- **Option B - sync from an HF Dataset at startup.** Upload the artifacts to a
  (private) HF Dataset, and have startup download them with `HF_TOKEN` via
  `huggingface_hub` before the health check runs. Keeps the Space repo small; the
  Dataset is the single source of truth for the index.

Either way, after the artifacts land, set `config/live_index.json` `active` to the
built version and `path` to the artifact directory so the health check passes.

### 4. Add the GitHub deploy secret

- In the GitHub repo -> Settings -> Secrets and variables -> Actions, add
  `HF_TOKEN` (a write-scoped HF token). The deploy workflow reads it as
  `secrets.HF_TOKEN`; it is never hardcoded.
- Edit `SPACE_ID` in `.github/workflows/deploy.yml` from the placeholder
  `your-hf-username/imdb-chatbot` to your real `<owner>/<space>`.

### 5. Deploy

Push to `main`. The `test` job runs ruff + pytest; on green, the `deploy` job
pushes the repo to the Space git remote:

```
git push --force "https://USER:$HF_TOKEN@huggingface.co/spaces/<owner>/<space>" HEAD:main
```

If the suite is RED, the deploy job never runs and the Space is left untouched.

## Health check reminder

The index MUST be present or the L3 health check will refuse to serve - that is
by design (degrade honestly, never serve broken). If the Space shows "index
unavailable", re-check step 3: the artifacts and the `config/live_index.json`
pointer.
