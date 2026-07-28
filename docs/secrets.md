# Secrets management (dotenvx)

Secrets live in the repo as an **encrypted** `.env`, using [dotenvx](https://dotenvx.com).
Values in the committed `.env` are ciphertext; the private key that decrypts them is
**never** committed.

## Files

| File | Committed? | Contents |
|------|-----------|----------|
| `.env` | **yes** | Variable names + **encrypted** values + a `DOTENV_PUBLIC_KEY` |
| `.env.keys` | **no** (git-ignored) | The `DOTENV_PRIVATE_KEY_*` that decrypts `.env` |
| `.env.example` | yes | Placeholder template (documentation only, no values) |

## Add or change a secret

```bash
# 1. Set the plaintext value (dotenvx encrypts it in place, keeps everything else encrypted)
npx @dotenvx/dotenvx set OPENROUTER_API_KEY "sk-or-..."
# 2. Commit the updated (still-encrypted) .env
git add .env && git commit -m "chore: update OPENROUTER_API_KEY"
```

Or edit `.env` values in plaintext then run `npx @dotenvx/dotenvx encrypt` before committing.

## Run the app / scripts with decrypted secrets

```bash
npx @dotenvx/dotenvx run -- python -m imdb_chatbot ...
```

dotenvx decrypts using `.env.keys` locally at runtime; nothing is written to disk in plaintext.

## CI and Hugging Face Spaces

These environments don't have `.env.keys`, so give them the private key as a platform secret:

- **GitHub Actions:** repo → Settings → Secrets → Actions → add `DOTENV_PRIVATE_KEY` (value from `.env.keys`). Then run steps under `npx @dotenvx/dotenvx run -- ...`.
- **Hugging Face Spaces:** Space → Settings → Secrets → add `DOTENV_PRIVATE_KEY`.

## Guardrails

- A pre-commit hook (`dotenvx ext precommit`) blocks committing an unencrypted `.env`.
- `.env.keys` and all plaintext `.env.*` variants are git-ignored.
- If a private key is ever exposed, rotate it (`dotenvx rotate`) **and** rotate the underlying provider keys.

## Which keys

| Key | Used by | Scope |
|-----|---------|-------|
| `OPENROUTER_API_KEY` | LLM calls | runtime (app + CI) |
| `HF_TOKEN` | durable memory (HF Dataset-sync, #22) | runtime |
| `TMDB_API_KEY` / `TMDB_READ_ACCESS_TOKEN` | offline ingestion pull (#14) | build-time only — not needed on Spaces |
