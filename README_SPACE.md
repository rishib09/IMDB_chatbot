---
title: IMDb Chatbot
emoji: "\U0001F3AC"
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# IMDb Chatbot - Hugging Face Space

This file is the Hugging Face Spaces "Space card". When you create the Space,
copy the YAML header above into the Space's own `README.md` (HF reads the config
from there). The Streamlit SDK runs `app.py`, which performs the L3 startup
health check before serving and degrades honestly if the index is unavailable.

## Secrets this Space expects (set VALUES in Space settings, never in git)

Set these under Space -> Settings -> Variables and secrets. Only the NAMES are
listed here; the values are entered in the Space UI and are never committed:

- `OPENROUTER_API_KEY` - OpenRouter key for the LLM slots (generation/extraction).
- `HF_TOKEN` - Hugging Face token, used to pull the prebuilt index artifacts from
  a private HF Dataset at startup (see docs/deploy.md), and by the deploy
  workflow to push to this Space.
- `DOTENV_PRIVATE_KEY` - dotenvx private key that decrypts the committed `.env`
  ciphertext at runtime.

## Index artifacts

The Space will only serve when the live index is present (the L3 health check
refuses to serve without it). Ship the index either by committing it to the Space
with git-lfs or by syncing it from an HF Dataset at startup. See `docs/deploy.md`
for the owner go-live checklist.
