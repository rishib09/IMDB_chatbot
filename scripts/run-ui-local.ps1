# Launch the Chat UI locally in LIVE mode.
#
# What it does:
#   1. Flips config/live_index.json to the on-disk MiniLM index so the L3 health
#      check passes. This is a LOCAL-ONLY edit - do NOT commit config/live_index.json
#      (git keeps it at active=null on purpose, for CI and deploy safety).
#   2. Launches Streamlit under dotenvx so OPENROUTER_API_KEY is decrypted and
#      injected, enabling real recommendations + token/cost telemetry.
#
# Usage (from the repo root):
#   .\scripts\run-ui-local.ps1
#
# Stop with Ctrl+C. To restore the committed pointer afterwards:
#   git checkout -- config/live_index.json

$ErrorActionPreference = "Stop"

$indexName = "dev_all-MiniLM-L6-v2_v1_one_per_movie"
$indexPath = "data/index/$indexName"

if (-not (Test-Path $indexPath)) {
    Write-Error "Local index not found at $indexPath. Build it first (ticket #15) or adjust the name in this script."
}

Write-Host "Flipping live-index pointer to $indexName (local only)..."
& .\.venv\Scripts\python.exe -c "from pathlib import Path; from imdb_chatbot.index.build import write_live_pointer; write_live_pointer('$indexName', Path('$indexPath'))"

Write-Host "Launching Streamlit under dotenvx (real OPENROUTER_API_KEY)..."
npx --yes "@dotenvx/dotenvx@latest" run -f .env -- .\.venv\Scripts\streamlit.exe run app.py
