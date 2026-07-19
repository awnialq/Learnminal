# Rust-native Learnminal architecture

Learnminal is a Rust-native Alacritty fork that communicates directly with the
Ollama daemon at `http://127.0.0.1:11434` (or `OLLAMA_HOST`). No Python
sidecar, local HTTP API, or IPC contract is required.

## Runtime flow

1. Start Ollama with `ollama serve`.
2. Press `Ctrl+Shift+E` to open the Chat overlay. Learnminal loads the active
   model into memory and keeps it resident while Chat is open, but does not
   send an automatic question or summarize a manual.
3. Submit a question. Learnminal gathers terminal context and can add a concise
   `man`/`--help` excerpt for the last command as hidden context.
4. The Rust client streams Ollama NDJSON responses directly into the overlay.

`/model` lists or selects installed Ollama models. The selected model is stored
in `~/.ai-cli-learning/settings.json`; the selection order is persisted model,
`LEARNMINAL_OLLAMA_MODEL`, the built-in default, then the first installed
model. `/info` reports locally collected system information.

Closing or replacing a chat request invalidates stale stream events and asks
Ollama to unload the active model. Chat replies are displayed as plain text;
there is no manual-summary path or actionable-command HUD.
