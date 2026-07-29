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
4. If web search is enabled (default), the model may call a `web_search` tool
   backed by DuckDuckGo (max two tool rounds). Set `LEARNMINAL_WEB_SEARCH=0` to
   disable. Flag verification still uses local Reference only.
5. The Rust client streams the final Ollama NDJSON answer into the overlay.

`/model` lists or selects installed Ollama models. The selected model is stored
in `~/.ai-cli-learning/settings.json`; the selection order is persisted model,
`LEARNMINAL_OLLAMA_MODEL`, the platform default (`gemma4:e4b-mlx` on macOS,
`gemma4:e4b` elsewhere — with the non-MLX tag as a macOS fallback), then the
first installed model. `/level` sets the user experience tier
(`beginner`/`novice`/`professional`/`expert`, default `beginner`) in the same
settings file so chat explanations match terminal knowledge. `/info` reports
locally collected system information.

Closing or replacing a chat request invalidates stale stream events and asks
Ollama to unload the active model. Chat replies are displayed as plain text;
there is no manual-summary path or actionable-command HUD.
