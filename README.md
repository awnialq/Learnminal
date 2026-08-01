# Learnminal

An agentic terminal you can learn with — an Alacritty fork with an Ollama-powered chat overlay (`Ctrl+Shift+E`).

## Build and run

```bash
# CLI binary → target/release/learnminal
cargo build -p alacritty --release
./target/release/learnminal

# macOS app bundle → target/release/osx/Learnminal.app
make app
open target/release/osx/Learnminal.app
```

The release binary is named **learnminal** (not `alacritty`) so you can tell it apart from a stock Alacritty install:

```bash
which learnminal   # should point at this repo's target/release/learnminal
```

## AI chat

```bash
# Start Ollama (separate terminal)
ollama serve
```

Then press `Ctrl+Shift+E` in Learnminal to open Chat and ask a question about your terminal. Opening Chat preloads the active model so the first response starts faster. The app talks directly to Ollama; no Python sidecar is required. When you submit a chat question, Learnminal may include a concise `man`/`--help` excerpt for the last command as hidden context.

The model can call a `web_search` tool (DuckDuckGo) when it needs up-to-date information. Disable with `LEARNMINAL_WEB_SEARCH=0`.

### Actions panel

Once an answer finishes, the model re-reads its own reply through a `list_actions` tool
call and pulls out the commands you can actually run. They appear in an **Actions** panel in
the top-right corner, numbered, each with a few words on what it is for:

```
 Actions
 1. ls -l        show as a list
 2. du -sh *     check dir sizes
```

The panel appears on its own when there is something to show, and stays up after you close
the chat overlay with `Esc` so you can read the commands while typing. Nothing is executed
or typed into your shell — the panel is display-only. Dismiss it with `/actions clear`, or
disable the feature entirely with `LEARNMINAL_ACTIONS=0`.

Slash commands in the overlay:

- `/model list` — show installed Ollama models
- `/model <name>` — switch the active model
- `/level` — show experience levels
- `/level <beginner|novice|professional|expert>` — set experience level for explanations
- `/info` — show cached system environment
- `/actions` — list the current actions in the transcript
- `/actions clear` — dismiss the Actions panel

Default model: `gemma4:e4b-mlx` on macOS, `gemma4:e4b` elsewhere. Override with
`LEARNMINAL_OLLAMA_MODEL` or persist a choice via `/model`.

Default experience level is `beginner`. Persist a choice via `/level` in
`~/.ai-cli-learning/settings.json`.
