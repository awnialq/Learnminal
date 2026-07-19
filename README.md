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

Slash commands in the overlay:

- `/model list` — show installed Ollama models
- `/model <name>` — switch the active model
- `/info` — show cached system environment

Optional: set a default model with `LEARNMINAL_OLLAMA_MODEL` or persist a choice via `/model`.
