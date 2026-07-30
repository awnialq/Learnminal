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

Slash commands in the overlay:

- `/model list` — show installed Ollama models
- `/model <name>` — switch the active model
- `/level` — show experience levels
- `/level <beginner|novice|professional|expert>` — set experience level for explanations
- `/info` — show cached system environment
- `/history` — show the recent commands the AI can see

## Shell integration (recommended)

Without it, Learnminal reconstructs your recent commands by scanning the terminal
screen for prompt characters, which cannot recover exit codes for anything but the
very last command. The shell hook records each command's exact text, exit code, and
working directory instead.

Learnminal writes the scripts to `~/.ai-cli-learning/shell/` on startup. Add one line
to your shell rc file:

```bash
# ~/.zshrc
[ -f "$HOME/.ai-cli-learning/shell/learnminal.zsh" ] && source "$HOME/.ai-cli-learning/shell/learnminal.zsh"

# ~/.bashrc
[ -f "$HOME/.ai-cli-learning/shell/learnminal.bash" ] && source "$HOME/.ai-cli-learning/shell/learnminal.bash"
```

Then open a new Learnminal window and run `/history` to confirm it says
`Source: shell hook`.

Each command appends one JSON line to `~/.ai-cli-learning/sessions/<session-id>.jsonl`
(directory mode 700). Files are rotated at 500 commands and swept after 24 hours.

**Privacy.** The commands you run, and the output visible on screen, are sent to your
local Ollama model when you ask a question. Secrets typed as command arguments are
included. Set `LEARNMINAL_NO_HISTORY=1` to turn recording off for a shell. Command
output is never written to disk — only to the model, and only from the live screen.

**bash caveats.** bash has no `preexec` hook, so the script installs a `DEBUG` trap and
prepends to `PROMPT_COMMAND`. It detects and cooperates with bash-preexec, starship, and
oh-my-bash; a hand-rolled `trap ... DEBUG` of your own will conflict. Shell history must
be enabled (the interactive default).

Default model: `gemma4:e4b-mlx` on macOS, `gemma4:e4b` elsewhere. Override with
`LEARNMINAL_OLLAMA_MODEL` or persist a choice via `/model`.

Default experience level is `beginner`. Persist a choice via `/level` in
`~/.ai-cli-learning/settings.json`.
