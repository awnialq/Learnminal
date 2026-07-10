# Learnminal

An agentic terminal you can learn with — an Alacritty fork with AI-powered explanations (`Ctrl+Shift+E`).

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

## AI backend (stub for testing)

```bash
pip install fastapi uvicorn pydantic
python ai-backend/server_stub.py
```

Then press `Ctrl+Shift+E` in Learnminal.
