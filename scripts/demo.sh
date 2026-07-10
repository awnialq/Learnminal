#!/usr/bin/env bash
# Learnminal demo scenarios (Task 8.3)
#
# Prerequisites:
#   Terminal 1: cd ai-backend && python3 server.py
#   Terminal 2: ollama serve   (if not already running)
#
# Scenario 1: Successful git rebase explanation
# Scenario 2: Failed command with error remediation

set -euo pipefail

BACKEND="http://127.0.0.1:8765"

_check_backend() {
    if ! curl -sf "$BACKEND/health" >/dev/null 2>&1; then
        echo "ERROR: Backend not running. Start with: cd ai-backend && python3 server.py" >&2
        exit 1
    fi
}

_stream() {
    local payload="$1"
    curl -sN \
        -H "Accept: text/event-stream" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BACKEND/explain"
}

# ── Scenario 1: Successful git rebase ────────────────────────────────────────
demo_scenario1() {
    echo "=== Scenario 1: git rebase -i HEAD~3 (success) ==="
    _stream '{
        "request_id": "demo-scenario-1",
        "timestamp": '"$(date +%s)"',
        "terminal": {
            "visible_text": "$ git rebase -i HEAD~3\nSuccessfully rebased and updated refs/heads/main.",
            "selected_text": null,
            "last_command": "git rebase -i HEAD~3",
            "cwd": "/home/user/project",
            "exit_code": 0,
            "rows": 24,
            "cols": 80
        }
    }'
    echo
}

# ── Scenario 2: Failed command with error remediation ────────────────────────
demo_scenario2() {
    echo "=== Scenario 2: git push origin main (failed — error remediation) ==="
    _stream '{
        "request_id": "demo-scenario-2",
        "timestamp": '"$(date +%s)"',
        "terminal": {
            "visible_text": "$ git push origin main\nerror: failed to push some refs to '"'"'https://github.com/user/repo'"'"'\nhint: Updates were rejected because the remote contains work that you do not have locally.",
            "selected_text": null,
            "last_command": "git push origin main",
            "cwd": "/home/user/project",
            "exit_code": 1,
            "rows": 24,
            "cols": 80
        }
    }'
    echo
}

# ── Main ─────────────────────────────────────────────────────────────────────
_check_backend

case "${1:-all}" in
    1) demo_scenario1 ;;
    2) demo_scenario2 ;;
    all)
        demo_scenario1
        echo
        demo_scenario2
        ;;
    *)
        echo "Usage: $0 [1|2|all]"
        exit 1
        ;;
esac
