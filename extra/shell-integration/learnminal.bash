# learnminal-shell-integration v1
#
# Records one JSONL line per command so the Learnminal overlay (Ctrl+Shift+E) can
# show the AI your last few commands with their exact exit codes. Command output
# is NOT written here — the terminal recovers that from its own screen buffer.
#
# Install by adding this line to ~/.bashrc:
#
#   [ -f "$HOME/.ai-cli-learning/shell/learnminal.bash" ] && source "$HOME/.ai-cli-learning/shell/learnminal.bash"
#
# Then open a new Learnminal window. Safe to source more than once.
# Set LEARNMINAL_NO_HISTORY=1 to disable recording.
#
# CAVEATS (bash has no preexec hook):
#   * We install a DEBUG trap. If you already set your own `trap ... DEBUG`
#     directly — rather than through bash-preexec / starship / oh-my-bash, which
#     we detect and cooperate with — whichever is sourced last wins.
#   * _learnminal_precmd must run FIRST in PROMPT_COMMAND to see the real $?,
#     so we prepend it.
#   * Requires shell history to be on (`set -o history`, the interactive default).
#
# Privacy: command lines are stored under ~/.ai-cli-learning/sessions/ (mode 700)
# and sent to the local Ollama model along with the visible screen text. Secrets
# typed as command arguments will be included.

[[ -n "$LEARNMINAL_SESSION_FILE" ]] || return 0
[[ "$LEARNMINAL_SESSION_VERSION" == "1" ]] || return 0
[[ -z "$LEARNMINAL_NO_HISTORY" ]] || return 0
[[ -n "$BASH_VERSION" && $- == *i* ]] || return 0

_LEARNMINAL_DIR="${LEARNMINAL_SESSION_FILE%/*}"
_LEARNMINAL_LEGACY="${_LEARNMINAL_DIR%/*}"
if [[ ! -d "$_LEARNMINAL_DIR" ]]; then
  mkdir -p "$_LEARNMINAL_DIR" 2>/dev/null && chmod 700 "$_LEARNMINAL_DIR" 2>/dev/null
fi
_LEARNMINAL_SEQ=0
_LEARNMINAL_RAN=0
_LEARNMINAL_START=0

# JSON-escape $1 into $REPLY. Pure parameter expansion, no forks.
_learnminal_json_escape() {
  local s=$1
  s=${s//\\/\\\\}              # backslash FIRST, before we introduce our own
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  s=${s//$'\r'/\\r}
  s=${s//$'\t'/\\t}
  s=${s//[$'\001'-$'\037']/}   # drop any remaining raw control characters
  REPLY=$s
}

# Fires per simple command, so `a | b` would trip it twice. We only record that
# *something* ran; the command text comes from `history 1` at prompt time.
_learnminal_debug() {
  [[ -n "$COMP_LINE" ]] && return                    # tab completion, not a command
  [[ "$BASH_COMMAND" == _learnminal_precmd* ]] && return
  (( _LEARNMINAL_RAN )) && return
  _LEARNMINAL_RAN=1
  _LEARNMINAL_START=${EPOCHSECONDS:-0}
}

_learnminal_precmd() {
  local code=$?
  (( _LEARNMINAL_RAN )) || return 0
  _LEARNMINAL_RAN=0

  local line
  line=$(HISTTIMEFORMAT='' builtin history 1)        # "  123  git status"
  line=${line#"${line%%[![:space:]]*}"}              # ltrim
  line=${line#* }                                    # drop the history number
  line=${line#"${line%%[![:space:]]*}"}              # ltrim again
  [[ -n "$line" ]] || return 0
  line=${line:0:512}

  (( _LEARNMINAL_SEQ++ ))
  local cmd_esc cwd_esc
  _learnminal_json_escape "$line"
  cmd_esc=$REPLY
  _learnminal_json_escape "${PWD:0:256}"
  cwd_esc=$REPLY

  # Kept under PIPE_BUF (4096) so O_APPEND writes stay atomic when several shells
  # (tmux panes) share one session file.
  printf '{"v":1,"seq":%d,"pid":%d,"cmd":"%s","exit":%d,"cwd":"%s","start":%d,"end":%d}\n' \
    "$_LEARNMINAL_SEQ" "$$" "$cmd_esc" "$code" "$cwd_esc" "$_LEARNMINAL_START" "${EPOCHSECONDS:-0}" \
    >> "$LEARNMINAL_SESSION_FILE" 2>/dev/null

  # Legacy single-value files, still read as a fallback by older code paths.
  printf '%s\n' "$line" > "$_LEARNMINAL_LEGACY/last_command"   2>/dev/null
  printf '%s\n' "$code" > "$_LEARNMINAL_LEGACY/last_exit_code" 2>/dev/null

  # Rotation: one fork per 500 commands keeps the file around ~100 KB.
  if (( _LEARNMINAL_SEQ % 500 == 0 )); then
    local tmp="$LEARNMINAL_SESSION_FILE.$$"
    if tail -n 50 "$LEARNMINAL_SESSION_FILE" > "$tmp" 2>/dev/null; then
      mv -f "$tmp" "$LEARNMINAL_SESSION_FILE" 2>/dev/null
    fi
    rm -f "$tmp" 2>/dev/null
  fi
}

# Cooperate with bash-preexec / starship / oh-my-bash instead of stealing their
# DEBUG trap. Guard against double-registration when sourced twice.
if [[ -n "${preexec_functions+x}" ]]; then
  [[ " ${preexec_functions[*]} " == *" _learnminal_debug "* ]] || preexec_functions+=(_learnminal_debug)
  [[ " ${precmd_functions[*]} " == *" _learnminal_precmd "* ]] || precmd_functions+=(_learnminal_precmd)
else
  trap '_learnminal_debug' DEBUG
  # bash 5.1+ allows PROMPT_COMMAND to be an array. Detect it with `declare -p`
  # rather than `${PROMPT_COMMAND@a}`, which is a parse error on bash 3.2 (still
  # the system bash on macOS) and would abort the whole script.
  if [[ "$(declare -p PROMPT_COMMAND 2>/dev/null)" == "declare -a"* ]]; then
    [[ " ${PROMPT_COMMAND[*]} " == *" _learnminal_precmd "* ]] ||
      PROMPT_COMMAND=(_learnminal_precmd "${PROMPT_COMMAND[@]}")
  elif [[ "$PROMPT_COMMAND" != *_learnminal_precmd* ]]; then
    PROMPT_COMMAND="_learnminal_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
  fi
fi
