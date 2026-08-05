# learnminal-shell-integration v1
#
# Records one JSONL line per command so the Learnminal overlay (Ctrl+Shift+E) can
# show the AI your last few commands with their exact exit codes. Command output
# is NOT written here — the terminal recovers that from its own screen buffer.
#
# Install by adding this line to ~/.zshrc:
#
#   [ -f "$HOME/.ai-cli-learning/shell/learnminal.zsh" ] && source "$HOME/.ai-cli-learning/shell/learnminal.zsh"
#
# Then open a new Learnminal window. Safe to source more than once.
# Set LEARNMINAL_NO_HISTORY=1 to disable recording.
#
# Privacy: command lines are stored under ~/.ai-cli-learning/sessions/ (mode 700)
# and sent to the local Ollama model along with the visible screen text. Secrets
# typed as command arguments will be included.

[[ -n "$LEARNMINAL_SESSION_FILE" ]] || return 0
[[ "$LEARNMINAL_SESSION_VERSION" == "1" ]] || return 0
[[ -z "$LEARNMINAL_NO_HISTORY" ]] || return 0
[[ -o interactive ]] || return 0
autoload -Uz add-zsh-hook 2>/dev/null || return 0
zmodload zsh/datetime 2>/dev/null

typeset -g  _LEARNMINAL_CMD=""
typeset -gi _LEARNMINAL_SEQ=0
typeset -gi _LEARNMINAL_START=0
typeset -g  _LEARNMINAL_DIR="${LEARNMINAL_SESSION_FILE:h}"
typeset -g  _LEARNMINAL_LEGACY="${_LEARNMINAL_DIR:h}"

if [[ ! -d "$_LEARNMINAL_DIR" ]]; then
  mkdir -p "$_LEARNMINAL_DIR" 2>/dev/null && chmod 700 "$_LEARNMINAL_DIR" 2>/dev/null
fi

# JSON-escape $1 into $REPLY. Pure parameter expansion: this runs on every prompt,
# and zsh forks a subshell for $(...).
_learnminal_json_escape() {
  local s=$1
  s=${s//\\/\\\\}              # backslash FIRST, before we introduce our own
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  s=${s//$'\r'/\\r}
  s=${s//$'\t'/\\t}
  s=${s//[$'\x01'-$'\x1f']/}   # drop any remaining raw control characters
  REPLY=$s
}

_learnminal_preexec() {
  # $1 is the full command line as typed, before alias/history expansion.
  _LEARNMINAL_CMD=${1[1,512]}
  _LEARNMINAL_START=${EPOCHSECONDS:-0}
}

_learnminal_precmd() {
  local code=$?
  [[ -n "$_LEARNMINAL_CMD" ]] || return 0
  local cmd=$_LEARNMINAL_CMD
  _LEARNMINAL_CMD=""
  (( _LEARNMINAL_SEQ++ ))

  local cmd_esc cwd_esc
  _learnminal_json_escape "$cmd"
  cmd_esc=$REPLY
  _learnminal_json_escape "${PWD[1,256]}"
  cwd_esc=$REPLY

  # Kept under PIPE_BUF (4096) so O_APPEND writes stay atomic when several shells
  # (tmux panes) share one session file.
  print -r -- "{\"v\":1,\"seq\":$_LEARNMINAL_SEQ,\"pid\":$$,\"cmd\":\"$cmd_esc\",\"exit\":$code,\"cwd\":\"$cwd_esc\",\"start\":$_LEARNMINAL_START,\"end\":${EPOCHSECONDS:-0}}" \
    >> "$LEARNMINAL_SESSION_FILE" 2>/dev/null

  # Legacy single-value files, still read as a fallback by older code paths.
  print -r -- "$cmd"  > "$_LEARNMINAL_LEGACY/last_command"   2>/dev/null
  print -r -- "$code" > "$_LEARNMINAL_LEGACY/last_exit_code" 2>/dev/null

  # Rotation: one fork per 500 commands keeps the file around ~100 KB.
  if (( _LEARNMINAL_SEQ % 500 == 0 )); then
    local tmp="$LEARNMINAL_SESSION_FILE.$$"
    if tail -n 50 "$LEARNMINAL_SESSION_FILE" > "$tmp" 2>/dev/null; then
      mv -f "$tmp" "$LEARNMINAL_SESSION_FILE" 2>/dev/null
    fi
    rm -f "$tmp" 2>/dev/null
  fi
}

# add-zsh-hook dedupes by function name, so re-sourcing does not double-register.
add-zsh-hook preexec _learnminal_preexec
add-zsh-hook precmd  _learnminal_precmd
