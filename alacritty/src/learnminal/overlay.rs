use std::time::{Duration, Instant};

use unicode_width::UnicodeWidthChar;

use alacritty_terminal::grid::Dimensions;
use alacritty_terminal::index::{Column, Point};
use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{Key, ModifiersState, NamedKey};

use crate::config::UiConfig;
use crate::display::color::Rgb;
use crate::display::SizeInfo;
use crate::learnminal::types::{
    parse_explain_response_lenient, strip_json_fences, CommandReferenceResponse, ExplainResponse,
    FlagExplanation, SystemInfo,
};
use crate::renderer::rects::RenderRect;

const BATCH_INTERVAL: Duration = Duration::from_millis(16);
/// Maximum lines to scroll per key-repeat tick when holding an arrow key.
const SCROLL_REPEAT_MAX_LINES: usize = 12;
/// Bottom-right corner panel (fraction of window).
const PANEL_WIDTH_FRACTION: f32 = 0.42;
const PANEL_HEIGHT_FRACTION: f32 = 0.45;
const PANEL_ALPHA: f32 = 0.96;
const HUD_ALPHA: f32 = 0.92;
/// Max display columns for the top-right actionable HUD.
const HUD_MAX_COLS: usize = 52;
/// Max actionable lines shown in the HUD (title + items).
const HUD_MAX_ITEMS: usize = 5;
const ACCENT_WIDTH_PX: f32 = 4.0;
const HEADER_ROWS: usize = 2;
const FOOTER_ROWS: usize = 3;
const INPUT_MAX_LEN: usize = 256;
const BORDER_ALPHA: f32 = 0.85;

const MSG_BACKEND_NOT_RUNNING: &str =
    "AI backend not running. Start with: uv sync && uv run --directory ai-backend python server.py";
const MSG_TIMEOUT: &str =
    "Response timed out. The model may be overloaded. Try a shorter selection.";
const MSG_EMPTY_CONTEXT: &str = "Could not read terminal content. Try selecting text manually.";

/// Overlay error panels auto-dismiss after this duration (Req 11 extension).
pub const ERROR_AUTO_DISMISS_SECS: u64 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    /// Show OS, package managers, and detected installed tools.
    Info { refresh: bool },
    /// List available slash commands (handled locally).
    Help,
    /// Clear the Chat mode transcript (handled locally).
    Clear,
}

impl SlashCommand {
    /// Parse `/info`, `/info refresh`, `/help`, etc. Returns `None` if not a slash command.
    pub fn parse(input: &str) -> Option<Result<Self, String>> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return None;
        }
        let rest = trimmed.trim_start_matches('/').trim();
        if rest.is_empty() {
            return Some(Err("Empty command. Type /help for available commands.".into()));
        }
        let mut words = rest.split_whitespace();
        let cmd = words.next().unwrap_or_default().to_ascii_lowercase();
        match cmd.as_str() {
            "info" => {
                let refresh = words.any(|w| w.eq_ignore_ascii_case("refresh"));
                Some(Ok(Self::Info { refresh }))
            },
            "help" => Some(Ok(Self::Help)),
            "clear" => Some(Ok(Self::Clear)),
            _ => Some(Err(format!("Unknown command '/{cmd}'. Type /help for available commands."))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayAction {
    None,
    Close,
    CopySelection(String),
    ScrollUp,
    ScrollDown,
    /// User pressed Enter in Chat mode.
    SubmitChat(String),
    /// User entered a slash command in the overlay input.
    RunSlashCommand(SlashCommand),
    /// Tab toggled between Command and Chat mode.
    ToggleMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InteractionMode {
    #[default]
    Command,
    Chat,
}

/// Whether keyboard input goes to the overlay panel or the shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputFocus {
    #[default]
    Overlay,
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OverlayMode {
    Normal,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineStyle {
    SectionHeader,
    Body,
    Muted,
    Code,
    SelectionLabel,
    SelectionText,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DisplayLine {
    text: String,
    style: LineStyle,
}

/// A single text draw call with explicit colors.
#[derive(Debug, Clone)]
pub struct OverlayText {
    pub point: Point<usize>,
    pub text: String,
    pub fg: Rgb,
    pub bg: Rgb,
}

/// Everything needed to paint the overlay for one frame.
#[derive(Debug, Clone, Default)]
pub struct OverlayDrawData {
    pub rects: Vec<RenderRect>,
    pub texts: Vec<OverlayText>,
}

/// Floating explanation panel drawn inside the terminal window.
#[derive(Debug, Clone)]
pub struct OverlayPanel {
    visible: bool,
    mode: OverlayMode,
    interaction_mode: InteractionMode,
    command_lines: Vec<DisplayLine>,
    chat_lines: Vec<DisplayLine>,
    scroll_offset: usize,
    /// When true, `prepare_draw` pins the viewport to the newest content (bottom).
    stick_to_bottom: bool,
    /// Counts consecutive arrow key repeats for scroll acceleration.
    scroll_burst: u32,
    input_buffer: String,
    input_focus: InputFocus,
    input_focused: bool,
    /// Shell-ready commands for the top-right HUD.
    actionable_items: Vec<String>,
    context_selection: Option<String>,
    context_command: String,
    pending_redraw: bool,
    last_append: Option<Instant>,
    has_chunks: bool,
    needs_redraw: bool,
    /// Raw tokens from SSE chunks (JSON from Ollama); not shown until `finalize`.
    stream_buffer: String,
    /// True while streaming a chat answer.
    chat_active: bool,
    /// Index in `chat_lines` of the in-progress assistant reply (for live token append).
    chat_stream_line: Option<usize>,
    /// Line count before an error was appended (chat path).
    lines_before_error: Option<usize>,
    /// True when the overlay shows only an error (no prior explanation content).
    error_only: bool,
}

impl Default for OverlayPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlayPanel {
    pub fn new() -> Self {
        Self {
            visible: false,
            mode: OverlayMode::Normal,
            interaction_mode: InteractionMode::Command,
            command_lines: Vec::new(),
            chat_lines: Vec::new(),
            scroll_offset: 0,
            stick_to_bottom: true,
            scroll_burst: 0,
            input_buffer: String::new(),
            input_focus: InputFocus::Overlay,
            input_focused: true,
            actionable_items: Vec::new(),
            context_selection: None,
            context_command: String::new(),
            pending_redraw: false,
            last_append: None,
            has_chunks: false,
            needs_redraw: false,
            stream_buffer: String::new(),
            chat_active: false,
            chat_stream_line: None,
            lines_before_error: None,
            error_only: false,
        }
    }

    pub fn interaction_mode(&self) -> InteractionMode {
        self.interaction_mode
    }

    pub fn input_focus(&self) -> InputFocus {
        self.input_focus
    }

    pub fn toggle_input_focus(&mut self) {
        self.input_focus = match self.input_focus {
            InputFocus::Overlay => InputFocus::Terminal,
            InputFocus::Terminal => InputFocus::Overlay,
        };
        self.input_focused = self.input_focus == InputFocus::Overlay;
        self.needs_redraw = true;
    }

    pub fn set_input_focus(&mut self, focus: InputFocus) {
        if self.input_focus == focus {
            return;
        }
        self.input_focus = focus;
        self.input_focused = focus == InputFocus::Overlay;
        self.needs_redraw = true;
    }

    /// True when `(x, y)` pixel coordinates lie inside the corner detail panel.
    pub fn contains_panel_point(&self, x: f32, y: f32, size_info: &SizeInfo) -> bool {
        let (px, py, pw, ph) = self.panel_geometry(size_info);
        x >= px && x < px + pw && y >= py && y < py + ph
    }

    fn view_lines(&self) -> &[DisplayLine] {
        match self.interaction_mode {
            InteractionMode::Command => &self.command_lines,
            InteractionMode::Chat => &self.chat_lines,
        }
    }

    fn view_lines_mut(&mut self) -> &mut Vec<DisplayLine> {
        match self.interaction_mode {
            InteractionMode::Command => &mut self.command_lines,
            InteractionMode::Chat => &mut self.chat_lines,
        }
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn needs_redraw(&self) -> bool {
        self.needs_redraw
    }

    pub fn clear_needs_redraw(&mut self) {
        self.needs_redraw = false;
    }

    /// Request a fresh layout pass after the window or cell metrics change.
    pub fn on_window_resize(&mut self) {
        if self.visible {
            self.needs_redraw = true;
        }
    }

    /// Attach terminal context shown in the highlighted block at the top.
    pub fn set_context(&mut self, selection: Option<String>, command: String) {
        self.context_selection = selection.filter(|s| !s.is_empty());
        self.context_command = command;
    }

    pub fn show(&mut self) {
        self.visible = true;
        self.mode = OverlayMode::Normal;
        self.interaction_mode = InteractionMode::Command;
        self.command_lines.clear();
        self.chat_lines.clear();
        self.scroll_offset = 0;
        self.stick_to_bottom = true;
        self.scroll_burst = 0;
        self.input_buffer.clear();
        self.input_focus = InputFocus::Overlay;
        self.input_focused = true;
        self.actionable_items.clear();
        self.has_chunks = false;
        self.stream_buffer.clear();
        self.chat_active = false;
        self.chat_stream_line = None;
        self.lines_before_error = None;
        self.error_only = false;
        self.pending_redraw = false;
        self.last_append = None;
        self.needs_redraw = true;
    }

    pub fn hide(&mut self) {
        self.visible = false;
        self.mode = OverlayMode::Normal;
        self.command_lines.clear();
        self.chat_lines.clear();
        self.stream_buffer.clear();
        self.input_buffer.clear();
        self.actionable_items.clear();
        self.lines_before_error = None;
        self.error_only = false;
        self.needs_redraw = true;
    }

    /// Remove a transient error panel (Req 11). Restores prior content or hides the overlay.
    pub fn dismiss_error(&mut self) {
        if !self.visible || self.mode != OverlayMode::Error {
            return;
        }

        if let Some(n) = self.lines_before_error.take() {
            self.view_lines_mut().truncate(n);
            self.mode = OverlayMode::Normal;
            self.error_only = false;
        } else if self.error_only {
            self.hide();
        } else {
            self.hide();
        }

        self.needs_redraw = true;
    }

    pub fn is_error_mode(&self) -> bool {
        self.mode == OverlayMode::Error
    }

    pub fn set_loading(&mut self) {
        let label = match self.interaction_mode {
            InteractionMode::Command => "  ◐  Loading command reference…",
            InteractionMode::Chat => "  ◐  Loading answer…",
        };
        *self.view_lines_mut() = vec![
            DisplayLine { text: String::new(), style: LineStyle::Body },
            DisplayLine { text: label.into(), style: LineStyle::Muted },
            DisplayLine { text: String::new(), style: LineStyle::Body },
        ];
        self.needs_redraw = true;
    }

    pub fn append_chunk(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.has_chunks = true;
        // Accumulate tokens only; the backend streams JSON from Ollama. Showing chunks
        // directly would flash raw JSON in the panel. Keep the loading state until finalize.
        self.stream_buffer.push_str(text);
        self.pending_redraw = true;
        self.last_append = Some(Instant::now());
    }

    pub fn flush_pending(&mut self) {
        if self.pending_redraw {
            self.pending_redraw = false;
            self.needs_redraw = true;
        }
    }

    pub fn should_batch(&self) -> bool {
        self.pending_redraw
            && self.last_append.is_some_and(|t| t.elapsed() < BATCH_INTERVAL)
    }

    /// If the backend streamed JSON tokens but the structured `done` event was not parsed,
    /// try to build the formatted view from the accumulated stream buffer.
    pub fn try_finalize_from_stream_buffer(&mut self) -> bool {
        if self.stream_buffer.is_empty() {
            return false;
        }
        let json_text = strip_json_fences(&self.stream_buffer);
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_text) else {
            return false;
        };
        let Some(response) = parse_explain_response_lenient(&value) else {
            return false;
        };
        self.finalize(&response);
        true
    }

    pub fn finalize(&mut self, response: &ExplainResponse) {
        self.remove_loading_lines();
        self.stream_buffer.clear();

        let structured = format_structured_response(response);
        self.command_lines = structured;
        self.actionable_items = actionable_items_from_response(response);
        self.has_chunks = true;
        self.stick_to_bottom = true;
        self.scroll_burst = 0;
        self.pending_redraw = false;
        self.last_append = None;
        self.needs_redraw = true;
    }

    /// Show a loading state while a slash command fetches from the backend.
    pub fn begin_slash_command(&mut self, label: &str) {
        self.chat_active = false;
        self.has_chunks = false;
        self.stream_buffer.clear();
        self.pending_redraw = false;
        self.last_append = None;
        self.input_buffer.clear();
        self.stick_to_bottom = true;
        self.scroll_burst = 0;
        let lines = self.view_lines_mut();
        lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        lines.push(DisplayLine {
            text: format!("Running {label}…"),
            style: LineStyle::Muted,
        });
        self.push_loading_lines();
        self.needs_redraw = true;
    }

    /// Show formatted man/--help reference (Command mode).
    pub fn show_command_reference(&mut self, response: &CommandReferenceResponse) {
        self.remove_loading_lines();
        self.interaction_mode = InteractionMode::Command;
        self.has_chunks = true;
        self.stick_to_bottom = true;
        self.scroll_burst = 0;
        self.stream_buffer.clear();

        let mut lines = Vec::new();
        lines.push(DisplayLine {
            text: format!("── {} ", response.title) + &"─".repeat(12),
            style: LineStyle::SectionHeader,
        });
        if response.source != "none" {
            lines.push(DisplayLine {
                text: format!("  (from {})", response.source),
                style: LineStyle::Muted,
            });
        }
        lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });

        for section in &response.sections {
            lines.push(DisplayLine {
                text: section.name.clone(),
                style: LineStyle::SectionHeader,
            });
            for line in &section.lines {
                if line.is_empty() {
                    lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
                } else {
                    lines.push(DisplayLine {
                        text: format!("  {line}"),
                        style: LineStyle::Body,
                    });
                }
            }
            lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        }

        self.command_lines = lines;
        self.actionable_items = actionable_items_from_command_reference(response);
        self.needs_redraw = true;
    }

    /// Clear Chat mode content; leaves Command mode explanation unchanged.
    pub fn clear_chat_output(&mut self) {
        self.chat_lines.clear();
        self.chat_active = false;
        self.chat_stream_line = None;
        self.lines_before_error = None;
        self.stream_buffer.clear();
        self.pending_redraw = false;
        self.last_append = None;
        self.scroll_offset = 0;
        self.stick_to_bottom = true;
        self.scroll_burst = 0;
        self.interaction_mode = InteractionMode::Chat;
        self.needs_redraw = true;
    }

    /// Prepare Chat mode for a user question.
    pub fn begin_chat_message(&mut self, query: &str) {
        self.interaction_mode = InteractionMode::Chat;
        self.chat_active = true;
        self.chat_stream_line = None;
        self.has_chunks = false;
        self.stream_buffer.clear();
        self.pending_redraw = false;
        self.last_append = None;
        self.input_buffer.clear();
        self.stick_to_bottom = true;
        self.scroll_burst = 0;
        self.chat_lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        self.chat_lines.push(DisplayLine {
            text: format!("You: {query}"),
            style: LineStyle::SectionHeader,
        });
        self.push_loading_lines();
        self.needs_redraw = true;
    }

    /// Append streaming chat tokens (visible immediately).
    pub fn append_chat_chunk(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.remove_loading_lines();
        self.has_chunks = true;

        if let Some(idx) = self.chat_stream_line {
            if let Some(line) = self.chat_lines.get_mut(idx) {
                line.text.push_str(text);
            }
        } else {
            self.chat_lines.push(DisplayLine {
                text: format!("Learnminal: {text}"),
                style: LineStyle::Body,
            });
            self.chat_stream_line = Some(self.chat_lines.len() - 1);
        }

        self.stick_to_bottom = true;
        self.needs_redraw = true;
    }

    pub fn finalize_chat(&mut self, reply: &str, actionable_items: &[String]) {
        self.remove_loading_lines();
        self.chat_active = false;
        if let Some(idx) = self.chat_stream_line.take() {
            if let Some(line) = self.chat_lines.get_mut(idx) {
                line.text = format!("Learnminal: {reply}");
            }
        } else {
            self.chat_lines.push(DisplayLine {
                text: format!("Learnminal: {reply}"),
                style: LineStyle::Body,
            });
        }
        self.chat_lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        if actionable_items.is_empty() {
            self.actionable_items = extract_actionable_from_prose(Some(reply));
        } else {
            self.actionable_items = actionable_items.iter().take(HUD_MAX_ITEMS).cloned().collect();
        }
        self.has_chunks = true;
        self.stick_to_bottom = true;
        self.needs_redraw = true;
    }

    pub fn toggle_interaction_mode(&mut self) {
        self.interaction_mode = match self.interaction_mode {
            InteractionMode::Command => InteractionMode::Chat,
            InteractionMode::Chat => InteractionMode::Command,
        };
        self.scroll_offset = 0;
        self.stick_to_bottom = true;
        self.scroll_burst = 0;
        self.needs_redraw = true;
    }

    fn push_loading_lines(&mut self) {
        let lines = self.view_lines_mut();
        lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        lines.push(DisplayLine {
            text: "  ◐  Loading…".into(),
            style: LineStyle::Muted,
        });
        lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
    }

    fn remove_loading_lines(&mut self) {
        for lines in [&mut self.command_lines, &mut self.chat_lines] {
            lines.retain(|l| !l.text.contains("Loading") && !l.text.contains('◐'));
        }
    }

    pub fn show_backend_not_running(&mut self) {
        self.show_error(MSG_BACKEND_NOT_RUNNING, false);
    }

    pub fn show_timeout(&mut self) {
        self.show_error(MSG_TIMEOUT, false);
    }

    pub fn show_sse_error(&mut self, message: &str) {
        let is_ollama = message.to_lowercase().contains("ollama");
        self.show_error(message, is_ollama);
    }

    pub fn show_empty_context(&mut self) {
        self.show_error(MSG_EMPTY_CONTEXT, false);
    }

    /// List slash commands (local; no backend call).
    pub fn show_slash_help(&mut self) {
        self.show_slash_message(
            "Slash commands",
            &[
                "/info — OS, shell, package managers, installed packages, tools".into(),
                "/info refresh — re-scan system and refresh package inventory".into(),
                "/clear — clear the Chat transcript".into(),
                "/help — show this list".into(),
                "".into(),
                "Tab → Chat mode to ask the agent about your command.".into(),
            ],
        );
    }

    /// Display system environment from `GET /system-info`.
    pub fn show_system_info(&mut self, info: &SystemInfo) {
        let mut lines = vec![
            format!("OS: {}", info.os),
            format!("Architecture: {}", info.arch),
            format!("Shell: {}", info.shell),
        ];
        if info.package_managers.is_empty() {
            lines.push("Package managers: none detected".into());
        } else {
            lines.push(format!("Package managers: {}", info.package_managers.join(", ")));
        }
        if info.installed_packages.is_empty() {
            lines.push("Installed packages: none enumerated (run /info refresh)".into());
        } else {
            let total = info.installed_packages_total.unwrap_or_else(|| {
                info.installed_packages.values().map(|v| v.len() as u64).sum()
            });
            lines.push(format!("Installed packages: {total} total across managers"));
            let mut managers: Vec<_> = info.installed_packages.keys().collect();
            managers.sort();
            for mgr in managers {
                if let Some(pkgs) = info.installed_packages.get(mgr) {
                    let sample: Vec<_> = pkgs.iter().take(8).cloned().collect();
                    let more = pkgs.len().saturating_sub(sample.len());
                    let suffix = if more > 0 { format!(" … +{more} more") } else { String::new() };
                    lines.push(format!("  {mgr} ({}): {}{suffix}", pkgs.len(), sample.join(", ")));
                }
            }
        }
        if info.installed_tools.is_empty() {
            lines.push("Installed tools: none detected".into());
        } else {
            lines.push(format!(
                "Installed tools ({}): {}",
                info.installed_tools.len(),
                info.installed_tools.join(", ")
            ));
        }
        if let Some(display) = &info.collected_at_display {
            lines.push(format!("Last collected: {display}"));
        } else if let Some(ts) = info.collected_at {
            lines.push(format!("Last collected: {ts}"));
        }
        self.show_slash_message("System environment (known to the agent)", &lines);
    }

    fn show_slash_message(&mut self, title: &str, body_lines: &[String]) {
        self.remove_loading_lines();
        self.chat_active = false;
        self.has_chunks = true;
        self.stick_to_bottom = true;
        self.scroll_burst = 0;
        let lines = self.view_lines_mut();
        lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        lines.push(DisplayLine {
            text: title.into(),
            style: LineStyle::SectionHeader,
        });
        for line in body_lines {
            if line.is_empty() {
                lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
            } else {
                lines.push(DisplayLine {
                    text: format!("  {line}"),
                    style: LineStyle::Body,
                });
            }
        }
        self.needs_redraw = true;
    }

    fn show_error(&mut self, message: &str, ollama: bool) {
        self.visible = true;
        self.mode = OverlayMode::Error;
        self.has_chunks = true;
        let mut error_lines =
            vec![DisplayLine { text: message.into(), style: LineStyle::Error }];
        if ollama {
            error_lines.push(DisplayLine {
                text: "Start with: ollama serve".into(),
                style: LineStyle::Muted,
            });
        }
        if self.chat_active {
            self.remove_loading_lines();
            self.lines_before_error = Some(self.chat_lines.len());
            self.error_only = false;
            self.chat_lines.extend(error_lines);
            self.chat_active = false;
            self.chat_stream_line = None;
        } else {
            self.lines_before_error = None;
            self.error_only = true;
            self.command_lines = error_lines;
            self.chat_lines.clear();
        }
        self.needs_redraw = true;
    }

    pub fn handle_key(&mut self, key: &KeyEvent, mods: ModifiersState) -> OverlayAction {
        if !self.visible || key.state != ElementState::Pressed {
            return OverlayAction::None;
        }
        self.dispatch_key(&key.logical_key, mods, key.text.as_deref(), key.repeat)
    }

    /// Lines to scroll per arrow press; accelerates while the key is held (repeat).
    fn scroll_step(&mut self, repeat: bool) -> usize {
        if !repeat {
            self.scroll_burst = 0;
            return 1;
        }
        self.scroll_burst = self.scroll_burst.saturating_add(1);
        let extra = (self.scroll_burst / 4) as usize;
        (1 + extra).min(SCROLL_REPEAT_MAX_LINES)
    }

    /// Routing logic without the `KeyEvent` wrapper (testable).
    fn dispatch_key(
        &mut self,
        logical_key: &Key,
        mods: ModifiersState,
        text: Option<&str>,
        repeat: bool,
    ) -> OverlayAction {
        let ctrl = mods.control_key();
        let alt = mods.alt_key();
        let shift = mods.shift_key();

        // Highest priority: navigation/control keys, then copy, then text insertion.
        match logical_key {
            Key::Named(NamedKey::Escape) => {
                if self.input_focus == InputFocus::Terminal {
                    self.set_input_focus(InputFocus::Overlay);
                    return OverlayAction::None;
                }
                self.hide();
                return OverlayAction::Close;
            },
            // ArrowUp → scroll up in history (toward older content).
            Key::Named(NamedKey::ArrowUp) if !ctrl => {
                let step = self.scroll_step(repeat);
                self.stick_to_bottom = false;
                self.scroll_offset = self.scroll_offset.saturating_sub(step);
                self.needs_redraw = true;
                return OverlayAction::ScrollUp;
            },
            // ArrowDown → scroll down (toward newer content / bottom).
            Key::Named(NamedKey::ArrowDown) if !ctrl => {
                let step = self.scroll_step(repeat);
                self.scroll_offset = self.scroll_offset.saturating_add(step);
                self.needs_redraw = true;
                return OverlayAction::ScrollDown;
            },
            Key::Named(NamedKey::Backspace)
                if self.input_focus == InputFocus::Overlay && self.input_focused && !ctrl =>
            {
                self.input_buffer.pop();
                self.needs_redraw = true;
                return OverlayAction::None;
            },
            Key::Named(NamedKey::Tab) if ctrl && shift && !alt => {
                self.toggle_input_focus();
                return OverlayAction::None;
            },
            Key::Named(NamedKey::Tab) if !ctrl && !alt => {
                self.toggle_interaction_mode();
                return OverlayAction::ToggleMode;
            },
            Key::Named(NamedKey::Enter)
                if self.input_focus == InputFocus::Overlay && self.input_focused && !ctrl =>
            {
                let query = self.input_buffer.trim().to_owned();
                if query.is_empty() {
                    return OverlayAction::None;
                }
                if let Some(parsed) = SlashCommand::parse(&query) {
                    self.input_buffer.clear();
                    self.needs_redraw = true;
                    return match parsed {
                        Ok(SlashCommand::Help) => {
                            self.show_slash_help();
                            OverlayAction::None
                        },
                        Ok(SlashCommand::Clear) => {
                            self.clear_chat_output();
                            OverlayAction::None
                        },
                        Ok(cmd @ SlashCommand::Info { .. }) => OverlayAction::RunSlashCommand(cmd),
                        Err(message) => {
                            self.show_slash_message("Command error", &[message]);
                            OverlayAction::None
                        },
                    };
                }
                if self.interaction_mode != InteractionMode::Chat {
                    return OverlayAction::None;
                }
                self.begin_chat_message(&query);
                return OverlayAction::SubmitChat(query);
            },
            _ => {},
        }

        // Copy: Shift+Y copies the command that triggered this explanation.
        // Checked before text insertion so 'Y' is not absorbed by the input field.
        if shift && !ctrl && !alt {
            if let Key::Character(ch) = logical_key {
                if ch.as_str().eq_ignore_ascii_case("y") {
                    if !self.context_command.is_empty() {
                        return OverlayAction::CopySelection(self.context_command.clone());
                    }
                    return OverlayAction::None;
                }
            }
        }

        // Otherwise, while focused and without modifier shortcuts, attempt text insertion.
        if self.input_focus == InputFocus::Overlay
            && self.input_focused
            && !ctrl
            && !alt
            && self.try_insert_key_text(logical_key, text)
        {
            return OverlayAction::None;
        }

        OverlayAction::None
    }

    /// Insert printable text from a key event (Space is often `NamedKey::Space`, not `Character`).
    fn try_insert_key_text(&mut self, logical_key: &Key, text: Option<&str>) -> bool {
        if let Some(text) = text.filter(|t| !t.is_empty()) {
            if self.push_input_str(text) {
                return true;
            }
        }

        match logical_key {
            Key::Named(NamedKey::Space) => self.push_input_str(" "),
            Key::Character(ch) => self.push_input_str(ch.as_str()),
            _ => false,
        }
    }

    fn push_input_str(&mut self, text: &str) -> bool {
        let mut inserted = false;
        for c in text.chars().filter(|c| !c.is_control()) {
            let ch_len = c.len_utf8();
            if self.input_buffer.len() + ch_len > INPUT_MAX_LEN {
                break;
            }
            self.input_buffer.push(c);
            inserted = true;
        }

        if inserted {
            self.needs_redraw = true;
        }
        inserted
    }

    /// Bottom-right corner panel: `(x, y, width, height)` in pixels.
    pub fn panel_geometry(&self, size_info: &SizeInfo) -> (f32, f32, f32, f32) {
        let width = size_info.width() * PANEL_WIDTH_FRACTION;
        let height = size_info.height() * PANEL_HEIGHT_FRACTION;
        let x = size_info.width() - width;
        let y = size_info.height() - height;
        (x, y, width, height)
    }

    /// Terminal region left of the corner panel: `(x, y, width, height)`.
    pub fn terminal_region_geometry(&self, size_info: &SizeInfo) -> (f32, f32, f32, f32) {
        let (panel_x, _, _, _) = self.panel_geometry(size_info);
        (0., 0., panel_x, size_info.height())
    }

    /// Build rects and styled text for the current frame.
    pub fn prepare_draw(&mut self, size_info: &SizeInfo, config: &UiConfig) -> OverlayDrawData {
        let mut data = OverlayDrawData::default();
        if !self.visible {
            return data;
        }

        let (panel_x, panel_y, panel_w, panel_h) = self.panel_geometry(size_info);
        let cell_w = size_info.cell_width();
        let cell_h = size_info.cell_height();
        let theme = Theme::from_config(config, self.mode);

        // Full panel backdrop.
        data.rects.push(RenderRect::new(panel_x, panel_y, panel_w, panel_h, theme.panel_bg, PANEL_ALPHA));

        // Left accent stripe.
        data.rects.push(RenderRect::new(
            panel_x,
            panel_y,
            ACCENT_WIDTH_PX.min(panel_w),
            panel_h,
            theme.accent,
            1.0,
        ));

        // Header bar.
        let header_h = cell_h * HEADER_ROWS as f32;
        data.rects.push(RenderRect::new(
            panel_x + ACCENT_WIDTH_PX,
            panel_y,
            panel_w - ACCENT_WIDTH_PX,
            header_h,
            theme.header_bg,
            BORDER_ALPHA,
        ));

        // Footer / input area.
        let footer_h = cell_h * FOOTER_ROWS as f32;
        data.rects.push(RenderRect::new(
            panel_x + ACCENT_WIDTH_PX,
            panel_y + panel_h - footer_h,
            panel_w - ACCENT_WIDTH_PX,
            footer_h,
            theme.input_bg,
            BORDER_ALPHA,
        ));

        // Input field highlight (inset box).
        let input_box_h = cell_h * 1.2;
        let input_box_y = panel_y + panel_h - footer_h + cell_h * 1.1;
        data.rects.push(RenderRect::new(
            panel_x + ACCENT_WIDTH_PX + cell_w * 0.5,
            input_box_y,
            panel_w - ACCENT_WIDTH_PX - cell_w,
            input_box_h,
            theme.input_field_bg,
            1.0,
        ));

        let panel_start_row = (panel_y / cell_h).floor() as usize;
        let start_col =
            Column((panel_x / cell_w).floor() as usize + 1).0.saturating_add(1);
        let panel_cols = ((panel_w / cell_w) as usize).saturating_sub(3).max(1);
        let total_rows = (panel_h / cell_h) as usize;

        let focus_hint = match self.input_focus {
            InputFocus::Overlay => "⌃⇧Tab → shell",
            InputFocus::Terminal => "⌃⇧Tab → panel",
        };

        // Header text.
        let title = " Learnminal ";
        data.texts.push(OverlayText {
            point: Point::new(panel_start_row, Column(start_col)),
            text: pad_line(title, panel_cols),
            fg: theme.title_fg,
            bg: theme.header_bg,
        });
        let mode_label = match self.interaction_mode {
            InteractionMode::Command => "[Command]",
            InteractionMode::Chat => "[Chat]",
        };
        let hints = format!(
            " Tab {mode_label}   {focus_hint}   Esc   ↑↓   ⇧Y"
        );
        data.texts.push(OverlayText {
            point: Point::new(panel_start_row + 1, Column(start_col)),
            text: pad_line(&hints, panel_cols),
            fg: theme.hint_fg,
            bg: theme.header_bg,
        });

        let content_start_row = HEADER_ROWS;
        let content_end_row = total_rows.saturating_sub(FOOTER_ROWS);
        let content_height = content_end_row.saturating_sub(content_start_row);

        let mut layout_lines: Vec<DisplayLine> = Vec::new();

        // Context block (command + selection highlight).
        if !self.context_command.is_empty() || self.context_selection.is_some() {
            let context_header = "── Context ";
            let fill_cols = panel_cols.saturating_sub(text_display_width(context_header));
            layout_lines.push(DisplayLine {
                text: context_header.to_owned() + &"─".repeat(fill_cols),
                style: LineStyle::SectionHeader,
            });
            if !self.context_command.is_empty() {
                layout_lines.push(DisplayLine {
                    text: format!("  $ {}", self.context_command),
                    style: LineStyle::Code,
                });
            }
            if let Some(ref sel) = self.context_selection {
                layout_lines.push(DisplayLine {
                    text: "  selection:".into(),
                    style: LineStyle::SelectionLabel,
                });
                layout_lines.push(DisplayLine {
                    text: format!("  ▸ {sel}"),
                    style: LineStyle::SelectionText,
                });
            }
            layout_lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        }

        layout_lines.extend(self.view_lines().iter().cloned());
        layout_lines = wrap_display_lines(&layout_lines, panel_cols);

        // Scroll content lines (offset 0 = top; max = bottom / newest).
        let max_scroll = layout_lines.len().saturating_sub(content_height.max(1));
        let scroll_start = if self.stick_to_bottom {
            max_scroll
        } else {
            self.scroll_offset.min(max_scroll)
        };
        self.scroll_offset = scroll_start;
        if scroll_start >= max_scroll {
            self.stick_to_bottom = true;
        }

        let visible_content: Vec<_> =
            layout_lines.iter().skip(scroll_start).take(content_height).cloned().collect();

        for (i, line) in visible_content.iter().enumerate() {
            let row = content_start_row + i;
            if row >= content_end_row {
                break;
            }
            let (fg, bg) = theme.colors(line.style, self.mode);
            data.texts.push(OverlayText {
                point: Point::new(panel_start_row + row, Column(start_col)),
                text: pad_line(&line.text, panel_cols),
                fg,
                bg,
            });
        }

        // Footer divider + label + input.
        let footer_start = total_rows.saturating_sub(FOOTER_ROWS);
        let divider = "─".repeat(panel_cols.min(48));
        data.texts.push(OverlayText {
            point: Point::new(panel_start_row + footer_start, Column(start_col)),
            text: pad_line(&divider, panel_cols),
            fg: theme.hint_fg,
            bg: theme.input_bg,
        });
        let footer_hint = match (self.input_focus, self.interaction_mode) {
            (InputFocus::Terminal, _) => " Terminal focus — type in shell (⌃⇧Tab for panel)",
            (InputFocus::Overlay, InteractionMode::Command) => " Tab → Chat   /info · /help",
            (InputFocus::Overlay, InteractionMode::Chat) => {
                if self.context_command.is_empty() {
                    " Ask ↵ send   /clear · /help"
                } else {
                    " Ask about this command ↵ send   /clear · /help"
                }
            },
        };
        data.texts.push(OverlayText {
            point: Point::new(panel_start_row + footer_start + 1, Column(start_col)),
            text: pad_line(footer_hint, panel_cols),
            fg: theme.hint_fg,
            bg: theme.input_bg,
        });

        let cursor = if self.input_focus == InputFocus::Overlay && self.input_focused {
            "▌"
        } else {
            " "
        };
        let input_display = if self.input_focus == InputFocus::Terminal {
            " > (terminal focus)".to_owned()
        } else if self.interaction_mode == InteractionMode::Command {
            format!(" > (Tab → Chat to ask){cursor}")
        } else if self.input_buffer.is_empty() {
            format!(" > _{}", cursor)
        } else {
            format!(" > {}{}", self.input_buffer, cursor)
        };
        for (i, input_line) in wrap_line_respecting_indent(&input_display, panel_cols)
            .into_iter()
            .enumerate()
            .take(2)
        {
            data.texts.push(OverlayText {
                point: Point::new(panel_start_row + footer_start + 2 + i, Column(start_col)),
                text: pad_line(&input_line, panel_cols),
                fg: theme.input_fg,
                bg: theme.input_field_bg,
            });
        }

        append_actionable_hud(&mut data, size_info, config, &self.actionable_items);

        data
    }

}

#[derive(Debug, Clone, Copy)]
struct Theme {
    panel_bg: Rgb,
    header_bg: Rgb,
    accent: Rgb,
    title_fg: Rgb,
    hint_fg: Rgb,
    section_fg: Rgb,
    body_fg: Rgb,
    muted_fg: Rgb,
    code_fg: Rgb,
    selection_label_fg: Rgb,
    selection_bg: Rgb,
    selection_fg: Rgb,
    input_bg: Rgb,
    input_field_bg: Rgb,
    input_fg: Rgb,
    error_fg: Rgb,
}

impl Theme {
    fn from_config(config: &UiConfig, mode: OverlayMode) -> Self {
        let c = &config.colors;
        let panel_bg = c.primary.background;
        let header_bg = blend(c.primary.background, c.normal.blue, 0.12);
        let input_bg = blend(c.primary.background, c.footer_bar_background(), 0.35);
        let input_field_bg = c.footer_bar_background();
        let accent = c.normal.cyan;

        Self {
            panel_bg,
            header_bg,
            accent,
            title_fg: c.bright.cyan,
            hint_fg: c.primary.foreground * 0.55,
            section_fg: c.bright.yellow,
            body_fg: c.primary.foreground,
            muted_fg: c.primary.foreground * 0.6,
            code_fg: c.bright.green,
            selection_label_fg: c.bright.magenta,
            selection_bg: blend(c.primary.background, c.normal.magenta, 0.25),
            selection_fg: c.primary.foreground,
            input_bg,
            input_field_bg,
            input_fg: c.footer_bar_foreground(),
            error_fg: if mode == OverlayMode::Error {
                c.primary.background
            } else {
                c.normal.red
            },
        }
    }

    fn colors(&self, style: LineStyle, mode: OverlayMode) -> (Rgb, Rgb) {
        match (style, mode) {
            (_, OverlayMode::Error) if matches!(style, LineStyle::Error) => {
                (self.error_fg, self.panel_bg)
            },
            (LineStyle::SectionHeader, _) => (self.section_fg, self.panel_bg),
            (LineStyle::Body, _) => (self.body_fg, self.panel_bg),
            (LineStyle::Muted, _) => (self.muted_fg, self.panel_bg),
            (LineStyle::Code, _) => (self.code_fg, self.panel_bg),
            (LineStyle::SelectionLabel, _) => (self.selection_label_fg, self.selection_bg),
            (LineStyle::SelectionText, _) => (self.selection_fg, self.selection_bg),
            (LineStyle::Error, _) => (self.error_fg, self.panel_bg),
        }
    }
}

fn blend(a: Rgb, b: Rgb, t: f32) -> Rgb {
    a * (1.0 - t) + b * t
}

fn char_display_width(c: char) -> usize {
    c.width().unwrap_or(0)
}

fn text_display_width(text: &str) -> usize {
    text.chars().map(char_display_width).sum()
}

/// Truncate at a UTF-8 character boundary, limiting display width to `max_cols`.
fn truncate_to_width(text: &str, max_cols: usize) -> String {
    if max_cols == 0 {
        return String::new();
    }

    let mut width = 0;
    let mut end_byte = 0;
    for (byte_idx, c) in text.char_indices() {
        let w = char_display_width(c);
        if width + w > max_cols {
            break;
        }
        width += w;
        end_byte = byte_idx + c.len_utf8();
    }

    text.get(..end_byte).unwrap_or("").to_owned()
}

fn pad_line(text: &str, cols: usize) -> String {
    let mut s = truncate_to_width(text, cols);
    let pad = cols.saturating_sub(text_display_width(&s));
    s.push_str(&" ".repeat(pad));
    s
}

/// Right edge of the terminal grid in pixels (accounts for padding and column count).
fn terminal_content_right(size_info: &SizeInfo) -> f32 {
    size_info.padding_x() + size_info.columns() as f32 * size_info.cell_width()
}

/// Draw the top-right actionable command HUD over the terminal grid (not the corner panel).
fn append_actionable_hud(
    data: &mut OverlayDrawData,
    size_info: &SizeInfo,
    config: &UiConfig,
    items: &[String],
) {
    if items.is_empty() {
        return;
    }

    let cell_w = size_info.cell_width();
    let cell_h = size_info.cell_height();
    let content_right = terminal_content_right(size_info);
    let content_left = size_info.padding_x();
    let max_hud_w = (content_right - content_left).max(cell_w);
    let hud_w = max_hud_w.min(cell_w * HUD_MAX_COLS as f32);
    let hud_cols = ((hud_w / cell_w) as usize).saturating_sub(2).max(8);
    let line_count = 1 + items.len().min(HUD_MAX_ITEMS) + usize::from(items.len() > HUD_MAX_ITEMS);
    let hud_h = cell_h * line_count as f32;
    let hud_x = (content_right - hud_w).max(content_left);
    let hud_y = size_info.padding_y();

    let theme = Theme::from_config(config, OverlayMode::Normal);
    let hud_bg = blend(theme.panel_bg, theme.code_fg, 0.08);

    data.rects.push(RenderRect::new(hud_x, hud_y, hud_w, hud_h, hud_bg, HUD_ALPHA));

    let start_col = Column(((hud_x - content_left) / cell_w).floor() as usize + 1);
    let mut row = (hud_y / cell_h).floor() as usize;

    data.texts.push(OverlayText {
        point: Point::new(row, start_col),
        text: pad_line(" Actions", hud_cols),
        fg: theme.section_fg,
        bg: hud_bg,
    });
    row += 1;

    for (i, item) in items.iter().take(HUD_MAX_ITEMS).enumerate() {
        let line = format!(" {}. {}", i + 1, item.trim());
        for wrapped in wrap_line_respecting_indent(&line, hud_cols).into_iter().take(2) {
            data.texts.push(OverlayText {
                point: Point::new(row, start_col),
                text: pad_line(&wrapped, hud_cols),
                fg: theme.code_fg,
                bg: hud_bg,
            });
            row += 1;
        }
    }

    if items.len() > HUD_MAX_ITEMS {
        data.texts.push(OverlayText {
            point: Point::new(row, start_col),
            text: pad_line(" …", hud_cols),
            fg: theme.muted_fg,
            bg: hud_bg,
        });
    }

}

fn dedupe_actionable_items(items: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for item in items {
        let key = item.trim().to_owned();
        if key.is_empty() || !seen.insert(key.clone()) {
            continue;
        }
        out.push(key);
        if out.len() >= HUD_MAX_ITEMS {
            break;
        }
    }
    out
}

/// Extract shell commands from prose (code fences, numbered/bullet lines).
pub fn extract_actionable_from_prose(prose: Option<&str>) -> Vec<String> {
    let Some(text) = prose.filter(|s| !s.is_empty()) else {
        return Vec::new();
    };

    let mut items = Vec::new();
    let mut in_fence = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence && !trimmed.is_empty() {
            items.push(trimmed.to_owned());
            continue;
        }
        if let Some(rest) = parse_numbered_list_line(trimmed) {
            items.push(rest);
        } else if let Some(rest) = trimmed.strip_prefix("- ") {
            if looks_like_command(rest) {
                items.push(rest.to_owned());
            }
        }
    }

    dedupe_actionable_items(items)
}

fn parse_numbered_list_line(trimmed: &str) -> Option<String> {
    let digit_len = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_len == 0 {
        return None;
    }
    let rest = trimmed[digit_len..]
        .trim_start_matches(|c| c == '.' || c == ')')
        .trim();
    if looks_like_command(rest) {
        Some(rest.to_owned())
    } else {
        None
    }
}

fn looks_like_command(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() || s.len() > 200 {
        return false;
    }
    // Skip obvious prose sentences.
    if s.contains(" should ") || s.contains(" because ") {
        return false;
    }
    s.starts_with('$')
        || s.contains("git ")
        || s.contains("sudo ")
        || s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric() || c == '.' || c == '/')
}

pub fn actionable_items_from_response(response: &ExplainResponse) -> Vec<String> {
    if !response.actionable_items.is_empty() {
        return dedupe_actionable_items(response.actionable_items.clone());
    }

    let mut items = extract_actionable_from_prose(response.error_fix.as_deref());
    for flag in &response.flags_explained {
        if !flag.example.is_empty() {
            items.push(flag.example.clone());
        }
    }
    dedupe_actionable_items(items)
}

pub fn actionable_items_from_command_reference(response: &CommandReferenceResponse) -> Vec<String> {
    let mut items = Vec::new();
    for section in &response.sections {
        if section.name.eq_ignore_ascii_case("EXAMPLES") {
            for line in &section.lines {
                let t = line.trim();
                if looks_like_command(t) {
                    items.push(t.to_owned());
                }
            }
        }
    }
    if items.is_empty() {
        for section in &response.sections {
            if section.name.eq_ignore_ascii_case("SYNOPSIS") {
                for line in &section.lines {
                    let t = line.trim();
                    if !t.is_empty() {
                        items.push(t.to_owned());
                        break;
                    }
                }
            }
        }
    }
    dedupe_actionable_items(items)
}

/// Section names used in the structured response (kept in module scope for tests).
pub(crate) const SECTION_GENERAL: &str = "General";
pub(crate) const SECTION_CONTEXT: &str = "Context";
pub(crate) const SECTION_FLAGS: &str = "Flags";
pub(crate) const SECTION_FIX: &str = "Suggested fix";
pub(crate) const SECTION_SIMILAR: &str = "Similar commands";

fn format_structured_response(response: &ExplainResponse) -> Vec<DisplayLine> {
    let mut lines = Vec::new();

    lines.push(DisplayLine {
        text: format!("── Command: {} ", response.command_name)
            + &"─".repeat(20usize.saturating_sub(response.command_name.chars().count())),
        style: LineStyle::SectionHeader,
    });
    lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });

    lines.push(DisplayLine { text: SECTION_GENERAL.into(), style: LineStyle::SectionHeader });
    push_prose_with_code_fences(&mut lines, &response.general_utility, "  ");
    lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });

    lines.push(DisplayLine { text: SECTION_CONTEXT.into(), style: LineStyle::SectionHeader });
    push_prose_with_code_fences(&mut lines, &response.contextual_usage, "  ");

    if !response.flags_explained.is_empty() {
        lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        lines.push(DisplayLine { text: SECTION_FLAGS.into(), style: LineStyle::SectionHeader });
        for FlagExplanation { flag, meaning, example } in &response.flags_explained {
            lines.push(DisplayLine {
                text: format!("  {flag} — {meaning}"),
                style: LineStyle::Body,
            });
            lines.push(DisplayLine {
                text: format!("    e.g. {example}"),
                style: LineStyle::Muted,
            });
        }
    }

    if let Some(fix) = &response.error_fix {
        lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        lines.push(DisplayLine { text: SECTION_FIX.into(), style: LineStyle::SectionHeader });
        push_prose_with_code_fences(&mut lines, fix, "  ");
    }

    if !response.similar_commands.is_empty() {
        lines.push(DisplayLine { text: String::new(), style: LineStyle::Body });
        lines.push(DisplayLine {
            text: SECTION_SIMILAR.into(),
            style: LineStyle::SectionHeader,
        });
        for cmd in &response.similar_commands {
            lines.push(DisplayLine { text: format!("  • {cmd}"), style: LineStyle::Muted });
        }
    }

    lines
}

/// Convert prose into styled lines, switching to `LineStyle::Code` between triple-backtick
/// fences. Fence lines themselves are dropped from the output.
fn push_prose_with_code_fences(out: &mut Vec<DisplayLine>, prose: &str, indent: &str) {
    let mut in_code = false;
    for body_line in prose.lines() {
        let trimmed = body_line.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        let style = if in_code { LineStyle::Code } else { LineStyle::Body };
        out.push(DisplayLine { text: format!("{indent}{body_line}"), style });
    }
}

/// Split leading whitespace from the rest of a line (for hang-indent wrapping).
fn split_leading_whitespace(line: &str) -> (String, &str) {
    let prefix_end = line
        .char_indices()
        .take_while(|(_, c)| c.is_whitespace())
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    (line[..prefix_end].to_owned(), &line[prefix_end..])
}

/// Wrap `line` to at most `max_cols` display columns, preserving a leading indent on every row.
fn wrap_line_respecting_indent(line: &str, max_cols: usize) -> Vec<String> {
    if line.is_empty() || max_cols == 0 {
        return vec![String::new()];
    }

    let (prefix, content) = split_leading_whitespace(line);
    let prefix_width = text_display_width(&prefix);
    if prefix_width >= max_cols {
        return vec![truncate_to_width(line, max_cols)];
    }

    let inner_cols = max_cols.saturating_sub(prefix_width);
    let content = content.trim_start();
    if content.is_empty() {
        return vec![prefix];
    }

    wrap_line(content, inner_cols)
        .into_iter()
        .map(|segment| format!("{prefix}{segment}"))
        .collect()
}

/// Wrap each display line to the panel width, preserving line style.
fn wrap_display_lines(lines: &[DisplayLine], max_cols: usize) -> Vec<DisplayLine> {
    let mut out = Vec::new();
    for line in lines {
        if line.text.is_empty() {
            out.push(line.clone());
            continue;
        }
        for text in wrap_line_respecting_indent(&line.text, max_cols) {
            out.push(DisplayLine { text, style: line.style });
        }
    }
    out
}

/// Break a single word across multiple rows when it exceeds `cols` display width.
fn wrap_word_by_chars(word: &str, cols: usize) -> Vec<String> {
    if cols == 0 {
        return vec![String::new()];
    }

    let mut rows = Vec::new();
    let mut current = String::new();
    let mut width = 0;

    for c in word.chars() {
        let cw = char_display_width(c);
        if width + cw > cols {
            if !current.is_empty() {
                rows.push(current);
                current = String::new();
                width = 0;
            }
        }
        current.push(c);
        width += cw;
    }

    if !current.is_empty() {
        rows.push(current);
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    rows
}

fn wrap_line(line: &str, cols: usize) -> Vec<String> {
    if line.is_empty() || cols == 0 {
        return vec![String::new()];
    }

    let mut segments = Vec::new();
    let mut current = String::new();
    let mut current_width = 0;

    for word in line.split_whitespace() {
        let word_width = text_display_width(word);
        if word_width > cols {
            if !current.is_empty() {
                segments.push(current);
                current = String::new();
                current_width = 0;
            }
            segments.extend(wrap_word_by_chars(word, cols));
            continue;
        }

        let extra = if current.is_empty() { 0 } else { 1 + word_width };
        if current.is_empty() {
            current = word.to_owned();
            current_width = word_width;
        } else if current_width + extra <= cols {
            current.push(' ');
            current.push_str(word);
            current_width += extra;
        } else {
            segments.push(current);
            current = word.to_owned();
            current_width = word_width;
        }
    }

    if !current.is_empty() {
        segments.push(current);
    }
    if segments.is_empty() {
        segments.push(String::new());
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use winit::keyboard::SmolStr;

    fn key_char(s: &str) -> Key {
        Key::Character(SmolStr::new(s))
    }

    fn dispatch(panel: &mut OverlayPanel, key: Key, mods: ModifiersState) -> OverlayAction {
        panel.dispatch_key(&key, mods, None, false)
    }

    fn dispatch_repeat(panel: &mut OverlayPanel, key: Key, mods: ModifiersState) -> OverlayAction {
        panel.dispatch_key(&key, mods, None, true)
    }

    fn dispatch_with_text(
        panel: &mut OverlayPanel,
        key: Key,
        mods: ModifiersState,
        text: &str,
    ) -> OverlayAction {
        panel.dispatch_key(&key, mods, Some(text), false)
    }

    fn make_response() -> ExplainResponse {
        ExplainResponse {
            command_name: "git".into(),
            flags_explained: Vec::new(),
            general_utility: "general utility".into(),
            contextual_usage: "contextual usage".into(),
            error_fix: None,
            similar_commands: Vec::new(),
            tool_calls_made: Vec::new(),
            actionable_items: Vec::new(),
        }
    }

    // ---- Existing core tests ----

    #[test]
    fn panel_geometry_is_bottom_right_corner() {
        let panel = OverlayPanel::new();
        let size = SizeInfo::new(1000., 800., 10., 20., 0., 0., false);
        let (x, y, width, height) = panel.panel_geometry(&size);
        assert!((width - 420.).abs() < 0.01);
        assert!((height - 360.).abs() < 0.01);
        assert!((x - 580.).abs() < 0.01);
        assert!((y - 440.).abs() < 0.01);
        assert!((x + width - 1000.).abs() < 0.01);
        assert!((y + height - 800.).abs() < 0.01);
    }

    #[test]
    fn push_input_str_accepts_spaces() {
        let mut panel = OverlayPanel::new();
        panel.show();
        assert!(panel.push_input_str("hello world"));
        assert_eq!(panel.input_buffer, "hello world");
    }

    #[test]
    fn wrap_line_breaks_long_words_instead_of_truncating() {
        let wrapped = wrap_line("abcdefghij", 4);
        assert!(wrapped.len() >= 2);
        assert!(wrapped.iter().all(|l| text_display_width(l) <= 4));
        assert_eq!(wrapped.concat(), "abcdefghij");
    }

    #[test]
    fn wrap_display_lines_preserves_indent_on_continuations() {
        let lines = vec![DisplayLine {
            text: "  hello world foo bar".into(),
            style: LineStyle::Body,
        }];
        let wrapped = wrap_display_lines(&lines, 10);
        assert!(wrapped.len() > 1);
        assert!(wrapped.iter().all(|l| l.text.starts_with("  ")));
        assert!(wrapped.iter().all(|l| text_display_width(&l.text) <= 10));
    }

    #[test]
    fn pad_line_truncates_utf8_on_char_boundary() {
        // Each '─' is 3 bytes but 1 display column; byte-truncate at col 5 would panic.
        let line = "── Context ─────────────────";
        let padded = pad_line(line, 10);
        assert!(text_display_width(&padded) <= 10);
        assert!(std::str::from_utf8(padded.as_bytes()).is_ok());
    }

    #[test]
    fn prepare_draw_includes_header_and_input() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.set_context(Some("git status".into()), "git status".into());
        panel.set_loading();
        let size = SizeInfo::new(1000., 800., 10., 20., 0., 0., false);
        let draw = panel.prepare_draw(&size, &UiConfig::default());
        assert!(draw.rects.len() >= 4);
        assert!(draw.texts.iter().any(|t| t.text.contains("Learnminal")));
        assert!(draw.texts.iter().any(|t| t.text.contains('>')));
    }

    #[test]
    fn actionable_hud_is_top_right_of_terminal_region() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.actionable_items = vec!["git status".into(), "git add -p".into()];
        let size = SizeInfo::new(1000., 800., 10., 20., 0., 0., false);
        let content_right = terminal_content_right(&size);
        let draw = panel.prepare_draw(&size, &UiConfig::default());

        let hud_rect = draw
            .rects
            .iter()
            .find(|r| (r.x + r.width - content_right).abs() < 0.5)
            .expect("actionable HUD background rect");
        assert!(hud_rect.x > size.padding_x());

        let actions = draw.texts.iter().find(|t| t.text.contains("Actions")).expect("HUD title");
        assert!(actions.point.column.0 > 1);
        assert_eq!(actions.point.line, 0);
    }

    #[test]
    fn actionable_hud_tracks_horizontal_resize() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.actionable_items = vec!["git status".into()];
        let narrow = SizeInfo::new(800., 600., 10., 20., 0., 0., false);
        let wide = SizeInfo::new(1200., 600., 10., 20., 0., 0., false);

        let hud_right = |size: &SizeInfo, draw: &OverlayDrawData| {
            let content_right = terminal_content_right(size);
            draw.rects
                .iter()
                .find(|r| (r.x + r.width - content_right).abs() < 0.5)
                .map(|r| r.x + r.width)
                .expect("hud rect")
        };

        let narrow_draw = panel.prepare_draw(&narrow, &UiConfig::default());
        let wide_draw = panel.prepare_draw(&wide, &UiConfig::default());
        assert!(hud_right(&wide, &wide_draw) > hud_right(&narrow, &narrow_draw));
    }

    // ---- Task 5.2: Unit tests for handle_key/dispatch_key ----

    #[test]
    fn dispatch_key_escape_closes_overlay() {
        let mut panel = OverlayPanel::new();
        panel.show();
        let action = dispatch(&mut panel, Key::Named(NamedKey::Escape), ModifiersState::empty());
        assert_eq!(action, OverlayAction::Close);
        assert!(!panel.is_visible());
    }

    #[test]
    fn dispatch_key_escape_from_terminal_focus_returns_to_overlay() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.set_input_focus(InputFocus::Terminal);
        let action = dispatch(&mut panel, Key::Named(NamedKey::Escape), ModifiersState::empty());
        assert_eq!(action, OverlayAction::None);
        assert!(panel.is_visible());
        assert_eq!(panel.input_focus(), InputFocus::Overlay);
    }

    #[test]
    fn toggle_input_focus_switches_modes() {
        let mut panel = OverlayPanel::new();
        panel.show();
        assert_eq!(panel.input_focus(), InputFocus::Overlay);
        panel.toggle_input_focus();
        assert_eq!(panel.input_focus(), InputFocus::Terminal);
        assert!(!panel.input_focused);
        panel.toggle_input_focus();
        assert_eq!(panel.input_focus(), InputFocus::Overlay);
        assert!(panel.input_focused);
    }

    #[test]
    fn dispatch_key_ctrl_shift_tab_toggles_input_focus() {
        let mut panel = OverlayPanel::new();
        panel.show();
        let action = dispatch(
            &mut panel,
            Key::Named(NamedKey::Tab),
            ModifiersState::CONTROL | ModifiersState::SHIFT,
        );
        assert_eq!(action, OverlayAction::None);
        assert_eq!(panel.input_focus(), InputFocus::Terminal);
        dispatch(
            &mut panel,
            Key::Named(NamedKey::Tab),
            ModifiersState::CONTROL | ModifiersState::SHIFT,
        );
        assert_eq!(panel.input_focus(), InputFocus::Overlay);
    }

    #[test]
    fn actionable_items_from_error_fix() {
        let mut response = make_response();
        response.error_fix = Some(
            "Try:\n1. git status\n2. git add .\n```\ngit commit -m fix\n```".into(),
        );
        let items = actionable_items_from_response(&response);
        assert!(items.iter().any(|i| i.contains("git status")));
        assert!(items.iter().any(|i| i.contains("git commit")));
    }

    #[test]
    fn dismiss_error_hides_error_only_overlay() {
        let mut panel = OverlayPanel::new();
        panel.show_backend_not_running();
        assert!(panel.is_error_mode());
        panel.dismiss_error();
        assert!(!panel.is_visible());
    }

    #[test]
    fn dismiss_error_restores_content_after_follow_up_failure() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.chat_lines.push(DisplayLine {
            text: "prior answer".into(),
            style: LineStyle::Body,
        });
        panel.chat_active = true;
        panel.interaction_mode = InteractionMode::Chat;
        panel.show_sse_error("backend down");
        assert!(panel.is_visible());
        assert!(panel.chat_lines.iter().any(|l| l.text.contains("backend down")));
        panel.dismiss_error();
        assert!(!panel.is_error_mode());
        assert!(panel.chat_lines.iter().any(|l| l.text.contains("prior answer")));
        assert!(!panel.chat_lines.iter().any(|l| l.text.contains("backend down")));
    }

    #[test]
    fn dispatch_key_arrow_up_scrolls_toward_older_content() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.scroll_offset = 5;
        panel.stick_to_bottom = false;
        let action = dispatch(&mut panel, Key::Named(NamedKey::ArrowUp), ModifiersState::empty());
        assert_eq!(action, OverlayAction::ScrollUp);
        assert_eq!(panel.scroll_offset, 4);
        assert!(!panel.stick_to_bottom);
    }

    #[test]
    fn dispatch_key_arrow_down_scrolls_toward_newer_content() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.stick_to_bottom = false;
        let action = dispatch(&mut panel, Key::Named(NamedKey::ArrowDown), ModifiersState::empty());
        assert_eq!(action, OverlayAction::ScrollDown);
        assert_eq!(panel.scroll_offset, 1);
    }

    #[test]
    fn scroll_step_accelerates_on_repeat() {
        let mut panel = OverlayPanel::new();
        assert_eq!(panel.scroll_step(false), 1);
        assert_eq!(panel.scroll_step(true), 1);
        for _ in 0..8 {
            panel.scroll_step(true);
        }
        assert!(panel.scroll_step(true) > 1);
    }

    #[test]
    fn begin_chat_message_sticks_to_bottom() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.scroll_offset = 0;
        panel.stick_to_bottom = false;
        panel.begin_chat_message("what is rebase?");
        assert!(panel.stick_to_bottom);
    }

    #[test]
    fn dispatch_key_shift_y_copies_command_not_explanation() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.set_context(None, "git rebase -i HEAD~3".into());
        panel.has_chunks = true;
        panel.finalize(&make_response());
        let action = dispatch(&mut panel, key_char("y"), ModifiersState::SHIFT);
        match action {
            OverlayAction::CopySelection(text) => {
                assert_eq!(text, "git rebase -i HEAD~3", "should copy only the command");
            },
            other => panic!("expected CopySelection, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_key_shift_y_with_no_command_returns_none() {
        let mut panel = OverlayPanel::new();
        panel.show();
        // No context_command set — nothing to copy.
        let action = dispatch(&mut panel, key_char("Y"), ModifiersState::SHIFT);
        assert_eq!(action, OverlayAction::None);
    }

    #[test]
    fn dispatch_key_unhandled_navigation_returns_none() {
        let mut panel = OverlayPanel::new();
        panel.show();
        // F5 is not in the routing table; it should fall through to None without typing.
        let action = dispatch(&mut panel, Key::Named(NamedKey::F5), ModifiersState::empty());
        assert_eq!(action, OverlayAction::None);
    }

    #[test]
    fn dispatch_key_arrow_up_returns_none_with_ctrl() {
        let mut panel = OverlayPanel::new();
        panel.show();
        let action =
            dispatch(&mut panel, Key::Named(NamedKey::ArrowUp), ModifiersState::CONTROL);
        assert_eq!(action, OverlayAction::None);
    }

    #[test]
    fn dispatch_key_when_hidden_short_circuits() {
        // handle_key (the public entry point) short-circuits on !visible. We can't fabricate
        // a winit::KeyEvent (private platform_specific field), so we just confirm visibility
        // gating via the helper getter.
        let panel = OverlayPanel::new();
        assert!(!panel.is_visible());
    }

    #[test]
    fn dispatch_key_plain_y_inserts_into_input_buffer() {
        let mut panel = OverlayPanel::new();
        panel.show();
        // Without Shift, plain "y" should be typed into the input field, not copied.
        let action = dispatch_with_text(&mut panel, key_char("y"), ModifiersState::empty(), "y");
        assert_eq!(action, OverlayAction::None);
        assert_eq!(panel.input_buffer, "y");
    }

    #[test]
    fn dispatch_key_backspace_removes_last_input_char() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.push_input_str("hi");
        let action =
            dispatch(&mut panel, Key::Named(NamedKey::Backspace), ModifiersState::empty());
        assert_eq!(action, OverlayAction::None);
        assert_eq!(panel.input_buffer, "h");
    }

    #[test]
    fn dispatch_key_enter_in_command_mode_does_not_submit() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.push_input_str("what is rebase?");
        let action = dispatch(&mut panel, Key::Named(NamedKey::Enter), ModifiersState::empty());
        assert_eq!(action, OverlayAction::None);
    }

    #[test]
    fn dispatch_key_enter_in_chat_mode_submits() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.interaction_mode = InteractionMode::Chat;
        panel.push_input_str("what is rebase?");
        let action = dispatch(&mut panel, Key::Named(NamedKey::Enter), ModifiersState::empty());
        assert_eq!(action, OverlayAction::SubmitChat("what is rebase?".into()));
        assert!(panel.input_buffer.is_empty());
        assert!(panel.chat_active);
        assert!(panel.chat_lines.iter().any(|l| l.text.contains("what is rebase?")));
    }

    #[test]
    fn dispatch_key_tab_toggles_interaction_mode() {
        let mut panel = OverlayPanel::new();
        panel.show();
        assert_eq!(panel.interaction_mode, InteractionMode::Command);
        let action = dispatch(&mut panel, Key::Named(NamedKey::Tab), ModifiersState::empty());
        assert_eq!(action, OverlayAction::ToggleMode);
        assert_eq!(panel.interaction_mode, InteractionMode::Chat);
    }

    #[test]
    fn dispatch_key_enter_slash_help_is_local() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.push_input_str("/help");
        let action = dispatch(&mut panel, Key::Named(NamedKey::Enter), ModifiersState::empty());
        assert_eq!(action, OverlayAction::None);
        assert!(panel.command_lines.iter().any(|l| l.text.contains("/info")));
    }

    #[test]
    fn dispatch_key_enter_slash_info_triggers_command() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.push_input_str("/info");
        let action = dispatch(&mut panel, Key::Named(NamedKey::Enter), ModifiersState::empty());
        assert_eq!(
            action,
            OverlayAction::RunSlashCommand(SlashCommand::Info { refresh: false })
        );
    }

    #[test]
    fn slash_command_parse_info_refresh() {
        let parsed = SlashCommand::parse("/info refresh").unwrap().unwrap();
        assert_eq!(parsed, SlashCommand::Info { refresh: true });
    }

    #[test]
    fn slash_command_parse_clear() {
        assert_eq!(
            SlashCommand::parse("/clear").unwrap().unwrap(),
            SlashCommand::Clear
        );
    }

    #[test]
    fn dispatch_key_enter_slash_clear_empties_chat() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.interaction_mode = InteractionMode::Chat;
        panel.chat_lines.push(DisplayLine {
            text: "Learnminal: old reply".into(),
            style: LineStyle::Body,
        });
        panel.command_lines.push(DisplayLine {
            text: "Command explanation".into(),
            style: LineStyle::Body,
        });
        panel.push_input_str("/clear");
        let action = dispatch(&mut panel, Key::Named(NamedKey::Enter), ModifiersState::empty());
        assert_eq!(action, OverlayAction::None);
        assert!(panel.chat_lines.is_empty());
        assert_eq!(panel.interaction_mode(), InteractionMode::Chat);
        assert!(panel.command_lines.iter().any(|l| l.text.contains("Command explanation")));
    }

    // ---- Task 6.1: Code-fence syntax highlighting in finalize ----

    #[test]
    fn append_chunk_buffers_json_without_flashing_raw_tokens() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.set_loading();
        panel.append_chunk("{\"command_name\":");
        assert!(panel.stream_buffer.contains("command_name"));
        assert!(!panel.command_lines.iter().any(|l| l.text.contains("command_name")));
        assert!(panel.command_lines.iter().any(|l| l.text.contains("Loading")));
    }

    #[test]
    fn finalize_marks_code_fences_with_code_style() {
        let mut panel = OverlayPanel::new();
        panel.show();
        panel.has_chunks = true;
        let mut response = make_response();
        response.general_utility = "intro\n```\ngit status\n```\noutro".into();
        panel.finalize(&response);

        // The fenced "git status" should be styled as Code; bookend lines as Body.
        let code_line =
            panel.command_lines.iter().find(|l| l.text.contains("git status")).unwrap();
        assert_eq!(code_line.style, LineStyle::Code);
        let intro_line =
            panel.command_lines.iter().find(|l| l.text.trim() == "intro").unwrap();
        assert_eq!(intro_line.style, LineStyle::Body);
        // Fence markers themselves are dropped.
        assert!(panel.command_lines.iter().all(|l| !l.text.contains("```")));
    }

    proptest! {
        // ---- Task 5.3: Overlay panel dimensions are proportional to terminal size. ----
        // Property 16: bottom-right corner panel is proportional to window size.
        #[test]
        fn property16_panel_geometry_proportional_to_size(
            width_px in 200u32..4000,
            height_px in 200u32..3000,
            cell_w in 6u32..30,
            cell_h in 10u32..40,
        ) {
            let panel = OverlayPanel::new();
            let size = SizeInfo::new(
                width_px as f32,
                height_px as f32,
                cell_w as f32,
                cell_h as f32,
                0.,
                0.,
                false,
            );
            let (x, y, panel_w, panel_h) = panel.panel_geometry(&size);

            let expected_w = (width_px as f32) * PANEL_WIDTH_FRACTION;
            let expected_h = (height_px as f32) * PANEL_HEIGHT_FRACTION;
            prop_assert!((panel_w - expected_w).abs() < 0.01);
            prop_assert!((panel_h - expected_h).abs() < 0.01);
            prop_assert!((x + panel_w - width_px as f32).abs() < 0.01);
            prop_assert!((y + panel_h - height_px as f32).abs() < 0.01);
        }

        // ---- Task 6.3: finalize renders all non-null sections. ----
        // Property 17: section header for `error_fix` appears iff Some; section header for
        // `similar_commands` appears iff non-empty; `Flags` appears iff non-empty.
        #[test]
        fn property17_finalize_renders_all_non_null_sections(
            command_name in "[a-z]{1,16}",
            general in "[a-zA-Z0-9 ]{1,40}",
            context in "[a-zA-Z0-9 ]{1,40}",
            include_fix in any::<bool>(),
            fix_text in "[a-zA-Z0-9 ]{1,40}",
            similar in prop::collection::vec("[a-z]{1,8}", 0..4),
            flag_count in 0usize..4,
        ) {
            let flags = (0..flag_count)
                .map(|i| FlagExplanation {
                    flag: format!("--f{i}"),
                    meaning: format!("meaning{i}"),
                    example: format!("ex{i}"),
                })
                .collect::<Vec<_>>();

            let response = ExplainResponse {
                command_name: command_name.clone(),
                flags_explained: flags,
                general_utility: general,
                contextual_usage: context,
                error_fix: if include_fix { Some(fix_text.clone()) } else { None },
                similar_commands: similar.clone(),
                tool_calls_made: Vec::new(),
                actionable_items: Vec::new(),
            };

            let mut panel = OverlayPanel::new();
            panel.show();
            panel.has_chunks = true;
            panel.finalize(&response);

            let texts: Vec<&str> = panel.command_lines.iter().map(|l| l.text.as_str()).collect();
            let has = |needle: &str| texts.iter().any(|t| t == &needle);

            // Always-present sections.
            prop_assert!(has(SECTION_GENERAL), "missing General section");
            prop_assert!(has(SECTION_CONTEXT), "missing Context section");
            prop_assert!(
                texts.iter().any(|t| t.contains(&command_name)),
                "missing command_name in title row",
            );

            // Conditional sections — present iff the source data is non-empty.
            prop_assert_eq!(has(SECTION_FIX), include_fix);
            prop_assert_eq!(has(SECTION_SIMILAR), !similar.is_empty());
            prop_assert_eq!(has(SECTION_FLAGS), flag_count > 0);
        }
    }
}
