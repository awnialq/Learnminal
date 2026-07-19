//! Learnminal AI learning overlay integration.

pub mod docs_fallback;
pub mod grid_extractor;
pub mod journal;
pub mod manpage;
pub mod ollama;
pub mod overlay;
pub mod prompt;
pub mod settings;
pub mod sysinfo;
pub mod types;
pub mod verify;

pub use grid_extractor::{extract_context, read_last_exit_code};
pub use ollama::{OllamaClient, OllamaError};
pub use overlay::SlashCommand;
pub use overlay::{InputFocus, OverlayAction, OverlayPanel};
pub use types::{ReferenceContext, ReferenceSource, SystemInfo, TerminalContext};
