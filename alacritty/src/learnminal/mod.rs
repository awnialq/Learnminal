//! Learnminal AI learning overlay integration.

pub mod grid_extractor;
pub mod ipc_client;
pub mod overlay;
pub mod types;

pub use grid_extractor::{extract_context, read_last_command, read_last_exit_code};
pub use ipc_client::{IpcClient, IpcError};
pub use overlay::{
    InputFocus, InteractionMode, OverlayAction, OverlayDrawData, OverlayPanel, OverlayText,
};
pub use overlay::SlashCommand;
pub use types::{ChatDoneEvent, ExplainResponse, SystemInfo, TerminalContext};
