//! Persistent user preferences (model selection) stored in
//! `~/.ai-cli-learning/settings.json`.
//!
//! The selected model is kept independently from the terminal configuration.

use std::io;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Directory holding Learnminal preferences under the user's home.
pub const SETTINGS_DIR_NAME: &str = ".ai-cli-learning";
const SETTINGS_FILE_NAME: &str = "settings.json";
const MODEL_KEY: &str = "ollama_model";

fn settings_dir() -> Option<PathBuf> {
    home::home_dir().map(|home| home.join(SETTINGS_DIR_NAME))
}

fn settings_path() -> Option<PathBuf> {
    settings_dir().map(|dir| dir.join(SETTINGS_FILE_NAME))
}

/// Preferred Ollama model from settings, if any.
pub fn get_preferred_model() -> Option<String> {
    read_preferred_model(&settings_path()?)
}

/// Persist the preferred Ollama model.
pub fn set_preferred_model(model: &str) -> io::Result<()> {
    let dir = settings_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "home directory not found"))?;
    write_preferred_model(&dir, model)
}

fn read_preferred_model(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    let model = value.get(MODEL_KEY)?.as_str()?.trim();
    if model.is_empty() {
        None
    } else {
        Some(model.to_owned())
    }
}

fn write_preferred_model(dir: &Path, model: &str) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(SETTINGS_FILE_NAME);

    let mut settings: Map<String, Value> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<Map<String, Value>>(&text).ok())
        .unwrap_or_default();
    settings.insert(MODEL_KEY.to_owned(), Value::String(model.trim().to_owned()));

    // Atomic write: serialize to a temp file in the same dir, then rename.
    let mut tmp = tempfile::Builder::new().prefix(".settings").suffix(".tmp").tempfile_in(dir)?;
    serde_json::to_writer_pretty(&mut tmp, &settings)?;
    use std::io::Write;
    tmp.as_file_mut().write_all(b"\n")?;
    tmp.persist(&path).map_err(|e| e.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_settings_file_yields_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        assert!(read_preferred_model(&path).is_none());
    }

    #[test]
    fn round_trips_preferred_model() {
        let dir = tempfile::tempdir().unwrap();
        write_preferred_model(dir.path(), "qwen3.6:35b-a3b").unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        assert_eq!(read_preferred_model(&path).as_deref(), Some("qwen3.6:35b-a3b"));
    }

    #[test]
    fn overwrite_preserves_other_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        std::fs::write(&path, r#"{"other":"keep","ollama_model":"old"}"#).unwrap();

        write_preferred_model(dir.path(), "new-model").unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value.get("other").and_then(Value::as_str), Some("keep"));
        assert_eq!(value.get("ollama_model").and_then(Value::as_str), Some("new-model"));
    }

    #[test]
    fn blank_model_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        std::fs::write(&path, r#"{"ollama_model":"   "}"#).unwrap();
        assert!(read_preferred_model(&path).is_none());
    }
}
