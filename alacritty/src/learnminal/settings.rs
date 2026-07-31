//! Persistent user preferences stored in `~/.ai-cli-learning/settings.json`.
//!
//! Preferences (model selection, experience level) are kept independently from
//! the terminal configuration.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde_json::{Map, Value};

/// Directory holding Learnminal preferences under the user's home.
pub const SETTINGS_DIR_NAME: &str = ".ai-cli-learning";
const SETTINGS_FILE_NAME: &str = "settings.json";
const MODEL_KEY: &str = "ollama_model";
const EXPERIENCE_LEVEL_KEY: &str = "experience_level";

/// User experience with the terminal and overall technical knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExperienceLevel {
    #[default]
    Beginner,
    Novice,
    Professional,
    Expert,
}

impl ExperienceLevel {
    /// All tiers in display order.
    pub const ALL: [Self; 4] = [Self::Beginner, Self::Novice, Self::Professional, Self::Expert];

    /// Canonical lowercase name used for persistence and slash commands.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Beginner => "beginner",
            Self::Novice => "novice",
            Self::Professional => "professional",
            Self::Expert => "expert",
        }
    }

    /// Human-readable title for overlay display.
    pub fn label(self) -> &'static str {
        match self {
            Self::Beginner => "Beginner",
            Self::Novice => "Novice",
            Self::Professional => "Professional",
            Self::Expert => "Expert",
        }
    }

    /// Short description of who this tier is for.
    pub fn description(self) -> &'static str {
        match self {
            Self::Beginner => "new to the terminal; explain basics step by step",
            Self::Novice => "some shell experience; explain non-obvious details",
            Self::Professional => "comfortable daily CLI user; stay concise",
            Self::Expert => "deep terminal knowledge; terse and high-signal",
        }
    }
}

impl fmt::Display for ExperienceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for ExperienceLevel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "beginner" => Ok(Self::Beginner),
            "novice" => Ok(Self::Novice),
            "professional" => Ok(Self::Professional),
            "expert" => Ok(Self::Expert),
            _ => Err(()),
        }
    }
}

/// `~/.ai-cli-learning`, or `None` when the home directory cannot be resolved.
///
/// Every Learnminal runtime file (settings, journal, sessions, shell scripts) hangs
/// off this directory; resolve it here rather than re-deriving it per module.
pub fn state_dir() -> Option<PathBuf> {
    home::home_dir().map(|home| home.join(SETTINGS_DIR_NAME))
}

fn settings_path() -> Option<PathBuf> {
    state_dir().map(|dir| dir.join(SETTINGS_FILE_NAME))
}

/// Write `bytes` to `path` atomically, via a temp file in `dir` and a rename.
///
/// The temp file must live in the destination's own directory: a rename across
/// filesystems is not atomic, and `~` may well be a different mount than `/tmp`.
pub fn atomic_write(dir: &Path, path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;

    let mut tmp = tempfile::Builder::new().prefix(".learnminal").suffix(".tmp").tempfile_in(dir)?;
    tmp.as_file_mut().write_all(bytes)?;
    tmp.persist(path).map_err(|err| err.error)?;
    Ok(())
}

/// Preferred Ollama model from settings, if any.
pub fn get_preferred_model() -> Option<String> {
    read_preferred_model(&settings_path()?)
}

/// Persist the preferred Ollama model.
pub fn set_preferred_model(model: &str) -> io::Result<()> {
    let dir = state_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "home directory not found"))?;
    write_preferred_model(&dir, model)
}

/// Experience level from settings, defaulting to [`ExperienceLevel::Beginner`].
pub fn get_experience_level() -> ExperienceLevel {
    settings_path()
        .and_then(|path| read_experience_level(&path))
        .unwrap_or_default()
}

/// Persist the experience level.
pub fn set_experience_level(level: ExperienceLevel) -> io::Result<()> {
    let dir = state_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "home directory not found"))?;
    write_experience_level(&dir, level)
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
    write_setting(dir, MODEL_KEY, Value::String(model.trim().to_owned()))
}

fn read_experience_level(path: &Path) -> Option<ExperienceLevel> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    let raw = value.get(EXPERIENCE_LEVEL_KEY)?.as_str()?.trim();
    if raw.is_empty() {
        None
    } else {
        ExperienceLevel::from_str(raw).ok()
    }
}

fn write_experience_level(dir: &Path, level: ExperienceLevel) -> io::Result<()> {
    write_setting(dir, EXPERIENCE_LEVEL_KEY, Value::String(level.as_str().to_owned()))
}

fn write_setting(dir: &Path, key: &str, value: Value) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(SETTINGS_FILE_NAME);

    let mut settings: Map<String, Value> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<Map<String, Value>>(&text).ok())
        .unwrap_or_default();
    settings.insert(key.to_owned(), value);

    let mut bytes = serde_json::to_vec_pretty(&settings)?;
    bytes.push(b'\n');
    atomic_write(dir, &path, &bytes)
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
        write_preferred_model(dir.path(), "gemma4:e4b-mlx").unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        assert_eq!(read_preferred_model(&path).as_deref(), Some("gemma4:e4b-mlx"));
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

    #[test]
    fn round_trips_experience_level() {
        let dir = tempfile::tempdir().unwrap();
        write_experience_level(dir.path(), ExperienceLevel::Professional).unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        assert_eq!(read_experience_level(&path), Some(ExperienceLevel::Professional));
    }

    #[test]
    fn missing_experience_level_defaults_to_beginner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        assert!(read_experience_level(&path).is_none());
        assert_eq!(ExperienceLevel::default(), ExperienceLevel::Beginner);
    }

    #[test]
    fn invalid_experience_level_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        std::fs::write(&path, r#"{"experience_level":"wizard"}"#).unwrap();
        assert!(read_experience_level(&path).is_none());
    }

    #[test]
    fn blank_experience_level_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        std::fs::write(&path, r#"{"experience_level":"   "}"#).unwrap();
        assert!(read_experience_level(&path).is_none());
    }

    #[test]
    fn experience_level_parse_is_case_insensitive() {
        assert_eq!("Beginner".parse(), Ok(ExperienceLevel::Beginner));
        assert_eq!("NOVICE".parse(), Ok(ExperienceLevel::Novice));
        assert_eq!("  Professional  ".parse(), Ok(ExperienceLevel::Professional));
        assert_eq!("expert".parse(), Ok(ExperienceLevel::Expert));
        assert!("intermediate".parse::<ExperienceLevel>().is_err());
    }

    #[test]
    fn writing_experience_level_preserves_other_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SETTINGS_FILE_NAME);
        std::fs::write(&path, r#"{"ollama_model":"gemma","other":"keep"}"#).unwrap();

        write_experience_level(dir.path(), ExperienceLevel::Expert).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value.get("ollama_model").and_then(Value::as_str), Some("gemma"));
        assert_eq!(value.get("other").and_then(Value::as_str), Some("keep"));
        assert_eq!(value.get("experience_level").and_then(Value::as_str), Some("expert"));
    }
}
