//! Detect the user's system environment for tailored AI responses and `/info`.
//!
//! Never panics — missing info is represented as empty strings or empty lists.
//! Package inventory is omitted; `installed_packages` is always empty.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::learnminal::types::SystemInfo;

const PACKAGE_MANAGERS: &[&str] = &[
    "brew", "apt", "apt-get", "dnf", "pacman", "zypper", "nix-env", "conda", "pip3", "cargo",
    "gem", "npm", "yarn", "pnpm",
];

const DEV_TOOLS: &[&str] = &[
    "git",
    "gh",
    "hg",
    "svn",
    "docker",
    "podman",
    "kubectl",
    "helm",
    "kind",
    "python3",
    "python",
    "node",
    "ruby",
    "go",
    "java",
    "rustc",
    "perl",
    "php",
    "make",
    "cmake",
    "ninja",
    "gradle",
    "mvn",
    "aws",
    "gcloud",
    "az",
    "terraform",
    "curl",
    "wget",
    "jq",
    "fzf",
    "rg",
    "fd",
    "bat",
    "eza",
    "htop",
    "tmux",
];

/// Return a snapshot of the user's system environment.
pub fn collect() -> SystemInfo {
    let shell = std::env::var("SHELL")
        .ok()
        .and_then(|path| Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "unknown".to_owned());

    let package_managers =
        PACKAGE_MANAGERS.iter().filter(|p| which(p)).map(|p| (*p).to_owned()).collect();
    let installed_tools = DEV_TOOLS.iter().filter(|t| which(t)).map(|t| (*t).to_owned()).collect();

    let collected_at =
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).ok();

    SystemInfo {
        os: detect_os(),
        arch: std::env::consts::ARCH.to_owned(),
        shell,
        package_managers,
        installed_packages: std::collections::HashMap::new(),
        installed_packages_total: None,
        installed_tools,
        collected_at,
        collected_at_display: None,
    }
}

fn detect_os() -> String {
    match std::env::consts::OS {
        "macos" => {
            macos_version().map(|v| format!("macOS {v}")).unwrap_or_else(|| "macOS".to_owned())
        },
        "linux" => linux_pretty_name().unwrap_or_else(|| "Linux".to_owned()),
        "windows" => "Windows".to_owned(),
        other => {
            let mut chars = other.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => "Unknown".to_owned(),
            }
        },
    }
}

fn macos_version() -> Option<String> {
    let output = std::process::Command::new("sw_vers").arg("-productVersion").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if version.is_empty() {
        None
    } else {
        Some(version)
    }
}

fn linux_pretty_name() -> Option<String> {
    let text = std::fs::read_to_string("/etc/os-release").ok()?;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("PRETTY_NAME=") {
            let trimmed = value.trim().trim_matches('"').trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
    }
    None
}

/// Whether `program` is found on `$PATH` as an executable (like `shutil.which`).
fn which(program: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| {
        let candidate = dir.join(program);
        is_executable(&candidate)
    })
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_populates_core_fields() {
        let info = collect();
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
        assert!(info.collected_at.is_some());
        assert!(info.is_complete());
    }

    #[test]
    fn which_finds_a_ubiquitous_binary() {
        // `sh` exists on every supported unix; on other platforms just assert no panic.
        #[cfg(unix)]
        assert!(which("sh"));
        #[cfg(not(unix))]
        let _ = which("cmd");
    }

    #[test]
    fn which_rejects_nonexistent_binary() {
        assert!(!which("definitely-not-a-real-binary-xyz-123"));
    }
}
