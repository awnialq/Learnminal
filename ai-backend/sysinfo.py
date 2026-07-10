"""
Detect the user's system environment for tailored AI responses.

collect() → dict is the public API.  It never raises — missing info
is represented as empty strings or empty lists.
"""
import os
import platform
import shutil
import time

from packages import collect_installed_packages, summarize_for_prompt, total_package_count

# Ordered: first match wins when reporting to the user.
_PACKAGE_MANAGERS = [
    "brew", "apt", "apt-get", "dnf", "pacman", "zypper",
    "nix-env", "conda", "pip3", "cargo", "gem", "npm", "yarn", "pnpm",
]

# Tools we probe for; presence shapes suggestions (e.g. "use brew install …").
_DEV_TOOLS = [
    # VCS
    "git", "gh", "hg", "svn",
    # Containers / orchestration
    "docker", "podman", "kubectl", "helm", "kind",
    # Languages / runtimes
    "python3", "python", "node", "ruby", "go", "java", "rustc", "perl", "php",
    # Build
    "make", "cmake", "ninja", "gradle", "mvn",
    # Cloud CLIs
    "aws", "gcloud", "az", "terraform",
    # Shell utilities
    "curl", "wget", "jq", "fzf", "rg", "fd", "bat", "eza", "htop", "tmux",
]


def _detect_os() -> str:
    system = platform.system()
    if system == "Darwin":
        ver, _, _ = platform.mac_ver()
        return f"macOS {ver}" if ver else "macOS"
    if system == "Linux":
        try:
            rel = platform.freedesktop_os_release()
            return rel.get("PRETTY_NAME") or f"Linux {platform.release()}"
        except (AttributeError, OSError):
            return f"Linux {platform.release()}"
    if system == "Windows":
        return f"Windows {platform.version()}"
    return f"{system} {platform.release()}"


def collect() -> dict:
    """Return a snapshot of the user's system environment.

    Keys: os, arch, shell, package_managers (list), installed_packages (dict),
          installed_packages_summary (str), installed_packages_total (int),
          installed_tools (list), collected_at (unix timestamp).
    """
    shell_path = os.environ.get("SHELL", "")
    shell_name = os.path.basename(shell_path) if shell_path else "unknown"
    package_managers = [pm for pm in _PACKAGE_MANAGERS if shutil.which(pm)]
    installed_packages = collect_installed_packages(package_managers)

    return {
        "os": _detect_os(),
        "arch": platform.machine(),
        "shell": shell_name,
        "package_managers": package_managers,
        "installed_packages": installed_packages,
        "installed_packages_summary": summarize_for_prompt(installed_packages),
        "installed_packages_total": total_package_count(installed_packages),
        "installed_tools": [t for t in _DEV_TOOLS if shutil.which(t)],
        "collected_at": int(time.time()),
    }
