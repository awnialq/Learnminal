"""
Enumerate installed packages per detected package manager.

Never raises — failures yield empty lists for that manager. Output is capped
before persistence to keep SQLite and LLM prompts bounded.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Callable

# Max packages stored per manager (full list in DB).
MAX_PACKAGES_PER_MANAGER = 500
# Max package names embedded in the agent prompt per manager.
MAX_PACKAGES_IN_PROMPT = 60
# Wall-clock limit for a single manager probe.
PROBE_TIMEOUT_SEC = 20


def _run_command(cmd: list[str], *, timeout: int = PROBE_TIMEOUT_SEC) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return 1, ""


def _cap(names: list[str], limit: int = MAX_PACKAGES_PER_MANAGER) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = name.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name.strip())
        if len(out) >= limit:
            break
    return sorted(out, key=str.lower)


def _lines_from_output(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _probe_brew() -> list[str]:
    code, out = _run_command(["brew", "list", "--formula"])
    names = _lines_from_output(out) if code == 0 else []
    code2, out2 = _run_command(["brew", "list", "--cask"])
    if code2 == 0:
        names.extend(f"cask:{n}" for n in _lines_from_output(out2))
    return _cap(names)


def _probe_dpkg() -> list[str]:
    code, out = _run_command(
        ["dpkg-query", "-W", "-f=${Package}\n"],
    )
    if code != 0:
        return []
    return _cap(_lines_from_output(out))


def _probe_rpm_names() -> list[str]:
    code, out = _run_command(["rpm", "-qa", "--queryformat", "%{NAME}\n"])
    if code != 0:
        return []
    return _cap(_lines_from_output(out))


def _probe_pacman() -> list[str]:
    code, out = _run_command(["pacman", "-Qq"])
    if code != 0:
        return []
    return _cap(_lines_from_output(out))


def _probe_nix_env() -> list[str]:
    code, out = _run_command(["nix-env", "-q"])
    if code != 0:
        return []
    names = []
    for line in _lines_from_output(out):
        # "pkgname-version" or "pkg name"
        names.append(line.split("-", 1)[0].strip() if "-" in line else line)
    return _cap(names)


def _probe_conda() -> list[str]:
    code, out = _run_command(["conda", "list", "-n", "base", "--json"])
    if code == 0 and out.strip():
        try:
            data = json.loads(out)
            names = [
                item.get("name", "")
                for item in data
                if isinstance(item, dict) and item.get("name")
            ]
            return _cap(names)
        except json.JSONDecodeError:
            pass
    code, out = _run_command(["conda", "list"])
    if code != 0:
        return []
    names = []
    for line in _lines_from_output(out):
        if line.startswith("#") or " " not in line:
            continue
        names.append(line.split()[0])
    return _cap(names)


def _probe_pip() -> list[str]:
    for exe in ("pip3", "pip"):
        if not shutil.which(exe):
            continue
        code, out = _run_command([exe, "list", "--format=freeze"])
        if code != 0:
            continue
        names = []
        for line in _lines_from_output(out):
            if "==" in line:
                names.append(line.split("==", 1)[0].strip())
            elif "@" in line:
                names.append(line.split("@", 1)[0].strip())
        return _cap(names)
    return []


def _probe_cargo() -> list[str]:
    code, out = _run_command(["cargo", "install", "--list"])
    if code != 0:
        return []
    names = []
    for line in _lines_from_output(out):
        if line.endswith(":"):
            names.append(line[:-1].strip().split()[0])
    return _cap(names)


def _probe_gem() -> list[str]:
    code, out = _run_command(["gem", "list"])
    if code != 0:
        return []
    names = []
    for line in _lines_from_output(out):
        if line.startswith("***"):
            continue
        names.append(line.split()[0])
    return _cap(names)


def _probe_npm() -> list[str]:
    code, out = _run_command(["npm", "list", "-g", "--depth=0", "--json"])
    if code == 0 and out.strip():
        try:
            data = json.loads(out)
            deps = data.get("dependencies") or {}
            if isinstance(deps, dict):
                return _cap(list(deps.keys()))
        except json.JSONDecodeError:
            pass
    code, out = _run_command(["npm", "list", "-g", "--depth=0"])
    if code != 0:
        return []
    names = []
    for line in _lines_from_output(out):
        if line.startswith(("npm", "└", "├", " ")):
            continue
        token = line.split()[0].lstrip("└─├│ ")
        if token and token != "npm":
            names.append(token)
    return _cap(names)


def _probe_yarn() -> list[str]:
    code, out = _run_command(["yarn", "global", "list", "--depth=0"])
    if code != 0:
        return []
    names = []
    for line in _lines_from_output(out):
        m = re.match(r"├─\s+(\S+)", line)
        if m:
            names.append(m.group(1))
    return _cap(names)


def _probe_pnpm() -> list[str]:
    code, out = _run_command(["pnpm", "list", "-g", "--depth=0", "--json"])
    if code == 0 and out.strip():
        try:
            data = json.loads(out)
            if isinstance(data, list) and data:
                deps = data[0].get("dependencies") or {}
                if isinstance(deps, dict):
                    return _cap(list(deps.keys()))
        except json.JSONDecodeError:
            pass
    return []


# Manager id → probe when that binary is on PATH.
_MANAGER_PROBES: dict[str, Callable[[], list[str]]] = {
    "brew": _probe_brew,
    "apt": _probe_dpkg,
    "apt-get": _probe_dpkg,
    "dnf": _probe_rpm_names,
    "pacman": _probe_pacman,
    "zypper": _probe_rpm_names,
    "nix-env": _probe_nix_env,
    "conda": _probe_conda,
    "pip3": _probe_pip,
    "pip": _probe_pip,
    "cargo": _probe_cargo,
    "gem": _probe_gem,
    "npm": _probe_npm,
    "yarn": _probe_yarn,
    "pnpm": _probe_pnpm,
}


def collect_installed_packages(detected_managers: list[str]) -> dict[str, list[str]]:
    """Return {manager: [package names]} for each detected manager."""
    result: dict[str, list[str]] = {}
    probed: set[str] = set()

    for manager in detected_managers:
        probe_key = manager
        if manager == "apt-get":
            probe_key = "apt"
        if probe_key in probed:
            continue
        probe = _MANAGER_PROBES.get(manager) or _MANAGER_PROBES.get(probe_key)
        if probe is None:
            continue
        probed.add(probe_key)
        packages = probe()
        if packages:
            result[probe_key] = packages

    return result


def summarize_for_prompt(
    installed_packages: dict[str, list[str]],
    *,
    max_per_manager: int = MAX_PACKAGES_IN_PROMPT,
) -> str:
    """Compact multi-line summary for LLM prompts."""
    if not installed_packages:
        return ""

    lines: list[str] = []
    for manager in sorted(installed_packages):
        pkgs = installed_packages[manager]
        if not pkgs:
            continue
        sample = pkgs[:max_per_manager]
        suffix = f" … (+{len(pkgs) - len(sample)} more)" if len(pkgs) > len(sample) else ""
        lines.append(f"  {manager} ({len(pkgs)}): {', '.join(sample)}{suffix}")
    return "\n".join(lines)


def total_package_count(installed_packages: dict[str, list[str]]) -> int:
    return sum(len(pkgs) for pkgs in installed_packages.values())
