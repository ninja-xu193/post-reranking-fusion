"""Check a small, explicitly scoped source release before uploading it."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FIXED = {
    ".gitignore", ".gitattributes", "LICENSE", "README.md", "README.zh-CN.md",
    "pyproject.toml", "requirements-tested.txt", ".github/workflows/ci.yml",
    "examples/synthetic.jsonl",
}
IGNORED = {".git", "__pycache__", ".venv", "build", "dist"}


def allowed(path):
    name = path.as_posix()
    if name in FIXED:
        return True
    return ((name.startswith("src/postfusion/") or name.startswith("tests/")
             or name.startswith("tools/")) and path.suffix == ".py") or (
                 name.startswith("docs/") and path.suffix == ".md")


def inspect(root):
    errors, files = [], []
    patterns = [
        ("machine path", re.compile(r"/(?:Us" + r"ers|ho" + r"me|ro" + r"ot)/")),
        ("private key", re.compile(r"-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
        ("access token", re.compile(r"(?:gh[pousr]_" + r"[A-Za-z0-9]{20,}|github_pat_" + r"[A-Za-z0-9_]{20,})")),
        ("cloud key", re.compile(r"AK" + r"IA[0-9A-Z]{16}")),
        ("internal IPv4", re.compile(r"\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b")),
    ]
    total = 0
    for file in sorted(root.rglob("*")):
        rel = file.relative_to(root)
        if any(part in IGNORED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if file.is_symlink():
            errors.append(f"{rel}: symbolic links are not permitted")
            continue
        if not file.is_file():
            continue
        if not allowed(rel):
            errors.append(f"{rel}: outside the release allowlist")
            continue
        payload = file.read_bytes()
        total += len(payload)
        if len(payload) > 200_000:
            errors.append(f"{rel}: exceeds the source-file size limit")
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{rel}: not UTF-8 text")
            continue
        if "\x00" in content:
            errors.append(f"{rel}: binary content")
        for label, pattern in patterns:
            if pattern.search(content):
                errors.append(f"{rel}: possible {label}; manual review required")
        files.append((rel.as_posix(), hashlib.sha256(payload).hexdigest()))
    for name in FIXED:
        if not (root / name).is_file():
            errors.append(f"{name}: required release file missing")
    if total > 1_000_000:
        errors.append("Release exceeds the total source size limit")
    return files, errors


def main():
    files, errors = inspect(ROOT)
    for name, digest in files:
        print(f"{digest}  {name}")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"PASS: {len(files)} allowlisted source files; no flagged content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
