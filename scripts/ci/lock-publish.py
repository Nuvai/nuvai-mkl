#!/usr/bin/env python3
"""Set `publish = false` on every crate in a Cargo repo (issue #2498, Gate 0).

Usage:  python3 lock-publish.py <repo-dir>          # dry run, prints what would change
        python3 lock-publish.py <repo-dir> --write  # edits Cargo.toml files in place
        python3 lock-publish.py <repo-dir> --check  # CI gate: exit 1 if any crate is publishable

- Adds `publish = false` right after `[package]` when no publish key exists.
- Rewrites `publish = [...]`, `publish = true` and `publish.workspace = true` to `publish = false`.
- Skips target/, vendor/, node_modules/ and .git/.
"""
import pathlib
import re
import sys

SKIP = {"target", "vendor", "node_modules", ".git"}
PUBLISH_LINE = re.compile(r"^\s*publish(\s*=|\.workspace\s*=)")


def package_section(lines):
    """Return (start, end) line indexes of the [package] table, or None."""
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "[package]":
            start = i
        elif start is not None and s.startswith("["):
            return start, i
    return (start, len(lines)) if start is not None else None


def fix(text):
    lines = text.splitlines(keepends=True)
    sec = package_section(lines)
    if sec is None:
        return None  # workspace-only manifest, nothing to publish
    start, end = sec
    for i in range(start + 1, end):
        if PUBLISH_LINE.match(lines[i]):
            if lines[i].strip() == "publish = false":
                return None
            # multi-line arrays: drop continuation lines up to the closing bracket
            j = i
            if "[" in lines[i] and "]" not in lines[i]:
                while j < end and "]" not in lines[j]:
                    j += 1
            lines[i : j + 1] = ["publish = false\n"]
            return "".join(lines)
    lines.insert(start + 1, "publish = false\n")
    return "".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = pathlib.Path(sys.argv[1]).resolve()
    write = "--write" in sys.argv
    changed = 0
    for p in sorted(root.rglob("Cargo.toml")):
        if SKIP & set(p.relative_to(root).parts):
            continue
        new = fix(p.read_text())
        if new is None:
            continue
        changed += 1
        print(("fixed   " if write else "would fix ") + str(p.relative_to(root)))
        if write:
            p.write_text(new)
    print(f"{changed} Cargo.toml file(s) {'changed' if write else 'to change'} in {root.name}")
    if "--check" in sys.argv and changed:
        sys.exit("ERROR: the crates above can be published. Run lock-publish.py --write.")


if __name__ == "__main__":
    main()
