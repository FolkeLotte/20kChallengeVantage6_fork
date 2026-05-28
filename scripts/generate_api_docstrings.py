#!/usr/bin/env python3
"""Generate docs/api.md from verbatim source docstrings.

This script extracts module and top-level function docstrings from selected
Python files and writes them to docs/api.md without summarizing content.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


DEFAULT_SOURCES = [
    "my-fl-project/20kLogRegChallenge/logreg_challenge_20k/central.py",
    "my-fl-project/20kLogRegChallenge/logreg_challenge_20k/partial.py",
]
DEFAULT_OUTPUT = "docs/api.md"


def build_markdown(repo_root: Path, sources: list[Path]) -> str:
    lines: list[str] = ["# API Reference (Verbatim Docstrings)", ""]
    lines.append(
        "This file is generated from source docstrings without summarizing their content."
    )
    lines.append("")

    for source in sources:
        src_text = source.read_text(encoding="utf-8")
        tree = ast.parse(src_text)
        rel = source.relative_to(repo_root)

        lines.append(f"## {rel.as_posix()}")
        lines.append("")

        module_doc = ast.get_docstring(tree, clean=False)
        if module_doc:
            lines.append("### Module Docstring")
            lines.append("")
            lines.append("```text")
            lines.append(module_doc)
            lines.append("```")
            lines.append("")

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue

            lines.append(f"### {node.name}")
            lines.append("")
            lines.append("```text")
            lines.append(doc)
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root path (defaults to parent of scripts/)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output markdown path relative to repo root",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source Python file path relative to repo root (can be repeated)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_rel_paths = args.source if args.source else DEFAULT_SOURCES
    sources = [repo_root / rel for rel in source_rel_paths]
    output_path = repo_root / args.output

    markdown = build_markdown(repo_root, sources)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(output_path)


if __name__ == "__main__":
    main()
