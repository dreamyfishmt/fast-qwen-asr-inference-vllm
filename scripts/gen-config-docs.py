#!/usr/bin/env python3
"""Regenerate the server settings table in docs/configuration.md from the env() declarations.

    uv run scripts/gen-config-docs.py           # rewrite the file
    uv run scripts/gen-config-docs.py --check   # exit 1 if it is out of date (used by the tests)
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "configuration.md"
BEGIN = "<!-- BEGIN GENERATED: scripts/gen-config-docs.py -->"
END = "<!-- END GENERATED -->"
MODULES = ["asr_server.app", "asr_server.engines.vllm_engine", "asr_server.engines.onnx_engine"]
GROUP_ORDER = ["Server", "Limits", "Streaming", "Languages", "OpenAI API", "ONNX backend", "vLLM backend"]


def render() -> str:
    sys.path.insert(0, str(ROOT))
    from asr_server import config

    for m in MODULES:
        importlib.import_module(m)  # registers their settings (the documented defaults ignore the environment)

    groups = {}
    for s in config.REGISTRY.values():
        groups.setdefault(s.group, []).append(s)
    order = GROUP_ORDER + sorted(g for g in groups if g not in GROUP_ORDER)

    out = [BEGIN, ""]
    for group in order:
        if group not in groups:
            continue
        out += [f"### {group}", "", "| Variable | Default | Description |", "|---|---|---|"]
        for s in sorted(groups[group], key=lambda s: s.name):
            default = f"`{s.default}`" if s.default != "" else "—"
            out.append(f"| `{s.name}` | {default} | {s.help} |")
        out.append("")
    out.append(END)
    return "\n".join(out)


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    start, end = text.index(BEGIN), text.index(END) + len(END)
    new = text[:start] + render() + text[end:]
    if "--check" in sys.argv:
        if new != text:
            print(f"{DOC.relative_to(ROOT)} is out of date: run scripts/gen-config-docs.py", file=sys.stderr)
            return 1
        return 0
    DOC.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
