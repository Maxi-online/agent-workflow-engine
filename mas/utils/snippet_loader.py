"""Load only explicitly named HTML snippets from the configured directory."""

import os
from pathlib import Path
import re


def load_snippet(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid snippet name")
    directory = Path(
        os.getenv("AGENT_SNIPPETS_DIR", str(Path(__file__).resolve().parents[1] / "snippets"))
    ).resolve()
    target = (directory / (name + ".html")).resolve()
    if target.parent != directory or not target.is_file():
        return ""
    return target.read_text(encoding="utf-8")
