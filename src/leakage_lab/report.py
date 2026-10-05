"""Plain-text, Markdown, and HTML renderings of a shelf run."""

from __future__ import annotations

import html
import shutil
from pathlib import Path

from leakage_lab.cards import ExperimentCard, PathResult

_COVER_NAME = "cover.jpg"


def _fmt_scores(path: PathResult) -> str:
    scores = path.scores
    return (
        f"ROC-AUC {scores.roc_auc:.4f}   "
        f"accuracy {scores.accuracy:.4f}   "
        f"AP {scores.average_precision:.4f}"
    )


def render_text(cards: tuple[ExperimentCard, ...], seed: int) -> str:
    lines = [
        "Leakage Lab Shelf",
        f"seed {seed}",
        "",
        "Accuracy uses a probability cutoff of 0.5. "
        "ROC-AUC and average precision use the positive-class probability. "
        "A high leaky score means that pipeline failed the audit.",
        "",
    ]
    for card in cards:
        lines.extend(
            [
                f"{card.number}. {card.title}",
                f"   {card.kicker}",
                f"   {card.population}",
                "",
                f"   leaky   {_fmt_scores(card.leaky)}",
                f"   clean   {_fmt_scores(card.clean)}",
                f"   gap     {card.roc_auc_gap:+.4f} ROC-AUC",
                "",
                _indent(card.mechanism),
                "",
                _indent(card.measured),
                "",
            ]
        )
        for label, value in card.facts:
            lines.append(f"   {label}: {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _indent(paragraph: str) -> str:
    return "\n".join(f"   {line}" if line else "" for line in _wrap(paragraph))


def _wrap(paragraph: str, width: int = 76) -> list[str]:
    words = paragraph.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_markdown(cards: tuple[ExperimentCard, ...], seed: int) -> str:
    parts = [
        "# Leakage Lab Shelf",
        "",
        f"Seed {seed}.",
        "",
        "Accuracy uses a probability cutoff of 0.5. "
        "ROC-AUC and average precision use the positive-class probability. "
        "A high leaky score means that pipeline failed the audit.",
        "",
    ]
    for card in cards:
        parts.extend(
            [
                f"## {card.number}. {card.title}",
                "",
                f"*{card.kicker}.* {card.population}.",
                "",
                "| Path | ROC-AUC | Accuracy | Average precision | Train rows | Test rows |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                _md_row("Leaky", card.leaky),
                _md_row("Clean", card.clean),
                "",
                f"ROC-AUC gap (leaky minus clean): {card.roc_auc_gap:+.4f}.",
                "",
                card.mechanism,
                "",
                card.measured,
                "",
            ]
        )
        if card.facts:
            parts.append("| | |")
            parts.append("| --- | --- |")
            for label, value in card.facts:
                parts.append(f"| {label} | {value} |")
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _md_row(name: str, path: PathResult) -> str:
    scores = path.scores
    return (
        f"| {name} | {scores.roc_auc:.4f} | {scores.accuracy:.4f} | "
        f"{scores.average_precision:.4f} | {path.n_train} | {path.n_test} |"
    )


def render_html(cards: tuple[ExperimentCard, ...], seed: int, *, cover_src: str | None) -> str:
    cover = ""
    if cover_src:
        cover = (
            f'<img class="cover" src="{html.escape(cover_src)}" '
            'alt="Leakage Lab Shelf cover: a cabinet of specimen jars, '
            'one of them cracked and leaking, labeled as experiments on '
            'ROC curves, splits, and validation.">'
        )
    articles = "\n".join(_article(card) for card in cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Leakage Lab Shelf</title>
<style>
:root {{
  --bg: #10161a;
  --panel: #182126;
  --ink: #f3eadc;
  --muted: #b7aea0;
  --line: #2c3840;
  --leak: #e15b68;
  --clean: #3cbfa0;
  --cream: #f6f0e4;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Georgia, "Iowan Old Style", Palatino, serif;
  line-height: 1.5;
}}
main {{
  max-width: 920px;
  margin: 0 auto;
  padding: 32px 20px 64px;
}}
.cover {{
  width: 100%;
  display: block;
  border-radius: 8px;
  margin-bottom: 28px;
}}
h1 {{
  font-weight: 500;
  font-size: 2.4rem;
  letter-spacing: 0.01em;
  margin: 0 0 8px;
}}
.tagline {{
  color: var(--muted);
  margin: 0 0 8px;
}}
.verbs {{
  color: var(--clean);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-family: "Segoe UI", sans-serif;
  font-size: 0.78rem;
  margin: 0 0 28px;
}}
.note {{
  color: var(--muted);
  max-width: 68ch;
}}
article {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 22px 22px 8px;
  margin: 22px 0;
}}
.kicker {{
  font-family: "Segoe UI", sans-serif;
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0;
}}
h2 {{
  margin: 4px 0 6px;
  font-weight: 500;
  font-size: 1.7rem;
}}
.population {{
  color: var(--muted);
  margin: 0 0 16px;
}}
.paths {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}}
.path {{
  background: #10171b;
  border-radius: 8px;
  padding: 12px 14px;
  border-left: 4px solid var(--line);
}}
.path.leaky {{ border-left-color: var(--leak); }}
.path.clean {{ border-left-color: var(--clean); }}
.path .name {{
  font-family: "Segoe UI", sans-serif;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 0;
}}
.path.leaky .name {{ color: var(--leak); }}
.path.clean .name {{ color: var(--clean); }}
.auc {{
  font-variant-numeric: tabular-nums;
  font-size: 2rem;
  margin: 2px 0;
}}
.sub {{
  font-family: "Segoe UI", sans-serif;
  color: var(--muted);
  font-size: 0.92rem;
  margin: 0;
}}
.gap {{
  font-family: "Segoe UI", sans-serif;
  margin: 12px 0 0;
}}
p {{ margin: 12px 0; }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 8px 0 16px;
  font-family: "Segoe UI", sans-serif;
  font-size: 0.92rem;
}}
th, td {{
  text-align: left;
  padding: 6px 8px 6px 0;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}}
td:last-child {{
  font-variant-numeric: tabular-nums;
  text-align: right;
  white-space: nowrap;
}}
footer {{
  color: var(--muted);
  font-size: 0.95rem;
}}
footer a {{ color: var(--cream); }}
@media (max-width: 700px) {{
  .paths {{ grid-template-columns: 1fr; }}
  h1 {{ font-size: 2rem; }}
}}
</style>
</head>
<body>
<main>
{cover}
<h1>Leakage Lab Shelf</h1>
<p class="tagline">Planted failure cases beside clean controls. Synthetic tables, fixed seed {seed}.</p>
<p class="verbs">Inspect data flow · Trace leakage · Build cleaner models</p>
<p class="note">Accuracy uses a probability cutoff of 0.5. ROC-AUC and average precision use the positive-class probability. A high leaky score means that pipeline failed the audit.</p>
{articles}
<footer>
<p>Companion checker for a split you already have: <a href="https://github.com/grummpy/SplitCheck">SplitCheck</a>. This shelf does not import it.</p>
</footer>
</main>
</body>
</html>
"""


def _article(card: ExperimentCard) -> str:
    facts = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in card.facts
    )
    return f"""
<article id="{html.escape(card.slug)}">
<p class="kicker">Experiment {card.number} · {html.escape(card.kicker)}</p>
<h2>{html.escape(card.title)}</h2>
<p class="population">{html.escape(card.population)}</p>
<div class="paths">
{_path_block("leaky", "Leaky", card.leaky)}
{_path_block("clean", "Clean", card.clean)}
</div>
<p class="gap">ROC-AUC gap (leaky minus clean): {card.roc_auc_gap:+.4f}</p>
<p>{html.escape(card.mechanism)}</p>
<p>{html.escape(card.measured)}</p>
<table>{facts}</table>
</article>
"""


def _path_block(kind: str, name: str, path: PathResult) -> str:
    scores = path.scores
    return f"""
<div class="path {kind}">
<p class="name">{name}</p>
<p class="auc">{scores.roc_auc:.4f}</p>
<p class="sub">ROC-AUC</p>
<p class="sub">Accuracy {scores.accuracy:.4f} · AP {scores.average_precision:.4f}</p>
<p class="sub">{path.n_train} train rows · {path.n_test} test rows</p>
</div>
"""


def find_cover(start: Path | None = None) -> Path | None:
    """Cover shipped at docs/cover.jpg in a checkout of this repo."""

    candidates = []
    if start is not None:
        candidates.append(start)
    cwd = Path.cwd()
    candidates.append(cwd / "docs" / _COVER_NAME)
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "docs" / _COVER_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def write_html(cards: tuple[ExperimentCard, ...], path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cover = find_cover()
    cover_src = None
    if cover is not None:
        destination = path.parent / _COVER_NAME
        if cover.resolve() != destination.resolve():
            shutil.copyfile(cover, destination)
        cover_src = _COVER_NAME
    path.write_text(render_html(cards, seed, cover_src=cover_src), encoding="utf-8")
