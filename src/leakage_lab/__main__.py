"""Command line for the shelf.

``python -m leakage_lab`` and ``leakage-lab`` both print the shelf.
"""

from __future__ import annotations

import argparse
import functools
import sys
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from leakage_lab import DEFAULT_SEED, __version__
from leakage_lab.catalog import run_shelf
from leakage_lab.exports import write_split_audit_exports
from leakage_lab.provenance import SUMMARY_SEEDS, write_manifest
from leakage_lab.report import render_markdown, render_multiseed_summary, render_text, write_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leakage-lab",
        description=(
            "Run the Leakage Lab Shelf: planted data leaks beside clean controls."
        ),
    )
    parser.add_argument("--version", action="version", version=f"leakage-lab {__version__}")
    sub = parser.add_subparsers(dest="command")

    shelf = sub.add_parser("shelf", help="Run every experiment and print the shelf")
    shelf.add_argument("--seed", type=int, default=DEFAULT_SEED)
    shelf.add_argument(
        "--output",
        type=Path,
        help="Also write a Markdown report to this path",
    )
    shelf.add_argument("--manifest", type=Path, help="Also write the versioned input/split/environment manifest as JSON.")
    shelf.add_argument(
        "--multi-seed-summary",
        action="store_true",
        help="Append the predeclared seed summary (0, 7, 42); it never selects seeds from results.",
    )
    shelf.add_argument(
        "--splitcheck-export",
        type=Path,
        help="Optionally write synthetic CSV fixtures for a separate split-audit tool; no tool dependency is imported.",
    )
    shelf.add_argument(
        "--html",
        type=Path,
        help="Also write an HTML shelf to this path",
    )

    serve = sub.add_parser("serve", help="Serve the HTML shelf on localhost")
    serve.add_argument("--seed", type=int, default=DEFAULT_SEED)
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _shelf(
    seed: int,
    output: Path | None,
    html_path: Path | None,
    manifest: Path | None,
    multi_seed_summary: bool,
    splitcheck_export: Path | None,
) -> int:
    cards = run_shelf(seed)
    sys.stdout.write(render_text(cards, seed))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(cards, seed), encoding="utf-8")
        print(f"Wrote {output}", file=sys.stderr)
    if html_path is not None:
        write_html(cards, html_path, seed)
        print(f"Wrote {html_path}", file=sys.stderr)
    if manifest is not None:
        write_manifest(manifest, cards, seed)
        print(f"Wrote {manifest}", file=sys.stderr)
    if splitcheck_export is not None:
        write_split_audit_exports(splitcheck_export, seed)
        print(f"Wrote synthetic CSV fixtures to {splitcheck_export}", file=sys.stderr)
    if multi_seed_summary:
        runs = tuple((summary_seed, run_shelf(summary_seed)) for summary_seed in SUMMARY_SEEDS)
        sys.stdout.write("\n" + render_multiseed_summary(runs))
    return 0


def _serve(seed: int, port: int) -> int:
    directory = Path(tempfile.mkdtemp(prefix="leakage-lab-"))
    html_path = directory / "index.html"
    cards = run_shelf(seed)
    write_html(cards, html_path, seed)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Leakage Lab Shelf at http://127.0.0.1:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    if not argv:
        argv = ["shelf"]
    args = parser.parse_args(argv)
    if args.command in (None, "shelf"):
        return _shelf(
            args.seed,
            args.output,
            args.html,
            args.manifest,
            args.multi_seed_summary,
            args.splitcheck_export,
        )
    if args.command == "serve":
        return _serve(args.seed, args.port)
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
