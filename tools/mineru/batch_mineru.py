#!/usr/bin/env python3
"""Batch wrapper for local MinerU document parsing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".docx",
    ".pptx",
    ".xlsx",
}


@dataclass
class ParseResult:
    source: str
    status: str
    backend: str
    method: str
    output_dir: str
    obsidian_dir: str | None
    markdown_files: list[str]
    json_files: list[str]
    started_at: str
    finished_at: str
    duration_seconds: float
    return_code: int
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_slug(path: Path) -> str:
    stem = re.sub(r"[^\w.-]+", "_", path.stem, flags=re.UNICODE).strip("._-")
    stem = stem or "document"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{digest}"


def expand_inputs(values: Iterable[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw in values:
        matches = sorted(Path().glob(raw)) if any(ch in raw for ch in "*?[]") else [Path(raw)]
        for match in matches:
            if match.is_dir():
                iterator = match.rglob("*") if recursive else match.iterdir()
                files.extend(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
            elif match.is_file() and match.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(match)
    return sorted({p.resolve() for p in files})


def collect_outputs(output_dir: Path) -> tuple[list[str], list[str]]:
    markdown = sorted(str(p) for p in output_dir.rglob("*.md"))
    json_files = sorted(str(p) for p in output_dir.rglob("*.json"))
    return markdown, json_files


def write_obsidian_index(
    source: Path,
    result: ParseResult,
    job_dir: Path,
    obsidian_root: Path,
    overwrite: bool,
) -> Path:
    note_dir = obsidian_root / stable_slug(source)
    raw_dir = note_dir / "mineru_raw"
    if raw_dir.exists() and overwrite:
        shutil.rmtree(raw_dir)
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    if not raw_dir.exists():
        shutil.copytree(job_dir, raw_dir)

    note_path = note_dir / f"{stable_slug(source)}.md"
    md_links = []
    for md in sorted(raw_dir.rglob("*.md")):
        rel = md.relative_to(note_dir).as_posix()
        md_links.append(f"- [[{rel}]]")

    json_links = []
    for json_file in sorted(raw_dir.rglob("*.json")):
        rel = json_file.relative_to(note_dir).as_posix()
        json_links.append(f"- `{rel}`")

    note = [
        "---",
        f'title: "{source.stem}"',
        f'source_path: "{source}"',
        "parser: mineru",
        f"backend: {result.backend}",
        f"method: {result.method}",
        f"parsed_at: {result.finished_at}",
        "---",
        "",
        f"# {source.stem}",
        "",
        "## Markdown",
        *(md_links or ["- No Markdown files found."]),
        "",
        "## JSON",
        *(json_links or ["- No JSON files found."]),
        "",
    ]
    if overwrite or not note_path.exists():
        note_path.write_text("\n".join(note), encoding="utf-8")
    return note_dir


def build_command(args: argparse.Namespace, source: Path, output_dir: Path) -> list[str]:
    cmd = [
        args.mineru_bin,
        "-p",
        str(source),
        "-o",
        str(output_dir),
        "-b",
        args.backend,
        "-m",
        args.method,
        "-f",
        str(args.formula).lower(),
        "-t",
        str(args.table).lower(),
    ]
    if args.backend == "pipeline" and args.lang:
        cmd.extend(["-l", args.lang])
    if args.effort and args.backend.startswith("hybrid"):
        cmd.extend(["--effort", args.effort])
    if args.start is not None:
        cmd.extend(["-s", str(args.start)])
    if args.end is not None:
        cmd.extend(["-e", str(args.end)])
    if args.api_url:
        cmd.extend(["--api-url", args.api_url])
    if args.url:
        cmd.extend(["-u", args.url])
    for extra in args.extra_arg:
        cmd.append(extra)
    return cmd


def parse_one(args: argparse.Namespace, source: Path) -> ParseResult:
    started_at = utc_now()
    started = time.monotonic()
    job_dir = args.output_root / stable_slug(source)
    done_file = job_dir / ".mineru_batch_done.json"

    if args.dry_run:
        cmd = build_command(args, source, job_dir)
        print(" ".join(cmd))
        return ParseResult(
            source=str(source),
            status="dry-run",
            backend=args.backend,
            method=args.method,
            output_dir=str(job_dir),
            obsidian_dir=None,
            markdown_files=[],
            json_files=[],
            started_at=started_at,
            finished_at=utc_now(),
            duration_seconds=0.0,
            return_code=0,
        )

    if done_file.exists() and not args.overwrite:
        markdown, json_files = collect_outputs(job_dir)
        return ParseResult(
            source=str(source),
            status="skipped",
            backend=args.backend,
            method=args.method,
            output_dir=str(job_dir),
            obsidian_dir=None,
            markdown_files=markdown,
            json_files=json_files,
            started_at=started_at,
            finished_at=utc_now(),
            duration_seconds=0.0,
            return_code=0,
        )

    if job_dir.exists() and args.overwrite:
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, source, job_dir)
    env = os.environ.copy()
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.model_source:
        env["MINERU_MODEL_SOURCE"] = args.model_source

    completed = subprocess.run(cmd, cwd=args.workdir, env=env, text=True)
    markdown, json_files = collect_outputs(job_dir)
    finished_at = utc_now()
    result = ParseResult(
        source=str(source),
        status="ok" if completed.returncode == 0 else "failed",
        backend=args.backend,
        method=args.method,
        output_dir=str(job_dir),
        obsidian_dir=None,
        markdown_files=markdown,
        json_files=json_files,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(time.monotonic() - started, 3),
        return_code=completed.returncode,
        error=None if completed.returncode == 0 else "mineru command failed",
    )

    if completed.returncode == 0:
        done_file.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        if args.obsidian_dir:
            note_dir = write_obsidian_index(source, result, job_dir, args.obsidian_dir, args.overwrite)
            result.obsidian_dir = str(note_dir)
            done_file.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def append_manifests(output_root: Path, results: list[ParseResult]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "manifest.jsonl"
    csv_path = output_root / "manifest.csv"

    with jsonl_path.open("a", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
        if write_header:
            writer.writeheader()
        for result in results:
            row = asdict(result)
            row["markdown_files"] = "|".join(row["markdown_files"])
            row["json_files"] = "|".join(row["json_files"])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch parse documents with MinerU.")
    parser.add_argument("inputs", nargs="+", help="Files, directories, or shell globs.")
    parser.add_argument("--output-root", type=Path, default=Path("archive/results/mineru"))
    parser.add_argument("--obsidian-dir", type=Path, default=None)
    parser.add_argument("--mineru-bin", default=os.environ.get("MINERU_BIN", "mineru"))
    parser.add_argument("--backend", default="pipeline", choices=["pipeline", "vlm-engine", "hybrid-engine", "vlm-http-client", "hybrid-http-client"])
    parser.add_argument("--method", default="auto", choices=["auto", "txt", "ocr"])
    parser.add_argument("--lang", default="cyrillic", choices=["ch", "ch_server", "korean", "ta", "te", "ka", "th", "el", "arabic", "east_slavic", "cyrillic", "devanagari", ""])
    parser.add_argument("--effort", default="medium", choices=["medium", "high"])
    parser.add_argument("--model-source", default=os.environ.get("MINERU_MODEL_SOURCE"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu"])
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--formula", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--workdir", default=str(Path.cwd()))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.obsidian_dir = args.obsidian_dir.resolve() if args.obsidian_dir else None

    sources = expand_inputs(args.inputs, args.recursive)
    if not sources:
        print("No supported input files found.", file=sys.stderr)
        return 2

    results: list[ParseResult] = []
    for source in sources:
        print(f"[mineru-batch] parsing {source}", flush=True)
        result = parse_one(args, source)
        results.append(result)
        print(f"[mineru-batch] {result.status}: {source.name} -> {result.output_dir}", flush=True)

    append_manifests(args.output_root, results)
    failed = [r for r in results if r.return_code != 0]
    print(f"[mineru-batch] processed={len(results)} failed={len(failed)} manifest={args.output_root / 'manifest.csv'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
