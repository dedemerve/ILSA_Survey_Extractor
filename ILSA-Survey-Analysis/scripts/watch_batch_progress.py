#!/usr/bin/env python3
"""
Live dashboard for ``run_batch_folders.py`` batch extraction.

Run in a second terminal while the batch is running (does not start or stop the batch)::

    python scripts/watch_batch_progress.py

Options::

    python scripts/watch_batch_progress.py --once     # print once and exit
    python scripts/watch_batch_progress.py --interval 3
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_LOG = PROJECT_ROOT / "outputs" / "batch_folders_extraction.log"
DEFAULT_DETAIL_LOG = PROJECT_ROOT / "outputs" / "logs" / "batch_folders.log"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"

# Batch log lines use second precision (no milliseconds).
BATCH_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| "
    r"(START|PROCESS|OK|FAIL|SKIP|PROGRESS|FINISH|LIMIT|DONE) \| (.+)$"
)
START_KV_RE = re.compile(r"(\w+)=([^\s|]+)")
PROCESS_RE = re.compile(
    r"^\[([^\]]+)\] (.+?) \((\d+)/(\d+)\)$"
)
OK_RE = re.compile(
    r"^\[([^\]]+)\] (.+?) \| cost=\$([0-9.]+) \| cumulative=\$([0-9.]+)"
)
RETRY_OK_RE = re.compile(r"^\[([^\]]+)\] (.+?) \| cost=\$([0-9.]+)$")
FAIL_RE = re.compile(r"^\[([^\]]+)\] (.+?) \| error=")
PROGRESS_RE = re.compile(
    r"^(\d+)/(\d+) \((\d+)%\) \| ok=(\d+) fail=(\d+) skip=(\d+) \| \$([0-9.]+) \| last=(.+)$"
)
TQDM_RE = re.compile(
    r"Extracting:.*?\|\s*(\d+)/(\d+).*?ok=(\d+),\s*fail=(\d+),\s*skip=(\d+),\s*cost_\$=([0-9.]+)"
)
RETRY_TQDM_RE = re.compile(
    r"Retrying:.*?\|\s*(\d+)/(\d+).*?ok=(\d+),\s*fail=(\d+)"
)
SUBDIRS_RE = re.compile(r"Subdirs: \[(.+)\]")


@dataclass
class BatchSnapshot:
    mode: str = "—"
    queue: int = 0
    total_pdfs: int = 0
    skipped_at_start: int = 0
    workers: int = 0
    ok: int = 0
    fail: int = 0
    skip: int = 0
    done: int = 0  # ok + fail since last START
    pct: float = 0.0
    last_process: list[str] = field(default_factory=list)
    last_ok_line: str = ""
    last_ok_ts: datetime | None = None
    files_per_min: float | None = None
    eta_min: float | None = None
    total_cost: float | None = None
    json_on_disk: int = 0
    json_expected: int = 0
    batch_running: bool = False
    finished: bool = False
    start_ts: datetime | None = None
    last_progress_line: str = ""


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def parse_start_fields(body: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in START_KV_RE.finditer(body)}


def top_folder_from_label(label: str) -> str:
    return label.split("/", 1)[0] if label else "—"


def read_log_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def count_json_files(outputs_root: Path, top_folder: str | None) -> int:
    if top_folder:
        base = outputs_root / top_folder
        if not base.is_dir():
            return 0
        return sum(1 for _ in base.rglob("*.json"))
    return sum(1 for _ in outputs_root.rglob("*.json"))


def is_batch_process_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "run_batch_folders|retry_failed_pdfs"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    for line in out.stdout.splitlines():
        if "watch_batch_progress" in line:
            continue
        if "run_batch_folders" in line or "retry_failed_pdfs" in line:
            return True
    return False


def dedupe_key(status: str, body: str) -> str:
    if status == "OK":
        m = OK_RE.match(body) or RETRY_OK_RE.match(body)
        if m:
            return f"OK|{m.group(1)}|{m.group(2)}"
    if status == "FAIL":
        m = FAIL_RE.match(body)
        if m:
            return f"FAIL|{m.group(1)}|{m.group(2)}"
    if status == "PROCESS":
        m = PROCESS_RE.match(body)
        if m:
            return f"PROCESS|{m.group(1)}|{m.group(2)}|{m.group(3)}"
    return f"{status}|{body}"


def files_per_minute(ok_times: list[datetime]) -> float | None:
    if len(ok_times) < 2:
        return None
    span = (ok_times[-1] - ok_times[0]).total_seconds()
    if span <= 0:
        return None
    return (len(ok_times) - 1) / (span / 60.0)


def parse_logs(
    extraction_log: Path,
    detail_log: Path | None,
    outputs_root: Path,
) -> BatchSnapshot:
    snap = BatchSnapshot()
    text = read_log_text(extraction_log)
    if not text:
        snap.batch_running = is_batch_process_running()
        return snap

    lines = text.splitlines()
    last_start_idx = -1
    start_fields: dict[str, str] = {}
    subdirs_from_detail: list[str] = []

    if detail_log and detail_log.is_file():
        for dline in read_log_text(detail_log).splitlines():
            m = SUBDIRS_RE.search(dline)
            if m:
                raw = m.group(1).strip()
                subdirs_from_detail = [
                    s.strip().strip("'\"")
                    for s in raw.split(",")
                    if s.strip()
                ]

    for i, line in enumerate(lines):
        m = BATCH_LINE_RE.match(line)
        if not m:
            continue
        ts_s, status, body = m.group(1), m.group(2), m.group(3)
        if status == "START":
            last_start_idx = i
            start_fields = parse_start_fields(body)
            snap.start_ts = parse_ts(ts_s)

    if last_start_idx < 0:
        snap.batch_running = is_batch_process_running()
        return snap

    snap.queue = int(
        start_fields.get("queue") or start_fields.get("retry_queue") or "0"
    )
    snap.total_pdfs = int(start_fields.get("total_pdfs", "0") or 0)
    snap.skipped_at_start = int(
        start_fields.get("skipped_existing")
        or start_fields.get("skipped")
        or "0"
    )
    snap.workers = int(start_fields.get("workers", "0") or 0)
    snap.skip = snap.skipped_at_start
    snap.json_expected = snap.total_pdfs or snap.queue

    seen: set[str] = set()
    ok_times: list[datetime] = []
    last_labels: list[str] = []
    top_folder: str | None = None

    for line in lines[last_start_idx:]:
        m = BATCH_LINE_RE.match(line)
        if m:
            ts_s, status, body = m.group(1), m.group(2), m.group(3)
            key = dedupe_key(status, body)
            if key in seen and status in ("OK", "FAIL", "PROCESS"):
                continue
            seen.add(key)

            if status == "OK":
                om = OK_RE.match(body) or RETRY_OK_RE.match(body)
                if om:
                    snap.ok += 1
                    snap.last_ok_line = f"{ts_s} | OK | [{om.group(1)}] {om.group(2)}"
                    snap.last_ok_ts = parse_ts(ts_s)
                    ok_times.append(snap.last_ok_ts)
                    top_folder = top_folder or top_folder_from_label(om.group(1))
                    try:
                        if om.lastindex and om.lastindex >= 4:
                            snap.total_cost = float(om.group(4))
                        elif om.lastindex and om.lastindex >= 3:
                            if snap.total_cost is None:
                                snap.total_cost = 0.0
                            snap.total_cost += float(om.group(3))
                    except (ValueError, TypeError):
                        pass
            elif status == "FAIL":
                if FAIL_RE.match(body):
                    snap.fail += 1
                    fm = FAIL_RE.match(body)
                    if fm:
                        top_folder = top_folder or top_folder_from_label(fm.group(1))
            elif status == "SKIP":
                snap.skip += 1
            elif status == "PROCESS":
                pm = PROCESS_RE.match(body)
                if pm:
                    label = pm.group(1)
                    fname = pm.group(2)
                    top_folder = top_folder or top_folder_from_label(label)
                    entry = f"[{label}] {fname} ({pm.group(3)}/{pm.group(4)})"
                    last_labels.append(entry)
            elif status == "PROGRESS":
                pr = PROGRESS_RE.match(body)
                if pr:
                    snap.last_progress_line = line
                    snap.ok = int(pr.group(4))
                    snap.fail = int(pr.group(5))
                    snap.skip = int(pr.group(6))
                    try:
                        snap.total_cost = float(pr.group(7))
                    except ValueError:
                        pass
            elif status == "FINISH":
                snap.finished = True
                for part in body.split("|"):
                    part = part.strip()
                    if part.startswith("extracted_ok=") or part.startswith("ok="):
                        snap.ok = int(part.split("=", 1)[1])
                    elif part.startswith("failed=") or part.startswith("fail="):
                        snap.fail = int(part.split("=", 1)[1])
                    elif part.startswith("skipped_at_start=") or part.startswith("skipped="):
                        snap.skip = int(part.split("=", 1)[1])
                    elif part.startswith("total_cost=$") or part.startswith("cost=$"):
                        try:
                            snap.total_cost = float(part.split("$", 1)[1])
                        except ValueError:
                            pass
            continue

        tm = TQDM_RE.search(line) or RETRY_TQDM_RE.search(line)
        if tm:
            done = int(tm.group(1))
            snap.done = max(snap.done, done)
            snap.ok = int(tm.group(3))
            snap.fail = int(tm.group(4))
            if tm.lastindex and tm.lastindex >= 5:
                snap.skip = int(tm.group(5))
            if tm.lastindex and tm.lastindex >= 6:
                try:
                    snap.total_cost = float(tm.group(6))
                except ValueError:
                    pass

    if subdirs_from_detail:
        snap.mode = ", ".join(subdirs_from_detail)
    elif top_folder:
        snap.mode = top_folder
    else:
        snap.mode = "—"

    snap.last_process = last_labels[-3:]
    snap.done = snap.ok + snap.fail
    if snap.queue:
        snap.pct = 100.0 * snap.ok / snap.queue
    snap.files_per_min = files_per_minute(ok_times[-20:])
    if snap.finished:
        snap.eta_min = None
    elif snap.files_per_min and snap.queue > snap.ok:
        remaining = snap.queue - snap.ok
        snap.eta_min = remaining / snap.files_per_min

    if top_folder:
        snap.json_on_disk = count_json_files(outputs_root, top_folder)
    else:
        snap.json_on_disk = count_json_files(outputs_root, None)

    snap.batch_running = is_batch_process_running()
    return snap


def render_dashboard(
    snap: BatchSnapshot,
    extraction_log: Path,
    detail_log: Path | None,
) -> str:
    lines: list[str] = []
    w = 72
    lines.append("=" * w)
    lines.append("  ILSA batch extraction — live progress")
    lines.append("=" * w)
    lines.append(f"  Mode (top-folder):  {snap.mode}")
    lines.append(f"  Workers:            {snap.workers or '—'}")
    lines.append(
        f"  Progress (OK/queue): {snap.ok}/{snap.queue or '—'} "
        f"({snap.pct:.1f}%)" if snap.queue else f"  Progress (OK/queue): {snap.ok}/—"
    )
    lines.append(
        f"  Completed (ok+fail): {snap.done}/{snap.queue or '—'} "
        f"| OK={snap.ok}  FAIL={snap.fail}  SKIP={snap.skip}"
    )
    lines.append(f"  Total PDFs (run):   {snap.total_pdfs or '—'}")
    lines.append("")
    lines.append("  Active (last PROCESS):")
    if snap.last_process:
        for p in snap.last_process:
            lines.append(f"    • {p}")
    else:
        lines.append("    (none yet)")
    lines.append("")
    if snap.last_ok_line:
        lines.append(f"  Last OK:  {snap.last_ok_line[: w - 2]}")
    else:
        lines.append("  Last OK:  —")
    if snap.finished:
        lines.append("  Status:   COMPLETED (FINISH in log)")
        lines.append("  Rate:     —")
    elif snap.files_per_min is not None:
        rate = f"{snap.files_per_min:.2f} OK/min"
        if snap.eta_min is not None:
            rate += f"  |  ETA ~{snap.eta_min:.0f} min (by OK rate)"
        lines.append(f"  Rate:     {rate}")
    else:
        lines.append("  Rate:     — (need ≥2 OK lines)")
    cost_s = f"${snap.total_cost:.2f}" if snap.total_cost is not None else "—"
    lines.append(f"  Cost:     {cost_s}")
    lines.append("")
    lines.append(
        f"  JSON on disk:       {snap.json_on_disk} / {snap.json_expected} expected"
    )
    run_s = "YES — batch/retry job running" if snap.batch_running else "NO"
    if snap.finished and not snap.batch_running:
        run_s = "NO — FINISH seen in log"
    lines.append(f"  Batch process:      {run_s}")
    if snap.last_progress_line:
        lines.append(f"  Last PROGRESS:      {snap.last_progress_line[: w - 2]}")
    lines.append("")
    lines.append(f"  Log:  {extraction_log}")
    if detail_log and detail_log.is_file():
        lines.append(f"        {detail_log}")
    lines.append("=" * w)
    lines.append("  Ctrl+C to exit  |  refreshes every few seconds")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_EXTRACTION_LOG,
        help=(
            "Main batch log (default: outputs/batch_folders_extraction.log; "
            "retry: outputs/retry_failed_pdfs.log)"
        ),
    )
    parser.add_argument(
        "--detail-log",
        type=Path,
        default=DEFAULT_DETAIL_LOG,
        help="Optional detail log (default: outputs/logs/batch_folders.log)",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=DEFAULT_OUTPUTS,
        help="Outputs root for JSON counts",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.5,
        help="Refresh interval in seconds (default: 2.5)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print dashboard once and exit (for testing)",
    )
    args = parser.parse_args()

    detail = args.detail_log if args.detail_log.is_file() else None

    while True:
        snap = parse_logs(args.log, detail, args.outputs.expanduser().resolve())
        clear_screen()
        print(render_dashboard(snap, args.log, detail))
        sys.stdout.flush()
        if args.once:
            break
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
