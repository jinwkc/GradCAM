"""터미널 및 리디렉션 출력용 배치 진행 표시 도우미."""

from __future__ import annotations

import sys
import time


def report_progress(
    label: str,
    batch_index: int,
    total_batches: int,
    started_at: float,
    detail: str = "",
    log_interval: int = 50,
) -> None:
    """TTY에서는 한 줄을 갱신하고, 파이프 출력에서는 일정 간격으로 진행을 기록합니다."""
    if total_batches < 1:
        raise ValueError("total_batches must be positive")
    if log_interval < 1:
        raise ValueError("log_interval must be positive")
    interactive = sys.stdout.isatty()
    if not (
        interactive
        or batch_index % log_interval == 0
        or batch_index == total_batches
    ):
        return

    elapsed = time.monotonic() - started_at
    eta = elapsed / batch_index * (total_batches - batch_index)
    width = 20
    filled = int(width * batch_index / total_batches)
    bar = "=" * filled + "." * (width - filled)
    message = (
        f"{label} {batch_index}/{total_batches} [{bar}] "
        f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
    )
    if detail:
        message = f"{message} {detail}"

    if interactive:
        end = "\n" if batch_index == total_batches else ""
        sys.stdout.write(f"\r{message}{end}")
        sys.stdout.flush()
    else:
        print(message, flush=True)
