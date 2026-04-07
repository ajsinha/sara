# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.7.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""
sara.core.progress
==================
Lightweight progress logging for long-running experiments.

Provides:
  SaraLogger   — structured logging with elapsed time and step tracking
  Heartbeat    — background thread that prints "still working..." if silent
  ProgressBar  — simple ASCII progress bar (no dependencies)
  phase()      — context manager that times a named phase

Usage
-----
    from sara.core.progress import SaraLogger, Heartbeat, phase

    log = SaraLogger("Ablation")
    log.section("Condition A — KD-SPAR")
    log.step("Harvesting teacher responses", total=40)
    for i, q in enumerate(queries):
        result = teacher.query(q)
        log.tick(i + 1)          # prints dot every 5, summary every 10
    log.done("Harvested 40 responses")

    with phase("SPAR iteration 1/3"):
        # ... long computation ...
        pass   # prints elapsed on exit
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Optional


# ── Colours (graceful fallback if terminal doesn't support ANSI) ──────────────
_HAVE_COLOUR = sys.stdout.isatty()
_G  = "\033[0;32m"  if _HAVE_COLOUR else ""  # green
_Y  = "\033[1;33m"  if _HAVE_COLOUR else ""  # yellow
_B  = "\033[0;34m"  if _HAVE_COLOUR else ""  # blue/cyan
_R  = "\033[0;31m"  if _HAVE_COLOUR else ""  # red
_DIM= "\033[2m"     if _HAVE_COLOUR else ""  # dim
_NC = "\033[0m"     if _HAVE_COLOUR else ""  # reset


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _now() -> float:
    return time.monotonic()


# ── Heartbeat thread ──────────────────────────────────────────────────────────

class Heartbeat:
    """
    Background thread that prints a "still working..." message if the main
    thread has been silent for more than `interval` seconds.

    Prevents the terminal from looking frozen during long Ollama calls.

    Usage
    -----
        hb = Heartbeat(interval=20, message="Waiting for Ollama response…")
        hb.start()
        # ... do slow work ...
        hb.stop()

    Or as a context manager:
        with Heartbeat(30):
            do_slow_thing()
    """

    def __init__(self, interval: float = 20.0, message: str = "Still working…"):
        self.interval = interval
        self.message  = message
        self._stop    = threading.Event()
        self._last    = _now()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def ping(self) -> None:
        """Call this whenever the main thread prints something to reset the timer."""
        self._last = _now()

    def start(self) -> "Heartbeat":
        self._stop.clear()
        self._last = _now()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(timeout=1.0):
            elapsed_since = _now() - self._last
            if elapsed_since >= self.interval:
                total = _now() - self._last
                print(f"{_DIM}  ⏳  {self.message}  ({_fmt_elapsed(elapsed_since)} elapsed){_NC}",
                      flush=True)
                self._last = _now()  # reset so it doesn't spam

    def __enter__(self) -> "Heartbeat":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ── Progress bar ──────────────────────────────────────────────────────────────

class ProgressBar:
    """
    Simple single-line ASCII progress bar, no external dependencies.

        [████████░░░░░░░░░░░░]  8/20  40%  12s elapsed
    """

    def __init__(self, total: int, width: int = 22, label: str = ""):
        self.total   = max(total, 1)
        self.width   = width
        self.label   = label
        self._start  = _now()
        self._done   = 0

    def update(self, n: int = 1) -> None:
        self._done = min(self._done + n, self.total)
        self._render()

    def set(self, n: int) -> None:
        self._done = min(n, self.total)
        self._render()

    def finish(self) -> None:
        self._done = self.total
        self._render()
        print()  # newline after the bar

    def _render(self) -> None:
        frac    = self._done / self.total
        filled  = int(self.width * frac)
        bar     = "█" * filled + "░" * (self.width - filled)
        elapsed = _fmt_elapsed(_now() - self._start)
        pct     = int(frac * 100)
        prefix  = f"  {self.label}  " if self.label else "  "
        line    = f"\r{prefix}[{bar}] {self._done}/{self.total}  {pct}%  {elapsed}"
        print(line, end="", flush=True)


# ── Phase context manager ─────────────────────────────────────────────────────

@contextmanager
def phase(name: str, heartbeat_interval: float = 25.0):
    """
    Context manager that prints a phase header with elapsed time on exit.
    Automatically runs a heartbeat thread inside the phase.

    Usage
    -----
        with phase("SPAR iteration 2/3"):
            do_stuff()
        # prints: ✓ SPAR iteration 2/3  (14s)
    """
    t0 = _now()
    print(f"\n{_B}  ┌─ {name}{_NC}", flush=True)
    hb = Heartbeat(heartbeat_interval, f"Still in: {name}")
    hb.start()
    try:
        yield hb
    finally:
        hb.stop()
        elapsed = _fmt_elapsed(_now() - t0)
        print(f"{_G}  └─ done  ({elapsed}){_NC}", flush=True)


# ── Main logger ───────────────────────────────────────────────────────────────

class SaraLogger:
    """
    Structured experiment logger with elapsed time tracking.

    Every message is timestamped relative to when the logger was created
    so you always know how long the experiment has been running.

    Usage
    -----
        log = SaraLogger("KD-SPAR Ablation")
        log.section("Condition A")
        log.step("Teacher harvest", total=40)
        for i in range(40):
            do_query()
            log.tick(i + 1)
        log.done("40 responses collected")
        log.info("KD score: 0.382")
        log.warn("Low score on query 7 — may indicate model mismatch")
        log.error("Ollama timed out")
    """

    def __init__(self, name: str = "Sara"):
        self.name   = name
        self._start = _now()
        self._step_total: Optional[int]  = None
        self._step_label: str            = ""
        self._pb:          Optional[ProgressBar] = None
        self._hb:          Optional[Heartbeat]   = None
        self._section_start = _now()

    def _elapsed(self) -> str:
        return _fmt_elapsed(_now() - self._start)

    def _sec_elapsed(self) -> str:
        return _fmt_elapsed(_now() - self._section_start)

    def _print(self, tag: str, colour: str, msg: str) -> None:
        self._ping_hb()
        ts = f"{_DIM}[{self._elapsed()}]{_NC}"
        print(f"{ts} {colour}{tag}{_NC} {msg}", flush=True)

    def _ping_hb(self) -> None:
        if self._hb:
            self._hb.ping()

    # ── Public API ────────────────────────────────────────────────────────────

    def banner(self, *lines: str) -> None:
        """Print a bordered banner — use for run-level headers."""
        width = max(len(l) for l in lines) + 4
        border = "═" * width
        print(f"\n{_G}╔{border}╗{_NC}", flush=True)
        for line in lines:
            pad = width - len(line)
            print(f"{_G}║  {line}{' '*pad}║{_NC}", flush=True)
        print(f"{_G}╚{border}╝{_NC}\n", flush=True)

    def section(self, title: str) -> None:
        """Print a section header — use for condition/phase boundaries."""
        self._section_start = _now()
        sep = "─" * 55
        print(f"\n{_B}{sep}{_NC}", flush=True)
        print(f"{_B}  {title}{_NC}", flush=True)
        print(f"{_B}{sep}{_NC}", flush=True)

    def step(self, label: str, total: Optional[int] = None) -> None:
        """
        Announce a step, optionally starting a progress bar.
        Call tick() to advance it.
        """
        self._step_label = label
        self._step_total = total
        self._ping_hb()
        if total:
            self._pb = ProgressBar(total, label=f"  {label}")
        else:
            ts = f"{_DIM}[{self._elapsed()}]{_NC}"
            print(f"{ts} {_B}  ▶{_NC}  {label}", flush=True)

    def tick(self, current: int, extra: str = "") -> None:
        """Advance progress bar to `current`. If no bar, print dots."""
        self._ping_hb()
        if self._pb:
            self._pb.set(current)
            if current >= (self._step_total or 1):
                self._pb.finish()
                self._pb = None
        else:
            # Fallback: dot every 5, newline every 50
            if current % 50 == 0:
                print(f" {current}", flush=True)
            elif current % 5 == 0:
                print("·", end="", flush=True)

    def done(self, msg: str = "") -> None:
        """Print a success line, finishing any open progress bar."""
        if self._pb:
            self._pb.finish()
            self._pb = None
        elapsed = self._sec_elapsed()
        text    = f"{msg}  ({elapsed})" if msg else f"done  ({elapsed})"
        self._print("  ✓", _G, text)

    def info(self, msg: str) -> None:
        self._print("  ·", _DIM, msg)

    def warn(self, msg: str) -> None:
        self._print("  ⚠", _Y, msg)

    def error(self, msg: str) -> None:
        self._print("  ✗", _R, msg)

    def metric(self, label: str, value: str, extra: str = "") -> None:
        """Print a key=value metric line."""
        self._ping_hb()
        ts = f"{_DIM}[{self._elapsed()}]{_NC}"
        extra_str = f"  {_DIM}{extra}{_NC}" if extra else ""
        print(f"{ts}     {_Y}{label}{_NC} = {value}{extra_str}", flush=True)

    def result(self, condition: str, kd: float, delta: float,
               cit: float, accepted: Optional[bool] = None) -> None:
        """Print a standardised result line for one ablation condition."""
        self._ping_hb()
        ts      = f"{_DIM}[{self._elapsed()}]{_NC}"
        delta_s = f"{delta:+.4f}"
        delta_c = _G if delta > 0 else (_R if delta < 0 else _DIM)
        acc_s   = ""
        if accepted is not None:
            acc_s = f"  {'✓ ACCEPTED' if accepted else '✗ REVERTED'}"
            acc_c = _G if accepted else _R
            acc_s = f"  {acc_c}{acc_s.strip()}{_NC}"
        print(f"{ts}  Cond {condition}  kd={kd:.4f}  Δ={delta_c}{delta_s}{_NC}"
              f"  cit={cit:.3f}{acc_s}", flush=True)

    def start_heartbeat(self, interval: float = 25.0,
                        message: str = "Waiting for Ollama…") -> "SaraLogger":
        """Start background heartbeat. Call stop_heartbeat() when done."""
        self._hb = Heartbeat(interval, message)
        self._hb.start()
        return self

    def stop_heartbeat(self) -> None:
        if self._hb:
            self._hb.stop()
            self._hb = None

    def summary(self, elapsed_total: Optional[float] = None) -> None:
        """Print final run summary."""
        t = elapsed_total or (_now() - self._start)
        print(f"\n{_G}{'═'*55}{_NC}", flush=True)
        print(f"{_G}  {self.name} completed  —  total time: {_fmt_elapsed(t)}{_NC}",
              flush=True)
        print(f"{_G}{'═'*55}{_NC}\n", flush=True)
