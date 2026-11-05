"""Small terminal timers for CPU work and synchronized CUDA stages."""

from contextlib import ContextDecorator, contextmanager, nullcontext
from contextvars import ContextVar
import os
import sys
import time


_phase = ContextVar("component_timing_phase", default=None)
_collector = ContextVar("component_timing_collector", default=None)
_listener = ContextVar("timing_listener", default=None)


def print_timing(part, message):
    """Use three terminal colors, with plain text for redirected output or NO_COLOR."""
    colors = {"WARMUP": "33", "TRACKING": "32", "INIT": "36", "REINIT": "36"}
    line = f"[{part}] {message}"
    if part in colors and sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb":
        line = f"\033[{colors[part]}m{line}\033[0m"
    print(line, flush=True)
    listener = _listener.get()
    if listener is not None:
        listener(part, message)


@contextmanager
def timing_listener(listener):
    """Forward the same summaries to an optional UI on the inference thread."""
    token = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(token)


class TimingCollector:
    """Collect component samples in this scope instead of printing each call."""

    def __init__(self, profile_pose=False):
        self.samples = {}
        self.profile_pose = profile_pose

    def __enter__(self):
        self._token = _collector.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _collector.reset(self._token)

    def record(self, name, elapsed_ms):
        total, count = self.samples.get(name, (0.0, 0))
        self.samples[name] = (total + elapsed_ms, count + 1)

    def format(self, name):
        if name not in self.samples:
            return f"{name}=skipped"
        total, count = self.samples[name]
        return f"{name}={total:.2f} ms"

    def report_warmup(self, total_ms):
        for name in ("GroundingDINO", "SAM2", "DINOv2"):
            if name in self.samples:
                total, count = self.samples[name]
                print_timing("WARMUP", f"{name}: total={total:.2f} ms | avg={total / count:.2f} ms | calls={count}")
            else:
                print_timing("WARMUP", f"{name}=skipped (no candidates)")
        print_timing("WARMUP", f"Detection total={total_ms:.2f} ms")


class TrackingTiming:
    """Report a mean over each wall-clock interval and flush partial windows on loss/exit."""

    def __init__(self, interval=5.0, profile=False):
        self.interval = interval
        self.profile = profile
        self.count = 0
        self.total_ms = 0.0
        self.started = None
        self.components = TimingCollector()

    def add(self, elapsed_ms, now=None, components=None):
        now = time.monotonic() if now is None else now
        if self.started is None:
            self.started = now
        self.count += 1
        self.total_ms += elapsed_ms
        if components is not None:
            for name, (total, calls) in components.samples.items():
                previous_total, previous_calls = self.components.samples.get(name, (0.0, 0))
                self.components.samples[name] = (previous_total + total, previous_calls + calls)
        if now - self.started >= self.interval:
            self.flush()

    def flush(self):
        if self.count:
            print_timing("TRACKING", f"WAPR avg={self.total_ms / self.count:.2f} ms | frames={self.count}")
            if self.profile and self.components.samples:
                stages = []
                measured = 0.0
                for name in ("WAPR", "SAPR", "WBPS"):
                    total = self.components.samples.get(name, (0.0, 0))[0]
                    measured += total
                    crop = self.components.samples.get(name + " crop", (0.0, 0))[0]
                    network = self.components.samples.get(name + " network", (0.0, 0))[0]
                    stages.append(f"{name}={total / self.count:.2f} ms "
                                  f"(crop={crop / self.count:.2f}, network={network / self.count:.2f})")
                other = max(0.0, self.total_ms - measured) / self.count
                print_timing("TRACKING", "stages avg/frame: " + " | ".join(stages)
                             + f" | other={other:.2f} ms | synchronized profiling")
        self.count = 0
        self.total_ms = 0.0
        self.started = None
        self.components.samples.clear()


def pose_component_timer(name, device):
    """Reuse ComponentTimer only inside an explicitly enabled pose collector.

    Detector collectors stay unaffected. Nested crop/network times are parts of
    the WAPR/SAPR/WBPS totals, not additional elapsed time.
    """
    collector = _collector.get()
    if collector is None or not collector.profile_pose:
        return nullcontext()
    return ComponentTimer(name, device)


@contextmanager
def timing_phase(name):
    """Label warmup calls without changing the component interfaces."""
    token = _phase.set(name)
    try:
        yield
    finally:
        _phase.reset(token)


class ComponentTimer(ContextDecorator):
    """Print elapsed milliseconds; an explicit CUDA device enables synchronization.

    Use as ``with ComponentTimer("SAM2", device):`` or as a decorator.
    A name of None measures silently. Torch is imported only for a CUDA timer.
    """

    def __init__(self, name, device=None):
        self.name = name
        self.device = device
        self.elapsed_ms = None

    def _recreate_cm(self):
        return type(self)(self.name, self.device)

    def _synchronize(self):
        if self.device is not None:
            import torch
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        self.elapsed_ms = None
        self._synchronize()
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False
        self._synchronize()
        self.elapsed_ms = (time.perf_counter() - self._started) * 1000
        collector = _collector.get()
        if self.name is not None and collector is not None:
            collector.record(self.name, self.elapsed_ms)
        elif self.name is not None:
            phase = _phase.get()
            prefix = "[TIMING]" if phase is None else f"[TIMING][{phase}]"
            print(f"{prefix} {self.name}: {self.elapsed_ms:.2f} ms", flush=True)
        return False
