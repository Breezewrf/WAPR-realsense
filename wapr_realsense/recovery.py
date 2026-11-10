"""Settings for bounded recovery after an established track becomes unreliable."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RecoverySettings:
    activate: bool = False
    timeout_s: float = 1.0
    confirm_frames: int = 2

    def __post_init__(self):
        if type(self.activate) is not bool:
            raise ValueError("Recovery activate must be a boolean")
        if isinstance(self.timeout_s, bool) or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Recovery timeout_s must be finite and positive")
        if type(self.confirm_frames) is not int or self.confirm_frames < 1:
            raise ValueError("Recovery confirm_frames must be a positive integer")


def recovery_from_args(args):
    return RecoverySettings(args.recovery_activate, args.recovery_timeout_s, args.recovery_confirm_frames)
