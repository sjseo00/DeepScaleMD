"""Progress lines on stderr, so a long conversion shows which step it is in.

    [   3.2s] reading PSF apoa1.psf
"""
import sys
import time

_START = time.monotonic()


def stage(message: str) -> None:
    print(f"[{time.monotonic() - _START:6.1f}s] {message}", file=sys.stderr, flush=True)


class Meter:
    """stage() line at every 10 % of a loop over `total` items: call tick(i) with the 0-based item index."""

    def __init__(self, message: str, total: int):
        self.message, self.total, self.next_step = message, max(total, 1), 1

    def tick(self, i: int) -> None:
        if (i + 1) * 10 >= self.next_step * self.total and self.next_step <= 10:
            stage(f"{self.message} {self.next_step * 10}%")
            self.next_step += 1
