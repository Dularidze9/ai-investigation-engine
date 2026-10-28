"""Per-IP rate limit for the public demo: at most LIMIT questions per WINDOW seconds (sliding window).

Kept in memory on purpose: one Render instance, one worker, so one dict is enough. It resets on a
restart, which is accepted; the hard money guarantee is the spending cap in the database (budget.py).
"""
import threading
import time
from collections import defaultdict, deque

LIMIT = 20
WINDOW = 3600.0


class RateLimiter:
    def __init__(self, limit: int = LIMIT, window: float = WINDOW, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool

    def allow(self, key: str) -> tuple[bool, int]:
        """Record one request for key if allowed. Returns (allowed, seconds until the next free slot)."""
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window:  # forget requests older than the window
                hits.popleft()
            if len(hits) >= self.limit:
                return False, int(hits[0] + self.window - now) + 1
            hits.append(now)
            return True, 0

    def remaining(self, key: str) -> int:
        now = self.clock()
        with self._lock:
            return self.limit - sum(1 for t in self._hits.get(key, ()) if t > now - self.window)


def client_ip(request) -> str:
    """Render's proxy puts the caller's address first in X-Forwarded-For. A client can forge this
    header to get a fresh bucket; that only bypasses the per-IP limit, never the global $ cap."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
