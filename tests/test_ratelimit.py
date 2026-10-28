from types import SimpleNamespace

from app.guard.answer_cache import key
from app.guard.ratelimit import RateLimiter, client_ip


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_allows_up_to_limit_then_blocks():
    lim = RateLimiter(limit=3, window=3600, clock=Clock())
    assert [lim.allow("a")[0] for _ in range(4)] == [True, True, True, False]


def test_limit_is_per_key():
    lim = RateLimiter(limit=1, window=3600, clock=Clock())
    assert lim.allow("a")[0] and lim.allow("b")[0] and not lim.allow("a")[0]


def test_window_slides():
    clock = Clock()
    lim = RateLimiter(limit=2, window=3600, clock=clock)
    lim.allow("a")                 # t=1000
    clock.t = 2000
    lim.allow("a")                 # t=2000
    ok, wait = lim.allow("a")
    assert not ok and wait == 2601  # the first request expires at 4600, we are at 2000 (+1 s rounding up)
    clock.t = 4600                  # first request leaves the window, one slot frees up
    assert lim.allow("a")[0] and not lim.allow("a")[0]


def test_blocked_requests_do_not_extend_the_wait():
    clock = Clock()
    lim = RateLimiter(limit=1, window=100, clock=clock)
    lim.allow("a")
    for _ in range(5):
        lim.allow("a")             # rejected: must not be recorded
    clock.t += 100
    assert lim.allow("a")[0]


def test_remaining():
    lim = RateLimiter(limit=3, window=3600, clock=Clock())
    lim.allow("a")
    assert lim.remaining("a") == 2 and lim.remaining("new") == 3


def test_client_ip_prefers_first_forwarded_address():
    req = SimpleNamespace(headers={"x-forwarded-for": "203.0.113.7, 10.0.0.1"}, client=SimpleNamespace(host="10.0.0.2"))
    assert client_ip(req) == "203.0.113.7"
    assert client_ip(SimpleNamespace(headers={}, client=SimpleNamespace(host="10.0.0.2"))) == "10.0.0.2"


def test_cache_key_ignores_case_and_spacing():
    assert key("Who directed  Sinister?") == key("  who directed sinister? ")
    assert key("Who directed Sinister?") != key("Who directed Ed Wood?")
