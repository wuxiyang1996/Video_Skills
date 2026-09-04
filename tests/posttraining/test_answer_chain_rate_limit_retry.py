"""Rate-limited rollouts are retried, everything else is not."""

import pytest

from scripts.eval.measure_answer_chain import _with_rate_limit_retry


class _Flaky:
    def __init__(self, fail_times, message):
        self.n, self.fail_times, self.message = 0, fail_times, message
    def __call__(self):
        self.n += 1
        if self.n <= self.fail_times:
            raise RuntimeError(self.message)
        return "ok"


def test_retries_on_429_then_succeeds() -> None:
    slept = []
    call = _Flaky(2, "429 error from https://openrouter.ai: rate limited")
    assert _with_rate_limit_retry(call, sleep=slept.append) == "ok"
    assert call.n == 3
    assert slept == [10.0, 20.0]          # exponential backoff


def test_gives_up_after_the_last_attempt() -> None:
    call = _Flaky(10, "HTTPError: 429")
    with pytest.raises(RuntimeError):
        _with_rate_limit_retry(call, attempts=3, sleep=lambda s: None)
    assert call.n == 3


def test_5xx_is_retried() -> None:
    slept = []
    call = _Flaky(1, "HTTPError: 502 error from https://openrouter.ai")
    assert _with_rate_limit_retry(call, sleep=slept.append) == "ok"
    assert call.n == 2 and slept == [10.0]


def test_transport_valueerror_is_retried() -> None:
    class _Bad:
        def __init__(self): self.n = 0
        def __call__(self):
            self.n += 1
            if self.n == 1:
                raise ValueError("Expecting value: line 1 column 1 (char 0)")   # requests .json() on an HTML body
            return "ok"
    call = _Bad()
    assert _with_rate_limit_retry(call, sleep=lambda s: None) == "ok"
    assert call.n == 2


def test_non_transient_errors_are_not_retried() -> None:
    slept = []
    call = _Flaky(1, "KeyError: 'choices'")
    with pytest.raises(RuntimeError):
        _with_rate_limit_retry(call, sleep=slept.append)
    assert call.n == 1 and slept == []


def test_success_needs_no_sleep() -> None:
    slept = []
    assert _with_rate_limit_retry(lambda: "ok", sleep=slept.append) == "ok"
    assert slept == []
