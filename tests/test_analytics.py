"""Tests for the Mixpanel analytics wrapper."""
from __future__ import annotations

import json

import pytest

from investor_intel.analytics import Analytics

pytest.importorskip("mixpanel")


class FakeConsumer:
    """Captures what mixpanel-python would POST, instead of hitting the network."""

    def __init__(self, fail: bool = False) -> None:
        self.sent: list[tuple[str, dict]] = []
        self.fail = fail

    def send(self, endpoint: str, json_message: str, api_key: str | None = None) -> None:
        if self.fail:
            raise ConnectionError("network down")
        self.sent.append((endpoint, json.loads(json_message)))


def test_disabled_without_token() -> None:
    analytics = Analytics("")
    assert analytics.enabled is False
    analytics.track("user-1", "Question Asked")  # no-op, must not raise
    analytics.drain()


def test_track_sends_event_with_properties() -> None:
    consumer = FakeConsumer()
    analytics = Analytics("test-token-123", consumer=consumer)
    assert analytics.enabled is True

    analytics.track("session-abc", "Report Uploaded", {"file_type": ".pdf", "chunks": 42})
    analytics.drain()

    assert len(consumer.sent) == 1
    endpoint, message = consumer.sent[0]
    assert endpoint == "events"
    assert message["event"] == "Report Uploaded"
    assert message["properties"]["token"] == "test-token-123"
    assert message["properties"]["distinct_id"] == "session-abc"
    assert message["properties"]["file_type"] == ".pdf"
    assert message["properties"]["chunks"] == 42


def test_send_failure_is_swallowed() -> None:
    analytics = Analytics("test-token-123", consumer=FakeConsumer(fail=True))
    analytics.track("session-abc", "Question Asked")  # must not raise
    analytics.drain()


def test_multiple_events_in_order() -> None:
    consumer = FakeConsumer()
    analytics = Analytics("test-token-123", consumer=consumer)
    analytics.track("s1", "Question Asked", {"question_length": 30})
    analytics.track("s1", "Answer Shown", {"citations_count": 2})
    analytics.drain()
    assert [m["event"] for _, m in consumer.sent] == ["Question Asked", "Answer Shown"]
