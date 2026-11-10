"""Product analytics via Mixpanel.

Sends usage events (sessions, report uploads, questions, answers) to a
Mixpanel project so real usage of the app can be reviewed in Mixpanel's
funnels, flows and retention reports.

Disabled by default: without ``MIXPANEL_PROJECT_TOKEN`` the tracker is a
no-op, so local development, CI and the public demo keep working with no
Mixpanel account. Events are sent on a background thread and every send
failure is swallowed - analytics must never break the app.

Set ``MIXPANEL_API_HOST=api-eu.mixpanel.com`` for projects on EU data
residency; defaults to the US host.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class Analytics:
    """Fire-and-forget Mixpanel event tracker.

    ``consumer`` is injectable for tests (any object with the mixpanel-python
    ``Consumer.send(endpoint, json_message)`` shape).
    """

    def __init__(self, token: str = "", api_host: str = "api.mixpanel.com", consumer=None) -> None:
        self._client = None
        self._threads: list[threading.Thread] = []
        token = (token or "").strip()
        if not token:
            return
        try:
            from mixpanel import Consumer, Mixpanel

            if consumer is None:
                consumer = Consumer(events_url=f"https://{api_host}/track")
            self._client = Mixpanel(token, consumer=consumer)
        except Exception as exc:  # noqa: BLE001 - analytics must never break the app
            logger.warning("Mixpanel analytics disabled: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def track(self, distinct_id: str, event: str, properties: dict | None = None) -> None:
        """Queue one event. Returns immediately; send failures are logged only."""
        if not self.enabled:
            return
        props = dict(properties or {})

        def _send() -> None:
            try:
                self._client.track(distinct_id, event, props)
            except Exception as exc:  # noqa: BLE001 - send failures are logged, never raised
                logger.warning("Mixpanel track failed for %s: %s", event, exc)

        thread = threading.Thread(target=_send, daemon=True)
        thread.start()
        self._threads.append(thread)

    def drain(self, timeout: float = 5.0) -> None:
        """Wait for queued sends to finish. Used by tests and CLI scripts."""
        for thread in self._threads:
            thread.join(timeout)
        self._threads.clear()
