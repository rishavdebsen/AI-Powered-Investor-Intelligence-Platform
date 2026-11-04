"""OpenAI-compatible chat client. Supports api.openai.com, a custom
OPENAI_BASE_URL, or Azure OpenAI (endpoint + deployment). Lazy-imported so the
local profile never needs the openai package or a key."""
from __future__ import annotations


class OpenAIChat:
    def __init__(self, api_key: str, model: str, base_url: str = "", azure_endpoint: str = "") -> None:
        if azure_endpoint:
            from openai import AzureOpenAI  # lazy

            self._client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version="2024-10-21")
        else:
            from openai import OpenAI  # lazy

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        self._model = model

    def complete(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(model=self._model, messages=messages, temperature=0)
        return resp.choices[0].message.content or ""
