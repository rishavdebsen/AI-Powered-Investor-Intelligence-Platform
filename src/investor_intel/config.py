"""Runtime configuration, driven by environment variables.

Two profiles are supported:
- local (default): pure-python TF-IDF retrieval + SQLite + extractive answers.
  Runs with no API keys, which is what CI and local development use.
- azure: Azure OpenAI + Azure AI Search + PostgreSQL, selected with II_PROFILE=azure
  and the corresponding environment variables. Never commit real values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    profile: str
    database_url: str
    data_dir: str
    embeddings_provider: str    # local | openai
    openai_embedding_model: str
    llm_provider: str           # none | openai | azure-openai
    openai_api_key: str
    openai_base_url: str
    openai_chat_model: str
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_chat_deployment: str
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index: str


def get_settings() -> Settings:
    return Settings(
        profile=_env("II_PROFILE", "local"),
        database_url=_env("DATABASE_URL", "sqlite:///./investor_intel.db"),
        data_dir=_env("II_DATA_DIR", "data"),
        embeddings_provider=_env("II_EMBEDDINGS_PROVIDER", "local"),
        openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_provider=_env("LLM_PROVIDER", "none"),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_base_url=_env("OPENAI_BASE_URL"),
        openai_chat_model=_env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        azure_openai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=_env("AZURE_OPENAI_API_KEY"),
        azure_openai_chat_deployment=_env("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
        azure_search_endpoint=_env("AZURE_SEARCH_ENDPOINT"),
        azure_search_api_key=_env("AZURE_SEARCH_API_KEY"),
        azure_search_index=_env("AZURE_SEARCH_INDEX", "investor-docs"),
    )
