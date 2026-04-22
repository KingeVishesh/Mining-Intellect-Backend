"""
LLM Factory — returns the primary (Grok) and fallback (Claude) LLM instances.
"""
from config import settings


def get_grok_llm(temperature: float = 0.0):
    """Return Grok via the OpenAI-compatible xAI endpoint."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="grok-3",
        api_key=settings.grok_api_key,
        base_url="https://api.x.ai/v1",
        temperature=temperature,
    )


def get_claude_llm(temperature: float = 0.0):
    """Return Claude Sonnet as a fallback LLM."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=settings.anthropic_api_key,
        temperature=temperature,
    )


def get_llm(temperature: float = 0.0):
    """Return the primary LLM, falling back to Claude if Grok key is absent."""
    if settings.grok_api_key:
        return get_grok_llm(temperature)
    return get_claude_llm(temperature)
