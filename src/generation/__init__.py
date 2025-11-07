"""Conversation generation utilities."""

from .conversation_generator import instantiate_conversation  # noqa: F401

try:  # pragma: no cover - optional dependency
    from .curator_utterances import CuratorUtteranceGenerator  # noqa: F401
except ImportError:  # pragma: no cover
    CuratorUtteranceGenerator = None  # type: ignore
