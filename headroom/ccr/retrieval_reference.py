"""Backward-compatible CCR marker references with opaque event handles."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HASH_PATTERN = r"[a-fA-F0-9]{12,24}"
_HANDLE_PATTERN = r"rh-[a-fA-F0-9]{32}"
CCR_REFERENCE_RE = re.compile(
    rf"<<ccr:(?P<hash>{_HASH_PATTERN})(?:@(?P<handle>{_HANDLE_PATTERN}))?"
)


@dataclass(frozen=True)
class RetrievalReference:
    """A content hash plus an optional event-scoped opaque handle."""

    hash_key: str
    retrieval_handle: str = ""


def references_in_text(text: str) -> tuple[RetrievalReference, ...]:
    """Return distinct references in marker order."""
    seen: set[tuple[str, str]] = set()
    references: list[RetrievalReference] = []
    for match in CCR_REFERENCE_RE.finditer(text):
        identity = (
            match.group("hash").lower(),
            (match.group("handle") or "").lower(),
        )
        if identity in seen:
            continue
        seen.add(identity)
        references.append(RetrievalReference(*identity))
    return tuple(references)


def attach_retrieval_handle(text: str, retrieval_handle: str) -> str:
    """Add ``retrieval_handle`` to legacy CCR markers without changing suffixes."""
    if not re.fullmatch(_HANDLE_PATTERN, retrieval_handle, flags=re.IGNORECASE):
        raise ValueError("retrieval_handle must be rh- followed by 32 hexadecimal characters")

    def replace(match: re.Match[str]) -> str:
        if match.group("handle"):
            return match.group(0)
        return f"<<ccr:{match.group('hash')}@{retrieval_handle.lower()}"

    return CCR_REFERENCE_RE.sub(replace, text)


def validate_retrieval_handle(value: object) -> str:
    """Return a normalized handle, or an empty string for absent/invalid input."""
    if not isinstance(value, str) or not re.fullmatch(_HANDLE_PATTERN, value, re.IGNORECASE):
        return ""
    return value.lower()
