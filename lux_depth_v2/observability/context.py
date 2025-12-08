from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

_REQUEST_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "lux_request_id",
    default=None,
)

# Header names we accept for incoming correlation
REQUEST_ID_HEADERS = ("x-request-id", "x-correlation-id")


def new_request_id() -> str:
    return str(uuid.uuid4())


def get_request_id() -> Optional[str]:
    return _REQUEST_ID.get()


def set_request_id(request_id: Optional[str]) -> contextvars.Token:
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    _REQUEST_ID.reset(token)


@contextmanager
def bind_request_id(request_id: Optional[str]) -> Iterator[None]:
    token = set_request_id(request_id)
    try:
        yield
    finally:
        reset_request_id(token)
