"""Generic typed envelope for orchestrator v1 responses.

Mirrors the wire shape of ``app.py:_api_envelope``:

    {"schema": schema, "success": success, "data": data, "error": error}

All four keys appear on every response — ``data`` and ``error`` are emitted as
JSON ``null`` when absent rather than being omitted. ``ApiEnvelope`` and
``ErrorEnvelope`` produce byte-identical output to the existing helper for any
input the helper accepts.

Generic over the ``data`` payload type so route models can declare e.g.
``ApiEnvelope[JobStatus]`` and have mypy/Pydantic enforce the shape.

Implementation note: the wire field is ``schema``, but ``schema`` shadows
``pydantic.BaseModel.schema()`` (deprecated in v2 but still defined). To avoid
the resulting warning while keeping the wire key correct, the Python attribute
is named ``schema_`` and uses ``schema`` as its alias. Callers can construct
with ``schema=`` via the alias regardless of config; ``populate_by_name=True``
*additionally* allows construction with the field name ``schema_=``, which
internal call sites that prefer the unambiguous python-name form may use.
``serialize_by_alias=True`` makes ``model_dump()`` emit ``schema`` by default.
"""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from transformation_portal.api.v1.errors import ErrorObject
from transformation_portal.api.v1.schemas import SchemaName

T = TypeVar("T")


class ApiEnvelope(BaseModel, Generic[T]):
    """Typed wrapper around the orchestrator's standard JSON envelope.

    ``model_dump(mode="json")`` produces the same dict that
    ``app.py:_api_envelope`` returns today.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    schema_: SchemaName = Field(alias="schema", serialization_alias="schema")
    success: bool
    data: T | None = None
    error: ErrorObject | None = None


class ErrorEnvelope(ApiEnvelope[None]):
    """Specialization of ``ApiEnvelope`` for error responses.

    Mirrors ``app.py:_error_response``: schema is always
    ``tp.orchestrator.error.v1``, success is always ``False``, data is always
    ``None``, and error is required (not optional).
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    schema_: Literal["tp.orchestrator.error.v1"] = Field(
        default="tp.orchestrator.error.v1",
        alias="schema",
        serialization_alias="schema",
    )
    success: Literal[False] = False
    data: None = None
    error: ErrorObject
