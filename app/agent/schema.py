"""Structured outputs. The JSON schemas are strict (all fields required, no extras) and
the same shapes are validated again with pydantic after the call."""
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["answered", "partial", "insufficient_evidence", "conflicting"]


class Plan(BaseModel):
    sub_queries: list[str] = Field(max_length=3)


class Claim(BaseModel):
    text: str
    citations: list[int]


class Conflict(BaseModel):
    description: str
    citations: list[int]


class Answer(BaseModel):
    status: Status
    answer: str
    claims: list[Claim]
    conflicts: list[Conflict]
    missing: list[str]


def _strict(name: str, schema: dict) -> dict:
    return {"name": name, "strict": True, "schema": schema}


_STR_LIST = {"type": "array", "items": {"type": "string"}}
_INT_LIST = {"type": "array", "items": {"type": "integer"}}

PLAN_SCHEMA = _strict("plan", {
    "type": "object", "additionalProperties": False, "required": ["sub_queries"],
    "properties": {"sub_queries": _STR_LIST},
})

ANSWER_SCHEMA = _strict("answer", {
    "type": "object", "additionalProperties": False,
    "required": ["status", "answer", "claims", "conflicts", "missing"],
    "properties": {
        "status": {"type": "string", "enum": ["answered", "partial", "insufficient_evidence", "conflicting"]},
        "answer": {"type": "string"},
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["text", "citations"],
            "properties": {"text": {"type": "string"}, "citations": _INT_LIST}}},
        "conflicts": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["description", "citations"],
            "properties": {"description": {"type": "string"}, "citations": _INT_LIST}}},
        "missing": _STR_LIST,
    },
})
