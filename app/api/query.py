"""POST /api/query: run the investigation agent on one question. GET /api/status: demo limits.

Order of checks for a question: per-IP rate limit -> answer cache (free) -> budget reservation -> agent.
"""
import logging

from fastapi import APIRouter, HTTPException, Request
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.agent.agent import Agent
from app.agent.retriever import DbRetriever, Embedder
from app.config import get_settings
from app.db import get_pool
from app.guard import answer_cache, budget
from app.guard.ratelimit import RateLimiter, client_ip
from app.llm.client import LLMClient, LLMError

log = logging.getLogger(__name__)
router = APIRouter()
limiter = RateLimiter()


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def _openai():
    from openai import OpenAI
    return OpenAI(api_key=get_settings().openai_api_key, timeout=30, max_retries=0)  # we retry ourselves


@router.post("/api/query")
def query(body: QueryIn, request: Request):
    allowed, retry_after = limiter.allow(client_ip(request))
    if not allowed:
        raise HTTPException(status_code=429, headers={"Retry-After": str(retry_after)},
                            detail=f"Rate limit: {limiter.limit} questions per hour. "
                                   f"Try again in {retry_after // 60 + 1} min.")
    question = body.question.strip()
    pool = get_pool()
    with pool.connection() as conn:
        cached = answer_cache.get(conn, question)
        if cached is not None:
            log.info("query served from cache")
            return {**cached, "cached": True, "cost_usd": 0.0}
        if not budget.reserve(conn):
            raise HTTPException(status_code=429, detail="The public demo budget is used up.")
    # If anything fails unexpectedly we keep the full reservation: overcharging the cap is safe,
    # undercharging it is not.
    reserved = actual = budget.MAX_QUERY_COST
    try:
        client = _openai()
        embedder = Embedder(client)
        with pool.connection() as conn:
            agent = Agent(LLMClient(client), DbRetriever(conn, embedder))
            result = agent.run(question)
        result.cost_usd += embedder.cost_usd
        actual = result.cost_usd
        log.info("query status=%s llm_calls=%d cost=$%.5f", result.status, result.llm_calls, result.cost_usd)
        out = result.to_dict()
        if result.status != "error":  # never cache a failure
            with pool.connection() as conn:
                answer_cache.put(conn, question, out)
        return {**out, "cached": False}
    except (LLMError, OpenAIError) as e:  # OpenAIError: query embedding failed after its retries
        raise HTTPException(status_code=503, detail="The language model service is unavailable. "
                                                    "Please try again in a minute.") from e
    finally:
        with pool.connection() as conn:
            budget.settle(conn, reserved, actual)


@router.get("/api/status")
def status(request: Request):
    with get_pool().connection() as conn:
        spend = budget.status(conn)
    return {**spend, "questions_left_this_hour": limiter.remaining(client_ip(request)),
            "limit_per_hour": limiter.limit}
