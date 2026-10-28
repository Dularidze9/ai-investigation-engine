"""POST /api/query: run the investigation agent on one question."""
import logging

from fastapi import APIRouter, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, Field

from app.agent.agent import Agent
from app.agent.retriever import DbRetriever, Embedder
from app.config import get_settings
from app.db import get_pool
from app.guard import budget
from app.llm.client import LLMClient, LLMError

log = logging.getLogger(__name__)
router = APIRouter()


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def _openai():
    from openai import OpenAI
    return OpenAI(api_key=get_settings().openai_api_key, timeout=30, max_retries=0)  # we retry ourselves


@router.post("/api/query")
def query(body: QueryIn):
    pool = get_pool()
    with pool.connection() as conn:
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
            result = agent.run(body.question.strip())
        result.cost_usd += embedder.cost_usd
        actual = result.cost_usd
        log.info("query status=%s llm_calls=%d cost=$%.5f", result.status, result.llm_calls, result.cost_usd)
        return result.to_dict()
    except (LLMError, OpenAIError) as e:  # OpenAIError: query embedding failed after its retries
        raise HTTPException(status_code=503, detail="The language model service is unavailable. "
                                                    "Please try again in a minute.") from e
    finally:
        with pool.connection() as conn:
            budget.settle(conn, reserved, actual)
