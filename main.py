"""Authenticated API and command-line entry point for Agent Workflow Engine."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import os
from typing import Literal, Optional
from fastapi import Depends, FastAPI, HTTPException, Request
import httpx
from pydantic import AliasChoices, BaseModel, Field, ConfigDict
from mas.agents.orchestrator_agent import OrchestratorAgent
from mas.utils.faiss_client import FaissClient
from mas.utils.memory_manager import MemoryManager


class ContextMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=10_000)
    user_id: str = Field(
        min_length=1, max_length=128, validation_alias=AliasChoices("user_id", "USER_ID")
    )
    context: list[ContextMessage] = Field(default_factory=list, max_length=20)
    image: Optional[str] = Field(
        default=None, max_length=1_000_000, pattern=r"^data:image/(?:png|jpeg|webp);base64,"
    )


class ChatResponse(BaseModel):
    answer: str


def create_app(orchestrator_factory=None, memory=None):
    @asynccontextmanager
    async def lifespan(application):
        application.state.memory = memory or MemoryManager()
        proxy = os.getenv("PROXY_URL") or None
        with httpx.Client(proxy=proxy, timeout=30) as sync_client:
            async with httpx.AsyncClient(proxy=proxy, timeout=60) as async_client:
                application.state.sync_client = sync_client
                application.state.async_client = async_client
                application.state.user_locks = {}
                yield

    app = FastAPI(title="Agent Workflow Engine", version="1.0.0", lifespan=lifespan)

    async def access(request: Request):
        expected = os.getenv("AGENT_API_TOKEN", "")
        if not expected:
            raise HTTPException(503, "AGENT_API_TOKEN must be configured")
        if not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + expected):
            raise HTTPException(401, "Invalid API access token")

    def build(user_id, state):
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY is required for live requests")
        knowledge = FaissClient(http_client=state.sync_client, openai_api_key=key)
        return OrchestratorAgent(user_id, key, knowledge, state.async_client, memory=state.memory)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "agent-workflow-engine"}

    @app.post("/telegram", response_model=ChatResponse, dependencies=[Depends(access)])
    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(access)])
    async def chat(body: ChatRequest, request: Request):
        state = request.app.state
        if body.user_id not in state.user_locks and len(state.user_locks) >= 1000:
            raise HTTPException(503, "Conversation capacity reached")
        lock = state.user_locks.setdefault(body.user_id, asyncio.Lock())
        async with lock:
            try:
                agent = (
                    orchestrator_factory(body.user_id, state.memory)
                    if orchestrator_factory
                    else await asyncio.to_thread(build, body.user_id, state)
                )
                answer = await agent.handle_message(
                    body.message, [m.model_dump() for m in body.context], body.image
                )
                return ChatResponse(answer=answer)
            except ValueError as error:
                raise HTTPException(
                    503, "Provider configuration or agent output is invalid"
                ) from error
            except Exception as error:
                raise HTTPException(502, "Agent workflow failed") from error

    @app.delete("/memory/{user_id}", dependencies=[Depends(access)])
    async def clear_memory(user_id: str, request: Request):
        state = request.app.state
        lock = state.user_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            state.memory.clear_memory(user_id)
        return {"message": "Memory cleared"}

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Agent Workflow Engine")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        from demo import run_demo

        print(json.dumps(asyncio.run(run_demo()), indent=2))
        return 0
    body = ChatRequest.model_validate_json(input())
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required; --demo runs without credentials")

    async def run():
        memory = MemoryManager()
        with httpx.Client(proxy=os.getenv("PROXY_URL") or None, timeout=30) as sync_client:
            async with httpx.AsyncClient(
                proxy=os.getenv("PROXY_URL") or None, timeout=60
            ) as async_client:
                knowledge = await asyncio.to_thread(
                    FaissClient, http_client=sync_client, openai_api_key=key
                )
                agent = OrchestratorAgent(body.user_id, key, knowledge, async_client, memory=memory)
                return await agent.handle_message(
                    body.message, [m.model_dump() for m in body.context], body.image
                )

    print(json.dumps({"answer": asyncio.run(run())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
