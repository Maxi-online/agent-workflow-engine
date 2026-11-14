"""Offline FAISS/BM25 and handoff example using deterministic local adapters."""

import asyncio
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from langchain_core.embeddings import Embeddings
from mas.agents.orchestrator_agent import OrchestratorAgent
from mas.utils.faiss_client import FaissClient
from mas.utils.memory_manager import MemoryManager


class HashEmbeddings(Embeddings):
    """A demo-only lexical hashing adapter, not a pretrained semantic model."""

    def embed_query(self, text):
        vector = [0.0] * 64
        for word in re.findall(r"[a-z]+", text.lower()):
            vector[int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 64] += 1
        norm = math.sqrt(sum(v * v for v in vector)) or 1
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class DemoRouter:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="General Agent"))]
        )


class DemoSummary:
    async def handle_message(self, message, context=None):
        return json.dumps({"short_summary": "Synthetic demonstration"})


class DemoAgent:
    def __init__(self, name, knowledge):
        self.name = name
        self.knowledge = knowledge

    async def handle_message(self, message, context=None, image=None):
        if self.name == "general agent":
            return "[Handoff] Services Agent"
        return self.knowledge.search("services", message, k=2, hybrid=True)


async def run_demo():
    with tempfile.TemporaryDirectory(prefix="agent-workflow-demo-") as folder:
        memory = MemoryManager(Path(folder) / "memory.sqlite3")
        knowledge = FaissClient(embeddings=HashEmbeddings())
        agent = OrchestratorAgent(
            "demo-reader",
            None,
            knowledge,
            memory=memory,
            routing_client=DemoRouter(),
            summary_agent=DemoSummary(),
            agent_factory=lambda name: DemoAgent(name, knowledge),
        )
        answer = await agent.handle_message(
            "What services cover knowledge retrieval and evaluation?"
        )
        reopened = MemoryManager(Path(folder) / "memory.sqlite3")
        return {
            "mode": "offline deterministic adapters",
            "live_model_requests": 0,
            "answer": answer,
            "final_agent": memory.get_last_agent("demo-reader"),
            "memory_survives_reinitialization": len(reopened.get_combined_memory("demo-reader"))
            == 2,
        }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo()), indent=2))
