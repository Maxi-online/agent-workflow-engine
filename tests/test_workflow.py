import json
from pathlib import Path
import secrets
from types import SimpleNamespace
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import create_app
from demo import HashEmbeddings, DemoRouter, DemoSummary
from mas.agents.orchestrator_agent import AGENTS, OrchestratorAgent
from mas.utils.faiss_client import FaissClient
from mas.utils.memory_manager import MemoryManager


def test_memory_survives_initialization_and_remains_user_scoped(tmp_path):
    path = tmp_path / "memory.sqlite3"
    first = MemoryManager(path)
    first.save_memory("one", "Question one", "Answer one")
    first.save_memory("two", "Question two", "Answer two")
    first.set_summary("one", {"short_summary": "Only user one"})
    second = MemoryManager(path)
    one = second.get_combined_memory("one")
    assert [m["role"] for m in one] == ["system", "user", "assistant"]
    assert all("two" not in m["content"] for m in one)
    second.clear_memory("one")
    assert second.get_combined_memory("one") == []
    assert len(second.get_combined_memory("two")) == 2


def test_scoped_retrieval_never_exposes_another_base(tmp_path):
    (tmp_path / "general.md").write_text("Public synthetic service information.", encoding="utf-8")
    (tmp_path / "restricted.md").write_text("A separate synthetic collection.", encoding="utf-8")
    embeddings = HashEmbeddings()
    FaissClient(bases_dir=tmp_path, embeddings=embeddings)
    scoped = FaissClient(bases_dir=tmp_path, embeddings=embeddings, allowed_bases=["general"])
    assert set(scoped.databases) == {"general"}
    assert scoped.search("restricted", "collection") == ""
    assert "Public synthetic" in scoped.search("general", "service")


def test_hybrid_search_accepts_numpy_scores_and_lexical_only_candidates():
    docs = [
        SimpleNamespace(page_content="Dense match", metadata={"chunk_id": "d", "source": "demo"}),
        SimpleNamespace(
            page_content="Second dense match", metadata={"chunk_id": "s", "source": "demo"}
        ),
        SimpleNamespace(
            page_content="Unique lexical evidence", metadata={"chunk_id": "l", "source": "demo"}
        ),
    ]

    class Dense:
        def similarity_search_with_score(self, query, k):
            return [(docs[0], 0.1), (docs[1], 2)]

    class Lexical:
        def get_scores(self, query):
            return np.array([0.1, 0.2, 4.0])

    client = FaissClient.__new__(FaissClient)
    client.databases = {"demo": Dense()}
    client.docs_texts = {"demo": docs}
    client.bm25_index = {"demo": Lexical()}
    result = client.search("demo", "unique", k=1, hybrid=True, weight_faiss=0, weight_bm25=1)
    assert "Unique lexical evidence" in result


def orchestrator(tmp_path, replies, max_handoffs=3):
    memory = MemoryManager(tmp_path / "memory.sqlite3")

    class Agent:
        def __init__(self, name):
            self.name = name

        async def handle_message(self, *args, **kwargs):
            return replies[self.name]

    agent = OrchestratorAgent(
        "reader",
        None,
        None,
        memory=memory,
        routing_client=DemoRouter(),
        summary_agent=DemoSummary(),
        agent_factory=Agent,
        max_handoffs=max_handoffs,
    )
    return agent, memory


@pytest.mark.asyncio
async def test_handoff_saves_only_the_final_answer_once(tmp_path):
    agent, memory = orchestrator(
        tmp_path,
        {"general agent": "[Handoff] Services Agent", "services agent": "A grounded final answer"},
    )
    assert await agent.handle_message("Question") == "A grounded final answer"
    assert len(memory.get_combined_memory("reader")) == 2
    assert memory.get_last_agent("reader") == "services agent"


@pytest.mark.asyncio
async def test_cyclic_handoff_does_not_enter_conversation_memory(tmp_path):
    agent, memory = orchestrator(
        tmp_path,
        {"general agent": "[Handoff] Services Agent", "services agent": "[Handoff] General Agent"},
    )
    with pytest.raises(RuntimeError, match="cyclic"):
        await agent.handle_message("Question")
    assert memory.get_combined_memory("reader") == []


@pytest.mark.asyncio
async def test_handoff_budget_is_enforced(tmp_path):
    agent, memory = orchestrator(
        tmp_path, {"general agent": "[Handoff] Services Agent"}, max_handoffs=0
    )
    with pytest.raises(RuntimeError, match="budget"):
        await agent.handle_message("Question")
    assert memory.get_combined_memory("reader") == []


@pytest.mark.asyncio
async def test_unknown_handoff_target_is_rejected(tmp_path):
    agent, memory = orchestrator(tmp_path, {"general agent": "[Handoff] Unknown Agent"})
    with pytest.raises(ValueError, match="unknown"):
        await agent.handle_message("Question")
    assert memory.get_combined_memory("reader") == []


@pytest.mark.asyncio
async def test_json_answers_are_unwrapped_before_persistence(tmp_path):
    agent, memory = orchestrator(
        tmp_path, {"general agent": '{"answer":"Grounded reply","emotions":{"neutral":1}}'}
    )
    assert await agent.handle_message("Question") == "Grounded reply"
    assert memory.get_combined_memory("reader")[-1]["content"] == "Grounded reply"


def test_api_uses_server_credentials_and_rejects_provider_keys_in_body(tmp_path, monkeypatch):
    access = secrets.token_urlsafe(24)
    monkeypatch.setenv("AGENT_API_TOKEN", access)
    memory = MemoryManager(tmp_path / "memory.sqlite3")

    class Agent:
        async def handle_message(self, message, context, image):
            return "Synthetic reply"

    app = create_app(lambda user, memory: Agent(), memory)
    with TestClient(app) as client:
        assert (
            client.post("/chat", json={"user_id": "one", "message": "Question"}).status_code == 401
        )
        headers = {"Authorization": "Bearer " + access}
        assert (
            client.post(
                "/chat", headers=headers, json={"user_id": "one", "message": "Question"}
            ).json()["answer"]
            == "Synthetic reply"
        )
        assert (
            client.post(
                "/chat",
                headers=headers,
                json={
                    "user_id": "one",
                    "message": "Question",
                    "OPENAI_API_KEY": secrets.token_urlsafe(16),
                },
            ).status_code
            == 422
        )


def test_every_routed_agent_has_a_valid_constructor_without_network():
    import httpx

    key = secrets.token_urlsafe(24)
    client = httpx.AsyncClient()
    agents = [cls("demo", key, SimpleNamespace(), client) for cls, _ in AGENTS.values()]
    assert len(agents) == 8
    import asyncio

    asyncio.run(client.aclose())


@pytest.mark.asyncio
async def test_summary_uses_the_injected_user_memory(tmp_path):
    from mas.agents.summarization_agent import SummarizationAgent

    value = {
        "short_summary": "A source fact",
        "long_summary": "A bounded conversation",
        "key_topics": ["retrieval"],
        "decisions": [],
    }

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))]
            )

    store = MemoryManager(tmp_path / "memory.sqlite3")
    agent = SummarizationAgent("one", None, completion_client=Client())
    agent.memory_store = store
    await agent.handle_message("Question", context=[])
    assert json.loads(store.get_summary("one")) == value
    assert store.get_summary("two") is None


@pytest.mark.parametrize(
    "value",
    [
        {"short_summary": "incomplete"},
        {"short_summary": "a", "long_summary": "b", "key_topics": "not an array", "decisions": []},
    ],
)
def test_malformed_summaries_are_not_accepted(value):
    from mas.agents.summarization_agent import validate_summary

    with pytest.raises(ValueError):
        validate_summary(value)
