# Agent Workflow Engine

A portfolio edition of a multi-agent assistant runtime with **bounded handoffs, persistent conversation memory, scoped knowledge retrieval and authenticated HTTP entry points**. The repository includes a complete offline path through FAISS, BM25, routing, handoff execution and SQLite persistence.

The engineering focus is the runtime around the model: which agent may read which collection, how a conversation survives a restart, how a handoff terminates, and how failures become visible. It is an integration and orchestration project. It does not claim a newly trained language model or measured production answer quality.

## What is implemented

- Eight routed specialists: general assistance, services, sales, technical consultation, articles, onboarding, interface guidance and issue handling.
- A router followed by a bounded specialist/handoff chain. Unknown targets, cycles and exhausted handoff budgets fail explicitly.
- User-scoped SQLite messages, summaries and the last successful specialist. Opening a database never resets its tables.
- Markdown ingestion into process-local FAISS indices. A specialist receives only its permitted named collection.
- Optional hybrid retrieval combining dense and BM25 rankings with weighted reciprocal-rank fusion.
- A fixed conversation window and automatic summarization once enough user turns accumulate.
- Authenticated FastAPI endpoints, a JSON command-line interface and an optional text/photo Telegram adapter.
- Additional analytics, SEO, content, localization, emotion and CMS proposal modules. These produce text or structured suggestions; they are not autonomous CMS executors.
- Offline regression checks and release checks for English source prose, configuration literals and secret patterns.

## Request lifecycle

~~~mermaid
flowchart LR
    API[Authenticated request] --> Lock[Per-user lock]
    Lock --> Memory[Load scoped memory]
    Memory --> Router[Select specialist]
    Router --> Scope[Select permitted knowledge base]
    Scope --> Retrieval[FAISS and optional BM25]
    Retrieval --> Agent[Specialist response]
    Agent --> Handoff{Handoff directive?}
    Handoff -->|Within budget| Scope
    Handoff -->|Final answer| Save[Persist one user/answer pair]
    Save --> Response[JSON response]
~~~

A handoff directive is an internal control message. It is not stored as the assistant's final answer. The successful final answer is written once after the chain completes.

The default routed path uses a router call and one specialist call. Handoffs and periodic summarization add calls. Initial knowledge indexing also uses the selected embedding provider in live mode. The offline demonstration uses neither a hosted language model nor hosted embeddings.

## Quick start: no credentials or network requests

Python 3.11 and 3.12 are the release-check targets.

~~~bash
git clone https://github.com/Maxi-online/agent-workflow-engine.git
cd agent-workflow-engine
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python main.py --demo
python -m pytest -q
~~~

On Windows, activate the same environment with:

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

The demonstration builds actual FAISS indices from the fictional English Markdown files, calculates BM25 scores, follows a general-to-services handoff and reopens its temporary SQLite database. Its router and specialist are deterministic adapters. Its hash-based embeddings are a local lexical stand-in, not a pretrained semantic model. The printed answer demonstrates integration behavior rather than model quality.

## Live HTTP service

Set the following process variables before starting a live service. No provider key is accepted in a request body.

| Variable | Purpose |
| --- | --- |
| OPENAI_API_KEY | Provider credential for live chat and embeddings |
| AGENT_API_TOKEN | Shared operator/gateway credential for the HTTP API |
| OPENAI_MODEL | Optional override for the agent chat model |
| OPENAI_EMBEDDING_MODEL | Optional embedding model override |
| AGENT_BASES_DIR | Optional directory of operator-owned Markdown collections |
| AGENT_MEMORY_PATH | Optional SQLite path; default is var/memory.sqlite3 |
| PROXY_URL | Optional operator-configured HTTP proxy; no built-in proxy |
| AGENT_SNIPPETS_DIR | Optional directory of approved named HTML snippets |

~~~bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
~~~

The health endpoint is available without a provider key:

~~~bash
curl http://127.0.0.1:8000/health
~~~

After the live environment is configured, this is a complete authenticated request:

~~~python
import os
import httpx

response = httpx.post(
    "http://127.0.0.1:8000/chat",
    headers={"Authorization": "Bearer " + os.environ["AGENT_API_TOKEN"]},
    json={
        "user_id": "demo-reader",
        "message": "Which services cover retrieval evaluation?",
        "context": [],
    },
    timeout=180,
)
response.raise_for_status()
print(response.json())
~~~

POST /telegram accepts the same schema. USER_ID remains an input alias for user_id. The response contains answer. Optional context entries accept only user or assistant roles; clients cannot insert a system/developer entry through that field. An optional image must be a bounded PNG, JPEG or WebP data URI.

DELETE /memory/demo-reader clears only that user's stored memory. The API credential authenticates an operator or gateway, not a complete end-user identity system. A deployed host must bind user_id to its own authenticated identity.

## Retrieval behavior

Each Markdown filename is a collection name. The routed specialists use general, services, sales, consultation, blog, onboarding, conflict or ui.

FaissClient.search supports semantic retrieval or hybrid=True. The hybrid branch takes candidates from both dense and lexical rankings and combines their rank contributions. Weights must be nonnegative and have a positive sum. The threshold applies to the normalized rank-fusion score; it is not a calibrated relevance probability.

Only the permitted collections are present in a specialist's client. Index caches are process-local and keyed by source content and embedding configuration. The runtime does not load client-supplied pickle files. Replace the fictional sources with documents you are authorized to process; generated indices and runtime state are excluded from Git.

## Repository guide

| Path | Responsibility |
| --- | --- |
| main.py | API, authentication and CLI |
| mas/agents/orchestrator_agent.py | Router, bounded handoffs and final-answer persistence |
| mas/agents/ | Specialist implementations and optional proposal modules |
| mas/utils/memory_manager.py | Non-destructive, user-scoped SQLite memory |
| mas/utils/faiss_builder.py | Markdown ingestion and index construction |
| mas/utils/faiss_client.py | Scoped semantic and hybrid retrieval |
| mas/utils/snippet_loader.py | Restricted named-snippet loading |
| mas/bases/ | Fictional English knowledge sources |
| telegram_bot.py | Optional text/photo Telegram client |
| demo.py | Deterministic offline adapters |
| tests/ | Memory, scope, handoff, API and retrieval regressions |

For the Telegram client, configure TELEGRAM_BOT_TOKEN, AGENT_API_TOKEN and optionally AGENT_API_URL, then run python telegram_bot.py. Starting that adapter intentionally connects to Telegram. It is not part of offline verification.

## Verification and limits

Tests cover persistence across initialization, user isolation, knowledge scopes, NumPy BM25 outputs, lexical-only candidates, handoff cycles, handoff limits, answer unwrapping and API credential boundaries. CI also compiles every tracked Python file and scans reachable Git history with a pinned, checksum-verified Gitleaks binary.

The service uses per-user in-process locks and a local SQLite store; the documented command uses one worker. Multi-process coordination, tenant authorization, background-job recovery, independent answer-quality evaluation and production load testing remain separate deployment work. HTML snippets are output content for a host to handle safely, not a complete browser interface.

This release removes original deployment settings, client corpora, vector caches and credentials. See [SECURITY.md](SECURITY.md) for the release checks and [architecture](docs/architecture.md) for the runtime contracts.

## License

Publicly visible for portfolio inspection under the restrictive [LICENSE](LICENSE). This is not an unrestricted open-source release. Third-party libraries retain their own licenses.
