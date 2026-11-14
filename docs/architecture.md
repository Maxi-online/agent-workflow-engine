# Runtime contracts

## Routing and completion

The router selects one of eight registered specialist names. An unknown router output falls back to General Agent. A specialist may produce a final answer or one strict handoff directive. A handoff must name a registered specialist, remain within the configured budget and not revisit a previously selected specialist.

Only a final nonempty answer is persisted. Intermediate directives are control state. Errors do not turn into a successful conversation record.

## Memory

MemoryManager opens short-lived SQLite connections and creates missing tables without removing existing tables. Reads are scoped by user_id. The latest summary is followed by the most recent message pairs in chronological order. A reopen must retain those rows.

The summary adapter uses the same injected memory store as the orchestrator. Tests use temporary stores; importing the module does not create or migrate a database.

## Knowledge boundaries

The runtime's knowledge namespace is a Markdown filename. A selected specialist obtains a client containing only the corresponding scope. A process-level cache does not grant a client access to additional collection names.

FAISS provides the dense ranking. BM25 provides a lexical ranking over the selected collection. Hybrid retrieval forms their candidate union and uses weighted reciprocal-rank fusion. Its score is a ranking convenience, not a probability or independently calibrated quality measure.

## Interfaces

The HTTP API uses an operator credential from the environment. Provider credentials are server-side settings. The host application must authenticate end users and decide the user_id namespace before forwarding requests.

The optional Telegram adapter forwards text or bounded image data to the same API. Optional CMS/SEO/content modules produce suggestions. They do not implement write access to a live website.

## Failure and deployment boundaries

SQLite snapshots and per-user locks target a single application process. The runtime does not provide a distributed actor system, a durable job broker or exactly-once external effects. Its tests establish deterministic control-flow and persistence properties, not a production throughput or language-model accuracy claim.
