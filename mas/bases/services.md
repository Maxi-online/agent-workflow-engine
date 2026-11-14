# Retrieval and Evaluation Services

Knowledge retrieval combines document ingestion, chunking and evidence selection. Evaluation checks whether answers are grounded and whether failures are visible. The demonstration includes FAISS search, BM25 lexical matching and reciprocal-rank fusion. Published examples use synthetic content.

## Integration design

Integration work connects authenticated HTTP endpoints, explicit external adapters and bounded background jobs. Operators configure their own destinations and credentials. The service does not execute website changes, make purchases or send messages unless a separate application adapter implements that action.
