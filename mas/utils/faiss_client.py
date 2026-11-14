"""Scoped semantic retrieval and weighted reciprocal-rank fusion with BM25."""

from rank_bm25 import BM25Okapi
from .faiss_builder import FaissBuilder


class FaissClient:
    def __init__(
        self,
        bases_dir=None,
        store_dir=None,
        http_client=None,
        openai_api_key=None,
        allowed_bases=None,
        embeddings=None,
    ):
        builder = FaissBuilder(
            bases_dir, store_dir, http_client, openai_api_key, embeddings, allowed_bases
        )
        built = builder.build_all()
        self.bases_dir = str(builder.bases_dir)
        self.store_dir = store_dir
        self.http_client = http_client
        self.embeddings = embeddings
        self.databases = {name: value[0] for name, value in built.items()}
        self.docs_texts = {name: value[1] for name, value in built.items()}
        self.bm25_index = {
            name: BM25Okapi([doc.page_content.lower().split() for doc in documents])
            for name, documents in self.docs_texts.items()
        }

    def search(
        self, base_name, query, k=3, hybrid=False, weight_faiss=0.5, weight_bm25=0.5, threshold=0.0
    ):
        if k < 1:
            raise ValueError("k must be positive")
        if base_name not in self.databases:
            return ""
        if weight_faiss < 0 or weight_bm25 < 0 or weight_faiss + weight_bm25 <= 0:
            raise ValueError("Retrieval weights must be nonnegative with a positive sum")
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be in the zero-to-one range")
        dense = self.databases[base_name].similarity_search_with_score(query, k=k)
        if not hybrid:
            return "\n\n".join(
                "Document "
                + str(i + 1)
                + " ["
                + doc.metadata.get("source", base_name)
                + "]\n"
                + doc.page_content
                for i, (doc, _) in enumerate(dense)
            )
        scores = self.bm25_index[base_name].get_scores(query.lower().split())
        documents = self.docs_texts[base_name]
        lexical = sorted(range(len(documents)), key=lambda i: float(scores[i]), reverse=True)
        lexical = [i for i in lexical if float(scores[i]) > 0][:k]

        def identity(doc):
            return doc.metadata.get("chunk_id") or (
                doc.metadata.get("source", ""),
                doc.page_content,
            )

        combined = {}
        lookup = {}
        total = weight_faiss + weight_bm25
        for rank, (doc, _) in enumerate(dense, 1):
            identifier = identity(doc)
            lookup[identifier] = doc
            combined[identifier] = combined.get(identifier, 0) + weight_faiss / total * 61 / (
                60 + rank
            )
        for rank, index in enumerate(lexical, 1):
            doc = documents[index]
            identifier = identity(doc)
            lookup[identifier] = doc
            combined[identifier] = combined.get(identifier, 0) + weight_bm25 / total * 61 / (
                60 + rank
            )
        selected = [
            (identifier, score)
            for identifier, score in sorted(
                combined.items(), key=lambda item: item[1], reverse=True
            )
            if score >= threshold
        ][:k]
        return "\n\n".join(
            "Document "
            + str(i + 1)
            + " ["
            + lookup[identifier].metadata.get("source", base_name)
            + "]\n"
            + lookup[identifier].page_content
            + "\nRank-fusion score: "
            + format(score, ".4f")
            for i, (identifier, score) in enumerate(selected)
        )
