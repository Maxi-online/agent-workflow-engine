"""Build process-local FAISS indices from explicit Markdown sources."""

import hashlib
import os
from pathlib import Path
import threading
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]
CACHE = {}
CACHE_LOCK = threading.RLock()


class FaissBuilder:
    def __init__(
        self,
        bases_dir=None,
        store_dir=None,
        http_client=None,
        openai_api_key=None,
        embeddings=None,
        allowed_bases=None,
    ):
        self.bases_dir = Path(
            bases_dir or os.getenv("AGENT_BASES_DIR", str(ROOT / "mas/bases"))
        ).resolve()
        self.store_dir = store_dir
        self.http_client = http_client
        self.openai_api_key = openai_api_key
        self.embeddings = embeddings
        self.allowed_bases = set(allowed_bases) if allowed_bases is not None else None

    def build_all(self):
        if not self.bases_dir.is_dir():
            raise ValueError("The configured knowledge directory does not exist")
        paths = [
            p
            for p in sorted(self.bases_dir.glob("*.md"))
            if self.allowed_bases is None or p.stem in self.allowed_bases
        ]
        if not paths:
            raise ValueError("No Markdown sources match the permitted knowledge scopes")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        signature = tuple((str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths)
        backend = (
            id(self.embeddings)
            if self.embeddings is not None
            else ("provider", os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_API_BASE", "")))
        )
        key = (signature, model, backend)
        with CACHE_LOCK:
            if key in CACHE:
                return CACHE[key]
            embeddings = self.embeddings
            if embeddings is None:
                credential = self.openai_api_key or os.getenv("OPENAI_API_KEY", "")
                if not credential:
                    raise ValueError("OPENAI_API_KEY is required for live embeddings")
                options = {"model": model, "api_key": credential}
                if self.http_client is not None:
                    options["http_client"] = self.http_client
                embeddings = OpenAIEmbeddings(**options)
            result = {}
            for path, source_signature in zip(paths, signature):
                file_key = ("file", source_signature, model, backend)
                if file_key in CACHE:
                    result[path.stem] = CACHE[file_key]
                    continue
                documents = []
                text = path.read_text(encoding="utf-8")
                for index, chunk in enumerate(
                    RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100).split_text(
                        text
                    )
                ):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": path.name,
                                "chunk_id": path.stem + ":" + str(index),
                            },
                        )
                    )
                if documents:
                    result[path.stem] = (FAISS.from_documents(documents, embeddings), documents)
                    CACHE[file_key] = result[path.stem]
            if not result:
                raise ValueError("The selected knowledge sources contain no usable text")
            CACHE[key] = result
            return result
