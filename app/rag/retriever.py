"""Local TF-IDF retriever for RAG over policy documents.

Loads markdown policies, splits them into heading-aware chunks, and indexes them with
scikit-learn's TF-IDF vectoriser (fully local, deterministic, no downloads). ``search``
returns the top-k chunks with cosine-similarity scores and citation metadata.

The ``BaseRetriever`` interface allows swapping to embeddings + a vector DB later
without touching the tool or agent code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc: str
    section: str
    text: str


class BaseRetriever:
    def search(self, query: str, k: int = 3) -> list[dict]:
        raise NotImplementedError


def _split_markdown(doc_name: str, content: str) -> list[Chunk]:
    """Split a markdown doc into chunks keyed by the nearest '##' heading."""
    chunks: list[Chunk] = []
    section = "General"
    buffer: list[str] = []

    def flush() -> None:
        text = " ".join(" ".join(buffer).split()).strip()
        if text:
            chunks.append(Chunk(doc=doc_name, section=section, text=text))

    for line in content.splitlines():
        if line.startswith("## "):
            flush()
            buffer = []
            section = line[3:].strip()
        elif line.startswith("# "):
            # top-level title; keep as section if no sub-heading yet
            section = line[2:].strip()
        else:
            buffer.append(line)
    flush()
    return chunks


class TfidfRetriever(BaseRetriever):
    def __init__(self, policies_dir: str | Path) -> None:
        self.policies_dir = Path(policies_dir)
        self.chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self.build()

    def build(self) -> int:
        """(Re)load documents and build the TF-IDF index. Returns chunk count."""
        self.chunks = []
        for path in sorted(self.policies_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            self.chunks.extend(_split_markdown(path.stem, content))

        if not self.chunks:
            self._vectorizer = None
            self._matrix = None
            return 0

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(c.text for c in self.chunks)
        return len(self.chunks)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.chunks or self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[dict] = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            results.append(
                {
                    "doc": c.doc,
                    "section": c.section,
                    "text": c.text,
                    "score": round(float(scores[i]), 4),
                }
            )
        return results
