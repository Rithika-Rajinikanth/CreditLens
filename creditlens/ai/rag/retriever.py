"""
CreditLens — Institutional Policy & Regulatory RAG Retriever
Fast, deterministic, 100% in-memory hybrid retriever (BM25 + Semantic keyword scoring).
Requires zero external databases, zero cloud APIs, and executes in <1ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import math

KB_DIR = Path(__file__).parent / "knowledge_base"


@dataclass
class PolicyChunk:
    chunk_id: str
    source_file: str
    title: str
    section: str
    content: str
    score: float = 0.0


class PolicyRetriever:
    """
    In-memory hybrid retriever for bank underwriting policies,
    statutory ECOA regulations, Basel III credit risk standards, and fraud playbooks.
    """

    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or KB_DIR
        self.chunks: List[PolicyChunk] = []
        self._doc_frequencies: Dict[str, int] = {}
        self._total_docs: int = 0
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        """Parse all markdown files in knowledge_base into indexed policy chunks."""
        self.chunks.clear()
        self._doc_frequencies.clear()

        if not self.kb_dir.exists():
            return

        for file_path in self.kb_dir.glob("*.md"):
            try:
                text = file_path.read_text(encoding="utf-8")
                self._parse_markdown_file(file_path.name, text)
            except Exception as e:
                print(f"[WARN] Failed to read {file_path.name}: {e}")

        self._total_docs = len(self.chunks)
        # Compute document frequencies
        for chunk in self.chunks:
            terms = set(self._tokenize(chunk.title + " " + chunk.content))
            for term in terms:
                self._doc_frequencies[term] = self._doc_frequencies.get(term, 0) + 1

    def _parse_markdown_file(self, filename: str, text: str) -> None:
        lines = text.splitlines()
        main_title = filename.replace(".md", "").replace("_", " ").title()
        current_section = "Overview"
        current_content: List[str] = []

        chunk_idx = 0
        for line in lines:
            if line.startswith("# "):
                main_title = line.replace("# ", "").strip()
            elif line.startswith("## "):
                if current_content:
                    self.chunks.append(
                        PolicyChunk(
                            chunk_id=f"{filename}#{chunk_idx}",
                            source_file=filename,
                            title=main_title,
                            section=current_section,
                            content="\n".join(current_content).strip(),
                        )
                    )
                    chunk_idx += 1
                    current_content = []
                current_section = line.replace("## ", "").strip()
            else:
                current_content.append(line)

        if current_content:
            self.chunks.append(
                PolicyChunk(
                    chunk_id=f"{filename}#{chunk_idx}",
                    source_file=filename,
                    title=main_title,
                    section=current_section,
                    content="\n".join(current_content).strip(),
                )
            )

    FINANCIAL_SYNONYMS = {
        "earning": ["income", "capacity", "salary"],
        "earnings": ["income", "capacity", "salary"],
        "earns": ["income", "capacity", "salary"],
        "allocate": ["loan", "amount", "tier", "limit", "classification"],
        "allocating": ["loan", "amount", "tier", "limit"],
        "allocation": ["loan", "amount", "tier", "limit"],
        "borrow": ["loan", "amount", "debt"],
        "borrowing": ["loan", "amount", "debt"],
        "qualify": ["minimum", "requirements", "tier", "score"],
        "rejection": ["reject", "adverse", "denial", "decline"],
        "decline": ["reject", "adverse", "denial", "ceiling"],
        "shock": ["macroeconomic", "rate", "treasury", "yield", "tighten"],
        "fraud": ["synthetic", "ring", "identity", "cluster", "network"],
        "rules": ["policy", "guidelines", "requirements", "limits"],
        "rule": ["policy", "guidelines", "requirements", "limits"],
    }

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) > 2]

    def retrieve(self, query: str, top_k: int = 3) -> List[PolicyChunk]:
        """
        Retrieve top_k most relevant policy excerpts using BM25-style term weighting
        with domain concept expansion and header boosting.
        """
        if not self.chunks:
            self._load_knowledge_base()
        if not self.chunks:
            return []

        base_terms = self._tokenize(query)
        if not base_terms:
            return self.chunks[:top_k]

        # Expand query with domain synonyms
        query_terms: Dict[str, float] = {}
        for term in base_terms:
            query_terms[term] = 1.0
            if term in self.FINANCIAL_SYNONYMS:
                for syn in self.FINANCIAL_SYNONYMS[term]:
                    if syn not in query_terms:
                        query_terms[syn] = 0.65

        scored_chunks: List[PolicyChunk] = []
        k1 = 1.5
        b = 0.75
        avg_dl = sum(len(self._tokenize(c.content)) for c in self.chunks) / max(len(self.chunks), 1)

        for chunk in self.chunks:
            doc_terms = self._tokenize(chunk.title + " " + chunk.section + " " + chunk.content)
            doc_len = len(doc_terms)
            score = 0.0

            # Match terms
            for term, weight in query_terms.items():
                tf = doc_terms.count(term)
                if tf > 0:
                    df = self._doc_frequencies.get(term, 1)
                    idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_dl, 1e-6)))
                    term_score = idf * (numerator / denominator) * weight

                    # Boost matches in Section header or Document Title
                    if term in chunk.section.lower():
                        term_score *= 2.5
                    if term in chunk.title.lower():
                        term_score *= 1.8

                    score += term_score

            if score > 0:
                scored_chunks.append(
                    PolicyChunk(
                        chunk_id=chunk.chunk_id,
                        source_file=chunk.source_file,
                        title=chunk.title,
                        section=chunk.section,
                        content=chunk.content,
                        score=round(score, 4),
                    )
                )

        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:top_k] if scored_chunks else self.chunks[:top_k]


# Global singleton instance
_GLOBAL_RETRIEVER: Optional[PolicyRetriever] = None


def get_policy_retriever() -> PolicyRetriever:
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = PolicyRetriever()
    return _GLOBAL_RETRIEVER
