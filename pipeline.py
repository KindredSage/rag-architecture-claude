"""
pipeline.py

Wires parser → chunker → indexer → retriever into one callable pipeline.
This is what your application code calls.

Usage:
    pipeline = RAGPipeline(os_client, pg_conn)
    
    # On PDF upload (async worker):
    pipeline.ingest_pdf("path/to/doc.pdf", doc_id="doc_123")

    # On reviewer approval:
    pipeline.approve_document("doc_123")

    # On user query:
    answer = pipeline.query("What is the refund policy?")
"""

import logging
import anthropic
from opensearchpy import OpenSearch

from pdf_parser import parse_pdf
from chunker import chunk_document, print_chunk_summary
from indexer import RAGIndexer, RAGRetriever, BGEEmbedder

logger = logging.getLogger(__name__)


RAG_SYSTEM_PROMPT = """
You are a precise document assistant. Answer questions using ONLY the provided source context.

Rules:
- After every factual claim, add a citation like [SOURCE 1] or [SOURCE 2]
- If the context doesn't contain the answer, respond exactly: "NOT_IN_DOCS"
- Never infer, assume, or use knowledge outside the provided context
- Be concise and direct
"""


class RAGPipeline:
    def __init__(self, os_client: OpenSearch, vision_llm_client=None):
        self.embedder = BGEEmbedder()
        self.indexer = RAGIndexer(os_client, self.embedder)
        self.retriever = RAGRetriever(os_client, self.embedder)
        self.vision_llm_client = vision_llm_client
        self.llm = anthropic.Anthropic()

    # ── Ingest ────────────────────────────────
    def ingest_pdf(self, pdf_path: str, doc_id: str) -> dict:
        """
        Full ingest pipeline for one PDF.
        Call this from your async worker (Celery, RQ, etc.) after upload.
        
        approved=False by default — document won't appear in queries
        until approve_document() is called.
        """
        logger.info(f"Ingesting {pdf_path} as doc_id={doc_id}")

        # 1. Parse PDF → structured markdown
        parsed = parse_pdf(
            pdf_path,
            doc_id=doc_id,
            vision_llm_client=self.vision_llm_client
        )
        logger.info(f"Parsed with {parsed.method_used.value}, {parsed.page_count} pages")

        # 2. Chunk markdown → hierarchical parent/child chunks
        chunks = chunk_document(parsed)
        print_chunk_summary(chunks)  # Remove in production

        # 3. Index into OpenSearch (approved=False)
        self.indexer.index_document(chunks, approved=False)

        return {
            "doc_id": doc_id,
            "parse_method": parsed.method_used.value,
            "page_count": parsed.page_count,
            "parent_chunks": len(chunks.parents),
            "child_chunks": len(chunks.children),
        }

    # ── Approval (called from your reviewer endpoint) ──
    def approve_document(self, doc_id: str):
        """
        Call this when a reviewer approves a document in your PG-backed UI.
        PG is already updated by your app — this syncs the approved flag to OpenSearch.
        """
        self.indexer.approve_document(doc_id)

    # ── Query ─────────────────────────────────
    def query(
        self,
        user_query: str,
        top_k: int = 5,
        min_score: float = 0.65,
    ) -> dict:
        """
        Full query pipeline:
          1. Retrieve relevant parent chunks (approved only)
          2. Confidence gate
          3. LLM generation with citation grounding
          4. Citation validation
        """
        # 1. Retrieve
        retrieval = self.retriever.retrieve(
            query=user_query,
            top_k_children=20,
            top_k_parents=top_k,
            min_score=min_score,
        )

        # 2. Confidence gate — no relevant docs found
        if retrieval["below_threshold"]:
            return {
                "answer": "I couldn't find relevant information in the approved documents.",
                "sources": [],
                "max_score": retrieval["max_score"],
                "gated": True,
            }

        # 3. Build context
        context = self.retriever.build_context_for_llm(retrieval)

        # 4. LLM generation
        user_message = f"""Context:
{context}

Question: {user_query}"""

        response = self.llm.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        answer = response.content[0].text

        # 5. Handle "not in docs" response from LLM
        if "NOT_IN_DOCS" in answer:
            return {
                "answer": "The approved documents don't contain enough information to answer this question.",
                "sources": [],
                "max_score": retrieval["max_score"],
                "gated": False,
                "llm_gated": True,
            }

        # 6. Return answer + source metadata
        sources = [
            {
                "doc_id":  p["doc_id"],
                "heading": p["heading"],
                "score":   round(p["score"], 3),
            }
            for p in retrieval["parents"]
        ]

        return {
            "answer": answer,
            "sources": sources,
            "max_score": round(retrieval["max_score"], 3),
            "gated": False,
            "llm_gated": False,
        }


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os_client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_auth=("admin", "admin"),
        use_ssl=False,
    )

    pipeline = RAGPipeline(os_client)

    # Ingest a PDF (unapproved)
    result = pipeline.ingest_pdf("sample.pdf", doc_id="doc_001")
    print("Ingest result:", result)

    # Simulate reviewer approving in PG, then sync to OpenSearch
    pipeline.approve_document("doc_001")

    # Query
    answer = pipeline.query("What are the payment terms?")
    print("\nAnswer:", answer["answer"])
    print("Sources:", answer["sources"])
    print("Max score:", answer["max_score"])
