"""
indexer.py

Indexes hierarchical chunks into OpenSearch and retrieves
parent context at query time using child embeddings.

Key design:
  - Only children are embedded + used for knn retrieval
  - Parents are stored (not embedded) and fetched by parent_id after retrieval
  - approved flag lives on both child docs (for fast filter) and synced from PG
"""

import json
import logging
from FlagEmbedding import BGEM3FlagModel
from opensearchpy import OpenSearch, helpers
from chunker import HierarchicalChunks, ParentChunk, ChildChunk

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Index mapping
# ─────────────────────────────────────────────
PARENT_INDEX = "rag_parents"
CHILD_INDEX  = "rag_children"

PARENT_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id":        {"type": "keyword"},
            "chunk_id":      {"type": "keyword"},
            "text":          {"type": "text", "index": False},  # stored, not searched
            "heading":       {"type": "text"},
            "heading_level": {"type": "integer"},
            "token_count":   {"type": "integer"},
            "chunk_index":   {"type": "integer"},
            "fingerprint":   {"type": "keyword"},
            "approved":      {"type": "boolean"},   # Synced from PG on approval
        }
    },
    "settings": {"number_of_shards": 2, "number_of_replicas": 1}
}

CHILD_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id":      {"type": "keyword"},
            "chunk_id":    {"type": "keyword"},
            "parent_id":   {"type": "keyword"},    # FK → parent chunk_id
            "text":        {"type": "text", "analyzer": "english"},  # for BM25
            "embedding":   {
                "type": "knn_vector",
                "dimension": 1024,                 # bge-m3 output dim
                "method": {
                    "name": "hnsw",
                    "engine": "nmslib",
                    "parameters": {"ef_construction": 128, "m": 16}
                }
            },
            "token_count":   {"type": "integer"},
            "chunk_index":   {"type": "integer"},
            "fingerprint":   {"type": "keyword"},
            "approved":      {"type": "boolean"},  # Denormalized from parent for fast filter
        }
    },
    "settings": {"index.knn": True, "number_of_shards": 2, "number_of_replicas": 1}
}


# ─────────────────────────────────────────────
# Embedder (bge-m3 — your existing model)
# ─────────────────────────────────────────────
class BGEEmbedder:
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of 1024-dim vectors."""
        output = self.model.encode(
            texts,
            batch_size=32,
            max_length=512,
            return_dense=True,
        )
        return output["dense_vecs"].tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]


# ─────────────────────────────────────────────
# Indexer
# ─────────────────────────────────────────────
class RAGIndexer:
    def __init__(self, os_client: OpenSearch, embedder: BGEEmbedder):
        self.os = os_client
        self.embedder = embedder
        self._ensure_indices()

    def _ensure_indices(self):
        for index, mapping in [
            (PARENT_INDEX, PARENT_MAPPING),
            (CHILD_INDEX, CHILD_MAPPING),
        ]:
            if not self.os.indices.exists(index=index):
                self.os.indices.create(index=index, body=mapping)
                logger.info(f"Created index: {index}")

    # ── Deduplication check ──────────────────
    def _chunk_exists(self, fingerprint: str, index: str) -> bool:
        """Returns True if a chunk with this fingerprint already exists."""
        res = self.os.count(
            index=index,
            body={"query": {"term": {"fingerprint": fingerprint}}}
        )
        return res["count"] > 0

    # ── Index a full document's chunks ───────
    def index_document(
        self,
        chunks: HierarchicalChunks,
        approved: bool = False,   # Default false — needs reviewer approval
    ):
        """
        Index all parent and child chunks for a document.
        approved=False by default — document appears in search only after approval.
        """
        # 1. Index parents (no embedding — just stored for context retrieval)
        parent_docs = []
        for parent in chunks.parents:
            if self._chunk_exists(parent.fingerprint, PARENT_INDEX):
                logger.debug(f"Skipping duplicate parent chunk: {parent.fingerprint}")
                continue

            parent_docs.append({
                "_index": PARENT_INDEX,
                "_id": parent.chunk_id,
                "_source": {
                    "doc_id":        parent.doc_id,
                    "chunk_id":      parent.chunk_id,
                    "text":          parent.text,
                    "heading":       parent.heading,
                    "heading_level": parent.heading_level,
                    "token_count":   parent.token_count,
                    "chunk_index":   parent.chunk_index,
                    "fingerprint":   parent.fingerprint,
                    "approved":      approved,
                }
            })

        if parent_docs:
            helpers.bulk(self.os, parent_docs)
            logger.info(f"Indexed {len(parent_docs)} parent chunks for doc {chunks.doc_id}")

        # 2. Embed and index children
        # Filter out duplicates first
        new_children = [
            c for c in chunks.children
            if not self._chunk_exists(c.fingerprint, CHILD_INDEX)
        ]

        if not new_children:
            logger.info("All child chunks already indexed (duplicates skipped)")
            return

        # Embed in batches
        texts = [c.text for c in new_children]
        embeddings = self.embedder.embed_batch(texts)

        child_docs = []
        for child, embedding in zip(new_children, embeddings):
            child_docs.append({
                "_index": CHILD_INDEX,
                "_id": child.chunk_id,
                "_source": {
                    "doc_id":      child.doc_id,
                    "chunk_id":    child.chunk_id,
                    "parent_id":   child.parent_id,
                    "text":        child.text,
                    "embedding":   embedding,
                    "token_count": child.token_count,
                    "chunk_index": child.chunk_index,
                    "fingerprint": child.fingerprint,
                    "approved":    approved,
                }
            })

        helpers.bulk(self.os, child_docs)
        logger.info(f"Indexed {len(child_docs)} child chunks for doc {chunks.doc_id}")

    # ── Approval sync (called when reviewer approves in PG) ──
    def approve_document(self, doc_id: str):
        """
        Sync approval from PG → OpenSearch.
        Single field update on all chunks for this doc.
        Called from your approval endpoint — not a scheduled job.
        """
        update_query = {
            "query": {"term": {"doc_id": doc_id}},
            "script": {"source": "ctx._source.approved = true", "lang": "painless"}
        }

        # Update both indices
        self.os.update_by_query(index=PARENT_INDEX, body=update_query)
        self.os.update_by_query(index=CHILD_INDEX, body=update_query)
        logger.info(f"Approved doc {doc_id} in OpenSearch")

    # ── Delete a document ────────────────────
    def delete_document(self, doc_id: str):
        delete_query = {"query": {"term": {"doc_id": doc_id}}}
        self.os.delete_by_query(index=PARENT_INDEX, body=delete_query)
        self.os.delete_by_query(index=CHILD_INDEX, body=delete_query)
        logger.info(f"Deleted doc {doc_id} from OpenSearch")


# ─────────────────────────────────────────────
# Retriever — hybrid search + parent fetch
# ─────────────────────────────────────────────
class RAGRetriever:
    def __init__(self, os_client: OpenSearch, embedder: BGEEmbedder):
        self.os = os_client
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k_children: int = 20,   # Retrieve more children, dedupe to fewer parents
        top_k_parents: int = 5,     # Final context chunks passed to LLM
        min_score: float = 0.65,    # Confidence gate — below this = no answer
    ) -> dict:
        """
        Full retrieval pipeline:
          1. Hybrid search (knn + BM25) on child index — approved only
          2. Deduplicate to unique parent IDs
          3. Fetch parent chunks (full context for LLM)
          4. Return with score metadata for confidence gating

        Returns:
          {
            "parents": [ParentContext, ...],
            "max_score": float,
            "below_threshold": bool
          }
        """
        query_vec = self.embedder.embed_query(query)

        # ── Hybrid search on children ────────
        search_body = {
            "size": top_k_children,
            "query": {
                "bool": {
                    "must": [
                        # Vector similarity
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vec,
                                    "k": top_k_children
                                }
                            }
                        }
                    ],
                    "should": [
                        # BM25 keyword boost (additive to knn score)
                        {"match": {"text": {"query": query, "boost": 0.3}}}
                    ],
                    "filter": [
                        # Hard permission gate — unapproved docs never surface
                        {"term": {"approved": True}}
                    ]
                }
            },
            "_source": ["chunk_id", "parent_id", "doc_id", "text", "token_count"]
        }

        response = self.os.search(index=CHILD_INDEX, body=search_body)
        hits = response["hits"]["hits"]

        if not hits:
            return {"parents": [], "max_score": 0.0, "below_threshold": True}

        max_score = hits[0]["_score"]

        # Below confidence threshold → don't answer
        if max_score < min_score:
            return {"parents": [], "max_score": max_score, "below_threshold": True}

        # ── Deduplicate: group children by parent ────
        # Multiple children may map to the same parent.
        # Keep the highest-scored child hit per parent.
        parent_scores: dict[str, float] = {}
        for hit in hits:
            pid = hit["_source"]["parent_id"]
            score = hit["_score"]
            if pid not in parent_scores or score > parent_scores[pid]:
                parent_scores[pid] = score

        # Sort parents by their best child score, take top_k_parents
        ranked_parent_ids = sorted(
            parent_scores, key=lambda pid: parent_scores[pid], reverse=True
        )[:top_k_parents]

        # ── Fetch parent chunks (full context) ──
        parent_docs = self.os.mget(
            index=PARENT_INDEX,
            body={"ids": ranked_parent_ids}
        )

        parents = []
        for doc in parent_docs["docs"]:
            if doc.get("found"):
                src = doc["_source"]
                parents.append({
                    "chunk_id":  src["chunk_id"],
                    "doc_id":    src["doc_id"],
                    "heading":   src["heading"],
                    "text":      src["text"],
                    "score":     parent_scores[src["chunk_id"]],
                })

        # Re-sort by score (mget doesn't preserve order)
        parents.sort(key=lambda x: x["score"], reverse=True)

        return {
            "parents": parents,
            "max_score": max_score,
            "below_threshold": False,
        }

    def build_context_for_llm(self, retrieval_result: dict) -> str:
        """
        Format retrieved parent chunks into LLM-ready context string.
        Each block includes heading + content + citation marker.
        """
        if retrieval_result["below_threshold"]:
            return ""

        context_parts = []
        for i, parent in enumerate(retrieval_result["parents"]):
            context_parts.append(
                f"[SOURCE {i+1} | doc:{parent['doc_id']} | chunk:{parent['chunk_id']}]\n"
                f"{parent['text']}\n"
                f"[END SOURCE {i+1}]"
            )

        return "\n\n---\n\n".join(context_parts)
