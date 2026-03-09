# Scalable RAG Architecture: 1000+ PDFs with Permissions

## The Core Problems You're Solving

| Problem | Root Cause | Solution Direction |
|---|---|---|
| Scale to 1000+ PDFs | Flat vector search degrades at scale | Hierarchical indexing + chunking strategy |
| Invalid/hallucinated responses | LLM fills gaps when context is weak | Confidence gating + citation grounding |
| Doc permission filtering | Approved status lives only in PG | Lightweight metadata sync to OpenSearch |

---

## 1. Chunking Strategy (Most Impactful Fix)

Your current approach likely uses naive fixed-size chunking. At 1000+ docs this causes:
- Retrieval of irrelevant mid-paragraph fragments
- Loss of structural context (headers, sections)
- Duplicate/near-duplicate chunks flooding top-k results

### Use Hierarchical Chunking

```
PDF
 └── Section (large chunk ~1500 tokens)  ← stored as "parent"
      └── Paragraph (small chunk ~300 tokens) ← stored as "child", used for retrieval
```

**How it works:**
1. Embed and index **child chunks** (small = precise retrieval)
2. When a child chunk is a top-k hit, fetch its **parent chunk** (large = rich LLM context)
3. Pass parent context to the LLM, not the child

```python
# Pseudocode
child_hits = opensearch.knn_search(query_embedding, k=10)

# Expand to parent chunks
parent_ids = {hit["parent_id"] for hit in child_hits}
parent_chunks = opensearch.mget(parent_ids)

context = build_context(parent_chunks)
```

**Why:** Small chunks = better semantic match. Large chunks = better LLM answer quality. You get both.

---

## 2. Retrieval Pipeline

### Step 1: Hybrid Search (BM25 + Vector)

Pure vector search misses exact keyword matches. Pure BM25 misses semantic similarity. Use both.

```python
# OpenSearch supports hybrid natively
query = {
  "query": {
    "bool": {
      "should": [
        {"match": {"text": user_query}},              # BM25
        {"knn": {"embedding": {"vector": query_vec, "k": 20}}}  # Vector
      ]
    }
  }
}
```

Combine scores with **Reciprocal Rank Fusion (RRF)**:
```
rrf_score(d) = Σ 1 / (k + rank_i(d))   where k=60
```

### Step 2: Reranking

After hybrid retrieval, rerank top-20 using a cross-encoder:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")  # Pairs well with bge-m3

pairs = [(user_query, chunk["text"]) for chunk in candidates]
scores = reranker.predict(pairs)

top_chunks = sorted(zip(scores, candidates), reverse=True)[:5]
```

**Why reranking:** Your bge-m3 embeddings capture semantic similarity, but a cross-encoder reads both query and chunk together — much higher precision for the final context passed to LLM.

### Step 3: MMR Deduplication

When 1000+ docs exist, top-k results often cluster around near-duplicate content. Use **Maximal Marginal Relevance** to diversify:

```python
def mmr(query_vec, candidate_vecs, top_k=5, lambda_=0.5):
    selected = []
    while len(selected) < top_k:
        mmr_scores = [
            lambda_ * sim(query_vec, c) - (1 - lambda_) * max(sim(c, s) for s in selected or [zeros])
            for c in candidates if c not in selected
        ]
        selected.append(candidates[argmax(mmr_scores)])
    return selected
```

---

## 3. Eliminating Invalid Responses

### A. Confidence Gating

Before returning any response, score how well the retrieved context actually answers the query:

```python
CONFIDENCE_PROMPT = """
Given this context and question, rate how well the context answers the question.
Return JSON: {"score": 0-10, "reason": "..."}

Question: {query}
Context: {context}
"""

confidence = llm(CONFIDENCE_PROMPT)

if confidence["score"] < 6:
    return "I don't have sufficient information in the approved documents to answer this."
```

### B. Citation-Grounded Responses

Force the LLM to cite its sources inline. If it can't cite, it can't claim:

```python
SYSTEM_PROMPT = """
Answer ONLY using the provided context. 
For every claim, add [doc_id:chunk_id] citation.
If the context doesn't contain the answer, say: "Not found in approved documents."
Never infer or assume beyond what's explicitly stated.
"""
```

Then validate post-generation:
```python
def validate_citations(response: str, retrieved_chunks: list) -> bool:
    cited_ids = extract_citation_ids(response)
    valid_ids = {chunk["id"] for chunk in retrieved_chunks}
    return all(cid in valid_ids for cid in cited_ids)
```

### C. Semantic Similarity Fallback Check

If the top retrieved chunk has low cosine similarity to the query, don't even attempt generation:

```python
TOP_K_MIN_SCORE = 0.65  # Tune this per your domain

if max_retrieval_score < TOP_K_MIN_SCORE:
    return "No relevant documents found."
```

---

## 4. The Permission Model — The Clean Answer

### Don't sync all metadata. Sync only one field.

**PG is your source of truth.** OpenSearch is your search index. The mistake to avoid is keeping them "in sync" — that creates a dual-write problem and eventual consistency bugs.

**The clean design:**

```
PostgreSQL                          OpenSearch
─────────────────────────           ─────────────────────────────────
docs table:                         chunk index:
  id                                  chunk_id
  title                               doc_id          ← FK reference
  uploaded_by                         text
  reviewer_id                         embedding
  review_notes                        approved        ← ONLY this field synced
  approved          ──────────►       
  approved_at       (one-way,
  created_at         event-driven)
  ...all other metadata
```

**Why this works:**
- OpenSearch only needs `approved: bool` to filter at query time
- All rich metadata (reviewer, notes, timestamps) stays in PG where it belongs
- Single-field sync = trivially simple to keep consistent
- No messy join queries at retrieval time

### Event-Driven Sync (Not Polling)

```python
# When reviewer approves in your app:
def approve_document(doc_id: str, reviewer_id: str):
    # 1. Update source of truth
    pg.execute("""
        UPDATE docs SET approved=true, approved_at=NOW(), reviewer_id=%s
        WHERE id=%s
    """, [reviewer_id, doc_id])
    
    # 2. Sync the ONE field to OpenSearch
    opensearch.update_by_query({
        "query": {"term": {"doc_id": doc_id}},
        "script": {"source": "ctx._source.approved = true"}
    })
    
    # Done. PG is truth. OpenSearch just has the flag.
```

### Query-Time Filtering

```python
def search(query: str, user_id: str) -> list:
    query_vec = embed(query)
    
    results = opensearch.search({
        "query": {
            "bool": {
                "must": [
                    {"knn": {"embedding": {"vector": query_vec, "k": 20}}}
                ],
                "filter": [
                    {"term": {"approved": True}}   # Hard filter, not scored
                ]
            }
        }
    })
    return results
```

**The `filter` clause in OpenSearch is a hard gate** — unapproved chunks never appear in results, and it doesn't affect relevance scoring. This is the right primitive.

---

## 5. Indexing Pipeline at Scale

### Async Ingestion Queue

Don't process PDFs synchronously. Use a queue:

```
PDF Upload → S3/MinIO → Queue (Redis/SQS) → Worker Pool → OpenSearch
                                                ↓
                                          PG metadata insert
                                          (approved=false)
```

```python
# Worker pseudocode
def process_pdf(doc_id: str, s3_path: str):
    pdf_text = extract_text(s3_path)           # PyMuPDF
    sections = hierarchical_chunk(pdf_text)    # Parent/child split
    
    for section in sections:
        embedding = bge_m3.encode(section.child_text)
        
        opensearch.index({
            "doc_id": doc_id,
            "parent_text": section.parent_text,
            "child_text": section.child_text,
            "embedding": embedding,
            "approved": False,                 # Default until reviewed
            "chunk_index": section.index
        })
    
    pg.execute("UPDATE docs SET indexed=true WHERE id=%s", [doc_id])
```

### Deduplication on Ingest

At 1000+ docs, you'll get duplicate content (same policy doc re-uploaded, etc.):

```python
import hashlib

def chunk_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

# Before indexing, check for exact duplicate chunks
existing = opensearch.search({"query": {"term": {"fingerprint": fp}}})
if existing["hits"]["total"]["value"] > 0:
    skip()  # Don't index duplicates
```

---

## 6. Complete Query Flow

```
User Query
    │
    ▼
[1] Embed with bge-m3
    │
    ▼
[2] OpenSearch hybrid search
    + filter: approved=true        ← Permission gate
    + knn + BM25
    top-20 child chunks
    │
    ▼
[3] Fetch parent chunks for top-20
    │
    ▼
[4] Cross-encoder rerank
    top-5 parent chunks
    │
    ▼
[5] MMR deduplication
    │
    ▼
[6] Confidence gate
    score < 0.65? → "No relevant docs found"
    │
    ▼
[7] LLM generation with citation prompt
    │
    ▼
[8] Validate citations post-generation
    │
    ▼
Response with source attribution
```

---

## 7. OpenSearch Index Mapping

```json
{
  "mappings": {
    "properties": {
      "doc_id":       { "type": "keyword" },
      "chunk_id":     { "type": "keyword" },
      "parent_id":    { "type": "keyword" },
      "child_text":   { "type": "text", "analyzer": "english" },
      "parent_text":  { "type": "text", "index": false },
      "embedding":    { 
        "type": "knn_vector", 
        "dimension": 1024,
        "method": { "name": "hnsw", "engine": "nmslib" }
      },
      "approved":     { "type": "boolean" },
      "fingerprint":  { "type": "keyword" },
      "chunk_index":  { "type": "integer" }
    }
  },
  "settings": {
    "index.knn": true,
    "index.knn.space_type": "cosinesimil"
  }
}
```

**Notes:**
- `parent_text` is `index: false` — stored but not searchable (saves index space)
- `approved` is `boolean` — ultra-fast filter, no scoring overhead
- `fingerprint` is `keyword` — exact match for dedup checks

---

## Summary: What to Build, In Order

1. **Fix chunking first** — hierarchical parent/child gives the biggest quality jump
2. **Add the `approved` field to OpenSearch** — sync it event-driven from PG on approval
3. **Add hybrid search + reranker** — BM25 + bge-m3 + bge-reranker-v2-m3
4. **Add confidence gating** — return "not found" rather than hallucinate
5. **Add citation grounding** — LLM must cite, validate citations post-gen
6. **Move ingestion to async queue** — never block on PDF processing
7. **Add fingerprint dedup** — prevents index bloat as docs accumulate
