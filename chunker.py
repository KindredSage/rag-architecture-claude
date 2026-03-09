"""
chunker.py

Hierarchical chunking: Markdown structure → Parent sections → Child paragraphs.

This is fundamentally different from LangChain's RecursiveCharTextSplitter:
  - RecursiveCharTextSplitter: splits blindly by character count, ignores structure
  - This chunker: uses document headers as natural section boundaries for parents,
    then splits within sections for children

Result:
  Parent chunk → full section with context (~1000-2000 tokens)
  Child chunk  → focused paragraph for precise retrieval (~200-400 tokens)
  Child stores parent_id → at query time, retrieve child, return parent to LLM
"""

import re
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from pdf_parser import ParsedDocument


@dataclass
class ParentChunk:
    chunk_id: str
    doc_id: str
    text: str
    heading: str           # The section heading this chunk belongs to
    heading_level: int     # 1=H1, 2=H2, 3=H3
    token_count: int
    chunk_index: int
    fingerprint: str       # SHA256 for dedup


@dataclass
class ChildChunk:
    chunk_id: str
    doc_id: str
    parent_id: str         # FK → ParentChunk.chunk_id
    text: str
    token_count: int
    chunk_index: int       # Position within parent
    fingerprint: str


@dataclass
class HierarchicalChunks:
    doc_id: str
    parents: list[ParentChunk]
    children: list[ChildChunk]


# ─────────────────────────────────────────────
# Token counting (approximate, no tokenizer needed)
# ─────────────────────────────────────────────
def approx_token_count(text: str) -> int:
    # ~1.3 tokens per word is accurate enough for chunking decisions
    return int(len(text.split()) * 1.3)


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


# ─────────────────────────────────────────────
# Step 1: Split markdown into sections by headers
# ─────────────────────────────────────────────
def split_by_headers(markdown: str) -> list[dict]:
    """
    Splits markdown into sections at every H1/H2/H3 boundary.
    Each section: { heading, heading_level, content }

    Why H3 and not deeper: H4+ usually represents minor formatting
    (callouts, notes) not true content sections. Tune per your docs.
    """
    # Match lines starting with 1–3 # characters
    header_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)

    sections = []
    matches = list(header_pattern.finditer(markdown))

    if not matches:
        # No headers found — treat whole document as one section
        # This happens with some Vision LLM outputs; handle gracefully
        return [{
            "heading": "Document",
            "heading_level": 1,
            "content": markdown.strip()
        }]

    # Extract content between consecutive headers
    for i, match in enumerate(matches):
        heading_level = len(match.group(1))
        heading_text = match.group(2).strip()

        # Content is from end of this header line to start of next header
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[content_start:content_end].strip()

        # Skip empty sections (header with no content)
        if not content:
            continue

        sections.append({
            "heading": heading_text,
            "heading_level": heading_level,
            "content": content,
        })

    return sections


# ─────────────────────────────────────────────
# Step 2: Merge small sections, split large ones
# ─────────────────────────────────────────────
def normalize_sections(
    sections: list[dict],
    min_parent_tokens: int = 100,
    max_parent_tokens: int = 1800,
) -> list[dict]:
    """
    After header-based splitting you often get:
      - Tiny sections (1-2 sentences) → merge with next
      - Giant sections (entire chapters) → split at paragraph boundaries

    This normalizes all sections into the ~100–1800 token parent range.
    """
    normalized = []

    i = 0
    while i < len(sections):
        section = sections[i]
        token_count = approx_token_count(section["content"])

        # Too small → merge forward into next section
        if token_count < min_parent_tokens and i + 1 < len(sections):
            next_section = sections[i + 1]
            merged = {
                "heading": section["heading"],  # Keep first heading
                "heading_level": section["heading_level"],
                "content": section["content"] + "\n\n" + next_section["content"],
            }
            sections[i + 1] = merged  # Replace next with merged
            i += 1
            continue

        # Too large → split at paragraph boundaries
        if token_count > max_parent_tokens:
            sub_sections = split_large_section(
                section, max_tokens=max_parent_tokens
            )
            normalized.extend(sub_sections)
        else:
            normalized.append(section)

        i += 1

    return normalized


def split_large_section(section: dict, max_tokens: int) -> list[dict]:
    """
    Split an oversized section at double-newline paragraph boundaries.
    Each split inherits the parent heading with a continuation marker.
    """
    paragraphs = re.split(r'\n\n+', section["content"])
    sub_sections = []
    current_content = ""
    part_num = 1

    for para in paragraphs:
        candidate = (current_content + "\n\n" + para).strip()
        if approx_token_count(candidate) > max_tokens and current_content:
            sub_sections.append({
                "heading": f"{section['heading']} (part {part_num})",
                "heading_level": section["heading_level"],
                "content": current_content.strip(),
            })
            current_content = para
            part_num += 1
        else:
            current_content = candidate

    if current_content.strip():
        sub_sections.append({
            "heading": f"{section['heading']} (part {part_num})" if part_num > 1 else section["heading"],
            "heading_level": section["heading_level"],
            "content": current_content.strip(),
        })

    return sub_sections


# ─────────────────────────────────────────────
# Step 3: Create child chunks within each parent
# ─────────────────────────────────────────────
def create_child_chunks(
    parent: ParentChunk,
    target_child_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[ChildChunk]:
    """
    Splits a parent section into child chunks at sentence boundaries,
    with a small overlap to avoid cutting context at chunk edges.

    Sentence boundary splitting >> character splitting for retrieval quality.
    """
    # Split into sentences (handles "e.g.", "Dr.", "Fig." edge cases)
    sentence_pattern = re.compile(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
    )
    sentences = sentence_pattern.split(parent.text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    children = []
    current_sentences = []
    current_tokens = 0
    child_index = 0

    # Keep last N overlap sentences to prepend to next child
    overlap_sentences = []

    for sentence in sentences:
        s_tokens = approx_token_count(sentence)

        # If adding this sentence exceeds target, flush current chunk
        if current_tokens + s_tokens > target_child_tokens and current_sentences:
            child_text = " ".join(current_sentences)

            children.append(ChildChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=parent.doc_id,
                parent_id=parent.chunk_id,
                text=child_text,
                token_count=approx_token_count(child_text),
                chunk_index=child_index,
                fingerprint=fingerprint(child_text),
            ))
            child_index += 1

            # Carry over overlap sentences into next child
            overlap_sentences = []
            overlap_token_count = 0
            for s in reversed(current_sentences):
                s_tok = approx_token_count(s)
                if overlap_token_count + s_tok <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_token_count += s_tok
                else:
                    break

            current_sentences = overlap_sentences.copy()
            current_tokens = sum(approx_token_count(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_tokens += s_tokens

    # Flush remaining sentences
    if current_sentences:
        child_text = " ".join(current_sentences)
        children.append(ChildChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=parent.doc_id,
            parent_id=parent.chunk_id,
            text=child_text,
            token_count=approx_token_count(child_text),
            chunk_index=child_index,
            fingerprint=fingerprint(child_text),
        ))

    return children


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def chunk_document(parsed_doc: ParsedDocument) -> HierarchicalChunks:
    """
    Full pipeline:
      ParsedDocument (markdown) → sections → normalize → parents + children

    Usage:
        parsed = parse_pdf("report.pdf", doc_id="doc_123")
        chunks = chunk_document(parsed)
        # chunks.parents → index text+embedding into OpenSearch (parent_text field)
        # chunks.children → index text+embedding into OpenSearch (child_text field, used for knn)
    """
    # 1. Split markdown by headers into raw sections
    raw_sections = split_by_headers(parsed_doc.markdown)

    # 2. Normalize section sizes
    sections = normalize_sections(
        raw_sections,
        min_parent_tokens=100,
        max_parent_tokens=1800,
    )

    parents = []
    children = []

    for section_idx, section in enumerate(sections):
        # Build parent text = heading + content (heading gives LLM context!)
        parent_text = f"{section['heading']}\n\n{section['content']}"

        parent = ParentChunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=parsed_doc.doc_id,
            text=parent_text,
            heading=section["heading"],
            heading_level=section["heading_level"],
            token_count=approx_token_count(parent_text),
            chunk_index=section_idx,
            fingerprint=fingerprint(parent_text),
        )
        parents.append(parent)

        # Create children within this parent
        child_chunks = create_child_chunks(parent)
        children.extend(child_chunks)

    return HierarchicalChunks(
        doc_id=parsed_doc.doc_id,
        parents=parents,
        children=children,
    )


# ─────────────────────────────────────────────
# Debug / inspection helper
# ─────────────────────────────────────────────
def print_chunk_summary(chunks: HierarchicalChunks):
    print(f"\nDocument: {chunks.doc_id}")
    print(f"  Parents: {len(chunks.parents)}")
    print(f"  Children: {len(chunks.children)}")
    print(f"  Avg children/parent: {len(chunks.children)/max(len(chunks.parents),1):.1f}")
    print()
    for p in chunks.parents[:3]:  # Show first 3
        kids = [c for c in chunks.children if c.parent_id == p.chunk_id]
        print(f"  [Parent] '{p.heading}' — {p.token_count} tokens, {len(kids)} children")
        for c in kids[:2]:
            preview = c.text[:80].replace('\n', ' ')
            print(f"    [Child] {c.token_count}t — '{preview}...'")
