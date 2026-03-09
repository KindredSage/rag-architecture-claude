"""
pdf_parser.py

Replaces raw PyMuPDF + Vision LLM with a structured markdown pipeline.

Strategy:
  1. Try docling first  — best for complex PDFs (tables, columns, mixed layouts)
  2. Fallback to pymupdf4llm — fast, good for clean/digital PDFs
  3. Fallback to Vision LLM — only for scanned/image-only PDFs

This gives you structured Markdown with headers intact,
which is what hierarchical chunking depends on.
"""

import re
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ParseMethod(Enum):
    DOCLING = "docling"
    PYMUPDF4LLM = "pymupdf4llm"
    VISION_LLM = "vision_llm"


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    markdown: str                    # Full structured markdown
    method_used: ParseMethod
    page_count: int
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Parser 1: Docling (best for complex PDFs)
# pip install docling
# ─────────────────────────────────────────────
def parse_with_docling(pdf_path: str) -> Optional[str]:
    """
    Docling preserves: headers, tables (as markdown), lists, reading order.
    Handles multi-column layouts, mixed text+image PDFs well.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)

        markdown = result.document.export_to_markdown()

        # Sanity check — docling sometimes returns near-empty for pure image PDFs
        if len(markdown.strip()) < 100:
            logger.warning("Docling returned minimal content, likely image-based PDF")
            return None

        return markdown

    except ImportError:
        logger.warning("docling not installed: pip install docling")
        return None
    except Exception as e:
        logger.error(f"Docling failed: {e}")
        return None


# ─────────────────────────────────────────────
# Parser 2: pymupdf4llm (fast, clean digital PDFs)
# pip install pymupdf4llm
# ─────────────────────────────────────────────
def parse_with_pymupdf4llm(pdf_path: str) -> Optional[str]:
    """
    pymupdf4llm is NOT the same as raw PyMuPDF.
    It uses heuristics to reconstruct headings, lists, and reading order
    and exports clean LLM-ready markdown — far better than fitz.get_text().
    """
    try:
        import pymupdf4llm

        markdown = pymupdf4llm.to_markdown(pdf_path)

        if len(markdown.strip()) < 100:
            return None

        return markdown

    except ImportError:
        logger.warning("pymupdf4llm not installed: pip install pymupdf4llm")
        return None
    except Exception as e:
        logger.error(f"pymupdf4llm failed: {e}")
        return None


# ─────────────────────────────────────────────
# Parser 3: Vision LLM (scanned / image PDFs only)
# Keep your existing Vision LLM but add structure prompting
# ─────────────────────────────────────────────
def parse_with_vision_llm(pdf_path: str, llm_client) -> Optional[str]:
    """
    Your existing Vision LLM path — but with a structured output prompt.
    The key change: instruct it to output Markdown with headers.
    This is what makes hierarchical chunking work downstream.
    """
    import base64
    import fitz  # PyMuPDF — only for image extraction here

    VISION_PROMPT = """
    You are a precise document parser. Convert this PDF page to clean Markdown.

    Rules:
    - Use # for document title, ## for major sections, ### for subsections
    - Preserve ALL text exactly — do not summarize or paraphrase
    - Format tables using Markdown table syntax
    - Format lists using - or 1. 2. 3.
    - Separate distinct sections with a blank line
    - Do NOT add commentary or explanation
    - Output ONLY the Markdown content
    """

    try:
        doc = fitz.open(pdf_path)
        all_pages_markdown = []

        for page_num, page in enumerate(doc):
            # Render page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for quality
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode()

            # Call your Vision LLM
            response = llm_client.messages.create(
                model="claude-opus-4-5",  # or your preferred vision model
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64
                            }
                        },
                        {"type": "text", "text": VISION_PROMPT}
                    ]
                }]
            )

            page_markdown = response.content[0].text
            all_pages_markdown.append(f"<!-- page:{page_num + 1} -->\n{page_markdown}")

        doc.close()
        return "\n\n".join(all_pages_markdown)

    except Exception as e:
        logger.error(f"Vision LLM failed: {e}")
        return None


# ─────────────────────────────────────────────
# Main parser with automatic fallback chain
# ─────────────────────────────────────────────
def parse_pdf(
    pdf_path: str,
    doc_id: str,
    vision_llm_client=None
) -> ParsedDocument:
    """
    Waterfall parser:
      docling → pymupdf4llm → Vision LLM

    Each fallback is triggered only if the previous returns None
    or insufficient content.
    """
    path = Path(pdf_path)
    filename = path.name

    # Attempt 1: Docling
    markdown = parse_with_docling(pdf_path)
    method = ParseMethod.DOCLING

    # Attempt 2: pymupdf4llm
    if markdown is None:
        logger.info(f"Falling back to pymupdf4llm for {filename}")
        markdown = parse_with_pymupdf4llm(pdf_path)
        method = ParseMethod.PYMUPDF4LLM

    # Attempt 3: Vision LLM (only if client is provided)
    if markdown is None and vision_llm_client is not None:
        logger.info(f"Falling back to Vision LLM for {filename} (likely scanned PDF)")
        markdown = parse_with_vision_llm(pdf_path, vision_llm_client)
        method = ParseMethod.VISION_LLM

    if markdown is None:
        raise ValueError(f"All parsers failed for {filename}. Check if PDF is corrupted.")

    # Page count via fitz (lightweight, just metadata)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
    except Exception:
        page_count = 0

    return ParsedDocument(
        doc_id=doc_id,
        filename=filename,
        markdown=markdown,
        method_used=method,
        page_count=page_count,
    )
