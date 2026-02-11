"""
RAG (Retrieval-Augmented Generation) system for Fuat_bot.

Provides PDF text extraction, chunking, indexing, and semantic search capabilities.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from .config import settings


class PDFExtractor:
    """Extract text from PDF files page-by-page with metadata."""

    @staticmethod
    def extract_text(
        pdf_path: Path,
        page_start: int | None = None,
        page_end: int | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            page_start: Starting page number (0-indexed), None for first page
            page_end: Ending page number (0-indexed, inclusive), None for last page
            include_metadata: Whether to include PDF metadata

        Returns:
            Dictionary containing:
                - pages: List of {page_number, text} dicts
                - total_pages: Total number of pages in PDF
                - metadata: PDF metadata (if include_metadata=True)
                - error: Error message (if extraction failed)
        """
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)

            # Determine page range
            start = page_start if page_start is not None else 0
            end = page_end + 1 if page_end is not None else total_pages

            # Validate range
            if start < 0 or end > total_pages or start >= end:
                return {
                    "error": f"Invalid page range: {start}-{end-1} (total pages: {total_pages})"
                }

            # Extract text from pages
            pages = []
            for i in range(start, end):
                page = reader.pages[i]
                text = page.extract_text()
                pages.append({"page_number": i, "text": text})

            result: dict[str, Any] = {
                "pages": pages,
                "total_pages": total_pages,
            }

            # Include metadata if requested
            if include_metadata and reader.metadata:
                metadata_dict = {}
                for key, value in reader.metadata.items():
                    # Convert metadata keys to standard format
                    clean_key = key.strip("/").lower()
                    metadata_dict[clean_key] = str(value) if value else None
                result["metadata"] = metadata_dict

            return result

        except FileNotFoundError:
            return {"error": f"PDF file not found: {pdf_path}"}
        except Exception as e:
            return {"error": f"Failed to extract PDF text: {str(e)}"}


class TextChunker:
    """
    Smart text chunking with overlap and sentence boundaries.

    Uses word-count approximation (1.3 tokens/word) instead of actual tokenization
    for performance. Splits on sentence boundaries to preserve semantic coherence.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size to keep (smaller chunks are merged or dropped)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_pages(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Chunk pages into smaller segments with page tracking.

        Chunks flow across page boundaries naturally - pages are just for citation tracking.

        Args:
            pages: List of {page_number, text} dicts from PDFExtractor

        Returns:
            List of chunks with:
                - text: Chunk text content
                - page_start: First page number in chunk
                - page_end: Last page number in chunk
                - chunk_index: Sequential chunk number
                - token_count: Approximate token count
        """
        chunks = []
        chunk_index = 0

        # Initialize chunk state BEFORE the page loop
        current_chunk = []
        current_tokens = 0
        chunk_start_page = pages[0]["page_number"] if pages else 0
        chunk_end_page = chunk_start_page

        for page_dict in pages:
            page_num = page_dict["page_number"]
            text = page_dict["text"]

            # Split page text into sentences
            sentences = self._split_on_sentences(text)

            for sentence in sentences:
                sentence_tokens = self._estimate_tokens(sentence)

                # Check if adding this sentence exceeds chunk size
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    if current_tokens >= self.min_chunk_size:
                        chunks.append({
                            "text": chunk_text,
                            "page_start": chunk_start_page,
                            "page_end": chunk_end_page,  # Track where chunk actually ended
                            "chunk_index": chunk_index,
                            "token_count": current_tokens,
                        })
                        chunk_index += 1

                    # Start new chunk with overlap
                    # Keep last few sentences for overlap
                    overlap_text = []
                    overlap_tokens = 0
                    for prev_sentence in reversed(current_chunk):
                        prev_tokens = self._estimate_tokens(prev_sentence)
                        if overlap_tokens + prev_tokens <= self.overlap:
                            overlap_text.insert(0, prev_sentence)
                            overlap_tokens += prev_tokens
                        else:
                            break

                    current_chunk = overlap_text
                    current_tokens = overlap_tokens
                    chunk_start_page = page_num  # New chunk starts at current page

                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                chunk_end_page = page_num  # Update end page as we add sentences

        # Save final chunk after all pages processed
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if current_tokens >= self.min_chunk_size:
                chunks.append({
                    "text": chunk_text,
                    "page_start": chunk_start_page,
                    "page_end": chunk_end_page,
                    "chunk_index": chunk_index,
                    "token_count": current_tokens,
                })

        return chunks

    def _split_on_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using simple heuristics.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Basic sentence splitting on common terminators
        # This is a simple approach; could be improved with NLTK or spaCy
        sentence_endings = re.compile(r'([.!?]+[\s\n]+)')
        sentences = sentence_endings.split(text)

        # Merge sentence text with its terminator
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)

        # Handle last sentence if it doesn't end with terminator
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

        return result

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using word count approximation.

        Uses 1.3 tokens per word heuristic (common for English text).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        words = len(text.split())
        return int(words * 1.3)


class DocumentIndexer:
    """High-level document indexing operations."""

    def __init__(self, memory_manager):
        """
        Initialize the document indexer.

        Args:
            memory_manager: MemoryManager instance for storage
        """
        self.memory = memory_manager
        self.extractor = PDFExtractor()
        self.chunker = TextChunker(
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
            min_chunk_size=settings.rag_min_chunk_size,
        )

    def index_document(
        self,
        pdf_path: Path,
        category: str = "documents",
        custom_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Index a single PDF document.

        Args:
            pdf_path: Path to PDF file
            category: Document category for organization
            custom_metadata: Optional custom metadata to attach

        Returns:
            Dictionary containing:
                - document_id: Unique document identifier
                - chunks_created: Number of chunks created
                - indexed_at: Timestamp of indexing
                - error: Error message (if indexing failed)
        """
        try:
            # Extract text from PDF
            extraction_result = self.extractor.extract_text(pdf_path, include_metadata=True)
            if "error" in extraction_result:
                return extraction_result

            pages = extraction_result["pages"]
            total_pages = extraction_result["total_pages"]
            pdf_metadata = extraction_result.get("metadata", {})

            # Chunk the pages
            chunks = self.chunker.chunk_pages(pages)

            if not chunks:
                return {"error": "No chunks created from PDF (possibly empty or too short)"}

            # Generate document ID (hash of file path for consistency)
            document_id = self._generate_document_id(pdf_path)

            # Get document title from metadata or filename
            document_title = pdf_metadata.get("title") or pdf_path.stem

            # Prepare document metadata
            doc_metadata = {
                "document_id": document_id,
                "source_file": str(pdf_path),
                "document_title": document_title,
                "category": category,
                "total_pages": total_pages,
                "chunk_count": len(chunks),
                "indexed_at": datetime.utcnow().isoformat(),
                "pdf_metadata": pdf_metadata,
                "custom_metadata": custom_metadata or {},
            }

            # Store chunks in ChromaDB and metadata in SQLite
            chunks_stored = self.memory.store_document_chunks(
                document_id=document_id,
                chunks=chunks,
                metadata=doc_metadata,
            )

            return {
                "document_id": document_id,
                "chunks_created": chunks_stored,
                "indexed_at": doc_metadata["indexed_at"],
            }

        except Exception as e:
            return {"error": f"Failed to index document: {str(e)}"}

    def index_directory(
        self,
        dir_path: Path,
        category: str = "documents",
        recursive: bool = False,
        pattern: str = "*.pdf",
    ) -> dict[str, Any]:
        """
        Batch index all PDFs in a directory.

        Args:
            dir_path: Path to directory containing PDFs
            category: Category for all documents
            recursive: Whether to search subdirectories
            pattern: Glob pattern for file matching

        Returns:
            Dictionary containing:
                - documents_indexed: Number of successfully indexed documents
                - documents_failed: Number of failed documents
                - results: List of individual indexing results
        """
        if not dir_path.is_dir():
            return {"error": f"Not a directory: {dir_path}"}

        # Find all matching PDF files
        if recursive:
            pdf_files = list(dir_path.rglob(pattern))
        else:
            pdf_files = list(dir_path.glob(pattern))

        if not pdf_files:
            return {"error": f"No PDF files found in {dir_path} with pattern '{pattern}'"}

        # Index each file
        results = []
        success_count = 0
        fail_count = 0

        for pdf_path in pdf_files:
            result = self.index_document(pdf_path, category=category)
            result["file"] = str(pdf_path)

            if "error" in result:
                fail_count += 1
            else:
                success_count += 1

            results.append(result)

        return {
            "documents_indexed": success_count,
            "documents_failed": fail_count,
            "results": results,
        }

    def _generate_document_id(self, pdf_path: Path) -> str:
        """
        Generate a unique document ID from file path.

        Uses hash of absolute path for consistency across indexing runs.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Document ID string
        """
        path_str = str(pdf_path.resolve())
        hash_obj = hashlib.sha256(path_str.encode())
        return f"doc_{hash_obj.hexdigest()[:16]}"


class DocumentSearcher:
    """Search indexed documents with hybrid semantic + keyword search and optional re-ranking."""

    def __init__(self, memory_manager):
        """
        Initialize the document searcher.

        Args:
            memory_manager: MemoryManager instance for retrieval
        """
        self.memory = memory_manager
        self._reranker = None  # Lazy-loaded cross-encoder
        self._bm25_index = None  # Lazy-loaded BM25 index
        self._bm25_documents = None  # Document list for BM25 (matches index order)
        self._bm25_corpus = None  # Tokenized corpus for BM25

    def _build_bm25_index(self, category: str | None = None):
        """
        Build BM25 index from all documents in ChromaDB.

        This is lazy-loaded on first search with use_bm25=True.

        Args:
            category: Optional category filter
        """
        try:
            # Get all chunks from ChromaDB
            all_chunks = self.memory.get_all_document_chunks(category=category)

            if not all_chunks:
                return

            # Build corpus (tokenized documents)
            self._bm25_documents = []
            self._bm25_corpus = []

            for chunk in all_chunks:
                content = chunk.get("content", "")
                # Simple tokenization (lowercase, split on whitespace and punctuation)
                tokens = re.findall(r'\w+', content.lower())

                self._bm25_corpus.append(tokens)
                self._bm25_documents.append(chunk)

            # Build BM25 index
            self._bm25_index = BM25Okapi(self._bm25_corpus)

        except Exception as e:
            print(f"Warning: Failed to build BM25 index: {e}")
            self._bm25_index = None

    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 5,
        min_score: float = 0.0,
        rerank: bool = True,
        use_bm25: bool = True,
        show_chunks: bool = False,
    ) -> dict[str, Any]:
        """
        Search indexed documents using hybrid semantic + keyword search.

        Args:
            query: Search query
            category: Optional category filter
            limit: Number of results to return
            min_score: Minimum similarity score (0-1)
            rerank: Whether to use cross-encoder re-ranking
            use_bm25: Whether to use BM25 keyword search alongside semantic search
            show_chunks: Whether to include detailed chunk information for debugging

        Returns:
            Dictionary containing:
                - results: List of results with content, score, and citation info
                - query: Original query
                - count: Number of results returned
                - debug_info: (if show_chunks=True) Detailed retrieval information
                - error: Error message (if search failed)
        """
        try:
            # Ensure numeric parameters are correct types
            limit = int(limit)
            min_score = float(min_score)

            # Determine retrieval limit (3x for re-ranking, direct otherwise)
            if rerank and settings.rag_rerank_enabled:
                retrieval_limit = min(
                    limit * settings.rag_rerank_multiplier,
                    settings.rag_max_retrieval_limit,
                )
            else:
                retrieval_limit = limit

            # Perform semantic vector search
            semantic_results = self.memory.search_document_chunks(
                query=query,
                category=category,
                limit=retrieval_limit,
            )

            # Perform BM25 keyword search if enabled
            bm25_results = []
            if use_bm25:
                bm25_results = self._bm25_search(query, category, retrieval_limit)

            # Combine results using Reciprocal Rank Fusion if both searches performed
            if use_bm25 and bm25_results:
                raw_results = self._combine_results_rrf(semantic_results, bm25_results, retrieval_limit)
            else:
                raw_results = semantic_results

            if not raw_results:
                return {
                    "results": [],
                    "query": query,
                    "count": 0,
                }

            # Store stage 1 results for debug info
            stage1_results = raw_results.copy() if show_chunks else None

            # Re-rank if enabled
            rerank_applied = False
            if rerank and settings.rag_rerank_enabled and len(raw_results) > limit:
                results = self._rerank_results(query, raw_results, limit)
                rerank_applied = True
            else:
                results = raw_results[:limit]

            # Filter by minimum score
            filtered_count = len(results)
            if min_score > 0:
                results = [r for r in results if float(r.get("score", 0)) >= min_score]

            # Build response
            response = {
                "results": results,
                "query": query,
                "count": len(results),
            }

            # Add debug information if requested
            if show_chunks:
                response["debug_info"] = {
                    "stage1_vector_search": {
                        "retrieval_limit": retrieval_limit,
                        "results_retrieved": len(stage1_results),
                        "chunks": [
                            {
                                "rank": i + 1,
                                "score": float(r.get("score", 0.0)),
                                "document": str(r.get("document_title", "Unknown")),
                                "pages": f"{r.get('page_start', '?')}-{r.get('page_end', '?')}",
                                "chunk_index": int(r.get("chunk_index", 0)) if r.get("chunk_index") not in [None, "?"] else "?",
                                "token_count": int(r.get("token_count", 0)) if r.get("token_count") not in [None, "?"] else "?",
                                "content_preview": str(r.get("content", ""))[:100] + "...",
                            }
                            for i, r in enumerate(stage1_results)
                        ],
                    },
                    "stage2_reranking": {
                        "enabled": rerank and settings.rag_rerank_enabled,
                        "applied": rerank_applied,
                        "model": settings.rag_rerank_model if rerank_applied else None,
                        "results_after_rerank": len(results) if rerank_applied else None,
                    },
                    "filtering": {
                        "min_score_threshold": min_score,
                        "results_before_filter": filtered_count,
                        "results_after_filter": len(results),
                        "filtered_out": filtered_count - len(results),
                    },
                    "final_results": [
                        {
                            "rank": i + 1,
                            "score": float(r.get("score", 0.0)),
                            "document": str(r.get("document_title", "Unknown")),
                            "pages": f"{r.get('page_start', '?')}-{r.get('page_end', '?')}",
                            "chunk_index": int(r.get("chunk_index", 0)) if r.get("chunk_index") not in [None, "?"] else "?",
                            "token_count": int(r.get("token_count", 0)) if r.get("token_count") not in [None, "?"] else "?",
                            "content": str(r.get("content", "")),
                        }
                        for i, r in enumerate(results)
                    ],
                }

            return response

        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    def _bm25_search(self, query: str, category: str | None, limit: int) -> list[dict[str, Any]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            category: Optional category filter
            limit: Number of results to return

        Returns:
            List of search results with BM25 scores
        """
        # Build index if not already built
        if self._bm25_index is None:
            self._build_bm25_index(category)

        if self._bm25_index is None or not self._bm25_documents:
            return []

        # Tokenize query
        query_tokens = re.findall(r'\w+', query.lower())

        # Get BM25 scores
        scores = self._bm25_index.get_scores(query_tokens)

        # Create results with scores
        results = []
        for i, score in enumerate(scores):
            if score > 0:  # Only include documents with non-zero scores
                doc = self._bm25_documents[i].copy()
                doc["score"] = float(score)
                results.append(doc)

        # Sort by score and return top N
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _combine_results_rrf(
        self,
        semantic_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        limit: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Combine semantic and BM25 results using Reciprocal Rank Fusion (RRF).

        RRF formula: RRF_score(d) = sum(1 / (k + rank(d)))
        where k is a constant (typically 60) and rank starts from 1.

        Args:
            semantic_results: Results from semantic vector search
            bm25_results: Results from BM25 keyword search
            limit: Number of final results to return
            k: RRF constant (default: 60)

        Returns:
            Combined and sorted list of results
        """
        # Build rank maps for both result sets
        # Use document content as key for matching
        semantic_ranks = {}
        bm25_ranks = {}

        for rank, result in enumerate(semantic_results, start=1):
            key = result.get("content", "")
            semantic_ranks[key] = rank

        for rank, result in enumerate(bm25_results, start=1):
            key = result.get("content", "")
            bm25_ranks[key] = rank

        # Collect all unique documents
        all_docs = {}
        for result in semantic_results + bm25_results:
            key = result.get("content", "")
            if key not in all_docs:
                all_docs[key] = result.copy()

        # Calculate RRF scores
        rrf_scores = {}
        for key, doc in all_docs.items():
            score = 0.0
            if key in semantic_ranks:
                score += 1.0 / (k + semantic_ranks[key])
            if key in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[key])
            rrf_scores[key] = score
            doc["score"] = score

        # Sort by RRF score and return top N
        combined = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        return combined[:limit]

    def _rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Re-rank results using cross-encoder for improved relevance.

        Args:
            query: Original search query
            results: Initial vector search results
            limit: Number of top results to keep

        Returns:
            Re-ranked list of results
        """
        # Lazy-load cross-encoder
        if self._reranker is None:
            self._init_reranker()

        # Prepare query-document pairs for cross-encoder
        pairs = [(query, r["content"]) for r in results]

        # Get cross-encoder scores
        scores = self._reranker.predict(pairs)

        # Update scores and sort
        for i, result in enumerate(results):
            result["score"] = float(scores[i])

        # Sort by new scores and return top N
        reranked = sorted(results, key=lambda x: x["score"], reverse=True)
        return reranked[:limit]

    def _init_reranker(self):
        """Initialize the cross-encoder model for re-ranking."""
        from sentence_transformers import CrossEncoder

        self._reranker = CrossEncoder(settings.rag_rerank_model)
