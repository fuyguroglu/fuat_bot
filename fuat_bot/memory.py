"""
Memory management system for Fuat_bot.

This module implements a multi-tiered memory system:
1. Working Memory (JSON files) - Recent context across sessions
2. Long-term Memory (SQLite) - Persistent facts and preferences
3. Semantic Memory (ChromaDB) - Embeddings for semantic search

The memory system provides:
- Manual control via tools (LLM decides what to remember)
- Automatic injection into system prompt
- Multiple storage backends for educational comparison
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import json
import sqlite3
import uuid

from .config import settings


@dataclass
class Memory:
    """Represents a single memory entry.

    Attributes:
        id: Unique identifier (UUID)
        content: The actual memory content/text
        memory_type: Type of memory (working, fact, semantic)
        category: Optional category for organization
        timestamp: When the memory was created
        metadata: Additional metadata (flexible dict)
    """
    id: str
    content: str
    memory_type: Literal["working", "fact", "semantic"]
    category: str | None
    timestamp: datetime
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class MemoryManager:
    """Manages all memory operations across three storage backends.

    Storage Backends:
    - JSON files: For working memory (recent, temporary context)
    - SQLite: For long-term facts (queryable, structured)
    - ChromaDB: For semantic memory (similarity search)

    The manager handles:
    - Storing memories in appropriate backend
    - Retrieving memories with various filters
    - Formatting memories for system prompt injection
    """

    def __init__(self, memory_dir: Path):
        """Initialize memory manager with all backends.

        Args:
            memory_dir: Directory for all memory storage
        """
        self.memory_dir = memory_dir
        self.working_dir = memory_dir / "working"
        self.db_path = memory_dir / "longterm.db"

        # Create directories
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._init_sqlite()

        # ChromaDB initialized lazily in Phase 4
        self._chroma_client = None
        self._chroma_collection = None

    # ========================================================================
    # Working Memory (JSON) - Phase 1
    # ========================================================================

    def store_working_memory(self, content: str, category: str = "general") -> str:
        """Store information in working memory (JSON file).

        Working memory is for recent, temporary context that's relevant
        to ongoing conversations but doesn't need long-term persistence.

        Storage format: Individual JSON files named with timestamp and ID.

        Args:
            content: The information to remember
            category: Category for organization (e.g., 'task', 'preference')

        Returns:
            Memory ID (UUID)
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Create filename with timestamp and short ID for easy sorting
        memory_file = self.working_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{memory_id[:8]}.json"

        memory_data = {
            "id": memory_id,
            "content": content,
            "memory_type": "working",
            "category": category,
            "timestamp": timestamp.isoformat(),
            "metadata": {}
        }

        memory_file.write_text(json.dumps(memory_data, indent=2))
        return memory_id

    def get_working_memories(
        self,
        limit: int = 10,
        category: str | None = None
    ) -> list[Memory]:
        """Get recent working memories from JSON files.

        Loads memories from JSON files, sorted by timestamp (newest first).
        Optionally filters by category.

        Args:
            limit: Maximum number of memories to return
            category: Optional category filter

        Returns:
            List of Memory objects, sorted by timestamp (newest first)
        """
        memories = []

        # Get all JSON files, sorted by filename (which includes timestamp)
        files = sorted(self.working_dir.glob("*.json"), reverse=True)

        # Load files until we have enough matching memories
        # Load extra to account for category filtering
        for file in files[:limit * 2]:
            try:
                data = json.loads(file.read_text())

                # Filter by category if specified
                if category and data.get("category") != category:
                    continue

                # Convert to Memory object
                memories.append(Memory(
                    id=data["id"],
                    content=data["content"],
                    memory_type=data["memory_type"],
                    category=data.get("category"),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    metadata=data.get("metadata", {})
                ))

                # Stop if we have enough
                if len(memories) >= limit:
                    break

            except (json.JSONDecodeError, KeyError) as e:
                # Skip corrupted files
                print(f"Warning: Skipping corrupted memory file {file}: {e}")
                continue

        return memories

    # ========================================================================
    # Long-term Memory (SQLite) - Phase 3
    # ========================================================================

    def _init_sqlite(self):
        """Initialize SQLite database with schema.

        Creates tables:
        - facts: Main storage for long-term facts and knowledge
        - indexed_documents: Registry of indexed PDF documents for RAG
        - Indexes for efficient querying by category and date
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_category
            ON facts(category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_created
            ON facts(created_at)
        """)

        # Create indexed documents table (for RAG system)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexed_documents (
                document_id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                document_title TEXT,
                category TEXT,
                total_pages INTEGER,
                chunk_count INTEGER,
                indexed_at TIMESTAMP,
                pdf_metadata TEXT,
                custom_metadata TEXT
            )
        """)

        # Create indexes for document queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_docs_category
            ON indexed_documents(category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_docs_source
            ON indexed_documents(source_file)
        """)

        conn.commit()
        conn.close()

    def store_fact(self, content: str, category: str = "general", metadata: dict | None = None) -> str:
        """Store long-term fact in SQLite database.

        Long-term memory is for persistent facts, preferences, and knowledge
        that should be retained across many sessions.

        Args:
            content: The fact to remember
            category: Category for organization
            metadata: Optional additional metadata

        Returns:
            Memory ID (UUID)
        """
        memory_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO facts (id, content, category, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (memory_id, content, category, json.dumps(metadata or {}))
        )

        conn.commit()
        conn.close()

        return memory_id

    def search_facts(
        self,
        query: str | None = None,
        category: str | None = None,
        limit: int = 20
    ) -> list[Memory]:
        """Search long-term facts with optional filters.

        Supports:
        - Full-text search (SQL LIKE) in content
        - Category filtering
        - Limit on number of results

        Args:
            query: Optional text to search for in content
            category: Optional category filter
            limit: Maximum number of results

        Returns:
            List of Memory objects matching criteria
        """
        # Ensure limit is a proper integer
        limit = int(limit) if limit is not None else 20

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build SQL query dynamically
        sql = """
            SELECT id, content, category, created_at, metadata
            FROM facts
            WHERE 1=1
        """
        params = []

        # Add category filter
        if category:
            sql += " AND category = ?"
            params.append(category)

        # Add text search
        if query:
            sql += " AND content LIKE ?"
            params.append(f"%{query}%")

        # Sort and limit - using integer directly to avoid parameter binding issues
        sql += f" ORDER BY created_at DESC LIMIT {limit}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        # Convert to Memory objects
        memories = []
        for row in rows:
            try:
                memories.append(Memory(
                    id=row[0],
                    content=row[1],
                    memory_type="fact",
                    category=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4])
                ))
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Warning: Skipping corrupted fact {row[0]}: {e}")
                continue

        return memories

    # ========================================================================
    # Semantic Memory (ChromaDB) - Phase 4
    # ========================================================================

    def _init_chromadb(self):
        """Lazy initialization of ChromaDB client and collection.

        Creates a persistent ChromaDB client and collection for semantic memory.
        Only imports ChromaDB when actually needed.

        Returns:
            ChromaDB collection
        """
        if self._chroma_client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                # Create persistent client
                self._chroma_client = chromadb.PersistentClient(
                    path=str(self.memory_dir / "chromadb"),
                    settings=ChromaSettings(anonymized_telemetry=False)
                )

                # Get or create collection
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name="semantic_memories",
                    metadata={"description": "Semantic memory with embeddings"}
                )
            except ImportError:
                raise ImportError(
                    "ChromaDB not installed. Install with: pip install chromadb sentence-transformers"
                )

        return self._chroma_collection

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using configured provider.

        Supports two providers:
        - sentence-transformers: Local embedding model (no API calls)
        - openai: OpenAI embedding API (requires API key)

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if settings.embedding_provider == "sentence-transformers":
            # Use local sentence-transformers model
            if not hasattr(self, '_embedding_model'):
                try:
                    from sentence_transformers import SentenceTransformer
                    self._embedding_model = SentenceTransformer(settings.embedding_model)
                except ImportError:
                    raise ImportError(
                        "sentence-transformers not installed. Install with: pip install sentence-transformers"
                    )

            return self._embedding_model.encode(text).tolist()

        elif settings.embedding_provider == "openai":
            # Use OpenAI embeddings API
            try:
                import openai
            except ImportError:
                raise ImportError("openai not installed. Install with: pip install openai")

            client = openai.OpenAI(api_key=settings.openai_api_key)
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding

        else:
            raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

    def store_semantic_memory(
        self,
        content: str,
        category: str = "general",
        metadata: dict | None = None
    ) -> str:
        """Store memory with embeddings for semantic search.

        Semantic memory allows finding related information based on meaning,
        not just keyword matching.

        Args:
            content: The information to remember
            category: Category for organization
            metadata: Optional additional metadata

        Returns:
            Memory ID (UUID)
        """
        memory_id = str(uuid.uuid4())
        collection = self._init_chromadb()

        # Generate embedding
        embedding = self._get_embedding(content)

        # Prepare metadata
        full_metadata = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }

        # Store in ChromaDB
        collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[full_metadata]
        )

        return memory_id

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None
    ) -> list[Memory]:
        """Search memories by semantic similarity.

        Uses vector similarity to find memories that are conceptually related
        to the query, even if they don't share exact keywords.

        Args:
            query: Search query (will be embedded and compared)
            limit: Maximum number of results
            category: Optional category filter

        Returns:
            List of Memory objects, sorted by relevance
        """
        collection = self._init_chromadb()

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Build filter
        where_filter = {"category": category} if category else None

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter
        )

        # Convert to Memory objects
        memories = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                memories.append(Memory(
                    id=results['ids'][0][i],
                    content=doc,
                    memory_type="semantic",
                    category=metadata.get('category'),
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    metadata=metadata
                ))

        return memories

    # ========================================================================
    # Document RAG (ChromaDB + SQLite)
    # ========================================================================

    def _init_document_collection(self):
        """Initialize ChromaDB collection for document chunks.

        Creates a separate collection from semantic_memories for RAG documents.
        This keeps document chunks isolated from regular semantic memories.

        Returns:
            ChromaDB collection for document chunks
        """
        # Initialize ChromaDB client if needed
        if self._chroma_client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self._chroma_client = chromadb.PersistentClient(
                    path=str(self.memory_dir / "chromadb"),
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            except ImportError:
                raise ImportError(
                    "ChromaDB not installed. Install with: pip install chromadb sentence-transformers"
                )

        # Get or create document chunks collection
        return self._chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "PDF document chunks for RAG"}
        )

    def store_document_chunks(
        self,
        document_id: str,
        chunks: list[dict[str, Any]],
        metadata: dict[str, Any]
    ) -> int:
        """Store document chunks in ChromaDB and metadata in SQLite.

        This method handles both storage backends for document indexing:
        1. ChromaDB: Stores chunks with embeddings for semantic search
        2. SQLite: Stores document metadata in indexed_documents table

        Args:
            document_id: Unique document identifier
            chunks: List of chunk dicts from TextChunker
            metadata: Document metadata from DocumentIndexer

        Returns:
            Number of chunks stored
        """
        # First, check if document already exists and delete if so
        existing = self._get_document_registry_entry(document_id)
        if existing:
            self.delete_document(document_id)

        # Get document collection
        collection = self._init_document_collection()

        # Prepare data for ChromaDB batch insert
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # Generate unique chunk ID
            chunk_id = f"{document_id}_chunk_{chunk['chunk_index']}"

            # Generate embedding for chunk text
            embedding = self._get_embedding(chunk["text"])

            # Prepare chunk metadata
            chunk_metadata = {
                "document_id": document_id,
                "source_file": metadata["source_file"],
                "document_title": metadata["document_title"],
                "chunk_index": chunk["chunk_index"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "category": metadata["category"],
                "content_type": "document_chunk",
                "indexed_at": metadata["indexed_at"],
                "token_count": chunk["token_count"],
            }

            # Add PDF metadata fields if present
            if metadata.get("pdf_metadata"):
                for key, value in metadata["pdf_metadata"].items():
                    chunk_metadata[f"pdf_{key}"] = value

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk["text"])
            metadatas.append(chunk_metadata)

        # Batch insert to ChromaDB (much faster than one-by-one)
        # Process in batches of 32 for memory efficiency
        batch_size = 32
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

        # Store document metadata in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO indexed_documents (
                document_id, source_file, document_title, category,
                total_pages, chunk_count, indexed_at, pdf_metadata, custom_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                metadata["source_file"],
                metadata["document_title"],
                metadata["category"],
                metadata["total_pages"],
                len(chunks),
                metadata["indexed_at"],
                json.dumps(metadata.get("pdf_metadata", {})),
                json.dumps(metadata.get("custom_metadata", {}))
            )
        )

        conn.commit()
        conn.close()

        return len(chunks)

    def search_document_chunks(
        self,
        query: str,
        category: str | None = None,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search document chunks by semantic similarity.

        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum number of results

        Returns:
            List of result dicts with:
                - content: Chunk text
                - score: Similarity score (distance from ChromaDB)
                - document_id: Source document ID
                - document_title: Document title
                - source_file: Path to source PDF
                - page_start: First page in chunk
                - page_end: Last page in chunk
                - chunk_index: Chunk sequence number
                - token_count: Approximate token count for chunk
        """
        collection = self._init_document_collection()

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Build filter for category
        if category:
            where_filter = {
                "$and": [
                    {"category": category},
                    {"content_type": "document_chunk"}
                ]
            }
        else:
            where_filter = {"content_type": "document_chunk"}

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter
        )

        # Convert to result dicts
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                # ChromaDB returns distance, convert to similarity score
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity = float(1 / (1 + float(distance)))  # Convert distance to similarity

                search_results.append({
                    "content": doc,
                    "score": float(similarity),
                    "document_id": str(metadata["document_id"]),
                    "document_title": str(metadata["document_title"]),
                    "source_file": str(metadata["source_file"]),
                    "page_start": int(metadata["page_start"]),
                    "page_end": int(metadata["page_end"]),
                    "chunk_index": int(metadata["chunk_index"]),
                    "token_count": int(metadata.get("token_count", 0)),
                })

        return search_results

    def get_all_document_chunks(self, category: str | None = None) -> list[dict[str, Any]]:
        """Get all document chunks (for BM25 indexing).

        Args:
            category: Optional category filter

        Returns:
            List of all chunks with content and metadata
        """
        collection = self._init_document_collection()

        # Build filter for category
        if category:
            where_filter = {
                "$and": [
                    {"category": category},
                    {"content_type": "document_chunk"}
                ]
            }
        else:
            where_filter = {"content_type": "document_chunk"}

        # Get all chunks (no limit)
        results = collection.get(
            where=where_filter
        )

        # Convert to result dicts
        all_chunks = []
        if results['documents']:
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i]

                all_chunks.append({
                    "content": doc,
                    "document_id": str(metadata.get("document_id", "")),
                    "document_title": str(metadata.get("document_title", "Unknown")),
                    "source_file": str(metadata.get("source_file", "")),
                    "page_start": int(metadata.get("page_start", 0)) if metadata.get("page_start") is not None else 0,
                    "page_end": int(metadata.get("page_end", 0)) if metadata.get("page_end") is not None else 0,
                    "chunk_index": int(metadata.get("chunk_index", 0)) if metadata.get("chunk_index") is not None else 0,
                    "token_count": int(metadata.get("token_count", 0)) if metadata.get("token_count") is not None else 0,
                })

        return all_chunks

    def list_documents(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all indexed documents.

        Args:
            category: Optional category filter

        Returns:
            List of document metadata dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if category:
            cursor.execute(
                """
                SELECT document_id, source_file, document_title, category,
                       total_pages, chunk_count, indexed_at, pdf_metadata, custom_metadata
                FROM indexed_documents
                WHERE category = ?
                ORDER BY indexed_at DESC
                """,
                (category,)
            )
        else:
            cursor.execute(
                """
                SELECT document_id, source_file, document_title, category,
                       total_pages, chunk_count, indexed_at, pdf_metadata, custom_metadata
                FROM indexed_documents
                ORDER BY indexed_at DESC
                """
            )

        rows = cursor.fetchall()
        conn.close()

        # Convert to dicts
        documents = []
        for row in rows:
            documents.append({
                "document_id": row[0],
                "source_file": row[1],
                "document_title": row[2],
                "category": row[3],
                "total_pages": row[4],
                "chunk_count": row[5],
                "indexed_at": row[6],
                "pdf_metadata": json.loads(row[7]) if row[7] else {},
                "custom_metadata": json.loads(row[8]) if row[8] else {},
            })

        return documents

    def delete_document(self, document_id: str) -> int:
        """Delete document and all its chunks.

        Removes from both ChromaDB (chunks) and SQLite (metadata).

        Args:
            document_id: Document to delete

        Returns:
            Number of chunks deleted
        """
        # Delete from ChromaDB
        collection = self._init_document_collection()

        # Get all chunk IDs for this document
        results = collection.get(
            where={"document_id": document_id}
        )

        chunks_deleted = 0
        if results['ids']:
            collection.delete(ids=results['ids'])
            chunks_deleted = len(results['ids'])

        # Delete from SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM indexed_documents WHERE document_id = ?",
            (document_id,)
        )

        conn.commit()
        conn.close()

        return chunks_deleted

    def _get_document_registry_entry(self, document_id: str) -> dict[str, Any] | None:
        """Get document metadata from SQLite registry.

        Args:
            document_id: Document ID to look up

        Returns:
            Document metadata dict or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT document_id, source_file, document_title, category,
                   total_pages, chunk_count, indexed_at, pdf_metadata, custom_metadata
            FROM indexed_documents
            WHERE document_id = ?
            """,
            (document_id,)
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "document_id": row[0],
                "source_file": row[1],
                "document_title": row[2],
                "category": row[3],
                "total_pages": row[4],
                "chunk_count": row[5],
                "indexed_at": row[6],
                "pdf_metadata": json.loads(row[7]) if row[7] else {},
                "custom_metadata": json.loads(row[8]) if row[8] else {},
            }

        return None

    # ========================================================================
    # Memory Injection (for System Prompt)
    # ========================================================================

    def get_relevant_memories_for_injection(self) -> str:
        """Build formatted memory section for system prompt injection.

        This method is called by the Agent to get memories that should be
        included in the system prompt on every turn. It formats memories
        from all backends into a human-readable format.

        Returns:
            Formatted string to inject into system prompt, or empty string if no memories
        """
        sections = []

        # Working Memory (recent context)
        try:
            working = self.get_working_memories(limit=settings.memory_working_limit)
            if working:
                lines = ["## Recent Context (Working Memory)"]
                for m in working:
                    ts = m.timestamp.strftime("%Y-%m-%d %H:%M")
                    lines.append(f"- [{ts}] {m.content}")
                sections.append("\n".join(lines))
        except Exception as e:
            print(f"Warning: Failed to load working memories: {e}")

        # Long-term Facts (Phase 3)
        try:
            facts = self.search_facts(limit=settings.memory_facts_limit)
            if facts:
                lines = ["## Long-term Knowledge"]

                # Group by category for better organization
                by_category: dict[str, list[Memory]] = {}
                for f in facts:
                    cat = f.category or "general"
                    by_category.setdefault(cat, []).append(f)

                # Format each category
                for cat, cat_facts in sorted(by_category.items()):
                    lines.append(f"\n### {cat.title()}")
                    for f in cat_facts:
                        lines.append(f"- {f.content}")

                sections.append("\n".join(lines))
        except Exception as e:
            print(f"Warning: Failed to load facts: {e}")

        # Semantic search (Phase 4)
        # For now, skip semantic search in injection
        # In future, could analyze recent conversation to determine query

        # Combine all sections
        if sections:
            return "\n\n".join(sections)
        else:
            return ""
