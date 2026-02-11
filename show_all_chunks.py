#!/usr/bin/env python3
"""Show all chunks from a specific document."""

from fuat_bot.config import settings
import chromadb

# Connect to ChromaDB
client = chromadb.PersistentClient(path=str(settings.memory_dir / 'chromadb'))
collection = client.get_collection('document_chunks')

# Get all chunks from graduate studies doc
results = collection.get(
    where={'document_title': '8zqlt4sA5V1TrzZk_CIU_GRADUATE_STUDIES_REGULATION...'}
)

print(f'Total chunks: {len(results["documents"])}')
print()

# Sort by chunk_index
chunks = []
for i in range(len(results['documents'])):
    meta = results['metadatas'][i]
    chunks.append({
        'index': meta['chunk_index'],
        'pages': f"{meta['page_start']}-{meta['page_end']}",
        'tokens': meta['token_count'],
        'content': results['documents'][i]
    })

chunks.sort(key=lambda x: x['index'])

# Show each chunk
for chunk in chunks:
    print(f"=== CHUNK {chunk['index']} (Pages {chunk['pages']}, {chunk['tokens']} tokens) ===")
    # Show more content for chunk 2 (grading table)
    preview_len = 2000 if chunk['index'] == 2 else 500
    print(chunk['content'][:preview_len])
    if len(chunk['content']) > preview_len:
        print(f"... [{len(chunk['content']) - preview_len} more characters]")
    print()
