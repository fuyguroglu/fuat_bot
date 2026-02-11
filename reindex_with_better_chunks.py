#!/usr/bin/env python3
"""
Re-index documents with better chunk settings.

This script will:
1. Delete all existing indexed documents
2. Re-index them with new chunk size settings
"""

import os
from pathlib import Path
from rich.console import Console
from rich.progress import track
from fuat_bot.tools import list_indexed_documents, delete_indexed_document, index_directory
from fuat_bot.config import settings

console = Console()


def main():
    console.print("[bold cyan]📚 Re-indexing Documents with Better Chunk Settings[/bold cyan]\n")

    # Show current settings
    console.print("[yellow]Current Settings:[/yellow]")
    console.print(f"  Chunk size: {settings.rag_chunk_size} tokens (~{int(settings.rag_chunk_size/1.3)} words)")
    console.print(f"  Overlap: {settings.rag_chunk_overlap} tokens (~{int(settings.rag_chunk_overlap/1.3)} words)")
    console.print(f"  Min chunk: {settings.rag_min_chunk_size} tokens\n")

    # Get currently indexed documents
    console.print("[yellow]Step 1: Getting list of indexed documents...[/yellow]")
    result = list_indexed_documents()

    if not result.get("success"):
        console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
        return

    docs = result.get("documents", [])
    console.print(f"Found [cyan]{len(docs)}[/cyan] indexed documents\n")

    # Store source files to re-index
    source_files = set()
    for doc in docs:
        source_file = doc.get("source_file")
        if source_file:
            source_files.add(Path(source_file).parent)

    console.print(f"[yellow]Step 2: Deleting old indices...[/yellow]")
    for doc in track(docs, description="Deleting"):
        doc_id = doc.get("document_id")
        delete_indexed_document(doc_id)
    console.print("[green]✓[/green] All old indices deleted\n")

    # Re-index directories
    console.print(f"[yellow]Step 3: Re-indexing with new settings...[/yellow]")
    for source_dir in source_files:
        console.print(f"\nIndexing: [cyan]{source_dir}[/cyan]")
        result = index_directory(
            path=str(source_dir),
            category="regulations",
            recursive=False,
            pattern="*.pdf"
        )

        if "error" in result:
            console.print(f"  [red]Error:[/red] {result['error']}")
        else:
            console.print(f"  [green]✓[/green] Indexed {result['documents_indexed']} documents")
            if result['documents_failed'] > 0:
                console.print(f"  [yellow]⚠[/yellow] Failed: {result['documents_failed']}")

    # Show new stats
    console.print(f"\n[yellow]Step 4: Verifying...[/yellow]")
    result = list_indexed_documents()
    new_docs = result.get("documents", [])

    total_chunks = sum(doc.get("chunk_count", 0) for doc in new_docs)
    console.print(f"\n[bold green]✅ Re-indexing Complete![/bold green]")
    console.print(f"  Documents: [cyan]{len(new_docs)}[/cyan]")
    console.print(f"  Total chunks: [cyan]{total_chunks}[/cyan]")
    console.print(f"  Avg chunks/doc: [cyan]{total_chunks/len(new_docs) if new_docs else 0:.1f}[/cyan]")


if __name__ == "__main__":
    main()
