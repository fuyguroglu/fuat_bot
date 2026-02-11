#!/usr/bin/env python3
"""
Debug Search Script - View RAG retrieval chunks directly without LLM summarization.

Usage:
    python debug_search.py "your search query"
    python debug_search.py "grading policies" --limit 5
    python debug_search.py "late homework" --no-rerank
"""

import sys
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from fuat_bot.tools import search_documents

console = Console()


def print_debug_search(query: str, limit: int = 3, rerank: bool = True, category: str = None):
    """Run search with debug mode and display results nicely."""

    console.print(f"\n[bold cyan]🔍 Searching for:[/bold cyan] {query}")
    console.print(f"[dim]limit={limit}, rerank={rerank}, category={category or 'all'}[/dim]\n")

    # Perform search with debug mode
    result = search_documents(
        query=query,
        limit=limit,
        rerank=rerank,
        show_chunks=True,
        category=category
    )

    # Check for errors
    if "error" in result:
        console.print(f"[red]❌ Error:[/red] {result['error']}")
        return

    debug_info = result.get("debug_info", {})

    # ============= STAGE 1: VECTOR SEARCH =============
    console.print("[bold yellow]📊 STAGE 1: VECTOR SEARCH[/bold yellow]")
    stage1 = debug_info.get("stage1_vector_search", {})

    console.print(f"  Retrieval limit: [cyan]{stage1.get('retrieval_limit')}[/cyan]")
    console.print(f"  Results retrieved: [cyan]{stage1.get('results_retrieved')}[/cyan]\n")

    # Create table for stage 1 chunks
    table1 = Table(title="Vector Search Results", show_header=True, header_style="bold magenta")
    table1.add_column("Rank", style="dim", width=4)
    table1.add_column("Score", justify="right", width=8)
    table1.add_column("Document", width=30)
    table1.add_column("Pages", width=8)
    table1.add_column("Chunk", width=6)
    table1.add_column("Tokens", width=7)
    table1.add_column("Preview", width=50)

    for chunk in stage1.get("chunks", []):
        table1.add_row(
            str(chunk["rank"]),
            f"{chunk['score']:.2f}",
            chunk["document"][:28] + "..." if len(chunk["document"]) > 28 else chunk["document"],
            chunk["pages"],
            str(chunk["chunk_index"]),
            str(chunk["token_count"]),
            chunk["content_preview"][:47] + "..." if len(chunk["content_preview"]) > 47 else chunk["content_preview"]
        )

    console.print(table1)
    console.print()

    # ============= STAGE 2: RE-RANKING =============
    console.print("[bold yellow]🔄 STAGE 2: RE-RANKING[/bold yellow]")
    stage2 = debug_info.get("stage2_reranking", {})

    console.print(f"  Enabled: [cyan]{stage2.get('enabled')}[/cyan]")
    console.print(f"  Applied: [cyan]{stage2.get('applied')}[/cyan]")
    if stage2.get("model"):
        console.print(f"  Model: [cyan]{stage2.get('model')}[/cyan]")
    console.print()

    # ============= FILTERING =============
    console.print("[bold yellow]🔍 FILTERING[/bold yellow]")
    filtering = debug_info.get("filtering", {})

    console.print(f"  Min score threshold: [cyan]{filtering.get('min_score_threshold')}[/cyan]")
    console.print(f"  Before filter: [cyan]{filtering.get('results_before_filter')}[/cyan]")
    console.print(f"  After filter: [cyan]{filtering.get('results_after_filter')}[/cyan]")
    console.print(f"  Filtered out: [cyan]{filtering.get('filtered_out')}[/cyan]")
    console.print()

    # ============= FINAL RESULTS =============
    console.print("[bold green]✅ FINAL RESULTS (After Re-ranking)[/bold green]\n")

    final_results = debug_info.get("final_results", [])

    for res in final_results:
        # Create panel for each result
        panel_content = f"""**Rank:** {res['rank']}
**Score:** {res['score']:.4f}
**Document:** {res['document']}
**Pages:** {res['pages']}
**Chunk Index:** {res['chunk_index']}
**Token Count:** {res['token_count']}

**Content:**
{res['content'][:500]}{"..." if len(res['content']) > 500 else ""}
"""

        console.print(Panel(
            Markdown(panel_content),
            title=f"[bold]Result #{res['rank']}[/bold]",
            border_style="green"
        ))
        console.print()

    # ============= SUMMARY =============
    console.print(f"[bold]📈 Summary:[/bold] Returned [cyan]{result['count']}[/cyan] results for query: [yellow]{result['query']}[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="Debug search tool - see RAG retrieval chunks directly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_search.py "grading policies"
  python debug_search.py "late homework" --limit 5
  python debug_search.py "student discipline" --no-rerank
  python debug_search.py "thesis requirements" --category regulations
        """
    )

    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=3, help="Number of results to return (default: 3)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder re-ranking")
    parser.add_argument("--category", type=str, help="Filter by category")

    args = parser.parse_args()

    try:
        print_debug_search(
            query=args.query,
            limit=args.limit,
            rerank=not args.no_rerank,
            category=args.category
        )
    except Exception as e:
        console.print(f"[red]❌ Error:[/red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
