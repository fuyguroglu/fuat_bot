#!/usr/bin/env python3
"""
Diagnose PDF indexing - show which files succeed/fail and why.
"""

from pathlib import Path
from rich.console import Console
from rich.table import Table
from fuat_bot.tools import index_document
from fuat_bot.config import settings

console = Console()


def main():
    console.print("[bold cyan]📋 PDF Indexing Diagnostics[/bold cyan]\n")

    # Get all PDFs in directory
    pdf_dir = settings.workspace_dir / "School Regulations"

    if not pdf_dir.exists():
        console.print(f"[red]Error:[/red] Directory not found: {pdf_dir}")
        return

    # Get all files matching *.pdf pattern
    all_files = sorted(pdf_dir.glob("*.pdf"))

    # Filter out Zone.Identifier files
    actual_pdfs = [f for f in all_files if ":Zone.Identifier" not in str(f)]
    zone_files = [f for f in all_files if ":Zone.Identifier" in str(f)]

    console.print(f"[yellow]Directory:[/yellow] {pdf_dir}")
    console.print(f"Total files matching *.pdf: [cyan]{len(all_files)}[/cyan]")
    console.print(f"  Actual PDFs: [green]{len(actual_pdfs)}[/green]")
    console.print(f"  Zone.Identifier files: [dim]{len(zone_files)}[/dim]")
    console.print()

    if zone_files:
        console.print("[dim]Zone.Identifier files (Windows metadata, safe to ignore):[/dim]")
        for f in zone_files[:3]:
            console.print(f"  [dim]{f.name}[/dim]")
        if len(zone_files) > 3:
            console.print(f"  [dim]... and {len(zone_files) - 3} more[/dim]")
        console.print()

    # Try to index each actual PDF
    console.print("[yellow]Testing each PDF:[/yellow]\n")

    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("File", width=50)
    results_table.add_column("Size", justify="right", width=10)
    results_table.add_column("Status", width=12)
    results_table.add_column("Chunks", justify="right", width=8)
    results_table.add_column("Details", width=40)

    success_count = 0
    fail_count = 0

    for pdf_path in actual_pdfs:
        file_size = pdf_path.stat().st_size
        size_str = f"{file_size / 1024:.0f}KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f}MB"

        # Try to index
        try:
            # Use relative path from workspace
            rel_path = pdf_path.relative_to(settings.workspace_dir)
            result = index_document(
                path=str(rel_path),
                category="test"
            )

            if "error" in result:
                fail_count += 1
                results_table.add_row(
                    pdf_path.name[:47] + "..." if len(pdf_path.name) > 47 else pdf_path.name,
                    size_str,
                    "[red]❌ Failed[/red]",
                    "-",
                    result["error"][:37] + "..." if len(result["error"]) > 37 else result["error"]
                )
            else:
                success_count += 1
                results_table.add_row(
                    pdf_path.name[:47] + "..." if len(pdf_path.name) > 47 else pdf_path.name,
                    size_str,
                    "[green]✅ Success[/green]",
                    str(result.get("chunks_created", "?")),
                    f"Indexed at {result.get('indexed_at', '')[:19]}"
                )
        except Exception as e:
            fail_count += 1
            results_table.add_row(
                pdf_path.name[:47] + "..." if len(pdf_path.name) > 47 else pdf_path.name,
                size_str,
                "[red]❌ Error[/red]",
                "-",
                str(e)[:37] + "..." if len(str(e)) > 37 else str(e)
            )

    console.print(results_table)
    console.print()

    # Summary
    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  Success: [green]{success_count}[/green] / {len(actual_pdfs)}")
    console.print(f"  Failed: [red]{fail_count}[/red] / {len(actual_pdfs)}")

    if zone_files:
        console.print(f"\n[yellow]💡 Tip:[/yellow] Remove Zone.Identifier files with:")
        console.print(f'    [dim]find "workspace/School Regulations" -name "*:Zone.Identifier" -delete[/dim]')


if __name__ == "__main__":
    main()
