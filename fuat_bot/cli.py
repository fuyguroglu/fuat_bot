"""
Command-line interface for Fuat_bot.

Run with: python -m fuat_bot.cli
"""

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from . import __version__
from .agent import create_agent
from .config import settings

app = typer.Typer(
    name="fuat_bot",
    help="Fuat's Personal AI Assistant",
    add_completion=False,
)
console = Console()


def _select_provider_interactive() -> tuple[str, str]:
    """
    Interactive provider and model selection.

    Returns:
        Tuple of (provider, model)
    """
    # Check which providers have API keys
    available_providers = []

    if settings.anthropic_api_key:
        available_providers.append(("anthropic", "Claude (Anthropic)"))
    if settings.gemini_api_key:
        available_providers.append(("gemini", "Gemini (Google)"))
    if settings.openai_api_key:
        available_providers.append(("openai", "OpenAI (GPT)"))
    # Ollama is always available (no API key needed)
    available_providers.append(("ollama", "Ollama (Local)"))

    if not available_providers:
        console.print("[red]No providers available. Please add API keys to .env[/red]")
        raise typer.Exit(1)

    # Show provider selection
    console.print("\n[bold cyan]Select LLM Provider:[/bold cyan]")
    for i, (key, name) in enumerate(available_providers, 1):
        console.print(f"  {i}. {name}")

    # Get provider selection
    while True:
        choice = Prompt.ask(
            "\n[bold]Choose provider",
            choices=[str(i) for i in range(1, len(available_providers) + 1)],
            default="1"
        )
        provider = available_providers[int(choice) - 1][0]
        break

    # Show model selection based on provider
    console.print(f"\n[bold cyan]Select Model for {provider}:[/bold cyan]")

    if provider == "ollama":
        models = [
            ("llama3.2:3b", "Llama 3.2 3B (Fast, 2GB)"),
            ("llama3.2:latest", "Llama 3.2 Latest (Balanced)"),
            ("llama3.3:70b", "Llama 3.3 70B (Powerful, slow)"),
            ("mistral:latest", "Mistral Latest"),
            ("phi3:latest", "Phi-3"),
            ("custom", "Enter custom model name"),
        ]
    elif provider == "gemini":
        models = [
            ("models/gemini-2.5-flash", "Gemini 2.5 Flash (Recommended)"),
            ("models/gemini-2.5-pro", "Gemini 2.5 Pro (Most capable)"),
            ("models/gemini-2.0-flash", "Gemini 2.0 Flash"),
        ]
    elif provider == "anthropic":
        models = [
            ("claude-sonnet-4-20250514", "Claude Sonnet 4 (Recommended)"),
            ("claude-opus-4-20250514", "Claude Opus 4 (Most capable)"),
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
        ]
    elif provider == "openai":
        models = [
            ("gpt-4o", "GPT-4o (Recommended)"),
            ("gpt-4o-mini", "GPT-4o Mini (Faster)"),
            ("gpt-4-turbo", "GPT-4 Turbo"),
        ]
    else:
        models = [("default", "Default model")]

    for i, (key, name) in enumerate(models, 1):
        console.print(f"  {i}. {name}")

    # Get model selection
    while True:
        choice = Prompt.ask(
            "\n[bold]Choose model",
            choices=[str(i) for i in range(1, len(models) + 1)],
            default="1"
        )
        model_key = models[int(choice) - 1][0]

        if model_key == "custom":
            model_key = Prompt.ask("[bold]Enter model name")

        break

    console.print(f"\n[green]✓[/green] Selected: [cyan]{provider}[/cyan] / [cyan]{model_key}[/cyan]\n")
    return provider, model_key


@app.command()
def chat(
    session: str = typer.Option(
        None,
        "--session", "-s",
        help="Session ID to resume (creates new if not specified)",
    ),
    message: str = typer.Option(
        None,
        "--message", "-m",
        help="Single message to send (non-interactive mode)",
    ),
    provider: str = typer.Option(
        None,
        "--provider", "-p",
        help="LLM provider (skips interactive selection if provided)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Model name (skips interactive selection if provided)",
    ),
):
    """
    Start an interactive chat session with the assistant.

    By default, shows an interactive menu to select provider and model.
    Use --provider and --model flags to skip the menu (useful for scripts).

    Examples:
        # Interactive mode (shows menu)
        python -m fuat_bot chat

        # Quick script mode (skip menu)
        python -m fuat_bot chat -p ollama --model llama3.2:3b -m "test query"
    """
    # If provider/model not specified, show interactive selection
    if not provider or not model:
        selected_provider, selected_model = _select_provider_interactive()
        provider = provider or selected_provider
        model = model or selected_model

    # Set provider and model
    settings.llm_provider = provider

    if settings.llm_provider == "ollama":
        settings.ollama_model = model
    else:
        settings.model_name = model

    # Check for API key
    try:
        api_key = settings.get_api_key()
    except ValueError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print(f"\nTo use {settings.llm_provider}, add the API key to your .env file:")

        if settings.llm_provider == "anthropic":
            console.print("  ANTHROPIC_API_KEY=sk-ant-...")
        elif settings.llm_provider == "gemini":
            console.print("  GEMINI_API_KEY=AIza...")
        elif settings.llm_provider == "openai":
            console.print("  OPENAI_API_KEY=sk-...")

        raise typer.Exit(1)

    # Create agent
    agent = create_agent(session)

    # Get the correct model name based on provider
    if settings.llm_provider == "ollama":
        model_display = settings.ollama_model
    else:
        model_display = settings.model_name

    provider_display = settings.llm_provider.capitalize()

    console.print(Panel.fit(
        f"[bold blue]Fuat_bot v{__version__}[/bold blue]\n"
        f"Provider: {provider_display}\n"
        f"Model: {model_display}\n"
        f"Session: {agent.session.session_id}\n"
        f"Workspace: {settings.workspace_dir.resolve()}",
        title="🤖 Welcome",
    ))
    
    # Non-interactive mode
    if message:
        response = agent.chat(message)
        console.print()
        console.print(Markdown(response))
        return
    
    # Interactive mode
    console.print("\n[dim]Type 'exit' or 'quit' to end the session.[/dim]\n")
    
    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
        
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        
        if not user_input.strip():
            continue
        
        # Get response
        console.print()
        with console.status("[bold blue]Thinking...[/bold blue]"):
            response = agent.chat(user_input)
        
        # Display response
        console.print()
        console.print(Panel(Markdown(response), title="[bold blue]Assistant[/bold blue]", border_style="blue"))
        console.print()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    category: str = typer.Option(
        None,
        "--category", "-c",
        help="Filter by document category",
    ),
    limit: int = typer.Option(
        5,
        "--limit", "-n",
        help="Number of results to return (default: 5)",
    ),
    rerank: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        help="Enable cross-encoder re-ranking (default: enabled)",
    ),
    show_content: bool = typer.Option(
        True,
        "--show-content/--no-content",
        help="Show chunk content (default: enabled)",
    ),
):
    """
    Search indexed documents directly (test RAG without LLM).

    This command shows raw search results from the RAG system,
    including chunk content, relevance scores, and citations.

    Examples:
        # Basic search
        python -m fuat_bot search "attendance policy"

        # Search with category filter
        python -m fuat_bot search "grading scale" -c regulations

        # Get more results without re-ranking
        python -m fuat_bot search "exam rules" -n 10 --no-rerank

        # Just see metadata (no content)
        python -m fuat_bot search "deadlines" --no-content
    """
    if not settings.rag_enabled:
        console.print("[red]RAG system is disabled in settings[/red]")
        raise typer.Exit(1)

    try:
        from .rag import DocumentSearcher
        from .memory import MemoryManager

        # Create searcher
        memory = MemoryManager(settings.memory_dir)
        searcher = DocumentSearcher(memory)

        # Perform search
        console.print(f"\n[bold]Searching for:[/bold] '{query}'\n")
        if category:
            console.print(f"[dim]Category filter: {category}[/dim]")
        console.print(f"[dim]Limit: {limit} | Re-ranking: {'enabled' if rerank else 'disabled'}[/dim]\n")

        with console.status("[bold blue]Searching...[/bold blue]"):
            result = searcher.search(
                query=query,
                category=category,
                limit=limit,
                rerank=rerank,
            )

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            raise typer.Exit(1)

        results = result.get("results", [])
        count = result.get("count", 0)

        if count == 0:
            console.print("[yellow]No results found[/yellow]")
            return

        console.print(f"[bold green]Found {count} result{'s' if count != 1 else ''}:[/bold green]\n")

        # Display results in a table
        for i, res in enumerate(results, 1):
            # Create header with rank and score
            score_color = "green" if res["score"] > 0.7 else "yellow" if res["score"] > 0.5 else "red"
            header = f"[bold]Result {i}[/bold] | Score: [{score_color}]{res['score']:.3f}[/{score_color}]"

            # Create citation info
            citation = (
                f"[cyan]{res['document_title']}[/cyan] "
                f"(pages {res['page_start']}-{res['page_end']})"
            )

            # Create content
            if show_content:
                content = f"{citation}\n\n{res['content'][:500]}{'...' if len(res['content']) > 500 else ''}"
            else:
                content = citation

            console.print(Panel(
                content,
                title=header,
                border_style="blue",
                padding=(0, 1),
            ))
            console.print()

    except ImportError as e:
        console.print(f"[red]RAG dependencies not installed: {e}[/red]")
        console.print("Install with: pip install pypdf chromadb sentence-transformers")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="calendar-setup")
def calendar_setup():
    """One-time Google Calendar OAuth setup.

    Downloads your credentials from Google Cloud Console first:
      console.cloud.google.com → APIs & Services → Credentials → OAuth 2.0 Client ID
      (choose Desktop app, download JSON, save as credentials.json in the project root)

    This command opens your browser to authorise access, then saves a token
    so the calendar tools work without re-authenticating.
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        console.print("[red]Google API libraries not installed.[/red]")
        console.print("Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        raise typer.Exit(1)

    creds_file = settings.google_calendar_credentials_file
    token_file = settings.google_calendar_token_file

    if not creds_file.exists():
        console.print(f"[red]credentials.json not found at:[/red] {creds_file.resolve()}\n")
        console.print("Steps to get it:")
        console.print("  1. Go to [link]https://console.cloud.google.com[/link]")
        console.print("  2. APIs & Services → Library → enable 'Google Calendar API'")
        console.print("  3. APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID")
        console.print("  4. Application type: Desktop app")
        console.print("  5. Download the JSON and save it as [bold]credentials.json[/bold] in the project root")
        raise typer.Exit(1)

    console.print("[bold blue]Starting Google OAuth flow...[/bold blue]")
    console.print("[dim]Your browser will open. Log in and grant calendar access.[/dim]\n")

    try:
        scopes = ["https://www.googleapis.com/auth/calendar"]
        flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), scopes)
        creds = flow.run_local_server(port=0)
    except Exception as e:
        console.print(f"[red]OAuth flow failed: {e}[/red]")
        raise typer.Exit(1)

    # Save the token
    token_file.write_text(creds.to_json())
    console.print(f"[green]Token saved to:[/green] {token_file.resolve()}\n")

    # Quick sanity check — list today's events
    try:
        from datetime import date
        service = build("calendar", "v3", credentials=creds)
        today = date.today().isoformat()
        result = service.events().list(
            calendarId=settings.google_calendar_id,
            timeMin=f"{today}T00:00:00Z",
            timeMax=f"{today}T23:59:59Z",
            maxResults=5,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
        count = len(events)

        console.print(Panel.fit(
            f"[bold green]Google Calendar connected![/bold green]\n\n"
            f"Calendar: [cyan]{settings.google_calendar_id}[/cyan]\n"
            f"Events today: {count}" +
            (("\n\n" + "\n".join(f"  • {e.get('summary', '(no title)')}" for e in events)) if events else ""),
            title="Setup Complete",
        ))
    except Exception as e:
        console.print(f"[yellow]Token saved but test call failed: {e}[/yellow]")
        console.print("[dim]Calendar tools should still work — try: python -m fuat_bot chat[/dim]")


@app.command(name="contacts-setup")
def contacts_setup():
    """One-time Google Contacts (People API) OAuth setup.

    Uses the same credentials.json as Calendar, but saves a separate
    contacts_token.json so the two OAuth tokens stay independent.

    Steps before running this command:
      1. Go to console.cloud.google.com → APIs & Services → Library
      2. Search for and enable "Google People API"
      3. Make sure credentials.json is already in the project root
         (same one used for Calendar; no need to create a new one)
      4. Run this command — your browser will open for authorisation
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        console.print("[red]Google API libraries not installed.[/red]")
        console.print("Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        raise typer.Exit(1)

    creds_file = settings.google_calendar_credentials_file  # reuse same credentials.json
    token_file = settings.google_contacts_token_file

    if not creds_file.exists():
        console.print(f"[red]credentials.json not found at:[/red] {creds_file.resolve()}\n")
        console.print("Follow the same steps as calendar-setup to obtain credentials.json.")
        raise typer.Exit(1)

    console.print("[bold blue]Starting Google Contacts OAuth flow...[/bold blue]")
    console.print("[dim]Your browser will open. Log in and grant Contacts (read-only) access.[/dim]\n")

    try:
        scopes = ["https://www.googleapis.com/auth/contacts.readonly"]
        flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), scopes)
        creds = flow.run_local_server(port=0)
    except Exception as e:
        console.print(f"[red]OAuth flow failed: {e}[/red]")
        raise typer.Exit(1)

    token_file.write_text(creds.to_json())
    console.print(f"[green]Token saved to:[/green] {token_file.resolve()}\n")

    # Quick sanity check — count total contacts
    try:
        service = build("people", "v1", credentials=creds)
        result = service.people().connections().list(
            resourceName="people/me",
            pageSize=10,
            personFields="names,emailAddresses",
        ).execute()
        total = result.get("totalItems", "?")
        sample = result.get("connections", [])[:5]
        names = [
            (p.get("names") or [{}])[0].get("displayName", "(no name)")
            for p in sample
        ]

        console.print(Panel.fit(
            f"[bold green]Google Contacts connected![/bold green]\n\n"
            f"Total contacts: [cyan]{total}[/cyan]\n\n"
            "First few:\n" + "\n".join(f"  • {n}" for n in names),
            title="Setup Complete",
        ))
    except Exception as e:
        console.print(f"[yellow]Token saved but test call failed: {e}[/yellow]")
        console.print("[dim]Contacts tools should still work — try: python -m fuat_bot chat[/dim]")


@app.command()
def sessions():
    """List all saved sessions."""
    sessions_dir = settings.sessions_dir
    
    if not sessions_dir.exists():
        console.print("[dim]No sessions found.[/dim]")
        return
    
    session_files = sorted(sessions_dir.glob("*.jsonl"), reverse=True)
    
    if not session_files:
        console.print("[dim]No sessions found.[/dim]")
        return
    
    console.print("[bold]Saved Sessions:[/bold]\n")
    for f in session_files[:20]:  # Show last 20
        size = f.stat().st_size
        console.print(f"  • {f.stem}  [dim]({size:,} bytes)[/dim]")
    
    if len(session_files) > 20:
        console.print(f"\n[dim]...and {len(session_files) - 20} more[/dim]")


@app.command()
def version():
    """Show version information."""
    console.print(f"Fuat_bot v{__version__}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
