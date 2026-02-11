# CLAUDE.md - Project Context

## Project Overview

**Fuat_bot** is a Python-based CLI agentic AI assistant (learning project). It implements the classic agentic loop: user message → LLM → tool calls → loop → text response.

- **Status**: Phase 1, 2, RAG, Phase 1.5 (Google Calendar), Email & Contacts complete
- **Python**: 3.11+ | **Env**: `fuat_bot` (conda)
- **Use case**: School/university teaching assistant — answers student questions from PDF regulation documents

## Architecture

```
CLI (cli.py) → Agent (agent.py) → Tools (tools.py)
                                → Memory (memory.py)
                                → RAG (rag.py)
                                → Calendar (calendar_tools.py)
                                → Web (web_tools.py)
                                → Email (email_tools.py)
                                → Contacts (contacts_tools.py)
```

## File Structure

```
fuat_bot/
├── __init__.py        # Package metadata
├── __main__.py        # Entry point
├── cli.py             # Typer + Rich CLI
├── config.py          # Pydantic Settings (loads .env)
├── agent.py           # Agent loop + session management + SYSTEM_PROMPT
├── tools.py           # TOOL_SCHEMAS + TOOL_IMPLEMENTATIONS dict
├── memory.py          # 3-tier memory (JSON / SQLite / ChromaDB)
├── rag.py             # RAG pipeline (extract → chunk → index → search)
├── calendar_tools.py  # Google Calendar CRUD + appointment slots
├── web_tools.py       # DuckDuckGo search + HTTP page fetching
├── email_tools.py     # SMTP send + IMAP read/search/delete/move/reply/forward + attachments, multi-account
└── contacts_tools.py  # Gmail (People API) + CardDAV contact lookup

workspace/           # Sandboxed file operations
sessions/            # JSONL session transcripts
memory/
├── working/         # JSON working memory files
├── longterm.db      # SQLite facts
└── chromadb/        # Semantic memory + document chunks
```

## CLI Commands

```
python -m fuat_bot chat             # Start interactive chat session
python -m fuat_bot search <query>   # Direct RAG search (no LLM)
python -m fuat_bot sessions         # List saved session transcripts
python -m fuat_bot calendar-setup   # One-time Google Calendar OAuth
python -m fuat_bot contacts-setup   # One-time Google Contacts OAuth
python -m fuat_bot version          # Show version
```

## Tools

**File** (sandboxed to `./workspace/`): `read_file`, `write_file`, `list_directory`

**System**: `bash` — ⚠️ unrestricted, remove for production

**Memory**: `remember_context` / `recall_context` (JSON), `remember_fact` / `recall_facts` (SQLite), `remember_semantically` / `search_memories` (ChromaDB)

**RAG**: `extract_pdf_text`, `index_document`, `index_directory`, `search_documents`, `list_indexed_documents`, `delete_indexed_document`

**Web**: `web_search` (DuckDuckGo via `ddgs`, no API key), `web_fetch` (reads a URL, strips HTML to plain text, truncates at 8000 chars)

**Google Calendar**: `calendar_list_events`, `calendar_add_event`, `calendar_update_event`, `calendar_delete_event`, `calendar_create_appointment_slots`, `calendar_mark_important_date` — requires one-time OAuth setup via `python -m fuat_bot calendar-setup`

**Email**: `send_email`, `list_emails`, `read_email`, `delete_email`, `search_emails`, `list_folders`, `create_folder`, `move_email`, `reply_email`, `forward_email`, `list_email_attachments`, `save_email_attachment` — multi-account SMTP/IMAP; configured via `EMAIL_ACCOUNTS` JSON in `.env`

**Contacts**: `search_contacts`, `list_contacts` — dual backend: Gmail People API or CardDAV (Zimbra/Outlook); requires one-time OAuth setup via `python -m fuat_bot contacts-setup` for Gmail

## Providers

Agent adapts to `LLM_PROVIDER` in `.env`:
- `anthropic` — native tool use
- `gemini` — schemas converted to Gemini function declarations; model names require `models/` prefix (e.g. `models/gemini-2.5-flash`)
- `ollama` — local LLM via native `ollama` library at `http://localhost:11434`; uses a more directive system prompt (`SYSTEM_PROMPT_OLLAMA`); Llama 3.x works reliably for tool calling
- `openai` — stub only, raises `NotImplementedError`

## Key Implementation Details

**Agent loop** (`agent.py`): max 10 iterations per user message; session stored as JSONL; system prompt injected with current datetime + memory context.

**Tool pattern**: all tools return `dict[str, Any]` — success has relevant keys, error has `{"error": "..."}`.

**Adding a tool**: add schema to `TOOL_SCHEMAS`, implement function, add to `TOOL_IMPLEMENTATIONS`.

**Config** (`config.py`): Pydantic Settings, all settings in `.env`. Update `Settings` class + `.env.example` for new settings.

**RAG search**: hybrid semantic (ChromaDB vectors) + keyword (BM25) combined with RRF (k=60), optional cross-encoder re-ranking. Default 10 results. `search_documents` supports `rerank`, `use_bm25`, `show_chunks` (debug), and `min_score` parameters.

**Memory injection**: working (up to `MEMORY_WORKING_LIMIT`) + facts (up to `MEMORY_FACTS_LIMIT`) + semantic results (up to `MEMORY_SEMANTIC_LIMIT`) are injected into system prompt automatically.

**Google Calendar auth** (`calendar_tools.py`): OAuth 2.0 desktop flow. `credentials.json` downloaded from Google Cloud Console, `token.json` auto-saved after first auth and auto-refreshed. Calendar ID configurable via `GOOGLE_CALENDAR_ID` in `.env` (defaults to `primary`).

**Google Contacts auth** (`contacts_tools.py`): Same OAuth desktop flow, separate `contacts_token.json`. CardDAV backend uses existing email credentials — no extra setup needed.

**Email** (`email_tools.py`): Standard library `smtplib`/`imaplib` only. Multi-account support via `EMAIL_ACCOUNTS` JSON (named accounts with address, password, SMTP/IMAP hosts). Supports HTML body, CC/BCC, and permanent vs. trash delete. `reply_email` sets proper threading headers (`In-Reply-To`, `References`) and supports reply-all. `forward_email` quotes the original message with an optional prepended note. `list_folders` discovers available IMAP mailboxes. `create_folder` creates a new mailbox (supports nested paths like `Students/2026`). `move_email` copies then expunges. `list_email_attachments` / `save_email_attachment` inspect and download attachments into the workspace. `_parse_message` exposes `cc`, `reply_to`, `in_reply_to`, and `references` headers.

**Security**: file/memory/RAG tools use `_resolve_path()` to prevent directory traversal. Bash tool has no restrictions — treat with caution.

## Code Style

- **Formatter**: Ruff (line length: 100)
- **Types**: Python 3.11+ type hints throughout
- **Docstrings**: on all public functions/classes
- **Paths**: always use `pathlib.Path`
- **Errors**: handle gracefully; never let exceptions surface to user raw

## Notes

1. Learning project — keep it simple, avoid premature abstraction
2. Security first — validate paths, sanitize inputs
3. Don't jump ahead to future phases without approval
4. Planned future: Phase 3 (skills system), Phase 4 (Telegram)
