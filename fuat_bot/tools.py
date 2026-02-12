"""
Tools that the agent can use to interact with the system.

Each tool is defined as:
1. A schema (for the LLM to understand how to call it)
2. An implementation function

Tools are the "hands" of the agent - they let it actually DO things.
"""

import subprocess
from pathlib import Path
from typing import Any

from .config import settings
from .web_tools import web_search, web_fetch
from .email_tools import (
    send_email,
    list_emails,
    read_email,
    delete_email,
    search_emails,
    list_folders,
    create_folder,
    move_email,
    reply_email,
    forward_email,
    list_email_attachments,
    save_email_attachment,
)
from .contacts_tools import search_contacts, list_contacts
from .calendar_tools import (
    calendar_list_events,
    calendar_add_event,
    calendar_update_event,
    calendar_delete_event,
    calendar_create_appointment_slots,
    calendar_mark_important_date,
)


# =============================================================================
# Global Memory Manager
# =============================================================================

# Lazy-initialized memory manager shared across all tool calls
_memory_manager = None


def _get_memory_manager():
    """Lazy initialization of memory manager singleton."""
    global _memory_manager
    if _memory_manager is None:
        from .memory import MemoryManager
        _memory_manager = MemoryManager(settings.memory_dir)
    return _memory_manager


# =============================================================================
# Tool Schemas (what the LLM sees)
# =============================================================================

TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Use this to examine files in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to workspace)",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative to workspace)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories in a path. Use this to explore the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to list (relative to workspace, use '.' for workspace root)",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "bash",
        "description": "Execute a bash command. Use for system operations, running scripts, etc. Be careful with destructive commands.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
    {
        "name": "remember_context",
        "description": "Store information in working memory (recent context). Use for temporary info relevant to recent conversations. Examples: current task, recent decisions, temporary preferences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember",
                },
                "category": {
                    "type": "string",
                    "description": "Category (e.g., 'task', 'preference', 'decision')",
                    "default": "general"
                }
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_context",
        "description": "Retrieve recent working memories. Use to recall temporary context from recent conversations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional: filter by category",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max memories to retrieve (default: 10)",
                    "default": 10
                }
            },
            "required": [],
        },
    },
    {
        "name": "remember_fact",
        "description": "Store permanent information (facts, preferences, important details). Use this for information that should persist long-term. Examples: user preferences, important facts about projects, learned information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact to remember",
                },
                "category": {
                    "type": "string",
                    "description": "Category (e.g., 'user_preference', 'project_info', 'technical_fact')",
                    "default": "general"
                }
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_facts",
        "description": "Search long-term memory for facts. Can filter by category or search content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional: search text to find in fact content",
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter by category",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of facts to retrieve (default: 20)",
                    "default": 20
                }
            },
            "required": [],
        },
    },
    {
        "name": "remember_semantically",
        "description": "Store information with semantic search capability. Use this when you want to find related information later based on meaning, not exact keywords. Examples: concepts, explanations, relationships.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember",
                },
                "category": {
                    "type": "string",
                    "description": "Category (e.g., 'concept', 'explanation', 'example')",
                    "default": "general"
                }
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memories",
        "description": "Search all semantic memories using similarity. Finds related memories based on meaning, not just keywords. This uses vector embeddings to find conceptually similar content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (will find semantically similar memories)",
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter by category",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "extract_pdf_text",
        "description": "Extract text from a PDF file in the workspace. Returns text organized by pages with metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to PDF file (relative to workspace)",
                },
                "page_start": {
                    "type": "integer",
                    "description": "Optional: starting page number (0-indexed)",
                },
                "page_end": {
                    "type": "integer",
                    "description": "Optional: ending page number (0-indexed, inclusive)",
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include PDF metadata (default: true)",
                    "default": True
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "index_document",
        "description": "Index a single PDF document for semantic search. Extracts text, chunks it intelligently, generates embeddings, and stores for retrieval. Use this to make PDFs searchable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to PDF file (relative to workspace)",
                },
                "category": {
                    "type": "string",
                    "description": "Category for organization (e.g., 'regulations', 'course_materials')",
                    "default": "documents"
                },
                "custom_metadata": {
                    "type": "object",
                    "description": "Optional custom metadata to attach to the document",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "index_directory",
        "description": "Batch index all PDF documents in a directory. Useful for indexing multiple files at once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to directory containing PDFs (relative to workspace)",
                },
                "category": {
                    "type": "string",
                    "description": "Category for all documents in this directory",
                    "default": "documents"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search subdirectories (default: false)",
                    "default": False
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for file matching (default: '*.pdf')",
                    "default": "*.pdf"
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_documents",
        "description": "Search indexed PDF documents using hybrid semantic + keyword search. Returns relevant passages with citations (document name and page numbers). Always use this when users ask questions about indexed PDFs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - what information are you looking for?",
                },
                "category": {
                    "type": "string",
                    "description": "Optional: filter by document category",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10, increased for better recall)",
                    "default": 10
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum similarity score 0-1 (default: 0.0)",
                    "default": 0.0
                },
                "rerank": {
                    "type": "boolean",
                    "description": "Use cross-encoder re-ranking for better relevance (default: true)",
                    "default": True
                },
                "use_bm25": {
                    "type": "boolean",
                    "description": "Use BM25 keyword search alongside semantic search for hybrid retrieval (default: true)",
                    "default": True
                },
                "show_chunks": {
                    "type": "boolean",
                    "description": "Debug mode: show detailed chunk information including token counts, chunk indices, and retrieval stages (default: false)",
                    "default": False
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_indexed_documents",
        "description": "List all indexed PDF documents with their metadata. Use this to see what documents are available to search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional: filter by category",
                }
            },
            "required": [],
        },
    },
    {
        "name": "delete_indexed_document",
        "description": "Remove an indexed document and all its chunks from the search system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document ID to delete (from list_indexed_documents)",
                }
            },
            "required": ["document_id"],
        },
    },
    # -------------------------------------------------------------------------
    # Web Tools
    # -------------------------------------------------------------------------
    {
        "name": "web_search",
        "description": (
            "Search the internet using DuckDuckGo and return titles, URLs, and snippets. "
            "Use this to find current information, news, documentation, or any topic "
            "not covered by indexed documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g. 'Python 3.13 release notes')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch and read the content of a web page by URL. "
            "Returns readable plain text by default (HTML tags stripped). "
            "Use this to read articles, documentation pages, or any URL found via web_search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to fetch (must start with http:// or https://)",
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Strip HTML and return plain text (default: true). Set false for raw HTML.",
                    "default": True,
                },
            },
            "required": ["url"],
        },
    },
    # -------------------------------------------------------------------------
    # Google Calendar Tools
    # -------------------------------------------------------------------------
    {
        "name": "calendar_list_events",
        "description": "List calendar events in a date range. Optionally filter by a text query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date or datetime in ISO format (e.g. '2026-02-10' or '2026-02-10T00:00:00')",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date or datetime in ISO format (e.g. '2026-02-17' or '2026-02-17T23:59:59')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of events to return (default: 20)",
                    "default": 20,
                },
                "query": {
                    "type": "string",
                    "description": "Optional free-text search to filter events by title, description, or location",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "calendar_add_event",
        "description": "Create a new event on Google Calendar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Event title/summary",
                },
                "start": {
                    "type": "string",
                    "description": "Start datetime in ISO format (e.g. '2026-02-15T14:00:00')",
                },
                "end": {
                    "type": "string",
                    "description": "End datetime in ISO format (e.g. '2026-02-15T15:00:00')",
                },
                "description": {
                    "type": "string",
                    "description": "Optional event description or notes",
                },
                "location": {
                    "type": "string",
                    "description": "Optional location (room, address, or meeting URL)",
                },
                "attendees": {
                    "type": "string",
                    "description": "Optional comma-separated email addresses to invite",
                },
            },
            "required": ["title", "start", "end"],
        },
    },
    {
        "name": "calendar_update_event",
        "description": "Update an existing calendar event. Only the fields you provide will be changed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The event ID (from calendar_list_events or calendar_add_event)",
                },
                "title": {
                    "type": "string",
                    "description": "New event title (optional)",
                },
                "start": {
                    "type": "string",
                    "description": "New start datetime in ISO format (optional)",
                },
                "end": {
                    "type": "string",
                    "description": "New end datetime in ISO format (optional)",
                },
                "description": {
                    "type": "string",
                    "description": "New event description (optional)",
                },
                "location": {
                    "type": "string",
                    "description": "New location (optional)",
                },
            },
            "required": ["event_id"],
        },
    },
    {
        "name": "calendar_delete_event",
        "description": "Delete a calendar event by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The event ID to delete (from calendar_list_events)",
                }
            },
            "required": ["event_id"],
        },
    },
    {
        "name": "calendar_mark_important_date",
        "description": (
            "Mark a date as important by creating an all-day banner event. "
            "It appears above the hourly grid in Google Calendar (not inside any time slot). "
            "Use this for deadlines, key academic dates, holidays, etc. "
            "Example: date='2026-06-22', title='Last day of classes'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in ISO format (e.g. '2026-06-22')",
                },
                "title": {
                    "type": "string",
                    "description": "Label for the date (e.g. 'Last day of classes')",
                },
                "description": {
                    "type": "string",
                    "description": "Optional extra notes about this date",
                },
                "color": {
                    "type": "string",
                    "description": (
                        "Optional color for visual distinction. "
                        "Choices: tomato, flamingo, tangerine, banana, sage, basil, "
                        "peacock, blueberry, lavender, grape, graphite. "
                        "Defaults to tomato (red)."
                    ),
                },
            },
            "required": ["date", "title"],
        },
    },
    {
        "name": "calendar_create_appointment_slots",
        "description": (
            "Create a series of bookable appointment slot events on a single day. "
            "Each slot becomes a separate calendar event. "
            "Example: date='2026-02-15', start_time='09:00', end_time='12:00', "
            "slot_duration_minutes=30 creates six 30-minute slots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date for the slots in ISO format (e.g. '2026-02-15')",
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in HH:MM format (e.g. '09:00')",
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in HH:MM format (e.g. '12:00')",
                },
                "slot_duration_minutes": {
                    "type": "integer",
                    "description": "Duration of each slot in minutes (e.g. 30)",
                },
                "title": {
                    "type": "string",
                    "description": "Title for each slot event (e.g. 'Office Hours Slot')",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description added to each slot event",
                },
                "location": {
                    "type": "string",
                    "description": "Optional location for all slots (room number, Zoom link, etc.)",
                },
            },
            "required": ["date", "start_time", "end_time", "slot_duration_minutes", "title"],
        },
    },
    # -------------------------------------------------------------------------
    # Email Tools
    # -------------------------------------------------------------------------
    {
        "name": "send_email",
        "description": (
            "Send an email via SMTP. Supports plain-text and optional HTML body, CC, BCC, "
            "and file attachments from the workspace. "
            "Use the 'account' parameter to choose which configured email account to send from."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address(es), comma-separated",
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                },
                "body": {
                    "type": "string",
                    "description": "Plain-text body of the email",
                },
                "account": {
                    "type": "string",
                    "description": "Named account to send from (e.g. 'personal', 'work'). Uses default if omitted.",
                },
                "cc": {
                    "type": "string",
                    "description": "Optional CC address(es), comma-separated",
                },
                "bcc": {
                    "type": "string",
                    "description": "Optional BCC address(es), comma-separated",
                },
                "html": {
                    "type": "string",
                    "description": "Optional HTML version of the body (sent alongside plain text)",
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of workspace-relative file paths to attach "
                        "(e.g. [\"reports/q1.pdf\", \"data.csv\"]). "
                        "Files must exist in the workspace directory."
                    ),
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "list_emails",
        "description": (
            "List emails in a mailbox folder via IMAP (newest first). "
            "Returns headers only (subject, from, date, UID) — use read_email for the full body."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "IMAP folder name (default: INBOX)",
                    "default": "INBOX",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of emails to return (default: 20)",
                    "default": 20,
                },
                "unread_only": {
                    "type": "boolean",
                    "description": "If true, only return unread/unseen messages (default: false)",
                    "default": False,
                },
                "account": {
                    "type": "string",
                    "description": "Named account to list from (e.g. 'personal', 'work'). Uses default if omitted.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_email",
        "description": "Read the full content (headers + body) of a specific email by its UID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "Email UID from list_emails or search_emails",
                },
                "folder": {
                    "type": "string",
                    "description": "IMAP folder containing the email (default: INBOX)",
                    "default": "INBOX",
                },
                "account": {
                    "type": "string",
                    "description": "Named account to read from (e.g. 'personal', 'work'). Uses default if omitted.",
                },
            },
            "required": ["uid"],
        },
    },
    {
        "name": "delete_email",
        "description": (
            "Delete an email by its UID. By default moves it to Trash (Gmail: [Gmail]/Trash). "
            "Set permanent=true to skip Trash and expunge immediately."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "Email UID from list_emails or search_emails",
                },
                "folder": {
                    "type": "string",
                    "description": "IMAP folder containing the email (default: INBOX)",
                    "default": "INBOX",
                },
                "permanent": {
                    "type": "boolean",
                    "description": "If true, permanently delete (skip Trash). Default: false.",
                    "default": False,
                },
                "account": {
                    "type": "string",
                    "description": "Named account to delete from (e.g. 'personal', 'work'). Uses default if omitted.",
                },
            },
            "required": ["uid"],
        },
    },
    {
        "name": "search_emails",
        "description": (
            "Search emails by a query string matched against subject, sender, and body text. "
            "Returns headers only — use read_email for full content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term to look for in subject, from address, or body",
                },
                "folder": {
                    "type": "string",
                    "description": "IMAP folder to search (default: INBOX)",
                    "default": "INBOX",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10,
                },
                "account": {
                    "type": "string",
                    "description": "Named account to search in (e.g. 'personal', 'work'). Uses default if omitted.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_folders",
        "description": (
            "List all IMAP folders/mailboxes available on the email account. "
            "Use this to discover folder names before calling move_email or list_emails with a non-default folder."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account": {
                    "type": "string",
                    "description": "Named account to inspect (e.g. 'personal', 'work'). Uses default if omitted.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_folder",
        "description": (
            "Create a new IMAP folder/mailbox on the email account. "
            "Supports nested folders using a separator (e.g. 'Students/2026'). "
            "Use list_folders to verify it was created."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Name of the folder to create (e.g. 'Archive', 'Students/2026')",
                },
                "account": {
                    "type": "string",
                    "description": "Named account (e.g. 'personal', 'work'). Uses default if omitted.",
                },
            },
            "required": ["folder"],
        },
    },
    {
        "name": "move_email",
        "description": (
            "Move an email from one folder to another (e.g. INBOX → Archive). "
            "Use list_folders first to find available folder names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "Email UID from list_emails or search_emails",
                },
                "destination_folder": {
                    "type": "string",
                    "description": "Target folder name (e.g. 'Archive', '[Gmail]/Starred')",
                },
                "source_folder": {
                    "type": "string",
                    "description": "Folder where the email currently lives (default: INBOX)",
                    "default": "INBOX",
                },
                "account": {
                    "type": "string",
                    "description": "Named account (e.g. 'personal', 'work'). Uses default if omitted.",
                },
            },
            "required": ["uid", "destination_folder"],
        },
    },
    {
        "name": "reply_email",
        "description": (
            "Reply to an existing email. Automatically sets threading headers (In-Reply-To, References) "
            "and prefixes the subject with 'Re:'. Use reply_all=true to CC all original recipients."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "UID of the email to reply to",
                },
                "body": {
                    "type": "string",
                    "description": "Plain-text reply body",
                },
                "folder": {
                    "type": "string",
                    "description": "Folder containing the original email (default: INBOX)",
                    "default": "INBOX",
                },
                "account": {
                    "type": "string",
                    "description": "Named account to send from. Uses default if omitted.",
                },
                "cc": {
                    "type": "string",
                    "description": "Optional extra CC address(es), comma-separated",
                },
                "reply_all": {
                    "type": "boolean",
                    "description": "If true, CC all original To/Cc recipients (default: false)",
                    "default": False,
                },
                "html": {
                    "type": "string",
                    "description": "Optional HTML version of the reply body",
                },
            },
            "required": ["uid", "body"],
        },
    },
    {
        "name": "forward_email",
        "description": (
            "Forward an existing email to a new recipient. "
            "Prefixes subject with 'Fwd:' and appends a quoted copy of the original message. "
            "An optional note can be added above the forwarded content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "UID of the email to forward",
                },
                "to": {
                    "type": "string",
                    "description": "Recipient address(es) to forward to, comma-separated",
                },
                "folder": {
                    "type": "string",
                    "description": "Folder containing the original email (default: INBOX)",
                    "default": "INBOX",
                },
                "account": {
                    "type": "string",
                    "description": "Named account to send from. Uses default if omitted.",
                },
                "note": {
                    "type": "string",
                    "description": "Optional introductory note to prepend above the forwarded message",
                },
                "cc": {
                    "type": "string",
                    "description": "Optional CC address(es), comma-separated",
                },
            },
            "required": ["uid", "to"],
        },
    },
    {
        "name": "list_email_attachments",
        "description": (
            "List attachments on an email without downloading them. "
            "Returns filename, content type, and size for each attachment. "
            "Use save_email_attachment to download a specific attachment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "Email UID from list_emails or search_emails",
                },
                "folder": {
                    "type": "string",
                    "description": "IMAP folder containing the email (default: INBOX)",
                    "default": "INBOX",
                },
                "account": {
                    "type": "string",
                    "description": "Named account (e.g. 'personal', 'work'). Uses default if omitted.",
                },
            },
            "required": ["uid"],
        },
    },
    {
        "name": "save_email_attachment",
        "description": (
            "Download and save a specific email attachment to the workspace. "
            "Use list_email_attachments first to find the exact filename."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "Email UID from list_emails or search_emails",
                },
                "filename": {
                    "type": "string",
                    "description": "Attachment filename to download (from list_email_attachments)",
                },
                "save_path": {
                    "type": "string",
                    "description": "Destination path inside the workspace (e.g. 'attachments/report.pdf')",
                },
                "folder": {
                    "type": "string",
                    "description": "IMAP folder containing the email (default: INBOX)",
                    "default": "INBOX",
                },
                "account": {
                    "type": "string",
                    "description": "Named account (e.g. 'personal', 'work'). Uses default if omitted.",
                },
            },
            "required": ["uid", "filename", "save_path"],
        },
    },
    # -------------------------------------------------------------------------
    # Contacts Tools
    # -------------------------------------------------------------------------
    {
        "name": "search_contacts",
        "description": (
            "Search contacts by name or email address. "
            "Gmail accounts use the Google People API (requires contacts-setup). "
            "Zimbra and other accounts use CardDAV with existing credentials. "
            "Use this before sending an email to look up someone's address."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Name or email fragment to search for (e.g. 'Sara', 'sara@ciu')",
                },
                "account": {
                    "type": "string",
                    "description": "Named account to search in (e.g. 'gmail', 'zimbra'). Searches all accounts if omitted.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_contacts",
        "description": (
            "List all contacts from one or all configured email accounts. "
            "Gmail accounts use the Google People API (requires contacts-setup). "
            "Zimbra and other accounts use CardDAV with existing credentials."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account": {
                    "type": "string",
                    "description": "Named account to list from (e.g. 'gmail', 'zimbra'). Lists all accounts if omitted.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum contacts per account (default: 50)",
                    "default": 50,
                },
            },
            "required": [],
        },
    },
]


# =============================================================================
# Tool Implementations
# =============================================================================


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the workspace, with safety checks."""
    workspace = settings.workspace_dir.resolve()
    
    # Ensure workspace exists
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Resolve the target path
    target = (workspace / path).resolve()
    
    # Safety: ensure we stay within workspace
    if not str(target).startswith(str(workspace)):
        raise ValueError(f"Path '{path}' escapes workspace directory")
    
    return target


def read_file(path: str) -> dict[str, Any]:
    """Read a file from the workspace."""
    try:
        target = _resolve_path(path)
        
        if not target.exists():
            return {"error": f"File not found: {path}"}
        
        if not target.is_file():
            return {"error": f"Not a file: {path}"}
        
        content = target.read_text(encoding="utf-8")
        return {"content": content, "path": str(path)}
    
    except Exception as e:
        return {"error": str(e)}


def write_file(path: str, content: str) -> dict[str, Any]:
    """Write content to a file in the workspace."""
    try:
        target = _resolve_path(path)
        
        # Create parent directories if needed
        target.parent.mkdir(parents=True, exist_ok=True)
        
        target.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(path), "bytes_written": len(content.encode())}
    
    except Exception as e:
        return {"error": str(e)}


def list_directory(path: str) -> dict[str, Any]:
    """List contents of a directory in the workspace."""
    try:
        target = _resolve_path(path)
        
        if not target.exists():
            return {"error": f"Directory not found: {path}"}
        
        if not target.is_dir():
            return {"error": f"Not a directory: {path}"}
        
        items = []
        for item in sorted(target.iterdir()):
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        
        return {"path": str(path), "items": items}
    
    except Exception as e:
        return {"error": str(e)}


def bash(command: str) -> dict[str, Any]:
    """Execute a bash command."""
    try:
        # Run from workspace directory
        workspace = settings.workspace_dir.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 60 seconds"}
    except Exception as e:
        return {"error": str(e)}


def remember_context(content: str, category: str = "general") -> dict[str, Any]:
    """Store information in working memory.

    Working memory is for recent, temporary context that's relevant to
    ongoing conversations but doesn't need long-term persistence.

    Args:
        content: The information to remember
        category: Category for organization (e.g., 'task', 'preference', 'decision')

    Returns:
        Dict with success status and memory ID, or error
    """
    try:
        if not settings.memory_enabled:
            return {"error": "Memory system is disabled"}

        memory_manager = _get_memory_manager()
        memory_id = memory_manager.store_working_memory(content, category)

        return {
            "success": True,
            "memory_id": memory_id,
            "content": content,
            "category": category,
            "message": "Memory stored successfully in working memory"
        }
    except Exception as e:
        return {"error": f"Failed to store memory: {str(e)}"}


def recall_context(category: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Retrieve recent working memories.

    Working memory contains recent, temporary context from recent conversations.

    Args:
        category: Optional category filter
        limit: Maximum number of memories to retrieve

    Returns:
        Dict with list of memories or error
    """
    try:
        if not settings.memory_enabled:
            return {"error": "Memory system is disabled"}

        # Ensure limit is an integer (LLM might pass as string)
        limit = int(limit) if limit is not None else 10

        memory_manager = _get_memory_manager()
        memories = memory_manager.get_working_memories(limit=limit, category=category)

        return {
            "success": True,
            "count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "category": m.category,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in memories
            ]
        }
    except Exception as e:
        return {"error": f"Failed to recall memories: {str(e)}"}


def remember_fact(content: str, category: str = "general") -> dict[str, Any]:
    """Store permanent information in long-term memory.

    Long-term memory (SQLite) is for persistent facts, preferences, and knowledge
    that should be retained across many sessions.

    Args:
        content: The fact to remember
        category: Category for organization

    Returns:
        Dict with success status and memory ID, or error
    """
    try:
        if not settings.memory_enabled:
            return {"error": "Memory system is disabled"}

        memory_manager = _get_memory_manager()
        memory_id = memory_manager.store_fact(content, category)

        return {
            "success": True,
            "memory_id": memory_id,
            "content": content,
            "category": category,
            "message": "Fact stored successfully in long-term memory"
        }
    except Exception as e:
        return {"error": f"Failed to store fact: {str(e)}"}


def recall_facts(query: str | None = None, category: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Search long-term facts with optional filters.

    Searches the SQLite database for facts matching the given criteria.

    Args:
        query: Optional text to search for in fact content
        category: Optional category filter
        limit: Maximum number of results

    Returns:
        Dict with list of facts or error
    """
    try:
        if not settings.memory_enabled:
            return {"error": "Memory system is disabled"}

        # Ensure limit is an integer (LLM might pass as string)
        limit = int(limit) if limit is not None else 20

        memory_manager = _get_memory_manager()
        facts = memory_manager.search_facts(query=query, category=category, limit=limit)

        return {
            "success": True,
            "count": len(facts),
            "facts": [
                {
                    "id": f.id,
                    "content": f.content,
                    "category": f.category,
                    "timestamp": f.timestamp.isoformat(),
                }
                for f in facts
            ]
        }
    except Exception as e:
        return {"error": f"Failed to recall facts: {str(e)}"}


def remember_semantically(content: str, category: str = "general") -> dict[str, Any]:
    """Store information with semantic embeddings for similarity search.

    Semantic memory uses vector embeddings to enable finding related information
    based on meaning, not just keyword matching.

    Args:
        content: The information to remember
        category: Category for organization

    Returns:
        Dict with success status and memory ID, or error
    """
    try:
        if not settings.memory_enabled:
            return {"error": "Memory system is disabled"}

        memory_manager = _get_memory_manager()
        memory_id = memory_manager.store_semantic_memory(content, category)

        return {
            "success": True,
            "memory_id": memory_id,
            "content": content,
            "category": category,
            "message": "Memory stored successfully in semantic memory (with embeddings)"
        }
    except ImportError as e:
        return {
            "error": f"ChromaDB or sentence-transformers not installed. Install with: pip install chromadb sentence-transformers. Details: {str(e)}"
        }
    except Exception as e:
        return {"error": f"Failed to store semantic memory: {str(e)}"}


def search_memories(query: str, category: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Search semantic memories by similarity.

    Uses vector embeddings to find memories that are conceptually related
    to the query, even if they don't share exact keywords.

    Args:
        query: Search query (will be embedded and compared)
        category: Optional category filter
        limit: Maximum number of results

    Returns:
        Dict with list of similar memories or error
    """
    try:
        if not settings.memory_enabled:
            return {"error": "Memory system is disabled"}

        # Ensure limit is an integer (LLM might pass as string)
        limit = int(limit) if limit is not None else 5

        memory_manager = _get_memory_manager()
        memories = memory_manager.semantic_search(query=query, category=category, limit=limit)

        return {
            "success": True,
            "count": len(memories),
            "query": query,
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "category": m.category,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in memories
            ]
        }
    except ImportError as e:
        return {
            "error": f"ChromaDB or sentence-transformers not installed. Install with: pip install chromadb sentence-transformers. Details: {str(e)}"
        }
    except Exception as e:
        return {"error": f"Failed to search semantic memories: {str(e)}"}


def extract_pdf_text(
    path: str,
    page_start: int | None = None,
    page_end: int | None = None,
    include_metadata: bool = True
) -> dict[str, Any]:
    """Extract text from a PDF file in the workspace.

    Args:
        path: Path to PDF file (relative to workspace)
        page_start: Optional starting page (0-indexed)
        page_end: Optional ending page (0-indexed, inclusive)
        include_metadata: Whether to include PDF metadata

    Returns:
        Dict with pages (list of {page_number, text}), total_pages, and optional metadata
    """
    try:
        if not settings.rag_enabled:
            return {"error": "RAG system is disabled"}

        from .rag import PDFExtractor

        # Resolve path with workspace sandboxing
        pdf_path = _resolve_path(path)

        if not pdf_path.exists():
            return {"error": f"PDF file not found: {path}"}

        if not pdf_path.suffix.lower() == ".pdf":
            return {"error": f"Not a PDF file: {path}"}

        # Extract text
        extractor = PDFExtractor()
        result = extractor.extract_text(
            pdf_path=pdf_path,
            page_start=page_start,
            page_end=page_end,
            include_metadata=include_metadata
        )

        return result

    except ImportError:
        return {"error": "pypdf not installed. Install with: pip install pypdf"}
    except Exception as e:
        return {"error": f"Failed to extract PDF text: {str(e)}"}


def index_document(
    path: str,
    category: str = "documents",
    custom_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Index a single PDF document for semantic search.

    Args:
        path: Path to PDF file (relative to workspace)
        category: Category for organization
        custom_metadata: Optional custom metadata

    Returns:
        Dict with document_id, chunks_created, and indexed_at timestamp
    """
    try:
        if not settings.rag_enabled:
            return {"error": "RAG system is disabled"}

        from .rag import DocumentIndexer

        # Resolve path with workspace sandboxing
        pdf_path = _resolve_path(path)

        if not pdf_path.exists():
            return {"error": f"PDF file not found: {path}"}

        if not pdf_path.suffix.lower() == ".pdf":
            return {"error": f"Not a PDF file: {path}"}

        # Index document
        memory_manager = _get_memory_manager()
        indexer = DocumentIndexer(memory_manager)

        result = indexer.index_document(
            pdf_path=pdf_path,
            category=category,
            custom_metadata=custom_metadata
        )

        return result

    except ImportError as e:
        return {"error": f"Missing dependency: {str(e)}. Install with: pip install pypdf chromadb sentence-transformers"}
    except Exception as e:
        return {"error": f"Failed to index document: {str(e)}"}


def index_directory(
    path: str,
    category: str = "documents",
    recursive: bool = False,
    pattern: str = "*.pdf"
) -> dict[str, Any]:
    """Batch index all PDF documents in a directory.

    Args:
        path: Path to directory (relative to workspace)
        category: Category for all documents
        recursive: Whether to search subdirectories
        pattern: Glob pattern for file matching

    Returns:
        Dict with documents_indexed, documents_failed, and results list
    """
    try:
        if not settings.rag_enabled:
            return {"error": "RAG system is disabled"}

        from .rag import DocumentIndexer

        # Resolve path with workspace sandboxing
        dir_path = _resolve_path(path)

        if not dir_path.exists():
            return {"error": f"Directory not found: {path}"}

        if not dir_path.is_dir():
            return {"error": f"Not a directory: {path}"}

        # Index directory
        memory_manager = _get_memory_manager()
        indexer = DocumentIndexer(memory_manager)

        result = indexer.index_directory(
            dir_path=dir_path,
            category=category,
            recursive=recursive,
            pattern=pattern
        )

        return result

    except ImportError as e:
        return {"error": f"Missing dependency: {str(e)}. Install with: pip install pypdf chromadb sentence-transformers"}
    except Exception as e:
        return {"error": f"Failed to index directory: {str(e)}"}


def search_documents(
    query: str,
    category: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    rerank: bool = True,
    use_bm25: bool = True,
    show_chunks: bool = False
) -> dict[str, Any]:
    """Search indexed PDF documents using hybrid semantic + keyword search.

    Args:
        query: Search query
        category: Optional category filter
        limit: Number of results to return (default: 10)
        min_score: Minimum similarity score (0-1)
        rerank: Use cross-encoder re-ranking
        use_bm25: Use BM25 keyword search alongside semantic search
        show_chunks: Show detailed chunk information for debugging

    Returns:
        Dict with results list (each with content, score, and citation info)
    """
    try:
        if not settings.rag_enabled:
            return {"error": "RAG system is disabled"}

        from .rag import DocumentSearcher

        # Ensure parameters are correct types
        limit = int(limit)
        min_score = float(min_score)

        # Validate limit
        if limit > settings.rag_max_retrieval_limit:
            return {"error": f"Limit exceeds maximum: {settings.rag_max_retrieval_limit}"}

        # Search documents
        memory_manager = _get_memory_manager()
        searcher = DocumentSearcher(memory_manager)

        result = searcher.search(
            query=query,
            category=category,
            limit=limit,
            min_score=min_score,
            rerank=rerank,
            use_bm25=use_bm25,
            show_chunks=show_chunks
        )

        return result

    except ImportError as e:
        return {"error": f"Missing dependency: {str(e)}. Install with: pip install chromadb sentence-transformers"}
    except Exception as e:
        return {"error": f"Failed to search documents: {str(e)}"}


def list_indexed_documents(category: str | None = None) -> dict[str, Any]:
    """List all indexed PDF documents.

    Args:
        category: Optional category filter

    Returns:
        Dict with documents list (each with metadata)
    """
    try:
        if not settings.rag_enabled:
            return {"error": "RAG system is disabled"}

        memory_manager = _get_memory_manager()
        documents = memory_manager.list_documents(category=category)

        return {
            "success": True,
            "count": len(documents),
            "documents": documents
        }

    except Exception as e:
        return {"error": f"Failed to list documents: {str(e)}"}


def delete_indexed_document(document_id: str) -> dict[str, Any]:
    """Delete an indexed document and all its chunks.

    Args:
        document_id: Document ID to delete

    Returns:
        Dict with success status and chunks_deleted count
    """
    try:
        if not settings.rag_enabled:
            return {"error": "RAG system is disabled"}

        memory_manager = _get_memory_manager()
        chunks_deleted = memory_manager.delete_document(document_id)

        return {
            "success": True,
            "document_id": document_id,
            "chunks_deleted": chunks_deleted,
            "message": f"Document deleted successfully ({chunks_deleted} chunks removed)"
        }

    except Exception as e:
        return {"error": f"Failed to delete document: {str(e)}"}


# =============================================================================
# Tool Dispatcher
# =============================================================================

TOOL_IMPLEMENTATIONS = {
    "web_search": web_search,
    "web_fetch": web_fetch,
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "bash": bash,
    "remember_context": remember_context,
    "recall_context": recall_context,
    "remember_fact": remember_fact,
    "recall_facts": recall_facts,
    "remember_semantically": remember_semantically,
    "search_memories": search_memories,
    "extract_pdf_text": extract_pdf_text,
    "index_document": index_document,
    "index_directory": index_directory,
    "search_documents": search_documents,
    "list_indexed_documents": list_indexed_documents,
    "delete_indexed_document": delete_indexed_document,
    "calendar_list_events": calendar_list_events,
    "calendar_add_event": calendar_add_event,
    "calendar_update_event": calendar_update_event,
    "calendar_delete_event": calendar_delete_event,
    "calendar_create_appointment_slots": calendar_create_appointment_slots,
    "calendar_mark_important_date": calendar_mark_important_date,
    "send_email": send_email,
    "list_emails": list_emails,
    "read_email": read_email,
    "delete_email": delete_email,
    "search_emails": search_emails,
    "list_folders": list_folders,
    "create_folder": create_folder,
    "move_email": move_email,
    "reply_email": reply_email,
    "forward_email": forward_email,
    "list_email_attachments": list_email_attachments,
    "save_email_attachment": save_email_attachment,
    "search_contacts": search_contacts,
    "list_contacts": list_contacts,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a tool by name with the given arguments.
    
    This is the main entry point called by the agent loop.
    """
    if name not in TOOL_IMPLEMENTATIONS:
        return {"error": f"Unknown tool: {name}"}
    
    try:
        return TOOL_IMPLEMENTATIONS[name](**arguments)
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}
