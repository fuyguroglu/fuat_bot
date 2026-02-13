"""
Telegram bot interface for Fuat_bot.

Bridges the synchronous agent loop to the async python-telegram-bot library.
Each Telegram user gets their own isolated session and memory.

Usage:
    python -m fuat_bot telegram-start
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import settings

# Telegram message limit (hard limit is 4096 chars)
_TG_MAX_LEN = 4000  # slight buffer below the hard limit


def _strip_markdown(text: str) -> str:
    """Strip common Markdown markers for plain-text Telegram delivery.

    Removes bold (**), italic (*), headers (##), inline code (`) and
    code fences (```). Leaves the text readable without any formatting noise.
    """
    # Code fences (``` … ```)
    text = re.sub(r"```[^\n]*\n?", "", text)
    text = re.sub(r"```", "", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold / italic (*** or ** or *)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    # ATX headers (### Title → Title)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Clean up excessive blank lines left after stripping
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _utf16_len(text: str) -> int:
    """Return the length of *text* in UTF-16 code units.

    Telegram measures message length in UTF-16 code units.  Characters in the
    Basic Multilingual Plane (U+0000–U+FFFF) count as 1 unit; characters
    outside the BMP (e.g. most emoji, U+10000+) count as 2 units.
    """
    return sum(2 if ord(c) > 0xFFFF else 1 for c in text)


def _utf16_safe_cut(text: str, max_units: int) -> int:
    """Return the character index at which *text* first exceeds *max_units* UTF-16 units.

    The returned index is the last character position whose prefix fits within
    *max_units*, so ``text[:result]`` is always a valid Python string and its
    UTF-16 length is ≤ *max_units*.
    """
    units = 0
    for i, ch in enumerate(text):
        ch_units = 2 if ord(ch) > 0xFFFF else 1
        if units + ch_units > max_units:
            return i
        units += ch_units
    return len(text)


def _split_message(text: str, max_len: int = _TG_MAX_LEN) -> list[str]:
    """Split *text* into chunks that each fit within *max_len* UTF-16 code units.

    Telegram's hard limit is 4096 UTF-16 code units; _TG_MAX_LEN = 4000 adds
    a safety margin.  Prefers splitting on paragraph (\\n\\n), line (\\n), or
    word boundaries, and falls back to a hard cut only as a last resort.
    """
    if _utf16_len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    while text:
        if _utf16_len(text) <= max_len:
            chunks.append(text)
            break

        # Find the character index that fits within max_len UTF-16 units
        cut = _utf16_safe_cut(text, max_len)
        slice_text = text[:cut]

        # Prefer a natural boundary within that slice
        idx = slice_text.rfind("\n\n")
        if idx == -1:
            idx = slice_text.rfind("\n")
        if idx == -1:
            idx = slice_text.rfind(" ")
        if idx == -1:
            # Hard cut at the UTF-16 boundary (already valid Unicode)
            idx = cut

        chunks.append(text[:idx].rstrip())
        text = text[idx:].lstrip()

    return [c for c in chunks if c]


class TelegramBot:
    """Telegram interface that wraps the Fuat_bot agent loop.

    Maintains a per-user agent cache so each user keeps their session in
    memory for the lifetime of the process (sessions are also persisted to
    JSONL files so they survive restarts).
    """

    def __init__(self) -> None:
        if not settings.telegram_bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is not set. "
                "Add it to .env and set TELEGRAM_ENABLED=true."
            )

        self._token: str = settings.telegram_bot_token
        self._allowed: set[int] = set(settings.telegram_allowed_users)
        self._open: bool = settings.telegram_open_access

        # user_id → Agent instance (lazy, created on first message)
        self._agents: dict[int, Any] = {}

        # Where to log unauthorized access attempts
        self._unauthorized_log: Path = Path("./telegram_unauthorized.jsonl")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_allowed(self, user_id: int) -> bool:
        """Return True if *user_id* is permitted to use the bot."""
        if self._open:
            return True
        return user_id in self._allowed

    def _get_or_create_agent(self, user_id: int) -> Any:
        """Return the cached Agent for *user_id*, creating one if needed."""
        if user_id not in self._agents:
            from .agent import create_agent
            session_id = f"tg_{user_id}"
            self._agents[user_id] = create_agent(session_id)
        return self._agents[user_id]

    def _log_unknown_user(self, tg_user: Any) -> None:
        """Append a record for an unauthorized access attempt.

        Stores only public Telegram profile metadata — not message content.
        """
        record = {
            "user_id": tg_user.id,
            "username": tg_user.username,
            "full_name": tg_user.full_name,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with self._unauthorized_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # Don't crash the bot if logging fails

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def handle_start(self, update: Any, context: Any) -> None:
        """/start — greet the user."""
        user = update.effective_user
        if not self._is_allowed(user.id):
            self._log_unknown_user(user)
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            return

        await update.message.reply_text(
            f"Hi {user.first_name}! I'm Fuat's personal AI assistant.\n\n"
            "Just send me a message and I'll help you out.\n\n"
            "Commands:\n"
            "/help — show this help\n"
            "/reset — start a fresh conversation"
        )

    async def handle_help(self, update: Any, context: Any) -> None:
        """/help — show available commands."""
        user = update.effective_user
        if not self._is_allowed(user.id):
            self._log_unknown_user(user)
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            return

        await update.message.reply_text(
            "Available commands:\n\n"
            "/start — introduction\n"
            "/reset — clear current session and start fresh\n"
            "/help — show this message\n\n"
            "For everything else, just type your message."
        )

    async def handle_reset(self, update: Any, context: Any) -> None:
        """/reset — drop the current session so the next message starts fresh."""
        user = update.effective_user
        if not self._is_allowed(user.id):
            self._log_unknown_user(user)
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            return

        self._agents.pop(user.id, None)
        await update.message.reply_text(
            "Session cleared. Your next message will start a fresh conversation."
        )

    async def handle_message(self, update: Any, context: Any) -> None:
        """Handle an incoming text message."""
        user = update.effective_user
        text = (update.message.text or "").strip()

        if not text:
            return

        if not self._is_allowed(user.id):
            self._log_unknown_user(user)
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            return

        # Show "typing…" while the agent thinks
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
        )

        # Run the synchronous agent loop in a thread to avoid blocking
        agent = self._get_or_create_agent(user.id)
        try:
            response: str = await asyncio.to_thread(agent.chat, text)
        except Exception as exc:
            await update.message.reply_text(
                f"Sorry, something went wrong: {exc}"
            )
            return

        # Clean up markdown and split if necessary
        clean = _strip_markdown(response)
        chunks = _split_message(clean)

        for chunk in chunks:
            await update.message.reply_text(chunk)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the bot using long-polling (blocks until Ctrl+C)."""
        try:
            from telegram.ext import Application, CommandHandler, MessageHandler, filters
        except ImportError:
            raise ImportError(
                "python-telegram-bot is not installed.\n"
                "Run: pip install 'python-telegram-bot>=21.0'"
            )

        from rich.console import Console
        console = Console()

        console.print(
            f"[bold green]Telegram bot starting...[/bold green]\n"
            f"Allowed users: {sorted(self._allowed) if self._allowed else '(open access)'}\n"
            f"Press Ctrl+C to stop."
        )

        app = Application.builder().token(self._token).build()

        app.add_handler(CommandHandler("start", self.handle_start))
        app.add_handler(CommandHandler("help", self.handle_help))
        app.add_handler(CommandHandler("reset", self.handle_reset))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        app.run_polling(drop_pending_updates=True)
