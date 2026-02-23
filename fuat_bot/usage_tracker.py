"""
Usage tracking and cost limits for LLM API calls.

Tracks token usage and estimated costs per provider/model.
Enforces daily/monthly cost limits with override capability.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional
import sqlite3


class UsageLimitExceeded(Exception):
    """Raised when usage limits are exceeded."""

    def __init__(
        self,
        limit_type: Literal["daily", "monthly", "user_daily"],
        current: float,
        limit: float,
        telegram_user_id: Optional[int] = None,
    ):
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        self.telegram_user_id = telegram_user_id
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        user_info = f" (Telegram user {self.telegram_user_id})" if self.telegram_user_id else ""
        return (
            f"{self.limit_type.replace('_', ' ').title()} limit exceeded{user_info}: "
            f"${self.current:.4f} / ${self.limit:.2f}"
        )


@dataclass
class ModelPricing:
    """Pricing per 1M tokens."""

    input_cost: float  # USD per 1M input tokens
    output_cost: float  # USD per 1M output tokens


# Pricing as of February 2026 (USD per 1M tokens)
# Sources: https://platform.claude.com/docs/en/about-claude/pricing
#          https://ai.google.dev/gemini-api/docs/pricing
DEFAULT_PRICING = {
    # Anthropic Claude (current models)
    "claude-3-5-haiku-20241022": ModelPricing(1.00, 5.00),
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00),
    "claude-opus-4-5": ModelPricing(5.00, 25.00),
    "claude-opus-4-6": ModelPricing(5.00, 25.00),
    # Legacy Anthropic models
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00),
    "claude-opus-4-0": ModelPricing(15.00, 75.00),
    "claude-opus-4-1": ModelPricing(15.00, 75.00),
    # Google Gemini
    "models/gemini-2.0-flash": ModelPricing(0.10, 0.40),
    "models/gemini-2.0-flash-exp": ModelPricing(0.10, 0.40),
    "models/gemini-2.5-flash-lite": ModelPricing(0.10, 0.40),
    "models/gemini-2.5-flash": ModelPricing(0.10, 0.40),
    "models/gemini-1.5-flash": ModelPricing(0.10, 0.40),
    "models/gemini-2.5-pro": ModelPricing(1.25, 10.00),
    "models/gemini-1.5-pro": ModelPricing(1.25, 10.00),
    "models/gemini-3.0-pro-preview": ModelPricing(2.00, 12.00),
    # Ollama (local, free)
    "ollama": ModelPricing(0.0, 0.0),
}


class UsageTracker:
    """Tracks API usage and enforces cost limits."""

    def __init__(
        self,
        db_path: Path,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None,
        telegram_user_daily_limit: Optional[float] = None,
        custom_pricing: Optional[dict[str, ModelPricing]] = None,
    ):
        self.db_path = db_path
        self.daily_limit = daily_limit  # None = no limit
        self.monthly_limit = monthly_limit  # None = no limit
        self.telegram_user_daily_limit = telegram_user_daily_limit  # None = no limit
        self.pricing = DEFAULT_PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Global daily usage
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS global_usage (
                date TEXT PRIMARY KEY,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cost REAL DEFAULT 0.0,
                requests INTEGER DEFAULT 0,
                overridden_at TEXT
            )
            """
        )

        # Per-user Telegram daily usage
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_usage (
                telegram_user_id INTEGER,
                date TEXT,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                cost REAL DEFAULT 0.0,
                requests INTEGER DEFAULT 0,
                overridden_at TEXT,
                PRIMARY KEY (telegram_user_id, date)
            )
            """
        )

        # Monthly usage (for monthly limits)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS monthly_usage (
                year_month TEXT PRIMARY KEY,
                cost REAL DEFAULT 0.0
            )
            """
        )

        conn.commit()
        conn.close()

    def _get_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost for given token usage."""
        pricing = self.pricing.get(model, ModelPricing(0.0, 0.0))
        cost = (tokens_in / 1_000_000 * pricing.input_cost) + (
            tokens_out / 1_000_000 * pricing.output_cost
        )
        return cost

    def check_limits(
        self, model: str, tokens_in: int, tokens_out: int, telegram_user_id: Optional[int] = None
    ) -> None:
        """
        Check if adding this usage would exceed limits.
        Raises UsageLimitExceeded if limits are exceeded and not overridden.
        """
        today = date.today().isoformat()
        this_month = date.today().strftime("%Y-%m")
        cost = self._get_cost(model, tokens_in, tokens_out)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check Telegram user limit first (if applicable)
        if telegram_user_id is not None and self.telegram_user_daily_limit is not None:
            cursor.execute(
                "SELECT cost, overridden_at FROM user_usage WHERE telegram_user_id = ? AND date = ?",
                (telegram_user_id, today),
            )
            row = cursor.fetchone()
            current_user_cost = row[0] if row else 0.0
            user_overridden = row[1] if row else None

            new_user_cost = current_user_cost + cost

            if new_user_cost > self.telegram_user_daily_limit and not user_overridden:
                conn.close()
                raise UsageLimitExceeded(
                    "user_daily", new_user_cost, self.telegram_user_daily_limit, telegram_user_id
                )

        # Check global daily limit
        if self.daily_limit is not None:
            cursor.execute(
                "SELECT cost, overridden_at FROM global_usage WHERE date = ?", (today,)
            )
            row = cursor.fetchone()
            current_daily_cost = row[0] if row else 0.0
            daily_overridden = row[1] if row else None

            new_daily_cost = current_daily_cost + cost

            if new_daily_cost > self.daily_limit and not daily_overridden:
                conn.close()
                raise UsageLimitExceeded("daily", new_daily_cost, self.daily_limit)

        # Check global monthly limit
        if self.monthly_limit is not None:
            cursor.execute(
                "SELECT cost FROM monthly_usage WHERE year_month = ?", (this_month,)
            )
            row = cursor.fetchone()
            current_monthly_cost = row[0] if row else 0.0

            new_monthly_cost = current_monthly_cost + cost

            if new_monthly_cost > self.monthly_limit:
                conn.close()
                raise UsageLimitExceeded("monthly", new_monthly_cost, self.monthly_limit)

        conn.close()

    def log_usage(
        self, model: str, tokens_in: int, tokens_out: int, telegram_user_id: Optional[int] = None
    ) -> float:
        """
        Log usage after a successful API call.
        Returns the cost of this request.
        """
        today = date.today().isoformat()
        this_month = date.today().strftime("%Y-%m")
        cost = self._get_cost(model, tokens_in, tokens_out)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update global daily usage
        cursor.execute(
            """
            INSERT INTO global_usage (date, tokens_in, tokens_out, cost, requests)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(date) DO UPDATE SET
                tokens_in = tokens_in + ?,
                tokens_out = tokens_out + ?,
                cost = cost + ?,
                requests = requests + 1
            """,
            (today, tokens_in, tokens_out, cost, tokens_in, tokens_out, cost),
        )

        # Update monthly usage
        cursor.execute(
            """
            INSERT INTO monthly_usage (year_month, cost)
            VALUES (?, ?)
            ON CONFLICT(year_month) DO UPDATE SET
                cost = cost + ?
            """,
            (this_month, cost, cost),
        )

        # Update per-user usage if Telegram
        if telegram_user_id is not None:
            cursor.execute(
                """
                INSERT INTO user_usage (telegram_user_id, date, tokens_in, tokens_out, cost, requests)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(telegram_user_id, date) DO UPDATE SET
                    tokens_in = tokens_in + ?,
                    tokens_out = tokens_out + ?,
                    cost = cost + ?,
                    requests = requests + 1
                """,
                (
                    telegram_user_id,
                    today,
                    tokens_in,
                    tokens_out,
                    cost,
                    tokens_in,
                    tokens_out,
                    cost,
                ),
            )

        conn.commit()
        conn.close()

        return cost

    def set_override(
        self, override_type: Literal["daily", "user_daily"], telegram_user_id: Optional[int] = None
    ) -> None:
        """Set override flag for today's limits."""
        today = date.today().isoformat()
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if override_type == "daily":
            cursor.execute(
                """
                INSERT INTO global_usage (date, overridden_at)
                VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET overridden_at = ?
                """,
                (today, now, now),
            )
        elif override_type == "user_daily" and telegram_user_id is not None:
            cursor.execute(
                """
                INSERT INTO user_usage (telegram_user_id, date, overridden_at)
                VALUES (?, ?, ?)
                ON CONFLICT(telegram_user_id, date) DO UPDATE SET overridden_at = ?
                """,
                (telegram_user_id, today, now, now),
            )

        conn.commit()
        conn.close()

    def clear_override(
        self, override_type: Literal["daily", "user_daily"], telegram_user_id: Optional[int] = None
    ) -> None:
        """Clear override flag (resets limit to trigger again at same threshold)."""
        today = date.today().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if override_type == "daily":
            cursor.execute(
                """
                UPDATE global_usage SET overridden_at = NULL WHERE date = ?
                """,
                (today,),
            )
        elif override_type == "user_daily" and telegram_user_id is not None:
            cursor.execute(
                """
                UPDATE user_usage SET overridden_at = NULL
                WHERE telegram_user_id = ? AND date = ?
                """,
                (telegram_user_id, today),
            )

        conn.commit()
        conn.close()

    def get_daily_stats(self, telegram_user_id: Optional[int] = None) -> dict:
        """Get today's usage statistics."""
        today = date.today().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Global stats
        cursor.execute(
            "SELECT tokens_in, tokens_out, cost, requests, overridden_at FROM global_usage WHERE date = ?",
            (today,),
        )
        row = cursor.fetchone()
        if row:
            stats["global"] = {
                "tokens_in": row[0],
                "tokens_out": row[1],
                "cost": row[2],
                "requests": row[3],
                "overridden": bool(row[4]),
                "limit": self.daily_limit,
            }
        else:
            stats["global"] = {
                "tokens_in": 0,
                "tokens_out": 0,
                "cost": 0.0,
                "requests": 0,
                "overridden": False,
                "limit": self.daily_limit,
            }

        # User stats (if applicable)
        if telegram_user_id is not None:
            cursor.execute(
                "SELECT tokens_in, tokens_out, cost, requests, overridden_at FROM user_usage WHERE telegram_user_id = ? AND date = ?",
                (telegram_user_id, today),
            )
            row = cursor.fetchone()
            if row:
                stats["user"] = {
                    "tokens_in": row[0],
                    "tokens_out": row[1],
                    "cost": row[2],
                    "requests": row[3],
                    "overridden": bool(row[4]),
                    "limit": self.telegram_user_daily_limit,
                }
            else:
                stats["user"] = {
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost": 0.0,
                    "requests": 0,
                    "overridden": False,
                    "limit": self.telegram_user_daily_limit,
                }

        conn.close()
        return stats

    def get_monthly_stats(self) -> dict:
        """Get this month's usage statistics."""
        this_month = date.today().strftime("%Y-%m")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT cost FROM monthly_usage WHERE year_month = ?", (this_month,)
        )
        row = cursor.fetchone()

        stats = {
            "cost": row[0] if row else 0.0,
            "limit": self.monthly_limit,
            "month": this_month,
        }

        conn.close()
        return stats

    def reset_daily(self) -> None:
        """Reset today's usage counters (mainly for testing)."""
        today = date.today().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM global_usage WHERE date = ?", (today,))
        cursor.execute("DELETE FROM user_usage WHERE date = ?", (today,))
        conn.commit()
        conn.close()

    def reset_monthly(self) -> None:
        """Reset this month's usage counter (mainly for testing)."""
        this_month = date.today().strftime("%Y-%m")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM monthly_usage WHERE year_month = ?", (this_month,))
        conn.commit()
        conn.close()
