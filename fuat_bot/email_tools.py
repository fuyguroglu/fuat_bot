"""
Email tools for Fuat_bot.

Provides SMTP (send) and IMAP (read / search / delete) capabilities using
Python's standard library only — no extra dependencies required.

Multiple accounts are supported. Define them in .env as a JSON object:

    EMAIL_ACCOUNTS={"personal": {"address": "me@gmail.com", "password": "app-pass",
                                  "smtp_host": "smtp.gmail.com", "smtp_port": 587,
                                  "imap_host": "imap.gmail.com", "imap_port": 993},
                    "work":     {"address": "me@company.com", "password": "app-pass2",
                                  "smtp_host": "smtp.gmail.com", "smtp_port": 587,
                                  "imap_host": "imap.gmail.com", "imap_port": 993}}
    EMAIL_DEFAULT_ACCOUNT=personal

smtp_host / smtp_port / imap_host / imap_port are optional per account and
fall back to the Gmail defaults shown above.

Setup for Gmail (per account):
    1. Enable 2-Step Verification on the Google account.
    2. Generate an App Password: Google Account → Security → App Passwords.
    3. Use that 16-character password as the account's "password" value.
"""

import email as email_lib
import imaplib
import re
import smtplib
import ssl
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, getaddresses
from pathlib import Path
from typing import Any

from .config import settings

# Default server values used when an account doesn't specify them
_DEFAULT_SMTP_HOST = "smtp.gmail.com"
_DEFAULT_SMTP_PORT = 587
_DEFAULT_IMAP_HOST = "imap.gmail.com"
_DEFAULT_IMAP_PORT = 993


# =============================================================================
# Internal helpers
# =============================================================================


def _resolve_account(account: str | None) -> dict[str, Any] | str:
    """Return the account config dict, or an error string if not found.

    Precedence:
      1. `account` parameter (name key in EMAIL_ACCOUNTS)
      2. EMAIL_DEFAULT_ACCOUNT setting
      3. First account in EMAIL_ACCOUNTS (if exactly one exists)
    """
    if not settings.email_enabled:
        return "Email is disabled. Set EMAIL_ENABLED=true in .env."

    accounts = settings.email_accounts
    if not accounts:
        return (
            "No email accounts configured. "
            "Set EMAIL_ACCOUNTS in .env as a JSON object of named accounts."
        )

    # Determine which account to use
    name = account or settings.email_default_account
    if name:
        if name not in accounts:
            available = ", ".join(accounts.keys())
            return f"Account '{name}' not found. Available accounts: {available}"
        cfg = accounts[name]
    elif len(accounts) == 1:
        cfg = next(iter(accounts.values()))
    else:
        available = ", ".join(accounts.keys())
        return (
            f"Multiple accounts configured but no default set. "
            f"Specify an account name or set EMAIL_DEFAULT_ACCOUNT. "
            f"Available: {available}"
        )

    # Validate required fields
    if not cfg.get("address"):
        return "Account config is missing 'address'."
    if not cfg.get("password"):
        return "Account config is missing 'password'."

    # Fill in defaults for optional server fields
    return {
        "address": cfg["address"],
        "password": cfg["password"],
        "smtp_host": cfg.get("smtp_host", _DEFAULT_SMTP_HOST),
        "smtp_port": int(cfg.get("smtp_port", _DEFAULT_SMTP_PORT)),
        "imap_host": cfg.get("imap_host", _DEFAULT_IMAP_HOST),
        "imap_port": int(cfg.get("imap_port", _DEFAULT_IMAP_PORT)),
    }


def _is_gmail(cfg: dict[str, Any]) -> bool:
    """Return True if this account uses Gmail's SMTP (which auto-saves to Sent)."""
    return cfg.get("smtp_host", "") == "smtp.gmail.com"


def _save_to_sent(cfg: dict[str, Any], raw_message: bytes) -> None:
    """Append a sent message to the IMAP Sent folder.

    Gmail's own SMTP server saves to Sent automatically, so this is only
    called for non-Gmail accounts (e.g. Zimbra, Outlook).

    Sent folder name defaults to 'Sent' but can be overridden per account
    via the 'sent_folder' key in EMAIL_ACCOUNTS config.
    """
    import time

    sent_folder = cfg.get("sent_folder", "Sent")
    conn = _imap_connect(cfg)
    try:
        conn.append(
            sent_folder,
            r"\Seen",
            imaplib.Time2Internaldate(time.time()),
            raw_message,
        )
    finally:
        conn.logout()


def _imap_connect(cfg: dict[str, Any]) -> imaplib.IMAP4_SSL:
    """Open an authenticated IMAP4_SSL connection using the given account config."""
    ctx = ssl.create_default_context()
    conn = imaplib.IMAP4_SSL(cfg["imap_host"], cfg["imap_port"], ssl_context=ctx)
    conn.login(cfg["address"], cfg["password"])
    return conn


def _parse_message(raw: bytes) -> dict[str, Any]:
    """Parse a raw RFC 2822 message into a plain dict."""
    msg = email_lib.message_from_bytes(raw)

    # Decode subject (may be encoded as RFC 2047)
    subject_parts = email_lib.header.decode_header(msg.get("Subject", ""))
    subject = ""
    for part, enc in subject_parts:
        if isinstance(part, bytes):
            subject += part.decode(enc or "utf-8", errors="replace")
        else:
            subject += part

    body = ""
    if msg.is_multipart():
        # Prefer plain text
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                charset = part.get_content_charset() or "utf-8"
                body = part.get_payload(decode=True).decode(charset, errors="replace")
                break
        # Fall back to HTML if no plain text
        if not body:
            for part in msg.walk():
                ct = part.get_content_type()
                cd = str(part.get("Content-Disposition", ""))
                if ct == "text/html" and "attachment" not in cd:
                    charset = part.get_content_charset() or "utf-8"
                    body = part.get_payload(decode=True).decode(charset, errors="replace")
                    break
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(charset, errors="replace")

    return {
        "subject": subject.strip(),
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "cc": msg.get("Cc", ""),
        "reply_to": msg.get("Reply-To", ""),
        "date": msg.get("Date", ""),
        "message_id": msg.get("Message-ID", "").strip(),
        "in_reply_to": msg.get("In-Reply-To", "").strip(),
        "references": msg.get("References", "").strip(),
        "body": body.strip(),
    }


# =============================================================================
# Public tool functions
# =============================================================================


def send_email(
    to: str,
    subject: str,
    body: str,
    account: str | None = None,
    cc: str | None = None,
    bcc: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    """Send an email via SMTP.

    Args:
        to: Recipient address(es), comma-separated.
        subject: Email subject line.
        body: Plain-text body.
        account: Named account from EMAIL_ACCOUNTS to send from (uses default if omitted).
        cc: Optional CC address(es), comma-separated.
        bcc: Optional BCC address(es), comma-separated.
        html: Optional HTML body (sent as an alternative part alongside plain text).

    Returns:
        Dict with success status or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        msg = MIMEMultipart("alternative") if html else MIMEMultipart()
        msg["From"] = cfg["address"]
        msg["To"] = to
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc

        if html:
            msg.attach(MIMEText(body, "plain", "utf-8"))
            msg.attach(MIMEText(html, "html", "utf-8"))
        else:
            msg.attach(MIMEText(body, "plain", "utf-8"))

        all_recipients = [addr for _, addr in getaddresses([to])]
        if cc:
            all_recipients += [addr for _, addr in getaddresses([cc])]
        if bcc:
            all_recipients += [addr for _, addr in getaddresses([bcc])]

        raw = msg.as_bytes()

        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ctx)
            smtp.login(cfg["address"], cfg["password"])
            smtp.sendmail(cfg["address"], all_recipients, raw)

        # Save to Sent folder via IMAP (Gmail does this automatically; others don't)
        saved_to_sent = False
        if not _is_gmail(cfg):
            try:
                _save_to_sent(cfg, raw)
                saved_to_sent = True
            except Exception:
                pass  # Don't fail the whole send just because Sent-copy failed

        return {
            "success": True,
            "from": cfg["address"],
            "to": to,
            "subject": subject,
            "saved_to_sent": saved_to_sent or _is_gmail(cfg),
            "message": "Email sent successfully.",
        }

    except smtplib.SMTPAuthenticationError:
        return {"error": "SMTP authentication failed. Check address and password for this account."}
    except Exception as e:
        return {"error": f"Failed to send email: {str(e)}"}


def list_emails(
    folder: str = "INBOX",
    limit: int = 20,
    unread_only: bool = False,
    account: str | None = None,
) -> dict[str, Any]:
    """List emails in a mailbox folder via IMAP (newest first).

    Returns headers only (subject, from, date, UID). Use read_email for the full body.

    Args:
        folder: IMAP folder name (default: INBOX).
        limit: Maximum number of emails to return.
        unread_only: If True, only return unseen messages.
        account: Named account from EMAIL_ACCOUNTS (uses default if omitted).

    Returns:
        Dict with emails list and count, or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        limit = int(limit)
        conn = _imap_connect(cfg)
        try:
            status, _ = conn.select(folder, readonly=True)
            if status != "OK":
                return {"error": f"Could not open folder '{folder}'."}

            criteria = "UNSEEN" if unread_only else "ALL"
            status, data = conn.search(None, criteria)
            if status != "OK":
                return {"error": "Failed to search mailbox."}

            uids = data[0].split()
            selected = list(reversed(uids[-limit:] if len(uids) > limit else uids))

            emails = []
            for uid in selected:
                status, msg_data = conn.fetch(uid, "(RFC822.HEADER)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue
                parsed = _parse_message(msg_data[0][1])
                parsed["uid"] = uid.decode()
                parsed.pop("body", None)
                emails.append(parsed)
        finally:
            conn.logout()

        return {
            "success": True,
            "account": cfg["address"],
            "folder": folder,
            "count": len(emails),
            "emails": emails,
        }

    except imaplib.IMAP4.error as e:
        return {"error": f"IMAP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to list emails: {str(e)}"}


def read_email(
    uid: str,
    folder: str = "INBOX",
    account: str | None = None,
) -> dict[str, Any]:
    """Read the full content of a specific email by its UID.

    Marks the message as read (Seen) after fetching.

    Args:
        uid: Email UID (from list_emails or search_emails).
        folder: IMAP folder containing the email (default: INBOX).
        account: Named account from EMAIL_ACCOUNTS (uses default if omitted).

    Returns:
        Dict with full email content or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        conn = _imap_connect(cfg)
        try:
            status, _ = conn.select(folder, readonly=False)
            if status != "OK":
                return {"error": f"Could not open folder '{folder}'."}

            status, msg_data = conn.fetch(uid.encode(), "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                return {"error": f"Email with UID {uid} not found."}

            parsed = _parse_message(msg_data[0][1])
            parsed["uid"] = uid
            conn.store(uid.encode(), "+FLAGS", "\\Seen")
        finally:
            conn.logout()

        return {"success": True, "account": cfg["address"], "folder": folder, **parsed}

    except imaplib.IMAP4.error as e:
        return {"error": f"IMAP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to read email: {str(e)}"}


def delete_email(
    uid: str,
    folder: str = "INBOX",
    permanent: bool = False,
    account: str | None = None,
) -> dict[str, Any]:
    """Delete an email by its UID.

    By default moves the message to Trash (Gmail: [Gmail]/Trash).
    Set permanent=True to expunge immediately without moving to Trash.

    Args:
        uid: Email UID (from list_emails or search_emails).
        folder: IMAP folder containing the email (default: INBOX).
        permanent: If True, permanently delete. Default: False (move to Trash).
        account: Named account from EMAIL_ACCOUNTS (uses default if omitted).

    Returns:
        Dict with success status or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        conn = _imap_connect(cfg)
        try:
            status, _ = conn.select(folder, readonly=False)
            if status != "OK":
                return {"error": f"Could not open folder '{folder}'."}

            if permanent:
                conn.store(uid.encode(), "+FLAGS", "\\Deleted")
                conn.expunge()
                action = "permanently deleted"
            else:
                # Try to move to Trash (Gmail uses [Gmail]/Trash)
                trash_folder = "[Gmail]/Trash"
                result = conn.copy(uid.encode(), trash_folder)
                if result[0] == "OK":
                    conn.store(uid.encode(), "+FLAGS", "\\Deleted")
                    conn.expunge()
                    action = f"moved to {trash_folder}"
                else:
                    # Fall back to permanent delete if Trash folder not available
                    conn.store(uid.encode(), "+FLAGS", "\\Deleted")
                    conn.expunge()
                    action = "permanently deleted (Trash folder not found)"
        finally:
            conn.logout()

        return {"success": True, "account": cfg["address"], "uid": uid, "action": action}

    except imaplib.IMAP4.error as e:
        return {"error": f"IMAP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to delete email: {str(e)}"}


def list_folders(account: str | None = None) -> dict[str, Any]:
    """List all IMAP folders/mailboxes available on the account.

    Args:
        account: Named account from EMAIL_ACCOUNTS (uses default if omitted).

    Returns:
        Dict with folders list or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        conn = _imap_connect(cfg)
        try:
            status, data = conn.list()
            if status != "OK":
                return {"error": "Failed to list folders."}

            folders = []
            for item in data:
                if not isinstance(item, bytes):
                    continue
                decoded = item.decode("utf-8", errors="replace")
                # Folder name is the last token — may be quoted or bare
                match = re.search(r'"([^"]+)"\s*$', decoded)
                if match:
                    folders.append(match.group(1))
                else:
                    parts = decoded.rsplit(" ", 1)
                    if len(parts) == 2:
                        folders.append(parts[1].strip())
        finally:
            conn.logout()

        return {
            "success": True,
            "account": cfg["address"],
            "count": len(folders),
            "folders": folders,
        }

    except imaplib.IMAP4.error as e:
        return {"error": f"IMAP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to list folders: {str(e)}"}


def create_folder(
    folder: str,
    account: str | None = None,
) -> dict[str, Any]:
    """Create a new IMAP folder/mailbox on the account.

    Args:
        folder: Name of the folder to create (e.g. 'Archive', 'Students/2026').
        account: Named account from EMAIL_ACCOUNTS (uses default if omitted).

    Returns:
        Dict with success status or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        conn = _imap_connect(cfg)
        try:
            status, data = conn.create(folder)
            if status != "OK":
                reason = data[0].decode() if data and isinstance(data[0], bytes) else str(data)
                return {"error": f"Failed to create folder '{folder}': {reason}"}
        finally:
            conn.logout()

        return {
            "success": True,
            "account": cfg["address"],
            "folder": folder,
            "message": f"Folder '{folder}' created successfully.",
        }

    except imaplib.IMAP4.error as e:
        return {"error": f"IMAP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to create folder: {str(e)}"}


def move_email(
    uid: str,
    destination_folder: str,
    source_folder: str = "INBOX",
    account: str | None = None,
) -> dict[str, Any]:
    """Move an email from one folder to another.

    Copies the message to the destination then removes it from the source.

    Args:
        uid: Email UID (from list_emails or search_emails).
        destination_folder: Target IMAP folder name (e.g. 'Archive', '[Gmail]/Starred').
        source_folder: IMAP folder the email is currently in (default: INBOX).
        account: Named account from EMAIL_ACCOUNTS (uses default if omitted).

    Returns:
        Dict with success status or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        conn = _imap_connect(cfg)
        try:
            status, _ = conn.select(source_folder, readonly=False)
            if status != "OK":
                return {"error": f"Could not open folder '{source_folder}'."}

            result = conn.copy(uid.encode(), destination_folder)
            if result[0] != "OK":
                return {
                    "error": (
                        f"Could not copy to '{destination_folder}'. "
                        "The folder may not exist — use list_folders to check."
                    )
                }

            conn.store(uid.encode(), "+FLAGS", "\\Deleted")
            conn.expunge()
        finally:
            conn.logout()

        return {
            "success": True,
            "uid": uid,
            "from_folder": source_folder,
            "to_folder": destination_folder,
            "message": f"Email moved from '{source_folder}' to '{destination_folder}'.",
        }

    except imaplib.IMAP4.error as e:
        return {"error": f"IMAP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to move email: {str(e)}"}


def search_emails(
    query: str,
    folder: str = "INBOX",
    limit: int = 10,
    account: str | None = None,
) -> dict[str, Any]:
    """Search emails by subject, sender, or body text.

    Args:
        query: Search term matched against subject, from address, and body.
        folder: IMAP folder to search (default: INBOX).
        limit: Maximum number of results to return (newest first).
        account: Named account from EMAIL_ACCOUNTS (uses default if omitted).

    Returns:
        Dict with matching email headers or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        limit = int(limit)
        conn = _imap_connect(cfg)
        try:
            status, _ = conn.select(folder, readonly=True)
            if status != "OK":
                return {"error": f"Could not open folder '{folder}'."}

            encoded_query = query.replace('"', '\\"')
            search_criteria = (
                f'(OR OR SUBJECT "{encoded_query}" FROM "{encoded_query}" TEXT "{encoded_query}")'
            )
            status, data = conn.search("UTF-8", search_criteria)

            if status != "OK":
                # Fall back to simpler ASCII search
                search_criteria = (
                    f'(OR SUBJECT "{encoded_query}" FROM "{encoded_query}")'
                )
                status, data = conn.search(None, search_criteria)

            if status != "OK":
                return {"error": "Failed to execute search."}

            uids = data[0].split()
            selected = list(reversed(uids[-limit:] if len(uids) > limit else uids))

            emails = []
            for uid in selected:
                status, msg_data = conn.fetch(uid, "(RFC822.HEADER)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue
                parsed = _parse_message(msg_data[0][1])
                parsed["uid"] = uid.decode()
                parsed.pop("body", None)
                emails.append(parsed)
        finally:
            conn.logout()

        return {
            "success": True,
            "account": cfg["address"],
            "query": query,
            "folder": folder,
            "count": len(emails),
            "emails": emails,
        }

    except imaplib.IMAP4.error as e:
        return {"error": f"IMAP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to search emails: {str(e)}"}


# =============================================================================
# Internal helpers for reply/forward/attachments
# =============================================================================


def _fetch_raw_message(uid: str, folder: str, cfg: dict[str, Any]) -> bytes:
    """Fetch raw RFC 2822 bytes for a UID. Raises ValueError on failure."""
    conn = _imap_connect(cfg)
    try:
        status, _ = conn.select(folder, readonly=True)
        if status != "OK":
            raise ValueError(f"Could not open folder '{folder}'.")
        status, msg_data = conn.fetch(uid.encode(), "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            raise ValueError(f"Email with UID {uid} not found.")
        return msg_data[0][1]
    finally:
        conn.logout()


def _decode_header_value(raw: str) -> str:
    """Decode an RFC 2047-encoded header value to a plain string."""
    parts = email_lib.header.decode_header(raw)
    result = ""
    for part, enc in parts:
        if isinstance(part, bytes):
            result += part.decode(enc or "utf-8", errors="replace")
        else:
            result += part
    return result


def _build_references(original_references: str, original_message_id: str) -> str:
    """Build a References header value for a reply."""
    refs = original_references.strip()
    mid = original_message_id.strip()
    if mid:
        refs = f"{refs} {mid}".strip()
    return refs


# =============================================================================
# Reply / Forward
# =============================================================================


def reply_email(
    uid: str,
    body: str,
    folder: str = "INBOX",
    account: str | None = None,
    cc: str | None = None,
    reply_all: bool = False,
    html: str | None = None,
) -> dict[str, Any]:
    """Reply to an existing email, preserving threading headers.

    Fetches the original message to extract Subject, From, Message-ID, and
    References, then sends a properly threaded reply.

    Args:
        uid: UID of the email to reply to (from list_emails / search_emails).
        body: Plain-text reply body.
        folder: IMAP folder containing the original email (default: INBOX).
        account: Named account to send from (uses default if omitted).
        cc: Optional extra CC recipients, comma-separated.
        reply_all: If True, CC all original To/Cc recipients (default: False).
        html: Optional HTML version of the reply body.

    Returns:
        Dict with success status or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        raw = _fetch_raw_message(uid, folder, cfg)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to fetch original email: {str(e)}"}

    try:
        original = _parse_message(raw)

        # Determine reply-to address
        reply_to_raw = original.get("reply_to") or original.get("from", "")
        to_addr = reply_to_raw or original["from"]

        # Subject: prefix with "Re: " if not already present
        orig_subject = original.get("subject", "")
        if not orig_subject.lower().startswith("re:"):
            subject = f"Re: {orig_subject}"
        else:
            subject = orig_subject

        # Build CC list for reply-all
        all_cc_parts = []
        if reply_all:
            for field in ("to", "cc"):
                val = original.get(field, "")
                if val:
                    # Exclude our own address from CC
                    for name, addr in getaddresses([val]):
                        if addr.lower() != cfg["address"].lower():
                            all_cc_parts.append(addr)
        if cc:
            all_cc_parts += [addr for _, addr in getaddresses([cc])]
        combined_cc = ", ".join(all_cc_parts) if all_cc_parts else None

        # Compose the reply
        msg = MIMEMultipart("alternative") if html else MIMEMultipart()
        msg["From"] = cfg["address"]
        msg["To"] = to_addr
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = subject
        if combined_cc:
            msg["Cc"] = combined_cc

        # Threading headers
        orig_mid = original.get("message_id", "")
        if orig_mid:
            msg["In-Reply-To"] = orig_mid
            msg["References"] = _build_references(original.get("references", ""), orig_mid)

        if html:
            msg.attach(MIMEText(body, "plain", "utf-8"))
            msg.attach(MIMEText(html, "html", "utf-8"))
        else:
            msg.attach(MIMEText(body, "plain", "utf-8"))

        all_recipients = [addr for _, addr in getaddresses([to_addr])]
        if combined_cc:
            all_recipients += [addr for _, addr in getaddresses([combined_cc])]

        raw_reply = msg.as_bytes()

        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ctx)
            smtp.login(cfg["address"], cfg["password"])
            smtp.sendmail(cfg["address"], all_recipients, raw_reply)

        if not _is_gmail(cfg):
            try:
                _save_to_sent(cfg, raw_reply)
            except Exception:
                pass

        return {
            "success": True,
            "from": cfg["address"],
            "to": to_addr,
            "subject": subject,
            "in_reply_to": orig_mid,
            "message": "Reply sent successfully.",
        }

    except smtplib.SMTPAuthenticationError:
        return {"error": "SMTP authentication failed."}
    except Exception as e:
        return {"error": f"Failed to send reply: {str(e)}"}


def forward_email(
    uid: str,
    to: str,
    folder: str = "INBOX",
    account: str | None = None,
    note: str | None = None,
    cc: str | None = None,
) -> dict[str, Any]:
    """Forward an existing email to a new recipient.

    Prepends an optional note above a quoted copy of the original message.

    Args:
        uid: UID of the email to forward.
        to: Recipient address(es) to forward to, comma-separated.
        folder: IMAP folder containing the original email (default: INBOX).
        account: Named account to send from (uses default if omitted).
        note: Optional introductory text to add above the forwarded message.
        cc: Optional CC address(es), comma-separated.

    Returns:
        Dict with success status or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        raw = _fetch_raw_message(uid, folder, cfg)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to fetch original email: {str(e)}"}

    try:
        original = _parse_message(raw)

        # Subject: prefix with "Fwd: " if not already present
        orig_subject = original.get("subject", "")
        if not orig_subject.lower().startswith("fwd:"):
            subject = f"Fwd: {orig_subject}"
        else:
            subject = orig_subject

        # Build forwarded body
        divider = "\n\n---------- Forwarded message ----------\n"
        orig_meta = (
            f"From: {original.get('from', '')}\n"
            f"Date: {original.get('date', '')}\n"
            f"Subject: {original.get('subject', '')}\n"
            f"To: {original.get('to', '')}\n\n"
        )
        fwd_body = (note or "") + divider + orig_meta + original.get("body", "")

        msg = MIMEMultipart()
        msg["From"] = cfg["address"]
        msg["To"] = to
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc

        msg.attach(MIMEText(fwd_body, "plain", "utf-8"))

        all_recipients = [addr for _, addr in getaddresses([to])]
        if cc:
            all_recipients += [addr for _, addr in getaddresses([cc])]

        raw_fwd = msg.as_bytes()

        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ctx)
            smtp.login(cfg["address"], cfg["password"])
            smtp.sendmail(cfg["address"], all_recipients, raw_fwd)

        if not _is_gmail(cfg):
            try:
                _save_to_sent(cfg, raw_fwd)
            except Exception:
                pass

        return {
            "success": True,
            "from": cfg["address"],
            "to": to,
            "subject": subject,
            "message": "Email forwarded successfully.",
        }

    except smtplib.SMTPAuthenticationError:
        return {"error": "SMTP authentication failed."}
    except Exception as e:
        return {"error": f"Failed to forward email: {str(e)}"}


# =============================================================================
# Attachment tools
# =============================================================================


def _decode_filename(raw: str) -> str:
    """Decode an RFC 2047-encoded attachment filename."""
    parts = email_lib.header.decode_header(raw)
    result = ""
    for part, enc in parts:
        if isinstance(part, bytes):
            result += part.decode(enc or "utf-8", errors="replace")
        else:
            result += part
    return result


def list_email_attachments(
    uid: str,
    folder: str = "INBOX",
    account: str | None = None,
) -> dict[str, Any]:
    """List attachments on an email without downloading them.

    Args:
        uid: Email UID (from list_emails or search_emails).
        folder: IMAP folder containing the email (default: INBOX).
        account: Named account from EMAIL_ACCOUNTS (uses default if omitted).

    Returns:
        Dict with attachments list (filename, content_type, size_bytes) or error.
    """
    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        raw = _fetch_raw_message(uid, folder, cfg)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to fetch email: {str(e)}"}

    try:
        msg = email_lib.message_from_bytes(raw)
        attachments = []
        for part in msg.walk():
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" not in cd:
                continue
            raw_fn = part.get_filename()
            if raw_fn:
                filename = _decode_filename(raw_fn)
            else:
                filename = "unnamed"
            payload = part.get_payload(decode=True)
            attachments.append({
                "filename": filename,
                "content_type": part.get_content_type(),
                "size_bytes": len(payload) if payload else 0,
            })

        return {
            "success": True,
            "uid": uid,
            "count": len(attachments),
            "attachments": attachments,
        }

    except Exception as e:
        return {"error": f"Failed to list attachments: {str(e)}"}


def save_email_attachment(
    uid: str,
    filename: str,
    save_path: str,
    folder: str = "INBOX",
    account: str | None = None,
) -> dict[str, Any]:
    """Download and save a specific email attachment to the workspace.

    Args:
        uid: Email UID (from list_emails or search_emails).
        filename: Attachment filename to save (from list_email_attachments).
        save_path: Destination path inside the workspace (e.g. 'attachments/file.pdf').
        folder: IMAP folder containing the email (default: INBOX).
        account: Named account from EMAIL_ACCOUNTS (uses default if omitted).

    Returns:
        Dict with success status and saved path, or error.
    """
    from .config import settings

    cfg = _resolve_account(account)
    if isinstance(cfg, str):
        return {"error": cfg}

    try:
        raw = _fetch_raw_message(uid, folder, cfg)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to fetch email: {str(e)}"}

    try:
        msg = email_lib.message_from_bytes(raw)
        payload_bytes: bytes | None = None

        for part in msg.walk():
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" not in cd:
                continue
            raw_fn = part.get_filename()
            if not raw_fn:
                continue
            decoded_fn = _decode_filename(raw_fn)
            if decoded_fn == filename:
                payload_bytes = part.get_payload(decode=True)
                break

        if payload_bytes is None:
            return {"error": f"Attachment '{filename}' not found in email UID {uid}."}

        # Resolve destination path safely within workspace
        workspace = settings.workspace_dir.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        dest = (workspace / save_path).resolve()
        if not str(dest).startswith(str(workspace)):
            return {"error": "save_path escapes the workspace directory."}

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload_bytes)

        return {
            "success": True,
            "uid": uid,
            "filename": filename,
            "saved_to": save_path,
            "size_bytes": len(payload_bytes),
        }

    except Exception as e:
        return {"error": f"Failed to save attachment: {str(e)}"}
