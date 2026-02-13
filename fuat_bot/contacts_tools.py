"""
Contacts tools for Fuat_bot.

Provides contact lookup across configured email accounts:
  - Gmail accounts (imap_host == imap.gmail.com): Google People API
    Requires one-time OAuth setup: python -m fuat_bot contacts-setup
  - All other accounts (Zimbra, Outlook, etc.): CardDAV
    Uses existing email credentials — no extra setup needed.

Tools:
  search_contacts(query, account?) — find contacts by name or email
  list_contacts(account?, limit?)  — list all contacts
"""

import re
import xml.etree.ElementTree as ET
from typing import Any

from .config import settings

_GMAIL_IMAP_HOST = "imap.gmail.com"
_CARDDAV_NS = "urn:ietf:params:xml:ns:carddav"
_DAV_NS = "DAV:"


# =============================================================================
# Helpers — account resolution
# =============================================================================


def _get_all_accounts() -> dict[str, dict]:
    """Return all configured email accounts, or {} if none."""
    return settings.email_accounts or {}


def _resolve_accounts(account: str | None) -> list[tuple[str, dict]] | str:
    """Return a list of (name, cfg) pairs to query, or an error string."""
    if not settings.email_enabled:
        return "Email is disabled. Set EMAIL_ENABLED=true in .env."

    all_accounts = _get_all_accounts()
    if not all_accounts:
        return "No email accounts configured in EMAIL_ACCOUNTS."

    if account:
        if account not in all_accounts:
            available = ", ".join(all_accounts.keys())
            return f"Account '{account}' not found. Available: {available}"
        return [(account, all_accounts[account])]

    return list(all_accounts.items())


def _is_gmail(cfg: dict) -> bool:
    return cfg.get("imap_host", "") == _GMAIL_IMAP_HOST


# =============================================================================
# Gmail — Google People API
# =============================================================================


def _get_people_service():
    """Return an authenticated Google People API service.

    Uses contacts_token.json (created by `contacts-setup`).
    Raises RuntimeError with a helpful message if not set up.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError(
            "Google API libraries not installed. Run: "
            "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )

    SCOPES = ["https://www.googleapis.com/auth/contacts.readonly"]
    token_path = settings.google_contacts_token_file
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds:
        raise RuntimeError(
            "Google Contacts not set up. Run: python -m fuat_bot contacts-setup"
        )

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        else:
            raise RuntimeError(
                "Google Contacts token expired. Re-run: python -m fuat_bot contacts-setup"
            )

    return build("people", "v1", credentials=creds)


def _search_gmail_contacts(query: str, limit: int) -> list[dict]:
    """Search Gmail contacts using the People API."""
    service = _get_people_service()

    results = service.people().searchContacts(
        query=query,
        readMask="names,emailAddresses,phoneNumbers,organizations",
        pageSize=min(limit, 30),
    ).execute()

    contacts = []
    for person in results.get("results", []):
        p = person.get("person", {})
        contacts.append(_parse_google_person(p))
    return contacts


def _list_gmail_contacts(limit: int) -> list[dict]:
    """List all Gmail contacts using the People API."""
    service = _get_people_service()

    results = service.people().connections().list(
        resourceName="people/me",
        pageSize=min(limit, 1000),
        personFields="names,emailAddresses,phoneNumbers,organizations",
        sortOrder="FIRST_NAME_ASCENDING",
    ).execute()

    contacts = []
    for person in results.get("connections", []):
        contacts.append(_parse_google_person(person))
    return contacts[:limit]


def _parse_google_person(person: dict) -> dict:
    """Flatten a Google People API person resource into a simple dict."""
    names = person.get("names", [])
    emails = person.get("emailAddresses", [])
    phones = person.get("phoneNumbers", [])
    orgs = person.get("organizations", [])

    name = names[0].get("displayName", "") if names else ""
    email_list = [e.get("value", "") for e in emails if e.get("value")]
    phone_list = [p.get("value", "") for p in phones if p.get("value")]
    org = orgs[0].get("name", "") if orgs else ""

    return {
        "name": name,
        "emails": email_list,
        "phones": phone_list,
        "organization": org,
    }


# =============================================================================
# CardDAV — Zimbra / generic
# =============================================================================


def _carddav_url(cfg: dict) -> str:
    """Build the CardDAV Contacts collection URL from the account config.

    Tries HTTPS first; falls back to the port already on imap_host if needed.
    Zimbra format: https://{imap_host}/dav/{email}/Contacts/
    """
    host = cfg.get("imap_host", "")
    email = cfg.get("address", "")
    return f"https://{host}/dav/{email}/Contacts/"


def _fetch_carddav_contacts(cfg: dict) -> list[dict]:
    """Fetch all contacts from a CardDAV server.

    Strategy:
      1. PROPFIND Depth:1 to list all .vcf resource URLs.
      2. addressbook-multiget REPORT to fetch all vCards in one request.
         If that fails, fall back to individual GET requests per URL.
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx not installed. Run: pip install httpx")

    base_url = _carddav_url(cfg)
    auth = (cfg["address"], cfg["password"])
    # Extract just the scheme+host for building absolute URLs
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host_root = f"{parsed.scheme}://{parsed.netloc}"

    # Step 1: PROPFIND Depth:1 to list contact URLs
    propfind_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<D:propfind xmlns:D=\"DAV:\"><D:prop><D:getetag/></D:prop></D:propfind>"
    )
    r = httpx.request(
        "PROPFIND", base_url,
        content=propfind_body.encode(),
        headers={"Content-Type": "application/xml", "Depth": "1"},
        auth=auth, timeout=15, follow_redirects=True,
    )
    if r.status_code not in (200, 207):
        raise RuntimeError(
            f"CardDAV PROPFIND failed with status {r.status_code}. URL: {base_url}"
        )

    root = ET.fromstring(r.text)
    hrefs = [
        el.text for el in root.iter(f"{{{_DAV_NS}}}href")
        if el.text and el.text.endswith(".vcf")
    ]
    if not hrefs:
        return []

    # Step 2a: Try addressbook-multiget REPORT (one request for all vCards)
    href_xml = "".join(f"<D:href>{h}</D:href>" for h in hrefs)
    multiget_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<C:addressbook-multiget xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:carddav">'
        "<D:prop><D:getetag/><C:address-data/></D:prop>"
        f"{href_xml}"
        "</C:addressbook-multiget>"
    )
    mr = httpx.request(
        "REPORT", base_url,
        content=multiget_body.encode(),
        headers={"Content-Type": "application/xml; charset=utf-8"},
        auth=auth, timeout=15, follow_redirects=True,
    )
    if mr.status_code in (200, 207):
        contacts = _parse_carddav_response(mr.text)
        if contacts:
            return contacts

    # Step 2b: Fall back to individual GET per contact URL
    contacts = []
    for href in hrefs:
        vcf_url = href if href.startswith("http") else host_root + href
        gr = httpx.get(vcf_url, auth=auth, timeout=10, follow_redirects=True)
        if gr.status_code == 200 and gr.text.strip():
            contact = _parse_vcard(gr.text)
            if contact:
                contacts.append(contact)
    return contacts


def _parse_carddav_response(xml_text: str) -> list[dict]:
    """Parse a multi-status CardDAV REPORT response into contact dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    contacts = []
    for response in root.iter(f"{{{_DAV_NS}}}response"):
        # Find address-data element
        addr_data = response.find(
            f".//{{{_CARDDAV_NS}}}address-data"
        )
        if addr_data is None or not addr_data.text:
            continue
        contact = _parse_vcard(addr_data.text)
        if contact:
            contacts.append(contact)

    return contacts


def _parse_vcard(vcard_text: str) -> dict | None:
    """Parse a single vCard string into a contact dict."""
    # Unfold continuation lines (RFC 6350 §3.2)
    unfolded = re.sub(r"\r?\n[ \t]", "", vcard_text)

    def get_values(field: str) -> list[str]:
        """Return all values for a given vCard field prefix."""
        pattern = rf"^{re.escape(field)}(?:[;:][^\r\n]*)$"
        values = []
        for line in unfolded.splitlines():
            if re.match(pattern, line, re.IGNORECASE):
                # Everything after the first ':'
                values.append(line.split(":", 1)[-1].strip())
        return values

    name = ""
    fn_values = get_values("FN")
    if fn_values:
        name = fn_values[0]

    # If FN is empty, try structured name (N field: Last;First;...)
    if not name:
        n_values = get_values("N")
        if n_values:
            parts = n_values[0].split(";")
            name = " ".join(p.strip() for p in reversed(parts[:2]) if p.strip())

    emails = []
    for line in unfolded.splitlines():
        if re.match(r"^EMAIL", line, re.IGNORECASE):
            val = line.split(":", 1)[-1].strip()
            if val and "@" in val:
                emails.append(val)

    phones = []
    for line in unfolded.splitlines():
        if re.match(r"^TEL", line, re.IGNORECASE):
            val = line.split(":", 1)[-1].strip()
            if val:
                phones.append(val)

    org = ""
    org_values = get_values("ORG")
    if org_values:
        org = org_values[0].split(";")[0].strip()

    # Skip empty / no-name entries
    if not name and not emails:
        return None

    return {"name": name, "emails": emails, "phones": phones, "organization": org}


def _search_carddav_contacts(query: str, cfg: dict, limit: int) -> list[dict]:
    """Fetch all CardDAV contacts then filter by query (client-side)."""
    all_contacts = _fetch_carddav_contacts(cfg)
    q = query.lower()
    matches = [
        c for c in all_contacts
        if q in c["name"].lower()
        or any(q in e.lower() for e in c["emails"])
        or q in c["organization"].lower()
    ]
    return matches[:limit]


# =============================================================================
# Public tool functions
# =============================================================================


def search_contacts(
    query: str,
    account: str | None = None,
) -> dict[str, Any]:
    """Search contacts by name or email address.

    Searches Gmail accounts via the Google People API and other accounts
    (Zimbra, etc.) via CardDAV. Requires one-time OAuth setup for Gmail
    accounts: python -m fuat_bot contacts-setup.

    Args:
        query: Name or email fragment to search for.
        account: Named account from EMAIL_ACCOUNTS (searches all if omitted).

    Returns:
        Dict with results list grouped by account, or error.
    """
    accounts = _resolve_accounts(account)
    if isinstance(accounts, str):
        return {"error": accounts}

    all_results: list[dict] = []
    errors: list[str] = []

    for name, cfg in accounts:
        try:
            if _is_gmail(cfg):
                contacts = _search_gmail_contacts(query, limit=20)
            else:
                contacts = _search_carddav_contacts(query, cfg, limit=20)

            for c in contacts:
                c["account"] = name
            all_results.extend(contacts)

        except RuntimeError as e:
            errors.append(f"{name}: {e}")
        except Exception as e:
            errors.append(f"{name}: {e}")

    result: dict[str, Any] = {
        "success": True,
        "query": query,
        "count": len(all_results),
        "contacts": all_results,
    }
    if errors:
        result["warnings"] = errors
    return result


def list_contacts(
    account: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List contacts from one or all configured accounts.

    Args:
        account: Named account from EMAIL_ACCOUNTS (lists all accounts if omitted).
        limit: Maximum contacts to return per account (default: 50).

    Returns:
        Dict with contacts list grouped by account, or error.
    """
    accounts = _resolve_accounts(account)
    if isinstance(accounts, str):
        return {"error": accounts}

    limit = int(limit)
    all_contacts: list[dict] = []
    errors: list[str] = []

    for name, cfg in accounts:
        try:
            if _is_gmail(cfg):
                contacts = _list_gmail_contacts(limit)
            else:
                contacts = _fetch_carddav_contacts(cfg)[:limit]

            for c in contacts:
                c["account"] = name
            all_contacts.extend(contacts)

        except RuntimeError as e:
            errors.append(f"{name}: {e}")
        except Exception as e:
            errors.append(f"{name}: {e}")

    result: dict[str, Any] = {
        "success": True,
        "count": len(all_contacts),
        "contacts": all_contacts,
    }
    if errors:
        result["warnings"] = errors
    return result
