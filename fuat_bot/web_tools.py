"""
Web tools for Fuat_bot.

Provides internet access via DuckDuckGo search and HTTP page fetching.
No API keys required.
"""

import ipaddress
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse


# =============================================================================
# Helpers
# =============================================================================

class _TextExtractor(HTMLParser):
    """Minimal HTML parser that strips tags and keeps visible text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        # Collapse excessive whitespace / blank lines
        lines = [line.strip() for line in raw.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines)


_FETCH_TRUNCATE = 8_000  # chars — keeps context manageable

# Hostnames that are always considered internal regardless of DNS resolution
_LOCAL_HOSTNAMES = {"localhost", "ip6-localhost", "ip6-loopback", "broadcasthost"}


def _is_ssrf_target(url: str) -> bool:
    """Return True if the URL contains a private/reserved address (SSRF prevention).

    Blocks explicit private IP addresses and well-known local hostnames in the
    URL itself. We intentionally do NOT do a live DNS resolution here because
    that causes false positives on WSL2, corporate VPNs, and other environments
    where legitimate public domains can resolve through private-range intermediaries.
    The main attack vectors (http://192.168.x.x, http://10.x.x.x, http://localhost)
    are fully covered by the literal checks below.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return True

        # Block well-known local hostnames
        if hostname.lower() in _LOCAL_HOSTNAMES:
            return True

        # If the URL contains a literal IP address, check it directly
        try:
            ip = ipaddress.ip_address(hostname)
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
            )
        except ValueError:
            pass  # It's a normal hostname — allow it through

        return False
    except Exception:
        return False


# =============================================================================
# Tool Implementations
# =============================================================================

def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web using DuckDuckGo and return titles, URLs, and snippets.

    Args:
        query: Search query string
        max_results: Number of results to return (default: 5, max: 10)

    Returns:
        Dict with "results" list (each has title, url, snippet) and "count"
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return {"error": "ddgs not installed. Run: pip install ddgs"}

    try:
        max_results = min(max(1, max_results), 10)
        results = []

        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

        return {"results": results, "count": len(results)}

    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


def web_fetch(url: str, extract_text: bool = True) -> dict[str, Any]:
    """Fetch the content of a web page by URL.

    Args:
        url: Full URL to fetch (must start with http:// or https://)
        extract_text: If True (default), strips HTML tags and returns readable
                      plain text. If False, returns raw HTML.

    Returns:
        Dict with "url", "content", "length", and "truncated" flag
    """
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed. Run: pip install httpx"}

    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    if _is_ssrf_target(url):
        return {"error": "URL resolves to a private or reserved address and cannot be fetched."}

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        with httpx.Client(follow_redirects=True, timeout=10.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

        raw = response.text

        if extract_text:
            parser = _TextExtractor()
            parser.feed(raw)
            content = parser.get_text()
        else:
            content = raw

        truncated = len(content) > _FETCH_TRUNCATE
        if truncated:
            content = content[:_FETCH_TRUNCATE]

        return {
            "url": str(response.url),
            "content": content,
            "length": len(content),
            "truncated": truncated,
        }

    except Exception as e:
        return {"error": f"Failed to fetch URL: {str(e)}"}
