"""
Google Calendar tools for Fuat_bot.

Provides read/write access to Google Calendar via the Calendar API v3.
Requires a one-time OAuth setup: `python -m fuat_bot calendar-setup`

All functions return dict[str, Any] with either success data or {"error": "..."}.
"""

from datetime import datetime, timedelta
from typing import Any

from .config import settings


# =============================================================================
# OAuth / Service Helper
# =============================================================================

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _get_calendar_service():
    """Build and return an authenticated Google Calendar service.

    Loads credentials from the token file (created by `calendar-setup`).
    Auto-refreshes expired tokens using the stored refresh token.

    Raises:
        RuntimeError: If credentials file or token file are missing / invalid.
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

    token_path = settings.google_calendar_token_file
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds:
        raise RuntimeError(
            "Google Calendar not set up. Run: python -m fuat_bot calendar-setup"
        )

    # Refresh if expired
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Persist the refreshed token
            token_path.write_text(creds.to_json())
        else:
            raise RuntimeError(
                "Google Calendar token is invalid. Re-run: python -m fuat_bot calendar-setup"
            )

    return build("calendar", "v3", credentials=creds)


def _parse_event(event: dict) -> dict:
    """Extract the fields we care about from a raw Calendar API event."""
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id", ""),
        "title": event.get("summary", "(no title)"),
        "start": start.get("dateTime") or start.get("date", ""),
        "end": end.get("dateTime") or end.get("date", ""),
        "location": event.get("location", ""),
        "description": event.get("description", ""),
        "status": event.get("status", "confirmed"),
        "link": event.get("htmlLink", ""),
    }


def _to_rfc3339(dt_str: str) -> str:
    """Convert 'YYYY-MM-DDTHH:MM:SS' to RFC3339 with UTC offset.

    Assumes local-naive datetime strings and appends 'Z' (UTC) if no
    timezone info is present, which is acceptable for school scheduling.
    """
    if dt_str.endswith("Z") or "+" in dt_str[10:]:
        return dt_str
    return dt_str + "Z"


# =============================================================================
# Tool Implementations
# =============================================================================

def calendar_list_events(
    start_date: str,
    end_date: str,
    max_results: int = 20,
    query: str | None = None,
) -> dict[str, Any]:
    """List calendar events in a date range.

    Args:
        start_date: Start date/datetime in ISO format (e.g. "2026-02-10" or "2026-02-10T00:00:00")
        end_date: End date/datetime in ISO format (e.g. "2026-02-17" or "2026-02-17T23:59:59")
        max_results: Maximum number of events to return (default: 20)
        query: Optional free-text search filter (searches title, description, location)

    Returns:
        Dict with "events" list and "count"
    """
    try:
        service = _get_calendar_service()

        # Ensure RFC3339 format
        time_min = _to_rfc3339(start_date if "T" in start_date else start_date + "T00:00:00")
        time_max = _to_rfc3339(end_date if "T" in end_date else end_date + "T23:59:59")

        kwargs: dict[str, Any] = {
            "calendarId": settings.google_calendar_id,
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": min(max_results, 250),
            "singleEvents": True,
            "orderBy": "startTime",
        }
        if query:
            kwargs["q"] = query

        result = service.events().list(**kwargs).execute()
        events = [_parse_event(e) for e in result.get("items", [])]

        return {"events": events, "count": len(events)}

    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to list events: {str(e)}"}


def calendar_add_event(
    title: str,
    start: str,
    end: str,
    description: str | None = None,
    location: str | None = None,
    attendees: str | None = None,
) -> dict[str, Any]:
    """Create a new calendar event.

    Args:
        title: Event title/summary
        start: Start datetime in ISO format (e.g. "2026-02-15T14:00:00")
        end: End datetime in ISO format (e.g. "2026-02-15T15:00:00")
        description: Optional event description / notes
        location: Optional location (room, URL, address)
        attendees: Optional comma-separated email addresses to invite

    Returns:
        Dict with created event details including id and link
    """
    try:
        service = _get_calendar_service()

        body: dict[str, Any] = {
            "summary": title,
            "start": {"dateTime": _to_rfc3339(start)},
            "end": {"dateTime": _to_rfc3339(end)},
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if attendees:
            emails = [e.strip() for e in attendees.split(",") if e.strip()]
            body["attendees"] = [{"email": e} for e in emails]

        event = service.events().insert(
            calendarId=settings.google_calendar_id,
            body=body,
            sendNotifications=bool(attendees),
        ).execute()

        return {
            "success": True,
            "event": _parse_event(event),
            "message": f"Event '{title}' created successfully",
        }

    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to create event: {str(e)}"}


def calendar_update_event(
    event_id: str,
    title: str | None = None,
    start: str | None = None,
    end: str | None = None,
    description: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Update an existing calendar event (only provided fields are changed).

    Args:
        event_id: The event ID (from calendar_list_events or calendar_add_event)
        title: New event title (optional)
        start: New start datetime ISO format (optional)
        end: New end datetime ISO format (optional)
        description: New description (optional)
        location: New location (optional)

    Returns:
        Dict with updated event details
    """
    try:
        service = _get_calendar_service()

        # Fetch the current event to patch selectively
        existing = service.events().get(
            calendarId=settings.google_calendar_id,
            eventId=event_id,
        ).execute()

        if title:
            existing["summary"] = title
        if start:
            existing["start"] = {"dateTime": _to_rfc3339(start)}
        if end:
            existing["end"] = {"dateTime": _to_rfc3339(end)}
        if description is not None:
            existing["description"] = description
        if location is not None:
            existing["location"] = location

        updated = service.events().update(
            calendarId=settings.google_calendar_id,
            eventId=event_id,
            body=existing,
        ).execute()

        return {
            "success": True,
            "event": _parse_event(updated),
            "message": f"Event '{updated.get('summary')}' updated successfully",
        }

    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to update event: {str(e)}"}


def calendar_delete_event(event_id: str) -> dict[str, Any]:
    """Delete a calendar event.

    Args:
        event_id: The event ID to delete (from calendar_list_events)

    Returns:
        Dict with success status
    """
    try:
        service = _get_calendar_service()

        service.events().delete(
            calendarId=settings.google_calendar_id,
            eventId=event_id,
        ).execute()

        return {
            "success": True,
            "event_id": event_id,
            "message": "Event deleted successfully",
        }

    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to delete event: {str(e)}"}


def calendar_create_appointment_slots(
    date: str,
    start_time: str,
    end_time: str,
    slot_duration_minutes: int,
    title: str,
    description: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Create a series of bookable appointment slot events on a given day.

    Each slot is a separate calendar event. Students can see them and RSVP.
    Note: Google's built-in Appointment Schedules feature has no API; this
    creates regular events as a practical equivalent.

    Args:
        date: Date for the slots in ISO format (e.g. "2026-02-15")
        start_time: Start time in HH:MM format (e.g. "09:00")
        end_time: End time in HH:MM format (e.g. "12:00")
        slot_duration_minutes: Duration of each slot in minutes (e.g. 30)
        title: Title for each slot event (e.g. "Office Hours Slot")
        description: Optional description added to each slot
        location: Optional location (room number, Zoom link, etc.)

    Returns:
        Dict with slots_created count and list of created event IDs
    """
    try:
        service = _get_calendar_service()

        # Parse start/end times
        start_dt = datetime.fromisoformat(f"{date}T{start_time}:00")
        end_dt = datetime.fromisoformat(f"{date}T{end_time}:00")
        delta = timedelta(minutes=slot_duration_minutes)

        if slot_duration_minutes <= 0:
            return {"error": "slot_duration_minutes must be greater than 0"}
        if start_dt >= end_dt:
            return {"error": "start_time must be before end_time"}

        created_ids: list[str] = []
        slot_start = start_dt
        slot_num = 0

        while slot_start + delta <= end_dt:
            slot_end = slot_start + delta
            slot_num += 1

            body: dict[str, Any] = {
                "summary": title,
                "start": {"dateTime": _to_rfc3339(slot_start.isoformat())},
                "end": {"dateTime": _to_rfc3339(slot_end.isoformat())},
            }
            if description:
                body["description"] = description
            if location:
                body["location"] = location

            event = service.events().insert(
                calendarId=settings.google_calendar_id,
                body=body,
            ).execute()
            created_ids.append(event["id"])

            slot_start = slot_end

        if not created_ids:
            return {"error": "No slots could be created — check that start_time is before end_time and duration fits"}

        return {
            "success": True,
            "slots_created": len(created_ids),
            "event_ids": created_ids,
            "first_slot": f"{date}T{start_time}",
            "last_slot": (start_dt + delta * (len(created_ids) - 1)).isoformat(),
            "message": f"Created {len(created_ids)} appointment slots on {date}",
        }

    except RuntimeError as e:
        return {"error": str(e)}
    except ValueError as e:
        return {"error": f"Invalid date/time format: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to create appointment slots: {str(e)}"}


# Google Calendar colorId map (name → API id)
_COLOR_NAMES: dict[str, str] = {
    "tomato": "11",
    "flamingo": "4",
    "tangerine": "6",
    "banana": "5",
    "sage": "2",
    "basil": "10",
    "peacock": "7",
    "blueberry": "9",
    "lavender": "1",
    "grape": "3",
    "graphite": "8",
}


def calendar_mark_important_date(
    date: str,
    title: str,
    description: str | None = None,
    color: str | None = None,
) -> dict[str, Any]:
    """Mark a date as important with an all-day banner event.

    Creates an all-day event (uses start.date / end.date) so it appears in the
    banner strip above the hourly grid in Google Calendar — not inside any time
    slot. Ideal for deadlines, holidays, and key academic dates.

    Args:
        date: Date in ISO format (e.g. "2026-06-22")
        title: Label for the date (e.g. "Last day of classes")
        description: Optional extra notes
        color: Optional color for visual distinction. Choices: tomato, flamingo,
               tangerine, banana, sage, basil, peacock, blueberry, lavender,
               grape, graphite. Defaults to tomato (red).

    Returns:
        Dict with created event details including id and link
    """
    try:
        service = _get_calendar_service()

        color_id = _COLOR_NAMES.get((color or "tomato").lower(), "11")

        body: dict[str, Any] = {
            "summary": title,
            "start": {"date": date},
            "end": {"date": date},
            "colorId": color_id,
            "transparency": "transparent",  # does not block time
        }
        if description:
            body["description"] = description

        event = service.events().insert(
            calendarId=settings.google_calendar_id,
            body=body,
        ).execute()

        return {
            "success": True,
            "event": _parse_event(event),
            "message": f"Marked '{title}' as an important date on {date}",
        }

    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to mark important date: {str(e)}"}
