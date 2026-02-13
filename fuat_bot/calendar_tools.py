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

    result: dict[str, Any] = {
        "id": event.get("id", ""),
        "title": event.get("summary", "(no title)"),
        "start": start.get("dateTime") or start.get("date", ""),
        "end": end.get("dateTime") or end.get("date", ""),
        "location": event.get("location", ""),
        "description": event.get("description", ""),
        "status": event.get("status", "confirmed"),
        "link": event.get("htmlLink", ""),
    }

    # Attendees (only included when present)
    attendees_raw = event.get("attendees", [])
    if attendees_raw:
        result["attendees"] = [
            {
                "email": a.get("email", ""),
                "name": a.get("displayName", ""),
                "status": a.get("responseStatus", "needsAction"),
            }
            for a in attendees_raw
        ]

    # Recurrence rule (only included for recurring master events)
    recurrence = event.get("recurrence", [])
    if recurrence:
        result["recurrence"] = recurrence

    # For instances: link back to the master series
    if event.get("recurringEventId"):
        result["recurring_event_id"] = event["recurringEventId"]

    return result


def _to_rfc3339(dt_str: str) -> str:
    """Strip any trailing timezone offset from a datetime string.

    Returns a naive 'YYYY-MM-DDTHH:MM:SS' string.  The timezone is supplied
    separately via the 'timeZone' field in the event start/end object.
    """
    # Drop a trailing 'Z'
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1]
    # Drop a trailing '+HH:MM' or '-HH:MM' offset
    if len(dt_str) > 10 and (dt_str[-6] in ("+", "-")):
        dt_str = dt_str[:-6]
    return dt_str


def _make_timed_field(dt_str: str) -> dict:
    """Build a Calendar API start/end timed-event object.

    Always includes the configured CALENDAR_TIMEZONE so that Google Calendar
    accepts the event (required for recurring events, recommended for all).
    """
    return {
        "dateTime": _to_rfc3339(dt_str),
        "timeZone": settings.calendar_timezone,
    }


_DAY_ABBR: dict[str, str] = {
    "monday": "MO", "tuesday": "TU", "wednesday": "WE", "thursday": "TH",
    "friday": "FR", "saturday": "SA", "sunday": "SU",
    "mo": "MO", "tu": "TU", "we": "WE", "th": "TH",
    "fr": "FR", "sa": "SA", "su": "SU",
}


def _build_rrule(
    freq: str,
    until: str | None = None,
    count: int | None = None,
    days: str | None = None,
    interval: int = 1,
) -> str:
    """Build an RFC 5545 RRULE string from friendly parameters.

    Args:
        freq: Recurrence frequency — "daily", "weekly", "monthly", or "yearly".
        until: Stop date in ISO format (e.g. "2026-06-30"). Mutually exclusive with count.
        count: Stop after this many occurrences. Mutually exclusive with until.
        days: Comma-separated days for weekly recurrence (e.g. "MO,WE,FR" or
              "monday,wednesday,friday"). Ignored for non-weekly frequencies.
        interval: Repeat every N periods (default 1). E.g. interval=2 with
                  freq="weekly" means every two weeks.

    Returns:
        RRULE string ready to insert into the Google Calendar API recurrence list.
    """
    freq_map = {
        "daily": "DAILY", "weekly": "WEEKLY",
        "monthly": "MONTHLY", "yearly": "YEARLY",
    }
    freq_upper = freq_map.get(freq.lower())
    if not freq_upper:
        raise ValueError(
            f"Invalid frequency '{freq}'. Must be daily, weekly, monthly, or yearly."
        )

    parts = [f"FREQ={freq_upper}"]

    if interval and interval > 1:
        parts.append(f"INTERVAL={interval}")

    if days and freq.lower() == "weekly":
        day_list = [
            _DAY_ABBR.get(d.strip().lower(), d.strip().upper())
            for d in days.split(",")
            if d.strip()
        ]
        parts.append(f"BYDAY={','.join(day_list)}")

    if until:
        # Google Calendar expects YYYYMMDDTHHMMSSZ
        until_clean = until.replace("-", "")[:8]
        parts.append(f"UNTIL={until_clean}T235959Z")
    elif count:
        parts.append(f"COUNT={count}")

    return "RRULE:" + ";".join(parts)


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
    recurrence_freq: str | None = None,
    recurrence_until: str | None = None,
    recurrence_count: int | None = None,
    recurrence_days: str | None = None,
    recurrence_interval: int = 1,
) -> dict[str, Any]:
    """Create a new calendar event, optionally recurring.

    Args:
        title: Event title/summary
        start: Start datetime in ISO format (e.g. "2026-02-15T14:00:00")
        end: End datetime in ISO format (e.g. "2026-02-15T15:00:00")
        description: Optional event description / notes
        location: Optional location (room, URL, address)
        attendees: Optional comma-separated email addresses to invite
        recurrence_freq: Repeat frequency — "daily", "weekly", "monthly", or "yearly".
                         Omit for a one-off event.
        recurrence_until: End date for recurrence in ISO format (e.g. "2026-06-30").
                          Mutually exclusive with recurrence_count.
        recurrence_count: Stop after this many occurrences.
                          Mutually exclusive with recurrence_until.
        recurrence_days: Comma-separated days for weekly recurrence
                         (e.g. "MO,WE,FR" or "monday,wednesday,friday").
        recurrence_interval: Repeat every N periods (default 1).
                             E.g. 2 with freq="weekly" means every two weeks.

    Returns:
        Dict with created event details including id and link
    """
    try:
        service = _get_calendar_service()

        body: dict[str, Any] = {
            "summary": title,
            "start": _make_timed_field(start),
            "end": _make_timed_field(end),
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if attendees:
            emails = [e.strip() for e in attendees.split(",") if e.strip()]
            body["attendees"] = [{"email": e} for e in emails]
        if recurrence_freq:
            rrule = _build_rrule(
                freq=recurrence_freq,
                until=recurrence_until,
                count=recurrence_count,
                days=recurrence_days,
                interval=recurrence_interval,
            )
            body["recurrence"] = [rrule]

        event = service.events().insert(
            calendarId=settings.google_calendar_id,
            body=body,
            sendNotifications=bool(attendees),
        ).execute()

        suffix = f" (repeating {recurrence_freq})" if recurrence_freq else ""
        return {
            "success": True,
            "event": _parse_event(event),
            "message": f"Event '{title}' created successfully{suffix}",
        }

    except RuntimeError as e:
        return {"error": str(e)}
    except ValueError as e:
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
            existing["start"] = _make_timed_field(start)
        if end:
            existing["end"] = _make_timed_field(end)
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


def calendar_delete_event(
    event_id: str,
    scope: str = "this",
) -> dict[str, Any]:
    """Delete a calendar event, with control over recurring series.

    Args:
        event_id: The event ID to delete (from calendar_list_events).
                  For recurring events this is an instance ID.
        scope: How much of a recurring series to delete:
               - "this"      — delete only this occurrence (default).
               - "all"       — delete the entire recurring series.
               - "following" — delete this occurrence and all future ones.
               Non-recurring events are always deleted fully regardless of scope.

    Returns:
        Dict with success status and description of what was deleted.
    """
    if scope not in ("this", "all", "following"):
        return {"error": "scope must be 'this', 'all', or 'following'"}

    try:
        service = _get_calendar_service()
        cal_id = settings.google_calendar_id

        if scope == "this":
            service.events().delete(calendarId=cal_id, eventId=event_id).execute()
            return {"success": True, "event_id": event_id, "message": "Event deleted successfully"}

        # For "all" and "following" we need the full event to inspect it
        event = service.events().get(calendarId=cal_id, eventId=event_id).execute()
        master_id = event.get("recurringEventId")

        if scope == "all":
            # Delete the master (or the event itself if it's not part of a series)
            target_id = master_id or event_id
            service.events().delete(calendarId=cal_id, eventId=target_id).execute()
            return {
                "success": True,
                "event_id": target_id,
                "message": "Entire recurring series deleted successfully",
            }

        # scope == "following"
        if not master_id:
            # Not a recurring instance — just delete it
            service.events().delete(calendarId=cal_id, eventId=event_id).execute()
            return {
                "success": True,
                "event_id": event_id,
                "message": "Event deleted (not part of a recurring series)",
            }

        # Determine the start time of this instance to compute UNTIL
        start = event.get("originalStartTime") or event.get("start", {})
        start_str = start.get("dateTime") or start.get("date", "")
        if not start_str:
            return {"error": "Could not determine instance start time"}

        if "T" in start_str:
            clean = start_str.replace("Z", "").split("+")[0]
            instance_dt = datetime.fromisoformat(clean)
            until_str = (instance_dt - timedelta(seconds=1)).strftime("%Y%m%dT%H%M%SZ")
        else:
            from datetime import date as _date
            d = _date.fromisoformat(start_str)
            until_str = (d - timedelta(days=1)).strftime("%Y%m%dT235959Z")

        # Patch the master event's RRULE to end just before this instance
        master = service.events().get(calendarId=cal_id, eventId=master_id).execute()
        new_recurrence = []
        for rule in master.get("recurrence", []):
            if rule.startswith("RRULE:"):
                parts = [
                    p for p in rule[len("RRULE:"):].split(";")
                    if not p.startswith("UNTIL=") and not p.startswith("COUNT=")
                ]
                parts.append(f"UNTIL={until_str}")
                new_recurrence.append("RRULE:" + ";".join(parts))
            else:
                new_recurrence.append(rule)
        master["recurrence"] = new_recurrence
        service.events().update(calendarId=cal_id, eventId=master_id, body=master).execute()

        return {
            "success": True,
            "event_id": master_id,
            "message": "Recurring series truncated — this and all following occurrences deleted",
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
                "start": _make_timed_field(slot_start.isoformat()),
                "end": _make_timed_field(slot_end.isoformat()),
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


def calendar_exclude_dates(
    event_id: str,
    dates: str,
) -> dict[str, Any]:
    """Exclude specific dates from a recurring event series (e.g. for holidays).

    Adds EXDATE entries to the master event's recurrence rule so that
    occurrences on those dates are skipped.  Works whether you pass the
    master event ID or any single-instance ID from the series.

    Args:
        event_id: ID of the recurring event or any of its instances.
        dates: Comma-separated dates to skip in ISO format
               (e.g. "2026-04-23,2026-05-19").

    Returns:
        Dict with success status and list of newly excluded dates.
    """
    try:
        service = _get_calendar_service()
        cal_id = settings.google_calendar_id

        # Resolve to the master event
        event = service.events().get(calendarId=cal_id, eventId=event_id).execute()
        master_id = event.get("recurringEventId", event_id)
        master = (
            service.events().get(calendarId=cal_id, eventId=master_id).execute()
            if master_id != event_id
            else event
        )

        if not master.get("recurrence"):
            return {"error": "Event is not a recurring event — no recurrence rule found."}

        # Extract the event's start time component to build EXDATE entries
        start = master.get("start", {})
        start_dt_str = start.get("dateTime", "")
        tz = start.get("timeZone", settings.calendar_timezone)

        if start_dt_str:
            # "2026-02-16T10:30:00" → "103000"
            time_part = _to_rfc3339(start_dt_str).split("T")[1].replace(":", "")[:6]
        else:
            time_part = None  # All-day recurring event

        # Parse the requested dates
        date_list = [d.strip() for d in dates.split(",") if d.strip()]
        if not date_list:
            return {"error": "No dates provided."}

        # Collect dates already excluded to avoid duplicates
        existing_recurrence = master.get("recurrence", [])
        already_excluded: set[str] = set()
        for rule in existing_recurrence:
            if rule.upper().startswith("EXDATE"):
                # Last colon-delimited segment holds the date(s)
                raw_dates = rule.split(":")[-1]
                for d in raw_dates.split(","):
                    already_excluded.add(d.strip()[:8])

        # Build new EXDATE lines for dates not already excluded
        new_exdates: list[str] = []
        skipped: list[str] = []
        for date_str in date_list:
            ymd = date_str.replace("-", "")[:8]
            if ymd in already_excluded:
                skipped.append(date_str)
                continue
            if time_part:
                new_exdates.append(f"EXDATE;TZID={tz}:{ymd}T{time_part}")
            else:
                new_exdates.append(f"EXDATE;VALUE=DATE:{ymd}")

        if not new_exdates:
            return {
                "success": True,
                "event_id": master_id,
                "excluded_dates": [],
                "message": "All requested dates were already excluded.",
            }

        master["recurrence"] = existing_recurrence + new_exdates
        service.events().update(
            calendarId=cal_id, eventId=master_id, body=master
        ).execute()

        result: dict[str, Any] = {
            "success": True,
            "event_id": master_id,
            "excluded_dates": date_list,
            "message": (
                f"Excluded {len(new_exdates)} date(s) from recurring series"
                + (f" ({len(skipped)} already excluded)." if skipped else ".")
            ),
        }
        return result

    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to exclude dates: {str(e)}"}


def calendar_get_event(event_id: str) -> dict[str, Any]:
    """Fetch full details of a single calendar event by its ID.

    Unlike calendar_list_events (which returns summary info), this returns
    the complete event including attendees, recurrence rule, and conference data.

    Args:
        event_id: The event ID (from calendar_list_events or calendar_add_event).

    Returns:
        Dict with full event details or error.
    """
    try:
        service = _get_calendar_service()

        event = service.events().get(
            calendarId=settings.google_calendar_id,
            eventId=event_id,
        ).execute()

        return {"success": True, "event": _parse_event(event)}

    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to get event: {str(e)}"}


def calendar_find_free_slots(
    date: str,
    duration_minutes: int = 60,
    time_min: str = "09:00",
    time_max: str = "18:00",
) -> dict[str, Any]:
    """Find free time slots in a given day using the freebusy API.

    Queries the configured calendar for busy periods and returns gaps that
    are at least *duration_minutes* long within the requested window.

    Args:
        date: Date to check in ISO format (e.g. "2026-03-15").
        duration_minutes: Minimum slot length needed, in minutes (default 60).
        time_min: Start of the search window in HH:MM format (default "09:00").
        time_max: End of the search window in HH:MM format (default "18:00").

    Returns:
        Dict with free_slots list (each has start, end, duration_minutes),
        busy_periods list, and counts.
    """
    try:
        service = _get_calendar_service()

        window_start = datetime.fromisoformat(f"{date}T{time_min}:00")
        window_end = datetime.fromisoformat(f"{date}T{time_max}:00")

        if window_start >= window_end:
            return {"error": "time_min must be before time_max"}
        if duration_minutes <= 0:
            return {"error": "duration_minutes must be positive"}
        if timedelta(minutes=duration_minutes) > (window_end - window_start):
            return {"error": "duration_minutes is longer than the search window"}

        freebusy_body = {
            "timeMin": window_start.isoformat() + "Z",
            "timeMax": window_end.isoformat() + "Z",
            "items": [{"id": settings.google_calendar_id}],
        }
        result = service.freebusy().query(body=freebusy_body).execute()

        # Parse busy periods returned by the API (UTC ISO strings)
        cal_data = result.get("calendars", {}).get(settings.google_calendar_id, {})
        busy: list[list[datetime]] = []
        for period in cal_data.get("busy", []):
            s = datetime.fromisoformat(period["start"].replace("Z", "").split("+")[0])
            e = datetime.fromisoformat(period["end"].replace("Z", "").split("+")[0])
            busy.append([s, e])

        # Sort and merge overlapping/adjacent busy periods
        busy.sort(key=lambda x: x[0])
        merged: list[list[datetime]] = []
        for s, e in busy:
            if merged and s < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        # Find free gaps of at least duration_minutes within the window
        delta = timedelta(minutes=duration_minutes)
        free_slots: list[dict[str, Any]] = []
        cursor = window_start

        for busy_start, busy_end in merged:
            # Clamp to window
            busy_start = max(busy_start, window_start)
            busy_end = min(busy_end, window_end)
            if cursor + delta <= busy_start:
                free_slots.append({
                    "start": cursor.strftime("%H:%M"),
                    "end": busy_start.strftime("%H:%M"),
                    "duration_minutes": int((busy_start - cursor).total_seconds() / 60),
                })
            cursor = max(cursor, busy_end)

        # Free time after the last busy period
        if cursor + delta <= window_end:
            free_slots.append({
                "start": cursor.strftime("%H:%M"),
                "end": window_end.strftime("%H:%M"),
                "duration_minutes": int((window_end - cursor).total_seconds() / 60),
            })

        return {
            "success": True,
            "date": date,
            "search_window": f"{time_min}–{time_max}",
            "duration_requested_minutes": duration_minutes,
            "free_slots": free_slots,
            "free_count": len(free_slots),
            "busy_periods": [
                {"start": s.strftime("%H:%M"), "end": e.strftime("%H:%M")}
                for s, e in merged
                if s < window_end and e > window_start
            ],
        }

    except RuntimeError as e:
        return {"error": str(e)}
    except ValueError as e:
        return {"error": f"Invalid date/time format: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to find free slots: {str(e)}"}
