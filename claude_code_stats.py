#!/usr/bin/env python3
"""
Claude Code Stats - Usage analytics for Claude Code CLI

Analyzes Claude Code conversation transcripts to provide usage statistics including:
- Active time vs wall-clock time (accounting for idle periods)
- Session counts and message breakdowns
- /clear and /compact command usage
- Model usage and token consumption

Usage:
    python claude_code_stats.py                    # Print report to stdout
    python claude_code_stats.py -o report.md       # Save to file
    python claude_code_stats.py --gap-threshold 10 # Custom idle threshold (minutes)

The tool reads data from ~/.claude/ which is where Claude Code stores:
- Conversation transcripts (projects/**/*.jsonl)
- SQLite database with response times (__store.db)
- Pre-computed statistics (stats-cache.json)

Methodology:
    Active time is estimated by summing gaps between messages that are <= the
    gap threshold (default: 15 minutes). Longer gaps indicate idle time such as
    bathroom breaks, meetings, or sessions left open overnight.
"""

import json
import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

__version__ = "0.2.0"

# Configuration - Claude Code data locations
CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"
STORE_DB = CLAUDE_DIR / "__store.db"
STATS_CACHE = CLAUDE_DIR / "stats-cache.json"

DEFAULT_GAP_THRESHOLD_MINUTES = 15

# Default patterns to extract repo name from cwd (first match wins)
DEFAULT_REPO_PATTERNS = [
    r'/Projects/([^/]+)',
    r'/code/([^/]+)',
    r'/repos/([^/]+)',
    r'/src/([^/]+)',
    r'~/([^/]+)',
]


def load_repo_pattern_from_env() -> Optional[str]:
    """Load custom REPO_PATTERN from .env file if it exists."""
    import re
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("REPO_PATTERN="):
                        pattern = line.split("=", 1)[1].strip()
                        # Remove surrounding quotes if present
                        if pattern.startswith('"') and pattern.endswith('"'):
                            pattern = pattern[1:-1]
                        elif pattern.startswith("'") and pattern.endswith("'"):
                            pattern = pattern[1:-1]
                        return pattern
        except IOError:
            pass
    return None


def get_repo_name(cwd: str, custom_pattern: str = None) -> str:
    """Extract repo name from cwd using configured or default patterns.

    Args:
        cwd: Working directory path from JSONL message
        custom_pattern: Optional regex pattern with capture group for repo name

    Returns:
        Repository name extracted from path, or fallback to last directory component
    """
    import re

    if not cwd:
        return "unknown"

    # Expand ~ to home directory for matching
    if cwd.startswith("~"):
        cwd = str(Path(cwd).expanduser())

    # Try custom pattern first if provided
    if custom_pattern:
        try:
            match = re.search(custom_pattern, cwd)
            if match and match.groups():
                return match.group(1)
        except re.error:
            pass  # Invalid regex, fall through to defaults

    # Try default patterns
    for pattern in DEFAULT_REPO_PATTERNS:
        try:
            match = re.search(pattern, cwd)
            if match and match.groups():
                return match.group(1)
        except re.error:
            continue

    # Fallback: last directory component
    path_parts = cwd.rstrip('/').split('/')
    return path_parts[-1] if path_parts and path_parts[-1] else "unknown"


def parse_timestamp(ts) -> Optional[datetime]:
    """Parse timestamp from various formats (ISO string or Unix timestamp)."""
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00')).replace(tzinfo=None)
        except ValueError:
            return None
    elif isinstance(ts, (int, float)):
        try:
            # Assume milliseconds if > 10 digits
            if ts > 10000000000:
                return datetime.fromtimestamp(ts / 1000)
            return datetime.fromtimestamp(ts)
        except (ValueError, OSError):
            return None
    return None


def load_jsonl_messages(repo_pattern: str = None) -> Tuple[List[Dict], Dict[str, int], Dict[str, Dict[str, int]]]:
    """Load all messages from JSONL conversation files.

    Args:
        repo_pattern: Optional regex pattern with capture group for extracting repo name from cwd

    Returns:
        Tuple of (messages list, session_summary_counts dict, daily_tokens dict)
        session_summary_counts maps session_id to count of summary entries (compactions)
        daily_tokens maps date string to {"input": int, "output": int}
    """
    messages = []
    session_summary_counts: Dict[str, int] = defaultdict(int)
    daily_tokens: Dict[str, Dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})

    if not PROJECTS_DIR.exists():
        return messages, dict(session_summary_counts), dict(daily_tokens)

    for jsonl_file in PROJECTS_DIR.rglob("*.jsonl"):
        # Extract session_id from filename (format: {session_id}.jsonl or agent-{id}.jsonl)
        filename = jsonl_file.stem
        file_session_id = filename if not filename.startswith("agent-") else None

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)

                        # Count file-level summaries (compaction markers without sessionId)
                        if data.get("type") == "summary" and "sessionId" not in data:
                            if file_session_id:
                                session_summary_counts[file_session_id] += 1
                            continue

                        if "timestamp" in data and "sessionId" in data:
                            dt = parse_timestamp(data["timestamp"])
                            if dt:
                                msg_content = data.get("message", {}).get("content", "")
                                content_str = str(msg_content)
                                is_clear = "<command-name>/clear" in content_str
                                is_compact = "<command-name>/compact" in content_str

                                # Extract token usage from message.usage field
                                usage = data.get("message", {}).get("usage", {})
                                if usage:
                                    date_str = dt.date().isoformat()
                                    # Input tokens = just direct input (not cache, which is context)
                                    input_tokens = usage.get("input_tokens", 0)
                                    output_tokens = usage.get("output_tokens", 0)
                                    daily_tokens[date_str]["input"] += input_tokens
                                    daily_tokens[date_str]["output"] += output_tokens

                                # Determine if this is actual user input (not tool results)
                                is_actual_user_input = False
                                is_tool_result = False
                                if data.get("type") == "user":
                                    if isinstance(msg_content, list):
                                        # Lists are usually tool results
                                        is_tool_result = any(
                                            isinstance(item, dict) and item.get("type") == "tool_result"
                                            for item in msg_content
                                        )
                                    elif isinstance(msg_content, str):
                                        # Filter out system messages and tool-related content
                                        is_tool_result = (
                                            "<tool_result>" in content_str or
                                            "tool_result" in content_str or
                                            "<local-command" in content_str or
                                            "Caveat:" in content_str
                                        )
                                    is_actual_user_input = not is_tool_result and content_str.strip() != ""

                                # Extract cwd and derive repo name
                                cwd = data.get("cwd", "")
                                repo_name = get_repo_name(cwd, repo_pattern)

                                messages.append({
                                    "timestamp": dt,
                                    "session_id": data["sessionId"],
                                    "type": data.get("type", "unknown"),
                                    "is_user": data.get("type") == "user",
                                    "is_actual_user_input": is_actual_user_input,
                                    "is_assistant": data.get("type") == "assistant",
                                    "is_clear": is_clear,
                                    "is_compact": is_compact,
                                    "cwd": cwd,
                                    "repo_name": repo_name,
                                })
                    except json.JSONDecodeError:
                        continue
        except (IOError, OSError):
            continue

    return messages, dict(session_summary_counts), dict(daily_tokens)


def load_assistant_durations() -> Dict[str, float]:
    """Load actual assistant response durations from SQLite database."""
    durations: Dict[str, float] = {}

    if not STORE_DB.exists():
        return durations

    try:
        conn = sqlite3.connect(STORE_DB)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT b.session_id, SUM(a.duration_ms)
            FROM assistant_messages a
            JOIN base_messages b ON a.uuid = b.uuid
            GROUP BY b.session_id
        """)
        for row in cursor.fetchall():
            durations[row[0]] = row[1] / 1000 / 60  # Convert to minutes
        conn.close()
    except sqlite3.Error:
        pass

    return durations


def load_stats_cache() -> Dict:
    """Load pre-computed stats from Claude Code's cache file."""
    if not STATS_CACHE.exists():
        return {}

    try:
        with open(STATS_CACHE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def calculate_session_stats(
    messages: List[Dict],
    gap_threshold_minutes: int,
    session_summary_counts: Dict[str, int]
) -> List[Dict]:
    """Calculate statistics for each session.

    Args:
        messages: List of message dicts
        gap_threshold_minutes: Gap threshold for active time calculation
        session_summary_counts: Dict mapping session_id to count of file-level summaries

    Returns:
        List of session statistics dicts
    """
    # Group messages by session
    sessions: Dict[str, List[Dict]] = defaultdict(list)
    for msg in messages:
        sessions[msg["session_id"]].append(msg)

    stats = []
    for session_id, msgs in sessions.items():
        msgs.sort(key=lambda x: x["timestamp"])

        if len(msgs) < 2:
            continue

        first_ts = msgs[0]["timestamp"]
        last_ts = msgs[-1]["timestamp"]
        wall_clock_minutes = (last_ts - first_ts).total_seconds() / 60

        # Calculate active time by summing gaps <= threshold
        active_minutes = 0.0
        for i in range(1, len(msgs)):
            gap = (msgs[i]["timestamp"] - msgs[i-1]["timestamp"]).total_seconds() / 60
            if gap <= gap_threshold_minutes:
                active_minutes += gap
            else:
                # Add small buffer for context switching
                active_minutes += 1

        user_msgs = sum(1 for m in msgs if m.get("is_actual_user_input"))
        assistant_msgs = sum(1 for m in msgs if m["is_assistant"])
        clear_count = sum(1 for m in msgs if m.get("is_clear"))

        # Compact count: /compact commands from messages + file-level summaries
        compact_commands = sum(1 for m in msgs if m.get("is_compact"))
        file_summaries = session_summary_counts.get(session_id, 0)
        compact_count = compact_commands + file_summaries

        # Determine primary repo for session (most common repo_name)
        repo_counts: Dict[str, int] = defaultdict(int)
        for m in msgs:
            repo_counts[m.get("repo_name", "unknown")] += 1
        primary_repo = max(repo_counts.keys(), key=lambda r: repo_counts[r]) if repo_counts else "unknown"

        stats.append({
            "session_id": session_id,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "wall_clock_minutes": wall_clock_minutes,
            "active_minutes": active_minutes,
            "message_count": len(msgs),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "clear_count": clear_count,
            "compact_count": compact_count,
            "date": first_ts.date(),
            "repo_name": primary_repo,
        })

    return stats


def aggregate_by_period(
    session_stats: List[Dict],
    period: str,
    reference_date: datetime
) -> Dict:
    """Aggregate session stats by time period."""
    if period == "week":
        start_date = (reference_date - timedelta(days=7)).date()
    elif period == "month":
        start_date = (reference_date - timedelta(days=30)).date()
    elif period == "3months":
        start_date = (reference_date - timedelta(days=90)).date()
    else:
        start_date = datetime.min.date()

    filtered = [s for s in session_stats if s["date"] >= start_date]

    if not filtered:
        return {
            "period": period,
            "start_date": start_date,
            "end_date": reference_date.date(),
            "sessions": 0,
            "messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "wall_clock_hours": 0,
            "active_hours": 0,
            "efficiency": 0,
            "avg_session_minutes": 0,
            "daily_breakdown": {}
        }

    total_wall = sum(s["wall_clock_minutes"] for s in filtered)
    total_active = sum(s["active_minutes"] for s in filtered)

    # Daily breakdown
    daily: Dict[str, Dict] = defaultdict(lambda: {
        "sessions": 0, "active_minutes": 0, "wall_clock_minutes": 0,
        "user_messages": 0, "assistant_messages": 0,
        "clear_count": 0, "compact_count": 0
    })
    for s in filtered:
        d = s["date"].isoformat()
        daily[d]["sessions"] += 1
        daily[d]["active_minutes"] += s["active_minutes"]
        daily[d]["wall_clock_minutes"] += s["wall_clock_minutes"]
        daily[d]["user_messages"] += s["user_messages"]
        daily[d]["assistant_messages"] += s["assistant_messages"]
        daily[d]["clear_count"] += s["clear_count"]
        daily[d]["compact_count"] += s["compact_count"]

    return {
        "period": period,
        "start_date": start_date,
        "end_date": reference_date.date(),
        "sessions": len(filtered),
        "messages": sum(s["message_count"] for s in filtered),
        "user_messages": sum(s["user_messages"] for s in filtered),
        "assistant_messages": sum(s["assistant_messages"] for s in filtered),
        "wall_clock_hours": total_wall / 60,
        "active_hours": total_active / 60,
        "efficiency": (total_active / total_wall * 100) if total_wall > 0 else 0,
        "avg_session_minutes": total_active / len(filtered) if filtered else 0,
        "daily_breakdown": dict(daily)
    }


def calculate_per_repo_stats(
    session_stats: List[Dict],
    period_days: int = None,
    reference_date: datetime = None
) -> Dict[str, Dict]:
    """Calculate statistics grouped by repository.

    Args:
        session_stats: List of session statistics (must include repo_name field)
        period_days: If set, only include sessions from last N days
        reference_date: Reference date for period calculation (default: now)

    Returns:
        Dict mapping repo_name to stats dict with keys:
        - sessions: number of sessions
        - active_hours: total active time in hours
        - wall_clock_hours: total wall-clock time in hours
        - efficiency: active/wall-clock percentage
        - messages: total message count
        - user_messages: user message count
        - assistant_messages: assistant message count
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Filter by period if specified
    if period_days:
        cutoff = (reference_date - timedelta(days=period_days)).date()
        filtered = [s for s in session_stats if s["date"] >= cutoff]
    else:
        filtered = session_stats

    # Group by repo
    repos: Dict[str, Dict] = defaultdict(lambda: {
        "sessions": 0,
        "active_minutes": 0,
        "wall_clock_minutes": 0,
        "messages": 0,
        "user_messages": 0,
        "assistant_messages": 0,
    })

    for s in filtered:
        repo = s.get("repo_name", "unknown")
        repos[repo]["sessions"] += 1
        repos[repo]["active_minutes"] += s["active_minutes"]
        repos[repo]["wall_clock_minutes"] += s["wall_clock_minutes"]
        repos[repo]["messages"] += s["message_count"]
        repos[repo]["user_messages"] += s["user_messages"]
        repos[repo]["assistant_messages"] += s["assistant_messages"]

    # Convert to final format with hours and efficiency
    result = {}
    for repo, data in repos.items():
        active_hours = data["active_minutes"] / 60
        wall_hours = data["wall_clock_minutes"] / 60
        efficiency = (data["active_minutes"] / data["wall_clock_minutes"] * 100) if data["wall_clock_minutes"] > 0 else 0

        result[repo] = {
            "sessions": data["sessions"],
            "active_hours": active_hours,
            "wall_clock_hours": wall_hours,
            "efficiency": efficiency,
            "messages": data["messages"],
            "user_messages": data["user_messages"],
            "assistant_messages": data["assistant_messages"],
        }

    return result


def get_top_sessions(session_stats: List[Dict], n: int = 10) -> List[Dict]:
    """Get top N sessions by active time."""
    sorted_stats = sorted(session_stats, key=lambda x: x["active_minutes"], reverse=True)
    return sorted_stats[:n]


def get_idle_sessions(session_stats: List[Dict], n: int = 5) -> List[Dict]:
    """Get sessions with most idle time (left open overnight)."""
    for s in session_stats:
        s["idle_minutes"] = s["wall_clock_minutes"] - s["active_minutes"]
    sorted_stats = sorted(session_stats, key=lambda x: x["idle_minutes"], reverse=True)
    return sorted_stats[:n]


def format_duration(minutes: float) -> str:
    """Format duration in human-readable format."""
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"


def format_tokens(tokens: int) -> str:
    """Format token count in human-readable format (K/M)."""
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1_000_000:
        return f"{tokens/1000:.1f}K"
    else:
        return f"{tokens/1_000_000:.1f}M"


def format_number(n: int) -> str:
    """Format number with comma thousands separators."""
    return f"{n:,}"


def generate_report(
    session_stats: List[Dict],
    assistant_durations: Dict[str, float],
    stats_cache: Dict,
    gap_threshold: int,
    daily_tokens: Dict[str, Dict[str, int]] = None,
    show_tokens: bool = False,
    period_days: int = None,
    per_repo_stats: Dict[str, Dict] = None
) -> str:
    """Generate markdown report.

    Args:
        session_stats: List of session statistics
        assistant_durations: Dict of session durations from SQLite
        stats_cache: Pre-computed stats from Claude Code
        gap_threshold: Gap threshold in minutes for idle detection
        daily_tokens: Dict mapping date to {"input": int, "output": int}
        show_tokens: Whether to include token columns in output
        period_days: If set, only show data for last N days (7, 30, 90, etc.)
        per_repo_stats: Dict mapping repo_name to stats (if --by-repo)
    """
    now = datetime.now()
    daily_tokens = daily_tokens or {}

    # Aggregate by periods
    week_stats = aggregate_by_period(session_stats, "week", now)
    month_stats = aggregate_by_period(session_stats, "month", now)
    three_month_stats = aggregate_by_period(session_stats, "3months", now)
    all_time = aggregate_by_period(session_stats, "all", now)

    # Get notable sessions
    top_sessions = get_top_sessions(session_stats)
    idle_sessions = get_idle_sessions(session_stats)

    # Calculate assistant response time
    total_assistant_minutes = sum(assistant_durations.values())

    report = f"""---
generated: {now.isoformat()}
gap_threshold_minutes: {gap_threshold}
---

# Claude Code Usage Report

Generated: **{now.strftime('%Y-%m-%d %H:%M')}**

> **Methodology**: Active time is estimated by summing gaps between messages that are <={gap_threshold} minutes.
> Longer gaps indicate idle time (bathroom break, meeting, left overnight, etc.)

---

## Summary

| Period | Sessions | Messages | Active Time | Wall-Clock | Efficiency |
|--------|----------|----------|-------------|------------|------------|
| Last 7 days | {week_stats['sessions']} | {week_stats['messages']} | **{week_stats['active_hours']:.1f}h** | {week_stats['wall_clock_hours']:.1f}h | {week_stats['efficiency']:.0f}% |
| Last 30 days | {month_stats['sessions']} | {month_stats['messages']} | **{month_stats['active_hours']:.1f}h** | {month_stats['wall_clock_hours']:.1f}h | {month_stats['efficiency']:.0f}% |
| Last 90 days | {three_month_stats['sessions']} | {three_month_stats['messages']} | **{three_month_stats['active_hours']:.1f}h** | {three_month_stats['wall_clock_hours']:.1f}h | {three_month_stats['efficiency']:.0f}% |
| All Time | {all_time['sessions']} | {all_time['messages']} | **{all_time['active_hours']:.1f}h** | {all_time['wall_clock_hours']:.1f}h | {all_time['efficiency']:.0f}% |

**Claude Response Time** (actual compute): {total_assistant_minutes/60:.1f}h

"""

    # Add per-repo breakdown if provided
    if per_repo_stats:
        period_label = f"Last {period_days} days" if period_days else "All time"
        report += f"""---

## Per-Repository Breakdown ({period_label})

| Repository | Active | Wall-Clock | Efficiency | Sessions | Messages |
|------------|--------|------------|------------|----------|----------|
"""
        # Sort by active hours descending
        sorted_repos = sorted(per_repo_stats.items(), key=lambda x: x[1]["active_hours"], reverse=True)
        for repo, stats in sorted_repos:
            report += f"| {repo} | {stats['active_hours']:.1f}h | {stats['wall_clock_hours']:.1f}h | {stats['efficiency']:.0f}% | {stats['sessions']} | {stats['messages']} |\n"

    report += """
---

## Last 7 Days - Daily Breakdown

"""

    # Build table header dynamically based on show_tokens flag
    if show_tokens:
        report += "| Date | Sessions | Active | Clock | User Msgs | Claude Msgs | Input Tokens | Output Tokens |\n"
        report += "|------|----------|--------|-------|-----------|-------------|--------------|---------------|\n"
    else:
        report += "| Date | Sessions | Active | Clock | User Msgs | Claude Msgs | Clears | Compacts |\n"
        report += "|------|----------|--------|-------|-----------|-------------|--------|----------|\n"

    # Add daily breakdown for last week
    for i in range(7):
        d = (now - timedelta(days=i)).date().isoformat()
        if d in week_stats["daily_breakdown"]:
            day = week_stats["daily_breakdown"][d]
            active_fmt = format_duration(day['active_minutes'])
            clock_fmt = format_duration(day['wall_clock_minutes'])
            if show_tokens:
                tokens = daily_tokens.get(d, {"input": 0, "output": 0})
                report += f"| {d} | {day['sessions']} | {active_fmt} | {clock_fmt} | {day['user_messages']} | {day['assistant_messages']} | {format_tokens(tokens['input'])} | {format_tokens(tokens['output'])} |\n"
            else:
                report += f"| {d} | {day['sessions']} | {active_fmt} | {clock_fmt} | {day['user_messages']} | {day['assistant_messages']} | {day['clear_count']} | {day['compact_count']} |\n"
        else:
            if show_tokens:
                tokens = daily_tokens.get(d, {"input": 0, "output": 0})
                report += f"| {d} | 0 | 0m | 0m | 0 | 0 | {format_tokens(tokens['input'])} | {format_tokens(tokens['output'])} |\n"
            else:
                report += f"| {d} | 0 | 0m | 0m | 0 | 0 | 0 | 0 |\n"

    report += """
---

## Last 30 Days - Weekly Breakdown

"""

    # Weekly aggregation for last month
    for week_num in range(4):
        week_start = now - timedelta(days=7 * (week_num + 1))
        week_end = now - timedelta(days=7 * week_num)
        week_sessions = [s for s in session_stats
                        if week_start.date() <= s["date"] < week_end.date()]

        if week_sessions:
            active = sum(s["active_minutes"] for s in week_sessions)
            wall = sum(s["wall_clock_minutes"] for s in week_sessions)
            user_msgs = sum(s["user_messages"] for s in week_sessions)
            claude_msgs = sum(s["assistant_messages"] for s in week_sessions)
            clears = sum(s["clear_count"] for s in week_sessions)
            compacts = sum(s["compact_count"] for s in week_sessions)
            active_fmt = format_duration(active)
            wall_fmt = format_duration(wall)
            report += f"- **Week of {week_start.strftime('%b %d')}**: {len(week_sessions)} sessions, {active_fmt} active / {wall_fmt} clock, {user_msgs} user / {claude_msgs} claude msgs, {clears} clears, {compacts} compacts\n"
        else:
            report += f"- **Week of {week_start.strftime('%b %d')}**: No activity\n"

    report += """
---

## Top 10 Most Active Sessions

| Session | Date | Active | Clock | User | Claude | Eff |
|---------|------|--------|-------|------|--------|-----|
"""

    for s in top_sessions:
        eff = (s["active_minutes"] / s["wall_clock_minutes"] * 100) if s["wall_clock_minutes"] > 0 else 100
        report += f"| `{s['session_id'][:8]}` | {s['date']} | {format_duration(s['active_minutes'])} | {format_duration(s['wall_clock_minutes'])} | {s['user_messages']} | {s['assistant_messages']} | {eff:.0f}% |\n"

    report += """
---

## Sessions Left Idle (overnight/long breaks)

These sessions have the largest gap between wall-clock and active time:

| Session | Date | Wall-Clock | Active | Idle Time |
|---------|------|------------|--------|-----------|
"""

    for s in idle_sessions:
        idle = s["wall_clock_minutes"] - s["active_minutes"]
        report += f"| `{s['session_id'][:8]}` | {s['date']} | {format_duration(s['wall_clock_minutes'])} | {format_duration(s['active_minutes'])} | {format_duration(idle)} |\n"

    report += """
---

## Model Usage (from stats cache)

"""

    if stats_cache and "modelUsage" in stats_cache:
        for model, usage in stats_cache["modelUsage"].items():
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            cache_read = usage.get("cacheReadInputTokens", 0)
            cache_create = usage.get("cacheCreationInputTokens", 0)
            report += f"**{model}**\n"
            report += f"- Input tokens: {input_tokens:,}\n"
            report += f"- Output tokens: {output_tokens:,}\n"
            report += f"- Cache read: {cache_read:,}\n"
            report += f"- Cache creation: {cache_create:,}\n\n"
    else:
        report += "_No model usage data available_\n"

    report += """
---

## Hourly Distribution

Peak usage hours (from stats cache):

"""

    if stats_cache and "hourCounts" in stats_cache:
        hour_counts = stats_cache["hourCounts"]
        sorted_hours = sorted(hour_counts.items(), key=lambda x: int(x[0]))
        max_count = max(int(v) for v in hour_counts.values()) if hour_counts else 1

        for hour, count in sorted_hours:
            bar_len = int((int(count) / max_count) * 20)
            bar = "#" * bar_len
            report += f"| {int(hour):02d}:00 | {bar} {count} |\n"
    else:
        report += "_No hourly data available_\n"

    report += f"""
---

## Notes

- **Session**: A new session is created when you start a new Claude Code process (new terminal, run `claude`). `/clear` stays in the same session; `/exit` ends it.
- **Active Time**: Estimated based on message timestamps with {gap_threshold}-minute gap threshold
- **Clock Time (Wall-Clock)**: Raw duration from first to last message (includes idle periods)
- **Efficiency**: Ratio of active to wall-clock time (higher = fewer idle gaps)
- **Claude Response Time**: Actual time spent waiting for Claude responses (from SQLite)

Data sources:
- JSONL files: `~/.claude/projects/**/*.jsonl`
- SQLite DB: `~/.claude/__store.db`
- Stats cache: `~/.claude/stats-cache.json`
"""

    return report


def generate_html_report(
    session_stats: List[Dict],
    stats_cache: Dict,
    daily_tokens: Dict[str, Dict[str, int]],
    period_days: int = 7,
    style: str = "card",
    light_mode: bool = False,
    username: str = None,
    per_repo_stats: Dict[str, Dict] = None
) -> str:
    """Generate HTML report for sharing.

    Args:
        session_stats: List of session statistics
        stats_cache: Pre-computed stats from Claude Code
        daily_tokens: Dict mapping date to {"input": int, "output": int}
        period_days: Number of days to include (7, 30, 90)
        style: "card" for compact card, "full" for detailed stats card
        light_mode: Use light theme with Anthropic brand colors
        username: GitHub username to display (optional)
        per_repo_stats: Dict mapping repo_name to stats (if --by-repo)

    Returns:
        HTML string
    """
    now = datetime.now()
    cutoff = (now - timedelta(days=period_days)).date()

    # Filter sessions for the period
    filtered = [s for s in session_stats if s["date"] >= cutoff]

    # Calculate stats
    total_sessions = len(filtered)
    total_messages = sum(s["message_count"] for s in filtered)
    total_active = sum(s["active_minutes"] for s in filtered)
    total_wall = sum(s["wall_clock_minutes"] for s in filtered)

    # Calculate total tokens for period
    total_input_tokens = 0
    total_output_tokens = 0
    for i in range(period_days):
        d = (now - timedelta(days=i)).date().isoformat()
        if d in daily_tokens:
            total_input_tokens += daily_tokens[d]["input"]
            total_output_tokens += daily_tokens[d]["output"]

    # Calculate user and claude messages
    total_user_messages = sum(s["user_messages"] for s in filtered)
    total_claude_messages = sum(s["assistant_messages"] for s in filtered)

    # Calculate clears and compacts
    total_clears = sum(s["clear_count"] for s in filtered)
    total_compacts = sum(s["compact_count"] for s in filtered)
    clears_per_session = total_clears / total_sessions if total_sessions > 0 else 0
    compacts_per_session = total_compacts / total_sessions if total_sessions > 0 else 0

    # Calculate tool results (total - user prompts - claude messages)
    total_tool_results = total_messages - total_user_messages - total_claude_messages
    tool_results_per_day = total_tool_results / period_days if period_days > 0 else 0

    # Calculate ratios
    token_ratio = total_output_tokens / total_input_tokens if total_input_tokens > 0 else 0
    message_ratio = total_claude_messages / total_user_messages if total_user_messages > 0 else 0

    # Calculate per-day averages
    active_per_day = (total_active / 60) / period_days if period_days > 0 else 0
    wall_per_day = (total_wall / 60) / period_days if period_days > 0 else 0
    sessions_per_day = total_sessions / period_days if period_days > 0 else 0
    messages_per_day = total_messages / period_days if period_days > 0 else 0
    user_messages_per_day = total_user_messages / period_days if period_days > 0 else 0
    claude_messages_per_day = total_claude_messages / period_days if period_days > 0 else 0
    input_tokens_per_day = total_input_tokens / period_days if period_days > 0 else 0
    output_tokens_per_day = total_output_tokens / period_days if period_days > 0 else 0
    total_tokens_per_day = (total_input_tokens + total_output_tokens) / period_days if period_days > 0 else 0

    # Period label
    period_label = f"Last {period_days} days"
    if period_days == 7:
        period_label = "Last 7 days"
    elif period_days == 30:
        period_label = "Last 30 days"
    elif period_days == 90:
        period_label = "Last 90 days"

    # Add username if provided
    if username:
        # Ensure @ prefix
        if not username.startswith("@"):
            username = f"@{username}"
        period_label = f"{period_label} · {username}"

    # Build day-of-week chart data (Sun=0 to Sat=6)
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    dow_hours = {i: [] for i in range(7)}  # Collect hours per day of week

    for i in range(period_days):
        d = (now - timedelta(days=i)).date()
        dow = (d.weekday() + 1) % 7  # Convert Monday=0 to Sunday=0
        sessions_for_day = [s for s in filtered if s["date"] == d]
        active_hours = sum(s["active_minutes"] for s in sessions_for_day) / 60
        dow_hours[dow].append(active_hours)

    # Calculate totals (7d) or averages (30d+)
    if period_days <= 7:
        # For 7 days: show totals (each day appears once)
        dow_values = {i: sum(dow_hours[i]) for i in range(7)}
        chart_label = "Hours by Day"
    else:
        # For 30d+: show averages per day of week
        dow_values = {i: (sum(dow_hours[i]) / len(dow_hours[i]) if dow_hours[i] else 0) for i in range(7)}
        chart_label = "Avg Hours by Day"

    max_dow_value = max(dow_values.values()) if dow_values else 1
    if max_dow_value == 0:
        max_dow_value = 1

    # Theme colors
    if light_mode:
        theme = {
            "bg": "#F4F3EE",
            "card_bg": "#FFFFFF",
            "card_border": "rgba(0,0,0,0.08)",
            "card_shadow": "rgba(0,0,0,0.08)",
            "stat_bg": "rgba(0,0,0,0.03)",
            "stat_border": "rgba(0,0,0,0.05)",
            "title": "#131314",
            "text": "#6b7280",
            "text_secondary": "#9ca3af",
            "accent": "#d97757",
            "accent_gradient": "linear-gradient(180deg, #d97757 0%, #c4684a 100%)",
            "green": "#059669",
            "blue": "#2563eb",
            "divider": "rgba(0,0,0,0.08)",
        }
    else:
        theme = {
            "bg": "#1a1a2e",
            "card_bg": "linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)",
            "card_border": "rgba(255,255,255,0.1)",
            "card_shadow": "rgba(0,0,0,0.3)",
            "stat_bg": "rgba(255,255,255,0.05)",
            "stat_border": "rgba(255,255,255,0.05)",
            "title": "#fff",
            "text": "#9ca3af",
            "text_secondary": "#6b7280",
            "accent": "#f59e0b",
            "accent_gradient": "linear-gradient(180deg, #f59e0b 0%, #d97706 100%)",
            "green": "#10b981",
            "blue": "#60a5fa",
            "divider": "rgba(255,255,255,0.1)",
        }

    if style == "card":
        # Compact card - great for Reddit/social sharing
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Code Stats</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: {theme["bg"]}; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }}
        .card {{ background: {theme["card_bg"]}; border-radius: 16px; padding: 24px; max-width: 520px; width: 100%; box-shadow: 0 8px 32px {theme["card_shadow"]}; border: 1px solid {theme["card_border"]}; }}
        .header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }}
        .header-left {{ display: flex; align-items: center; gap: 12px; }}
        .header-right {{ margin-left: auto; text-align: right; }}
        .ratio {{ color: {theme["green"]}; font-size: 12px; font-weight: 600; }}
        .ratio-label {{ color: {theme["text_secondary"]}; font-size: 9px; text-transform: uppercase; }}
        .logo {{ width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; }}
        .title {{ color: {theme["title"]}; font-size: 18px; font-weight: 600; }}
        .period {{ color: {theme["text"]}; font-size: 12px; }}
        .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
        .stat {{ background: {theme["stat_bg"]}; border-radius: 10px; padding: 12px 10px; text-align: center; }}
        .stat-label {{ color: {theme["text"]}; font-size: 9px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
        .stat-value {{ color: {theme["accent"]}; font-size: 20px; font-weight: 700; margin-bottom: 4px; }}
        .stat-daily {{ color: {theme["text"]}; font-size: 11px; }}
        .footer {{ margin-top: 20px; padding-top: 16px; border-top: 1px solid {theme["divider"]}; }}
        .date {{ color: {theme["text_secondary"]}; font-size: 11px; }}
        .eye {{ animation: blink 4s ease-in-out infinite; }}
        .arm-right {{ transform-origin: 13px 7px; animation: wave 5s ease-in-out infinite; }}
        @keyframes blink {{ 0%, 92%, 100% {{ transform: scaleY(1); }} 95%, 97% {{ transform: scaleY(0.1); }} }}
        @keyframes wave {{ 0%, 85%, 100% {{ transform: rotate(0deg); }} 88%, 92% {{ transform: rotate(-25deg); }} 90% {{ transform: rotate(-15deg); }} }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div class="header-left">
                <div class="logo"><svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 16 16"><rect width="16" height="16" fill="none"/><rect x="3" y="2" width="10" height="6" fill="#d97757"/><rect x="4" y="8" width="1" height="3" fill="#d97757"/><rect x="5.5" y="8" width="1" height="3" fill="#d97757"/><rect x="9.5" y="8" width="1" height="3" fill="#d97757"/><rect x="11" y="8" width="1" height="3" fill="#d97757"/><rect x="2" y="4" width="1" height="3" fill="#d97757"/><rect class="arm-right" x="13" y="4" width="1" height="3" fill="#d97757"/><rect class="eye" x="5" y="3" width="1" height="2" fill="#141413"/><rect class="eye" x="10" y="3" width="1" height="2" fill="#141413"/></svg></div>
                <div>
                    <div class="title">Claude Code Stats</div>
                    <div class="period">{period_label}</div>
                </div>
            </div>
            <div class="header-right">
                <div class="ratio">{message_ratio:.1f}x</div>
                <div class="ratio-label">msg ratio</div>
                <div class="ratio" style="margin-top: 4px;">{token_ratio:.2f}x</div>
                <div class="ratio-label">token ratio</div>
            </div>
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-label">Active</div>
                <div class="stat-value">{total_active/60:.1f}h</div>
                <div class="stat-daily">{active_per_day:.1f}h/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">Clock</div>
                <div class="stat-value">{total_wall/60:.1f}h</div>
                <div class="stat-daily">{wall_per_day:.1f}h/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">Sessions</div>
                <div class="stat-value">{format_number(total_sessions)}</div>
                <div class="stat-daily">{sessions_per_day:.1f}/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">User Prompts</div>
                <div class="stat-value">{format_number(total_user_messages)}</div>
                <div class="stat-daily">{user_messages_per_day:.0f}/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">Claude Msgs</div>
                <div class="stat-value">{format_number(total_claude_messages)}</div>
                <div class="stat-daily">{claude_messages_per_day:.0f}/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">Tool Results</div>
                <div class="stat-value">{format_number(total_tool_results)}</div>
                <div class="stat-daily">{tool_results_per_day:.0f}/day</div>
            </div>
        </div>
        <div class="footer">
            <div class="date">{now.strftime('%Y-%m-%d')}</div>
        </div>
    </div>
</body>
</html>"""

    else:  # style == "full"
        # Build chart bars for day of week
        chart_bars = ""
        for i in range(7):
            height_pct = (dow_values[i] / max_dow_value) * 100 if max_dow_value > 0 else 0
            value_label = f"{dow_values[i]:.1f}h" if dow_values[i] >= 0.1 else ""
            chart_bars += f'''<div class="bar-container">
                <div class="bar-value">{value_label}</div>
                <div class="bar" style="height: {height_pct}%"></div>
                <div class="bar-label">{day_names[i]}</div>
            </div>'''

        # Chart background for light mode
        chart_bg = "rgba(0,0,0,0.03)" if light_mode else "rgba(0,0,0,0.2)"

        # Build per-repo section if provided
        repo_section = ""
        if per_repo_stats and len(per_repo_stats) > 0:
            # Sort by active hours descending, take top 6
            sorted_repos = sorted(per_repo_stats.items(), key=lambda x: x[1]["active_hours"], reverse=True)[:6]
            repo_items = ""
            for repo, stats in sorted_repos:
                repo_items += f'''<div class="repo-item">
                    <div class="repo-name">{repo}</div>
                    <div class="repo-hours">{stats["active_hours"]:.1f}h</div>
                    <div class="repo-sessions">{stats["sessions"]} sessions</div>
                </div>'''
            repo_section = f'''<div class="section-title">Top Repositories</div>
        <div class="repo-grid">{repo_items}</div>'''

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Code Stats - Full Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: {theme["bg"]}; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }}
        .card {{ background: {theme["card_bg"]}; border-radius: 20px; padding: 32px; max-width: 640px; width: 100%; box-shadow: 0 12px 48px {theme["card_shadow"]}; border: 1px solid {theme["card_border"]}; }}
        .header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 28px; }}
        .header-left {{ display: flex; align-items: center; gap: 16px; }}
        .header-right {{ margin-left: auto; text-align: right; }}
        .ratio {{ color: {theme["green"]}; font-size: 14px; font-weight: 600; }}
        .ratio-label {{ color: {theme["text_secondary"]}; font-size: 10px; text-transform: uppercase; }}
        .logo {{ width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; }}
        .header-text .title {{ color: {theme["title"]}; font-size: 24px; font-weight: 700; }}
        .header-text .period {{ color: {theme["text"]}; font-size: 14px; margin-top: 4px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 28px; }}
        .stat {{ background: {theme["stat_bg"]}; border-radius: 12px; padding: 14px 12px; text-align: center; border: 1px solid {theme["stat_border"]}; }}
        .stat-label {{ color: {theme["text"]}; font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }}
        .stat-value {{ color: {theme["accent"]}; font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
        .stat-daily {{ color: {theme["text"]}; font-size: 12px; }}
        .section-title {{ color: {theme["title"]}; font-size: 14px; font-weight: 600; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px; }}
        .chart {{ display: flex; gap: 8px; align-items: flex-end; height: 120px; margin-bottom: 28px; padding: 16px; background: {chart_bg}; border-radius: 12px; }}
        .bar-container {{ flex: 1; display: flex; flex-direction: column; align-items: center; height: 100%; justify-content: flex-end; }}
        .bar {{ width: 100%; background: {theme["accent_gradient"]}; border-radius: 4px 4px 0 0; min-height: 2px; }}
        .bar-value {{ color: {theme["accent"]}; font-size: 10px; margin-bottom: 4px; font-weight: 600; }}
        .bar-label {{ color: {theme["text"]}; font-size: 11px; margin-top: 8px; font-weight: 500; }}
        .token-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
        .token-stat {{ background: {theme["stat_bg"]}; border-radius: 12px; padding: 16px; border: 1px solid {theme["stat_border"]}; text-align: center; }}
        .token-label {{ color: {theme["text"]}; font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }}
        .token-value {{ color: {theme["blue"]}; font-size: 22px; font-weight: 600; }}
        .token-daily {{ color: {theme["text"]}; font-size: 12px; margin-top: 4px; }}
        .repo-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px; }}
        .repo-item {{ background: {theme["stat_bg"]}; border-radius: 12px; padding: 14px 12px; text-align: center; border: 1px solid {theme["stat_border"]}; }}
        .repo-name {{ color: {theme["title"]}; font-size: 12px; font-weight: 600; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .repo-hours {{ color: {theme["accent"]}; font-size: 18px; font-weight: 700; }}
        .repo-sessions {{ color: {theme["text"]}; font-size: 10px; margin-top: 4px; }}
        .footer {{ padding-top: 20px; border-top: 1px solid {theme["divider"]}; }}
        .meta {{ color: {theme["text_secondary"]}; font-size: 12px; }}
        .eye {{ animation: blink 4s ease-in-out infinite; }}
        .arm-right {{ transform-origin: 13px 7px; animation: wave 5s ease-in-out infinite; }}
        @keyframes blink {{ 0%, 92%, 100% {{ transform: scaleY(1); }} 95%, 97% {{ transform: scaleY(0.1); }} }}
        @keyframes wave {{ 0%, 85%, 100% {{ transform: rotate(0deg); }} 88%, 92% {{ transform: rotate(-25deg); }} 90% {{ transform: rotate(-15deg); }} }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div class="header-left">
                <div class="logo"><svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 16 16"><rect width="16" height="16" fill="none"/><rect x="3" y="2" width="10" height="6" fill="#d97757"/><rect x="4" y="8" width="1" height="3" fill="#d97757"/><rect x="5.5" y="8" width="1" height="3" fill="#d97757"/><rect x="9.5" y="8" width="1" height="3" fill="#d97757"/><rect x="11" y="8" width="1" height="3" fill="#d97757"/><rect x="2" y="4" width="1" height="3" fill="#d97757"/><rect class="arm-right" x="13" y="4" width="1" height="3" fill="#d97757"/><rect class="eye" x="5" y="3" width="1" height="2" fill="#141413"/><rect class="eye" x="10" y="3" width="1" height="2" fill="#141413"/></svg></div>
                <div class="header-text">
                    <div class="title">Claude Code Stats</div>
                    <div class="period">{period_label}</div>
                </div>
            </div>
            <div class="header-right">
                <div class="ratio">{message_ratio:.1f}x</div>
                <div class="ratio-label">msg ratio</div>
                <div class="ratio" style="margin-top: 6px;">{token_ratio:.2f}x</div>
                <div class="ratio-label">token ratio</div>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat">
                <div class="stat-label">Active</div>
                <div class="stat-value">{total_active/60:.1f}h</div>
                <div class="stat-daily">{active_per_day:.1f}h/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">Clock</div>
                <div class="stat-value">{total_wall/60:.1f}h</div>
                <div class="stat-daily">{wall_per_day:.1f}h/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">Sessions</div>
                <div class="stat-value">{format_number(total_sessions)}</div>
                <div class="stat-daily">{sessions_per_day:.1f}/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">User Prompts</div>
                <div class="stat-value">{format_number(total_user_messages)}</div>
                <div class="stat-daily">{user_messages_per_day:.0f}/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">Claude Msgs</div>
                <div class="stat-value">{format_number(total_claude_messages)}</div>
                <div class="stat-daily">{claude_messages_per_day:.0f}/day</div>
            </div>
            <div class="stat">
                <div class="stat-label">Tool Results</div>
                <div class="stat-value">{format_number(total_tool_results)}</div>
                <div class="stat-daily">{tool_results_per_day:.0f}/day</div>
            </div>
        </div>

        <div class="section-title">{chart_label}</div>
        <div class="chart">
            {chart_bars}
        </div>

        <div class="section-title">Token Usage</div>
        <div class="token-stats">
            <div class="token-stat">
                <div class="token-label">Input Tokens</div>
                <div class="token-value">{format_tokens(total_input_tokens)}</div>
                <div class="token-daily">{format_tokens(int(input_tokens_per_day))}/day</div>
            </div>
            <div class="token-stat">
                <div class="token-label">Output Tokens</div>
                <div class="token-value">{format_tokens(total_output_tokens)}</div>
                <div class="token-daily">{format_tokens(int(output_tokens_per_day))}/day</div>
            </div>
        </div>

        <div class="section-title">Session Behavior</div>
        <div class="token-stats">
            <div class="token-stat">
                <div class="token-label">Clears</div>
                <div class="token-value">{format_number(total_clears)}</div>
                <div class="token-daily">{clears_per_session:.1f}/session</div>
            </div>
            <div class="token-stat">
                <div class="token-label">Compacts</div>
                <div class="token-value">{format_number(total_compacts)}</div>
                <div class="token-daily">{compacts_per_session:.1f}/session</div>
            </div>
        </div>

        {repo_section}

        <div class="footer">
            <div class="meta">Generated {now.strftime('%Y-%m-%d %H:%M')}</div>
        </div>
    </div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Claude Code usage time and generate statistics report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Print report to stdout
  %(prog)s -o report.md        Save report to file
  %(prog)s -g 10               Use 10-minute gap threshold (default: 15)
  %(prog)s --tokens            Include token usage columns in daily breakdown
  %(prog)s --html card         Generate compact HTML card for sharing
  %(prog)s --html full         Generate full HTML stats card with chart
  %(prog)s --period 30         Show stats for last 30 days only
  %(prog)s --by-repo           Show per-repository breakdown
  %(prog)s --repo my-project   Filter stats to a single repo
  %(prog)s --repo-pattern '/code/([^/]+)'  Custom repo extraction pattern

The tool reads data from ~/.claude/ directory where Claude Code stores
conversation transcripts and statistics.
        """
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--gap-threshold", "-g",
        type=int,
        default=DEFAULT_GAP_THRESHOLD_MINUTES,
        help=f"Gap threshold in minutes for idle detection (default: {DEFAULT_GAP_THRESHOLD_MINUTES})"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages (only output report)"
    )
    parser.add_argument(
        "--tokens", "-t",
        action="store_true",
        help="Include token usage (input/output) in daily breakdown instead of clears/compacts"
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Use light theme with Anthropic brand colors (default: dark theme)"
    )
    parser.add_argument(
        "--username", "-u",
        type=str,
        default=None,
        help="GitHub username to display (e.g., @joshroman)"
    )
    parser.add_argument(
        "--html",
        choices=["card", "full"],
        default=None,
        help="Generate HTML output: 'card' for compact shareable card, 'full' for detailed stats with chart"
    )
    parser.add_argument(
        "--period", "-p",
        type=int,
        choices=[7, 30, 90],
        default=None,
        help="Limit report to last N days (7, 30, or 90)"
    )
    parser.add_argument(
        "--by-repo",
        action="store_true",
        help="Show per-repository breakdown in report"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Filter stats to a single repository name"
    )
    parser.add_argument(
        "--repo-pattern",
        type=str,
        default=None,
        help="Regex pattern to extract repo name from cwd (e.g., '/code/([^/]+)')"
    )

    args = parser.parse_args()

    # Load username from .env if not provided via CLI
    if args.username is None:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GITHUB_USERNAME="):
                            args.username = line.split("=", 1)[1].strip()
                            break
            except IOError:
                pass

    # Check if Claude Code data exists
    if not CLAUDE_DIR.exists():
        print(f"Error: Claude Code data directory not found at {CLAUDE_DIR}", file=sys.stderr)
        print("Make sure you have Claude Code installed and have used it at least once.", file=sys.stderr)
        sys.exit(1)

    # Load repo pattern from .env if not provided via CLI
    repo_pattern = args.repo_pattern
    if repo_pattern is None:
        repo_pattern = load_repo_pattern_from_env()

    if not args.quiet:
        print("Loading conversation data...", file=sys.stderr)

    messages, session_summary_counts, daily_tokens = load_jsonl_messages(repo_pattern=repo_pattern)
    assistant_durations = load_assistant_durations()
    stats_cache = load_stats_cache()

    if not messages:
        print("No conversation data found. Have you used Claude Code yet?", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"  Found {len(messages)} messages", file=sys.stderr)
        print(f"  Found {sum(session_summary_counts.values())} compaction events", file=sys.stderr)
        if daily_tokens:
            total_tokens = sum(d["input"] + d["output"] for d in daily_tokens.values())
            print(f"  Found {format_tokens(total_tokens)} tokens across {len(daily_tokens)} days", file=sys.stderr)
        print("Calculating session statistics...", file=sys.stderr)

    session_stats = calculate_session_stats(messages, args.gap_threshold, session_summary_counts)

    # Filter to single repo if requested
    if args.repo:
        original_count = len(session_stats)
        session_stats = [s for s in session_stats if s.get("repo_name") == args.repo]
        if not args.quiet:
            print(f"  Filtered to repo '{args.repo}': {len(session_stats)} of {original_count} sessions", file=sys.stderr)

    if not args.quiet:
        print(f"  Analyzed {len(session_stats)} sessions", file=sys.stderr)
        if args.by_repo:
            repos = set(s.get("repo_name", "unknown") for s in session_stats)
            print(f"  Found {len(repos)} repositories", file=sys.stderr)
        print("Generating report...", file=sys.stderr)

    # Calculate per-repo stats if requested
    per_repo_stats = None
    if args.by_repo:
        period_days = args.period if args.period else 7
        per_repo_stats = calculate_per_repo_stats(session_stats, period_days=period_days)

    # Determine output format
    if args.html:
        # HTML output
        period_days = args.period if args.period else 7
        report = generate_html_report(
            session_stats,
            stats_cache,
            daily_tokens,
            period_days=period_days,
            style=args.html,
            light_mode=args.light,
            username=args.username,
            per_repo_stats=per_repo_stats
        )
    else:
        # Markdown output
        report = generate_report(
            session_stats,
            assistant_durations,
            stats_cache,
            args.gap_threshold,
            daily_tokens=daily_tokens,
            show_tokens=args.tokens,
            period_days=args.period,
            per_repo_stats=per_repo_stats
        )

    # Output report
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        if not args.quiet:
            print(f"\nReport saved to: {args.output}", file=sys.stderr)
    else:
        print(report)

    # Print quick summary to stderr if outputting to file
    if args.output and not args.quiet:
        now = datetime.now()
        week_sessions = [s for s in session_stats
                        if s["date"] >= (now - timedelta(days=7)).date()]
        week_active = sum(s["active_minutes"] for s in week_sessions)
        print(f"\nQuick Stats (last 7 days):", file=sys.stderr)
        print(f"  Sessions: {len(week_sessions)}", file=sys.stderr)
        print(f"  Active time: {week_active/60:.1f} hours", file=sys.stderr)


if __name__ == "__main__":
    main()
