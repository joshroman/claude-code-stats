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

__version__ = "0.1.0"

# Configuration - Claude Code data locations
CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"
STORE_DB = CLAUDE_DIR / "__store.db"
STATS_CACHE = CLAUDE_DIR / "stats-cache.json"

DEFAULT_GAP_THRESHOLD_MINUTES = 15


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


def load_jsonl_messages() -> Tuple[List[Dict], Dict[str, int]]:
    """Load all messages from JSONL conversation files.

    Returns:
        Tuple of (messages list, session_summary_counts dict)
        session_summary_counts maps session_id to count of summary entries (compactions)
    """
    messages = []
    session_summary_counts: Dict[str, int] = defaultdict(int)

    if not PROJECTS_DIR.exists():
        return messages, dict(session_summary_counts)

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

                                messages.append({
                                    "timestamp": dt,
                                    "session_id": data["sessionId"],
                                    "type": data.get("type", "unknown"),
                                    "is_user": data.get("type") == "user",
                                    "is_actual_user_input": is_actual_user_input,
                                    "is_assistant": data.get("type") == "assistant",
                                    "is_clear": is_clear,
                                    "is_compact": is_compact,
                                })
                    except json.JSONDecodeError:
                        continue
        except (IOError, OSError):
            continue

    return messages, dict(session_summary_counts)


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
            "date": first_ts.date()
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


def generate_report(
    session_stats: List[Dict],
    assistant_durations: Dict[str, float],
    stats_cache: Dict,
    gap_threshold: int
) -> str:
    """Generate markdown report."""
    now = datetime.now()

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

---

## Last 7 Days - Daily Breakdown

| Date | Sessions | Active | Clock | User Msgs | Claude Msgs | Clears | Compacts |
|------|----------|--------|-------|-----------|-------------|--------|----------|
"""

    # Add daily breakdown for last week
    for i in range(7):
        d = (now - timedelta(days=i)).date().isoformat()
        if d in week_stats["daily_breakdown"]:
            day = week_stats["daily_breakdown"][d]
            active_fmt = format_duration(day['active_minutes'])
            clock_fmt = format_duration(day['wall_clock_minutes'])
            report += f"| {d} | {day['sessions']} | {active_fmt} | {clock_fmt} | {day['user_messages']} | {day['assistant_messages']} | {day['clear_count']} | {day['compact_count']} |\n"
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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Claude Code usage time and generate statistics report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Print report to stdout
  %(prog)s -o report.md        Save report to file
  %(prog)s -g 10               Use 10-minute gap threshold (default: 15)
  %(prog)s --json              Output raw stats as JSON (not yet implemented)

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

    args = parser.parse_args()

    # Check if Claude Code data exists
    if not CLAUDE_DIR.exists():
        print(f"Error: Claude Code data directory not found at {CLAUDE_DIR}", file=sys.stderr)
        print("Make sure you have Claude Code installed and have used it at least once.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("Loading conversation data...", file=sys.stderr)

    messages, session_summary_counts = load_jsonl_messages()
    assistant_durations = load_assistant_durations()
    stats_cache = load_stats_cache()

    if not messages:
        print("No conversation data found. Have you used Claude Code yet?", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"  Found {len(messages)} messages", file=sys.stderr)
        print(f"  Found {sum(session_summary_counts.values())} compaction events", file=sys.stderr)
        print("Calculating session statistics...", file=sys.stderr)

    session_stats = calculate_session_stats(messages, args.gap_threshold, session_summary_counts)

    if not args.quiet:
        print(f"  Analyzed {len(session_stats)} sessions", file=sys.stderr)
        print("Generating report...", file=sys.stderr)

    report = generate_report(
        session_stats,
        assistant_durations,
        stats_cache,
        args.gap_threshold
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
