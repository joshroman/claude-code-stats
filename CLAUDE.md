# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-file Python CLI tool that analyzes Claude Code usage data from `~/.claude/` to generate statistics reports (markdown or HTML).

## Commands

```bash
# Run directly
python claude_code_stats.py

# Run with options
python claude_code_stats.py --html full -p 7 -o stats.html
python claude_code_stats.py --tokens -o report.md

# Install locally for development
pip install -e .
```

## Architecture

Single module (`claude_code_stats.py`) with no external dependencies. Key data sources from `~/.claude/`:
- `projects/**/*.jsonl` - Conversation transcripts with timestamps and token usage
- `__store.db` - SQLite with response timing data
- `stats-cache.json` - Pre-computed model usage stats

Key functions:
- `load_jsonl_messages()` - Extracts messages, token counts, and compaction events from JSONL files
- `calculate_session_stats()` - Computes per-session metrics using gap-based active time estimation
- `generate_report()` - Markdown output
- `generate_html_report()` - HTML card output (card/full styles) with animated Claude icon
