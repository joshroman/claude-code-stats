# Claude Code Stats

Usage analytics for [Claude Code](https://claude.ai/code) CLI - see how much time you actually spend using Claude Code.

## Features

- **Active Time Tracking**: Estimates actual usage time by analyzing gaps between messages (accounts for idle periods, breaks, and sessions left open overnight)
- **Session Analytics**: Track session counts, message breakdowns, and efficiency metrics
- **Command Usage**: Monitor `/clear` and `/compact` command frequency
- **Model Statistics**: View token usage by model from Claude Code's cache
- **Hourly Distribution**: See when you're most active with Claude Code

## Installation

### Quick Start (No Installation)

```bash
# Clone and run directly
git clone https://github.com/joshroman/claude-code-stats.git
cd claude-code-stats
python3 claude_code_stats.py
```

### Install as Package

```bash
pip install claude-code-stats
```

Or install from source:

```bash
git clone https://github.com/joshroman/claude-code-stats.git
cd claude-code-stats
pip install .
```

## Usage

```bash
# Print report to terminal
claude-code-stats

# Save report to file
claude-code-stats -o report.md

# Custom idle threshold (default: 15 minutes)
claude-code-stats --gap-threshold 10

# Quiet mode (no progress messages)
claude-code-stats -q -o report.md
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Save report to file (default: print to stdout) |
| `-g, --gap-threshold MINS` | Minutes of inactivity before counting as idle (default: 15) |
| `-q, --quiet` | Suppress progress messages |
| `-V, --version` | Show version number |

## Sample Output

```markdown
# Claude Code Usage Report

Generated: **2025-12-31 10:59**

## Summary

| Period | Sessions | Messages | Active Time | Wall-Clock | Efficiency |
|--------|----------|----------|-------------|------------|------------|
| Last 7 days | 273 | 35938 | **68.4h** | 453.6h | 15% |
| Last 30 days | 871 | 99885 | **191.9h** | 2370.9h | 8% |

## Last 7 Days - Daily Breakdown

| Date | Sessions | Active | Clock | User Msgs | Claude Msgs | Clears | Compacts |
|------|----------|--------|-------|-----------|-------------|--------|----------|
| 2025-12-31 | 22 | 9.4h | 21.2h | 260 | 3026 | 31 | 42 |
| 2025-12-30 | 34 | 11.9h | 2.5d | 306 | 3251 | 6 | 63 |
...
```

## How It Works

### Data Sources

Claude Code stores conversation data in `~/.claude/`:

- **`projects/**/*.jsonl`** - Conversation transcripts with timestamps
- **`__store.db`** - SQLite database with response timing data
- **`stats-cache.json`** - Pre-computed usage statistics

### Methodology

**Active Time** is calculated by:
1. Loading all message timestamps from conversation files
2. Calculating gaps between consecutive messages
3. Summing gaps that are ≤ threshold (default: 15 minutes)
4. Longer gaps indicate idle time (bathroom breaks, meetings, left overnight)

This prevents inflated numbers from sessions that were left open but not actively used.

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **Active Time** | Estimated actual usage (gaps ≤ threshold) |
| **Wall-Clock Time** | Total time from first to last message (includes idle) |
| **Efficiency** | Ratio of active to wall-clock time |
| **Clears** | Number of `/clear` commands used |
| **Compacts** | Number of `/compact` commands + auto-compactions |

## Requirements

- Python 3.8+
- Claude Code installed and used at least once

No external dependencies required - uses only Python standard library.

## Privacy

This tool only reads data from your local `~/.claude/` directory. No data is sent anywhere - all processing happens locally on your machine.

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related

- [Claude Code](https://claude.ai/code) - Anthropic's official CLI for Claude
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
