"""Cursor IDE data collector.

Parses data from ~/.cursor/ai-tracking/ai-code-tracking.db:
- ai_code_hashes — AI-generated code (hash, model, file, timestamp)
- scored_commits — AI vs human lines per commit
- conversation_summaries — title, TLDR, model, mode per conversation
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from agenttop.collectors.base import BaseCollector
from agenttop.config import CURSOR_DIR
from agenttop.models import Event, Session, ToolName, ToolStats

TOKENS_PER_CONVERSATION_ESTIMATE = 2000
COST_PER_TOKEN = 0.000003  # Cursor uses mostly cheaper models

# Directories that are containers, not project names
_CONTAINER_DIRS = {"repo", "repos", "desktop", "projects", "dev", "src", "code", "work", "documents"}


def _extract_project(filepath: str) -> str | None:
    """Extract project name from an absolute file path.

    Walks past the home directory and any container dirs (repo/, Desktop/, etc.)
    to find the actual project directory name.
    """
    if not filepath or not filepath.startswith("/"):
        return None
    from pathlib import Path as _P

    home = str(_P.home())
    rel = filepath[len(home) + 1:] if filepath.startswith(home + "/") else filepath.lstrip("/")
    parts = rel.split("/")

    # Skip container directories to find the real project name
    for i, part in enumerate(parts):
        if part.lower() not in _CONTAINER_DIRS:
            # Don't return if it looks like a file (has extension) with no subdirectory
            if "." in part and i == len(parts) - 1:
                return None
            return part
    return None


class CursorCollector(BaseCollector):
    """Collects data from Cursor's local SQLite database."""

    def __init__(self, cursor_dir: Path | None = None) -> None:
        self._dir = cursor_dir or CURSOR_DIR
        self._db_path = self._dir / "ai-tracking" / "ai-code-tracking.db"

    @property
    def tool_name(self) -> ToolName:
        return ToolName.CURSOR

    def is_available(self) -> bool:
        return self._db_path.exists()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # -- Conversations --

    def _get_conversations(self, since_ms: int = 0) -> list[dict]:
        """Fetch conversation summaries from Cursor DB."""
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM conversation_summaries WHERE updatedAt >= ? ORDER BY updatedAt DESC",
                (since_ms,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except (sqlite3.Error, OSError):
            return []

    # -- AI code hashes --

    def _get_ai_code_hashes(self, since_ms: int = 0) -> list[dict]:
        """Fetch AI-generated code records."""
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM ai_code_hashes WHERE createdAt >= ? ORDER BY createdAt DESC",
                (since_ms,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except (sqlite3.Error, OSError):
            return []

    # -- Scored commits --

    def _get_scored_commits(self, since_ms: int = 0) -> list[dict]:
        """Fetch commit scores (AI vs human lines)."""
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM scored_commits WHERE scoredAt >= ? ORDER BY scoredAt DESC",
                (since_ms,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except (sqlite3.Error, OSError):
            return []

    # -- BaseCollector interface --

    def collect_events(self) -> list[Event]:
        """Collect events from Cursor AI tracking DB."""
        events = []
        for code in self._get_ai_code_hashes():
            ts_ms = code.get("createdAt") or code.get("timestamp") or 0
            if not ts_ms:
                continue
            events.append(
                Event(
                    tool=ToolName.CURSOR,
                    event_type="ai_code",
                    timestamp=datetime.fromtimestamp(ts_ms / 1000),
                    session_id=code.get("conversationId"),
                    data={
                        "source": code.get("source", ""),
                        "file": code.get("fileName", ""),
                        "model": code.get("model", ""),
                    },
                )
            )
        return events

    def collect_sessions(self) -> list[Session]:
        """Build sessions from conversation summaries and ai_code_hashes.

        Uses conversation_summaries when available, otherwise groups
        ai_code_hashes by conversationId and extracts project from file paths.
        """
        sessions = []

        # Try conversation summaries first
        for conv in self._get_conversations():
            updated_ms = conv.get("updatedAt", 0)
            if not updated_ms:
                continue
            ts = datetime.fromtimestamp(updated_ms / 1000)
            sessions.append(
                Session(
                    id=conv.get("conversationId", "unknown"),
                    tool=ToolName.CURSOR,
                    start_time=ts,
                    end_time=ts,
                    message_count=1,
                    total_tokens=TOKENS_PER_CONVERSATION_ESTIMATE,
                    estimated_cost_usd=TOKENS_PER_CONVERSATION_ESTIMATE * COST_PER_TOKEN,
                    prompts=[conv.get("title", ""), conv.get("tldr", "")],
                )
            )

        if sessions:
            return sessions

        # Fallback: build sessions from ai_code_hashes grouped by conversationId
        return self._sessions_from_code_hashes()

    def _sessions_from_code_hashes(self) -> list[Session]:
        """Build sessions from ai_code_hashes, extracting project from file paths."""
        groups: dict[str, list[dict]] = {}
        for code in self._get_ai_code_hashes():
            cid = code.get("conversationId") or "unknown"
            groups.setdefault(cid, []).append(code)

        sessions = []
        for cid, entries in groups.items():
            timestamps = []
            project = None
            for entry in entries:
                ts_ms = entry.get("createdAt") or entry.get("timestamp") or 0
                if ts_ms:
                    timestamps.append(datetime.fromtimestamp(ts_ms / 1000))
                if not project:
                    project = _extract_project(entry.get("fileName", ""))

            if not timestamps:
                continue
            start = min(timestamps)
            end = max(timestamps)
            msg_count = len(entries)
            tokens = msg_count * TOKENS_PER_CONVERSATION_ESTIMATE
            sessions.append(
                Session(
                    id=cid,
                    tool=ToolName.CURSOR,
                    project=project,
                    start_time=start,
                    end_time=end,
                    message_count=msg_count,
                    total_tokens=tokens,
                    estimated_cost_usd=tokens * COST_PER_TOKEN,
                )
            )
        return sessions

    def get_stats(self, days: int = 0) -> ToolStats:
        """Aggregate stats for the dashboard.

        Args:
            days: Number of days to aggregate. 0 = all available data.
        """
        stats = ToolStats(tool=ToolName.CURSOR)
        if days > 0:
            since = datetime.now() - timedelta(days=days)
        else:
            since = datetime(2000, 1, 1)
        since_ms = int(since.timestamp() * 1000)

        convs = self._get_conversations(since_ms=since_ms)
        codes = self._get_ai_code_hashes(since_ms=since_ms)

        stats.sessions_today = len(convs)
        stats.messages_today = len(codes)
        stats.tool_calls_today = len(codes)
        stats.tokens_today = len(convs) * TOKENS_PER_CONVERSATION_ESTIMATE
        stats.estimated_cost_today = stats.tokens_today * COST_PER_TOKEN

        # Build hourly distribution from code hashes
        hourly = [0] * 24
        for code in self._get_ai_code_hashes(since_ms=since_ms):
            ts_ms = code.get("createdAt", 0)
            if ts_ms:
                hour = datetime.fromtimestamp(ts_ms / 1000).hour
                hourly[hour] += TOKENS_PER_CONVERSATION_ESTIMATE // 10
        stats.hourly_tokens = hourly

        if codes or convs:
            stats.status = "active"

        return stats

    def get_ai_vs_human_ratio(self) -> dict:
        """Calculate AI vs human code contribution ratio from scored commits."""
        commits = self._get_scored_commits()
        total_ai = 0
        total_human = 0
        for c in commits:
            tab = c.get("tabLinesAdded") or 0
            composer = c.get("composerLinesAdded") or 0
            human = c.get("humanLinesAdded") or 0
            total_ai += tab + composer
            total_human += human
        total = total_ai + total_human
        return {
            "ai_lines": total_ai,
            "human_lines": total_human,
            "ai_percentage": (total_ai / total * 100) if total > 0 else 0,
        }
