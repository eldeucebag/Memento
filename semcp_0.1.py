#!/usr/bin/env python3
"""Corrected MCP memory server using the official MCP library.

Memory-engineering discipline:
- bounded retrieval
- time-based decay
- bounded reinforcement
- forgetting
- provenance journal
- migration backup
- FTS with fallback
- finite floats only
- bounded responses
- event-based wake/sleep state
- sleep-cycle maintenance and novel synthesis (dream tool)

Install:
    pip install mcp

Run:
    python memory_mcp_corrected.py
"""

import json
import math
import os
import re
import random
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DB_PATH = os.environ.get("MCP_MEMORY_DB", "memory_v4.db")
DEFAULT_NAMESPACE = os.environ.get("MCP_MEMORY_NAMESPACE", "default")

CURRENT_SCHEMA_VERSION = 4

MAX_NAME = 300
MAX_TEXT = 4000
MAX_QUERY = 500
MAX_RECALL_LIMIT = 100
MAX_GRAPH_DEPTH = 4
MAX_TOP_NODES = 500
MAX_NAMESPACES = 1000  # FIX #6: bound the namespace listing

DECAY_RATE = float(os.environ.get("MCP_MEMORY_DECAY_RATE", "0.05"))
SALIENCE_HALF_LIFE_HOURS = 48.0
AFFECT_HALF_LIFE_HOURS = 24.0

# --- Synthesis config ---
MAX_SYNTHESIS_CANDIDATES = 500
JOURNAL_COOCCURRENCE_WINDOW_HOURS = float(
    os.environ.get("MCP_MEMORY_JOURNAL_WINDOW_HOURS", "2.0")
)
JOURNAL_COOCCURRENCE_LOOKBACK_DAYS = 30
SCHEMA_NODE_PREFIX = "__schema__::"
MAX_SCHEMAS_PER_DREAM = 10
RELATION_ATTENUATION = 0.7   # used for typed transitive inference


_db_dir = os.path.dirname(os.path.abspath(DB_PATH))
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

FTS_AVAILABLE = False


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: Any, low: float, high: float) -> float:
    if value is None:
        return low
    try:
        x = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(x):
        return low
    return max(low, min(high, x))


def clamp_int(value: Any, low: int, high: int) -> int:
    try:
        x = int(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, x))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def safe_bool_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    s = str(value).strip().lower()
    return 1 if s in ("1", "true", "yes", "y") else 0


def clean_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:limit]


def require_text(value: Any, field: str, limit: int) -> str:
    s = clean_text(value, limit)
    if not s:
        raise ValueError(f"{field} is required")
    return s


def as_finite_float(value: Any, field: str, required: bool = False) -> Optional[float]:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be numeric")
    if not math.isfinite(x):
        raise ValueError(f"{field} must be finite")
    return x


def canonical_name(name: str) -> str:
    return " ".join(name.strip().casefold().split())


def parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since(value: Any) -> float:
    dt = parse_ts(value)
    if not dt:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)


def fts_escape(text: str) -> str:
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    if not tokens:
        return '""'
    tokens = tokens[:20]
    return " OR ".join(f'"{token}"' for token in tokens)


# FIX #7: helper to inspect the latest event without duplicating SQL
def latest_event_type(conn: sqlite3.Connection, namespace: str) -> Optional[str]:
    row = conn.execute(
        "SELECT type FROM events WHERE namespace = ? ORDER BY ts DESC LIMIT 1",
        (namespace,),
    ).fetchone()
    return row["type"] if row else None


# ---------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def get_columns(conn: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def backup_database() -> Optional[str]:
    if not os.path.exists(DB_PATH):
        return None

    backup_path = f"{DB_PATH}.backup-{int(time.time())}"
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(backup_path)

    with dst:
        src.backup(dst)

    src.close()
    dst.close()
    return backup_path


# ---------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------


def create_config_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS config (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            ts TEXT NOT NULL,
            PRIMARY KEY(namespace, key)
        )
        """
    )


def create_nodes_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY,
            namespace TEXT NOT NULL DEFAULT 'default',
            canonical TEXT NOT NULL,
            display_name TEXT NOT NULL,
            objective TEXT NOT NULL DEFAULT '',
            flow TEXT NOT NULL DEFAULT '',
            corpus TEXT NOT NULL DEFAULT '',
            verified INTEGER NOT NULL DEFAULT 0,
            suppressed INTEGER NOT NULL DEFAULT 0,
            base_strength REAL NOT NULL DEFAULT 0.35,
            salience REAL NOT NULL DEFAULT 0.0,
            affect REAL NOT NULL DEFAULT 0.0,
            salience_ts TEXT,
            affect_ts TEXT,
            last_accessed TEXT NOT NULL,
            created TEXT NOT NULL,
            UNIQUE(namespace, canonical)
        )
        """
    )


def create_features_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS features (
            node_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value REAL NOT NULL,
            contributes INTEGER NOT NULL DEFAULT 0,
            ts TEXT NOT NULL,
            PRIMARY KEY(node_id, key),
            FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )
        """
    )


def create_edges_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY,
            namespace TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            evidence TEXT NOT NULL DEFAULT '',
            valid_from TEXT,
            valid_to TEXT,
            ts TEXT NOT NULL,
            UNIQUE(namespace, source_id, target_id, relation),
            FOREIGN KEY(source_id) REFERENCES nodes(id) ON DELETE CASCADE,
            FOREIGN KEY(target_id) REFERENCES nodes(id) ON DELETE CASCADE
        )
        """
    )


def create_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            namespace TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('wake', 'sleep_start')),
            quality REAL,
            ts TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_ns_ts
        ON events(namespace, ts DESC)
        """
    )


def create_journal_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY,
            namespace TEXT NOT NULL,
            tool TEXT NOT NULL,
            node_id INTEGER,
            detail TEXT NOT NULL,
            ts TEXT NOT NULL,
            FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_journal_ns_ts
        ON journal(namespace, ts DESC)
        """
    )


def create_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_nodes_ns_canonical
        ON nodes(namespace, canonical)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_edges_ns_source
        ON edges(namespace, source_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_edges_ns_target
        ON edges(namespace, target_id)
        """
    )


# ---------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------


def insert_node_row(
    conn: sqlite3.Connection,
    namespace: str,
    display_name: str,
    objective: str = "",
    flow: str = "",
    corpus: str = "",
    base_strength: float = 0.35,
    salience: float = 0.0,
    affect: float = 0.0,
    verified: int = 0,
    suppressed: int = 0,
    last_accessed: Optional[str] = None,
    created: Optional[str] = None,
) -> Optional[int]:
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    display_name = clean_text(display_name, MAX_NAME)
    if not display_name:
        return None

    canonical = canonical_name(display_name)
    now = now_utc()

    conn.execute(
        """
        INSERT OR IGNORE INTO nodes (
            namespace, canonical, display_name, objective, flow, corpus,
            verified, suppressed, base_strength, salience, affect,
            salience_ts, affect_ts, last_accessed, created
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            namespace,
            canonical,
            display_name,
            clean_text(objective, MAX_TEXT),
            clean_text(flow, MAX_TEXT),
            clean_text(corpus, MAX_TEXT),
            1 if verified else 0,
            1 if suppressed else 0,
            clamp(base_strength, 0.0, 1.0),
            clamp(salience, -1.0, 1.0),
            clamp(affect, -1.0, 1.0),
            now,
            now,
            last_accessed or now,
            created or now,
        ),
    )

    row = conn.execute(
        "SELECT id FROM nodes WHERE namespace = ? AND canonical = ?",
        (namespace, canonical),
    ).fetchone()

    return row["id"] if row else None


def ensure_node_id(
    conn: sqlite3.Connection,
    namespace: str,
    name: str,
    corpus: str = "MIGRATED",
) -> Optional[int]:
    name = clean_text(name, MAX_NAME)
    if not name:
        return None

    canonical = canonical_name(name)
    row = conn.execute(
        "SELECT id FROM nodes WHERE namespace = ? AND canonical = ?",
        (namespace, canonical),
    ).fetchone()

    if row:
        return row["id"]

    return insert_node_row(
        conn,
        namespace=namespace,
        display_name=name,
        corpus=corpus,
        base_strength=0.2,
    )


def copy_legacy_nodes(conn: sqlite3.Connection, legacy_table: str) -> None:
    rows = conn.execute(f"SELECT * FROM {legacy_table}").fetchall()

    for row in rows:
        d = dict(row)

        name = str(d.get("name") or d.get("display_name") or "").strip()
        if not name:
            continue

        namespace = str(d.get("namespace") or DEFAULT_NAMESPACE).strip() or DEFAULT_NAMESPACE

        base = safe_float(
            d.get("base_strength", d.get("base_vitality", d.get("vitality", 0.35))),
            0.35,
        )

        insert_node_row(
            conn,
            namespace=namespace,
            display_name=name,
            objective=str(d.get("objective") or ""),
            flow=str(d.get("flow") or ""),
            corpus=str(d.get("corpus") or ""),
            base_strength=base,
            salience=safe_float(d.get("salience"), 0.0),
            affect=safe_float(d.get("affect"), 0.0),
            verified=safe_bool_int(d.get("verified")),
            suppressed=safe_bool_int(d.get("suppressed")),
            last_accessed=d.get("last_accessed") or d.get("ts") or now_utc(),
            created=d.get("created") or now_utc(),
        )


def copy_legacy_edges(conn: sqlite3.Connection, legacy_table: str) -> None:
    rows = conn.execute(f"SELECT * FROM {legacy_table}").fetchall()

    for row in rows:
        d = dict(row)

        source = str(d.get("source") or d.get("source_name") or "").strip()
        target = str(d.get("target") or d.get("target_name") or "").strip()
        if not source or not target:
            continue

        namespace = str(d.get("namespace") or DEFAULT_NAMESPACE).strip() or DEFAULT_NAMESPACE

        source_id = ensure_node_id(conn, namespace, source, corpus="MIGRATED")
        target_id = ensure_node_id(conn, namespace, target, corpus="MIGRATED")
        if source_id is None or target_id is None:
            continue

        relation = clean_text(d.get("relation") or "related_to", 100).lower()
        weight = clamp(safe_float(d.get("weight"), 0.5), -1.0, 1.0)
        confidence = clamp(safe_float(d.get("confidence"), 0.5), 0.0, 1.0)
        evidence = clean_text(d.get("evidence") or "migrated", MAX_TEXT)
        ts = d.get("ts") or now_utc()

        conn.execute(
            """
            INSERT OR IGNORE INTO edges (
                namespace, source_id, target_id, relation,
                weight, confidence, evidence, valid_from, valid_to, ts
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                namespace,
                source_id,
                target_id,
                relation,
                weight,
                confidence,
                evidence,
                ts,
                None,
                ts,
            ),
        )


def copy_legacy_weights(conn: sqlite3.Connection, legacy_table: str) -> None:
    rows = conn.execute(f"SELECT * FROM {legacy_table}").fetchall()

    for row in rows:
        d = dict(row)

        node = str(d.get("node") or "").strip()
        key = clean_text(d.get("key") or "", 200)
        if not node or not key:
            continue

        namespace = str(d.get("namespace") or DEFAULT_NAMESPACE).strip() or DEFAULT_NAMESPACE
        node_id = ensure_node_id(conn, namespace, node, corpus="MIGRATED")
        if node_id is None:
            continue

        value = safe_float(d.get("value"), 0.0)

        conn.execute(
            """
            INSERT OR IGNORE INTO features (
                node_id, key, value, contributes, ts
            )
            VALUES(?, ?, ?, 0, ?)
            """,
            (node_id, key, value, d.get("ts") or now_utc()),
        )


def copy_legacy_journal(conn: sqlite3.Connection, legacy_table: str) -> None:
    rows = conn.execute(f"SELECT * FROM {legacy_table}").fetchall()

    for row in rows:
        d = dict(row)

        tool = clean_text(d.get("flow") or d.get("tool") or "legacy", 100)
        namespace = str(d.get("namespace") or DEFAULT_NAMESPACE).strip() or DEFAULT_NAMESPACE
        node_name = str(d.get("node") or "").strip()

        node_id = None
        if node_name:
            canonical = canonical_name(node_name)
            node_row = conn.execute(
                "SELECT id FROM nodes WHERE namespace = ? AND canonical = ?",
                (namespace, canonical),
            ).fetchone()
            if node_row:
                node_id = node_row["id"]

        detail = str(d.get("detail") or "")
        if not detail:
            detail = json.dumps(
                {"objective": d.get("objective") or ""},
                allow_nan=False,
            )

        conn.execute(
            """
            INSERT INTO journal (namespace, tool, node_id, detail, ts)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                namespace,
                tool,
                node_id,
                detail,
                d.get("ts") or now_utc(),
            ),
        )


def migrate_old_brain_events(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "brain"):
        return

    cols = get_columns(conn, "brain")
    if "wake_ts" not in cols or "sleep_onset_ts" not in cols:
        return

    row = conn.execute("SELECT * FROM brain LIMIT 1").fetchone()
    if not row:
        return

    d = dict(row)

    wake_ts = parse_ts(d.get("wake_ts"))
    if wake_ts:
        conn.execute(
            "INSERT INTO events(namespace, type, quality, ts) VALUES(?, 'wake', ?, ?)",
            (DEFAULT_NAMESPACE, clamp(d.get("sleep_quality"), 0.0, 1.0), wake_ts.isoformat()),
        )

    sleep_ts = parse_ts(d.get("sleep_onset_ts"))
    if sleep_ts:
        conn.execute(
            "INSERT INTO events(namespace, type, quality, ts) VALUES(?, 'sleep_start', ?, ?)",
            (DEFAULT_NAMESPACE, clamp(d.get("sleep_quality"), 0.0, 1.0), sleep_ts.isoformat()),
        )


def migrate_to_v4(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TRIGGER IF EXISTS nodes_ai")
    conn.execute("DROP TRIGGER IF EXISTS nodes_ad")
    conn.execute("DROP TRIGGER IF EXISTS nodes_au")
    conn.execute("DROP TABLE IF EXISTS nodes_fts")

    conn.execute("PRAGMA foreign_keys = OFF")

    create_config_table(conn)

    # Nodes
    if table_exists(conn, "nodes"):
        cols = get_columns(conn, "nodes")
        if "canonical" not in cols:
            legacy = f"nodes_legacy_{int(time.time_ns())}"
            conn.execute(f"ALTER TABLE nodes RENAME TO {legacy}")
            create_nodes_table(conn)
            copy_legacy_nodes(conn, legacy)
        else:
            create_nodes_table(conn)
    else:
        create_nodes_table(conn)

    # Edges
    if table_exists(conn, "edges"):
        cols = get_columns(conn, "edges")
        if "source_id" not in cols:
            legacy = f"edges_legacy_{int(time.time_ns())}"
            conn.execute(f"ALTER TABLE edges RENAME TO {legacy}")
            create_edges_table(conn)
            copy_legacy_edges(conn, legacy)
        else:
            create_edges_table(conn)
    else:
        create_edges_table(conn)

    # Features / old weights
    if table_exists(conn, "weights"):
        cols = get_columns(conn, "weights")
        if "node_id" not in cols:
            legacy = f"weights_legacy_{int(time.time_ns())}"
            conn.execute(f"ALTER TABLE weights RENAME TO {legacy}")
            create_features_table(conn)
            copy_legacy_weights(conn, legacy)
        else:
            create_features_table(conn)
    else:
        create_features_table(conn)

    # Journal
    if table_exists(conn, "journal"):
        cols = get_columns(conn, "journal")
        if "tool" not in cols:
            legacy = f"journal_legacy_{int(time.time_ns())}"
            conn.execute(f"ALTER TABLE journal RENAME TO {legacy}")
            create_journal_table(conn)
            copy_legacy_journal(conn, legacy)
        else:
            create_journal_table(conn)
    else:
        create_journal_table(conn)

    create_events_table(conn)
    create_indexes(conn)
    migrate_old_brain_events(conn)

    conn.execute("PRAGMA foreign_keys = ON")


def setup_fts(conn: sqlite3.Connection) -> bool:
    conn.execute("DROP TRIGGER IF EXISTS nodes_ai")
    conn.execute("DROP TRIGGER IF EXISTS nodes_ad")
    conn.execute("DROP TRIGGER IF EXISTS nodes_au")
    conn.execute("DROP TABLE IF EXISTS nodes_fts")

    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE nodes_fts USING fts5(
                display_name,
                objective,
                corpus,
                content='nodes',
                content_rowid='id'
            )
            """
        )

        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS nodes_ai
            AFTER INSERT ON nodes
            BEGIN
                INSERT INTO nodes_fts(rowid, display_name, objective, corpus)
                VALUES (new.id, new.display_name, new.objective, new.corpus);
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_ad
            AFTER DELETE ON nodes
            BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, display_name, objective, corpus)
                VALUES ('delete', old.id, old.display_name, old.objective, old.corpus);
            END;

            CREATE TRIGGER IF NOT EXISTS nodes_au
            AFTER UPDATE ON nodes
            WHEN old.display_name IS NOT new.display_name
              OR old.objective IS NOT new.objective
              OR old.corpus IS NOT new.corpus
            BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, display_name, objective, corpus)
                VALUES ('delete', old.id, old.display_name, old.objective, old.corpus);

                INSERT INTO nodes_fts(rowid, display_name, objective, corpus)
                VALUES (new.id, new.display_name, new.objective, new.corpus);
            END;
            """
        )

        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
        return True

    except sqlite3.OperationalError:
        return False


def init_db() -> None:
    global FTS_AVAILABLE

    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()

    if version != CURRENT_SCHEMA_VERSION and os.path.exists(DB_PATH):
        backup_database()

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    try:
        if version != CURRENT_SCHEMA_VERSION:
            migrate_to_v4(conn)
            conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
            conn.commit()

        if not table_exists(conn, "nodes_fts"):
            FTS_AVAILABLE = setup_fts(conn)
        else:
            FTS_AVAILABLE = True

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Memory math
# ---------------------------------------------------------------------


# FIX #5: renamed from get_config so the public tool name is free
def read_config(conn: sqlite3.Connection, namespace: str, key: str, default: str) -> str:
    row = conn.execute(
        "SELECT value FROM config WHERE namespace = ? AND key = ?",
        (namespace, key),
    ).fetchone()
    return row["value"] if row else default


def brain_state(conn: sqlite3.Connection, namespace: str) -> Dict[str, Any]:
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE

    local_now = datetime.now().astimezone()
    hour = local_now.hour + local_now.minute / 60.0

    peak_raw = read_config(conn, namespace, "circadian_peak_hour", "10.0")
    peak_hour = clamp(safe_float(peak_raw, 10.0), 0.0, 24.0)

    circadian = 0.5 + 0.5 * math.cos(2.0 * math.pi * ((hour - peak_hour) / 24.0))
    circadian = clamp(circadian, 0.0, 1.0)

    latest = conn.execute(
        """
        SELECT type, quality, ts
        FROM events
        WHERE namespace = ?
        ORDER BY ts DESC
        LIMIT 1
        """,
        (namespace,),
    ).fetchone()

    now = datetime.now(timezone.utc)

    if not latest:
        state_name = "awake"
        pressure = 0.0
        budget = 1.0
        quality = 0.7

    elif latest["type"] == "wake":
        state_name = "awake"
        wake_dt = parse_ts(latest["ts"]) or now
        hours_awake = max(0.0, (now - wake_dt).total_seconds() / 3600.0)
        quality = clamp(latest["quality"], 0.0, 1.0) if latest["quality"] is not None else 0.7

        pressure = clamp(hours_awake / 16.0 - 0.10 * quality, 0.0, 1.0)
        budget = clamp((1.0 - pressure) * (0.5 + 0.5 * quality), 0.0, 1.0)

    else:
        state_name = "asleep"
        sleep_dt = parse_ts(latest["ts"]) or now

        prev_wake = conn.execute(
            """
            SELECT ts, quality
            FROM events
            WHERE namespace = ? AND type = 'wake' AND ts <= ?
            ORDER BY ts DESC
            LIMIT 1
            """,
            (namespace, sleep_dt.isoformat()),
        ).fetchone()

        if prev_wake:
            wake_dt = parse_ts(prev_wake["ts"]) or sleep_dt
            pressure_at_sleep = clamp((sleep_dt - wake_dt).total_seconds() / 3600.0 / 16.0, 0.0, 1.0)
            quality = clamp(prev_wake["quality"], 0.0, 1.0) if prev_wake["quality"] is not None else 0.7
        else:
            pressure_at_sleep = 0.5
            quality = 0.7

        sleep_quality = clamp(latest["quality"], 0.0, 1.0) if latest["quality"] is not None else quality
        hours_asleep = max(0.0, (now - sleep_dt).total_seconds() / 3600.0)
        recovery_hours = 8.0 / max(0.25, sleep_quality)

        pressure = clamp(pressure_at_sleep - hours_asleep / recovery_hours, 0.0, 1.0)
        budget = clamp(1.0 - pressure, 0.0, 1.0)

    plasticity = clamp(
        (0.2 + 0.8 * circadian)
        * (1.0 - 0.65 * pressure)
        * (0.5 + 0.5 * budget),
        0.0,
        1.0,
    )

    return {
        "namespace": namespace,
        "state": state_name,
        "local_hour": hour,
        "circadian": circadian,
        "sleep_pressure": pressure,
        "budget": budget,
        "plasticity": plasticity,
        "ts": now_utc(),
    }


def feature_sum(conn: sqlite3.Connection, node_id: int) -> float:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(value), 0.0) AS total
        FROM features
        WHERE node_id = ? AND contributes = 1
        """,
        (node_id,),
    ).fetchone()

    return safe_float(row["total"], 0.0) if row else 0.0


def effective_strength(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    feature_total: Optional[float] = None,
) -> float:
    base = clamp(row["base_strength"], 0.0, 1.0)
    decay = math.exp(-DECAY_RATE * hours_since(row["last_accessed"] or row["created"]))
    if feature_total is None:
        contribution = feature_sum(conn, row["id"])
    else:
        contribution = safe_float(feature_total, 0.0)
    feature_factor = 1.0 + 0.5 * math.tanh(contribution)
    return clamp(base * decay * feature_factor, 0.0, 1.0)


def effective_salience(row: sqlite3.Row) -> float:
    raw = clamp(row["salience"], -1.0, 1.0)
    age = hours_since(row["salience_ts"] or row["last_accessed"])
    return clamp(raw * math.exp(-age / SALIENCE_HALF_LIFE_HOURS), -1.0, 1.0)


def effective_affect(row: sqlite3.Row) -> float:
    raw = clamp(row["affect"], -1.0, 1.0)
    age = hours_since(row["affect_ts"] or row["last_accessed"])
    return clamp(raw * math.exp(-age / AFFECT_HALF_LIFE_HOURS), -1.0, 1.0)


def node_view(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    feature_total: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "namespace": row["namespace"],
        "name": row["display_name"],
        "canonical": row["canonical"],
        "objective": row["objective"],
        "flow": row["flow"],
        "corpus": row["corpus"],
        "verified": bool(row["verified"]),
        "suppressed": bool(row["suppressed"]),
        "base_strength": clamp(row["base_strength"], 0.0, 1.0),
        "effective_strength": effective_strength(conn, row, feature_total),
        "salience": effective_salience(row),
        "affect": effective_affect(row),
        "last_accessed": row["last_accessed"],
        "created": row["created"],
    }


def write_journal(
    conn: sqlite3.Connection,
    namespace: str,
    tool: str,
    node_id: Optional[int],
    detail: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO journal(namespace, tool, node_id, detail, ts)
        VALUES(?, ?, ?, ?, ?)
        """,
        (
            namespace,
            tool,
            node_id,
            json.dumps(detail, allow_nan=False, default=str),
            now_utc(),
        ),
    )


# ---------------------------------------------------------------------
# Node / graph helpers
# ---------------------------------------------------------------------


def get_node_row(conn: sqlite3.Connection, namespace: str, name: str) -> Optional[sqlite3.Row]:
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    canonical = canonical_name(clean_text(name, MAX_NAME))
    if not canonical:
        return None

    return conn.execute(
        "SELECT * FROM nodes WHERE namespace = ? AND canonical = ?",
        (namespace, canonical),
    ).fetchone()


def get_node_row_by_id(conn: sqlite3.Connection, node_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()


def touch_node(conn: sqlite3.Connection, node_id: int) -> None:
    conn.execute(
        "UPDATE nodes SET last_accessed = ? WHERE id = ?",
        (now_utc(), node_id),
    )


def strengthen_node(conn: sqlite3.Connection, node_id: int, delta: float) -> None:
    row = conn.execute("SELECT base_strength FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if not row:
        return

    current = clamp(row["base_strength"], 0.0, 1.0)
    gain = clamp(delta, -1.0, 1.0) * (1.0 - current)
    new_base = clamp(current + gain, 0.0, 1.0)

    conn.execute(
        "UPDATE nodes SET base_strength = ?, last_accessed = ? WHERE id = ?",
        (new_base, now_utc(), node_id),
    )


def adjust_emotion(conn: sqlite3.Connection, node_id: int, d_salience: float, d_affect: float) -> None:
    row = conn.execute("SELECT salience, affect FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if not row:
        return

    salience = clamp(row["salience"] + d_salience, -1.0, 1.0)
    affect = clamp(row["affect"] + d_affect, -1.0, 1.0)
    now = now_utc()

    conn.execute(
        """
        UPDATE nodes
        SET salience = ?, affect = ?, salience_ts = ?, affect_ts = ?, last_accessed = ?
        WHERE id = ?
        """,
        (salience, affect, now, now, now, node_id),
    )


def ensure_node(
    conn: sqlite3.Connection,
    namespace: str,
    name: str,
    plasticity: float = 0.0,
    objective: str = "",
    flow: str = "",
    corpus: str = "",
) -> sqlite3.Row:
    row = get_node_row(conn, namespace, name)
    if row:
        if not row["suppressed"]:
            touch_node(conn, row["id"])
        return row

    display_name = require_text(name, "node name", MAX_NAME)
    canonical = canonical_name(display_name)
    now = now_utc()

    base_strength = clamp(0.10 + 0.10 * plasticity, 0.0, 1.0)

    conn.execute(
        """
        INSERT INTO nodes (
            namespace, canonical, display_name, objective, flow, corpus,
            verified, suppressed, base_strength, salience, affect,
            salience_ts, affect_ts, last_accessed, created
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE,
            canonical,
            display_name,
            clean_text(objective, MAX_TEXT),
            clean_text(flow, MAX_TEXT),
            clean_text(corpus, MAX_TEXT) or "ABSENT",
            0,
            0,
            base_strength,
            0.0,
            0.0,
            now,
            now,
            now,
            now,
        ),
    )

    row = get_node_row(conn, namespace, display_name)
    if not row:
        raise ValueError("node creation failed")
    return row


def would_create_cycle(
    conn: sqlite3.Connection,
    namespace: str,
    source_id: int,
    target_id: int,
    max_depth: int = 4,
) -> bool:
    if source_id == target_id:
        return True

    queue = [(target_id, 0)]
    seen = set()

    while queue:
        node_id, depth = queue.pop(0)

        if node_id == source_id:
            return True

        if depth >= max_depth or node_id in seen:
            continue

        seen.add(node_id)

        rows = conn.execute(
            "SELECT target_id FROM edges WHERE namespace = ? AND source_id = ?",
            (namespace, node_id),
        ).fetchall()

        for r in rows:
            queue.append((r["target_id"], depth + 1))

    return False


def upsert_edge(
    conn: sqlite3.Connection,
    namespace: str,
    source_id: int,
    target_id: int,
    relation: str,
    weight: float,
    confidence: float,
    evidence: str,
) -> None:
    now = now_utc()

    conn.execute(
        """
        INSERT INTO edges (
            namespace, source_id, target_id, relation,
            weight, confidence, evidence, valid_from, valid_to, ts
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(namespace, source_id, target_id, relation) DO UPDATE SET
            weight = excluded.weight,
            confidence = excluded.confidence,
            evidence = excluded.evidence,
            ts = excluded.ts
        """,
        (
            namespace,
            source_id,
            target_id,
            relation,
            clamp(weight, -1.0, 1.0),
            clamp(confidence, 0.0, 1.0),
            clean_text(evidence, MAX_TEXT),
            now,
            None,
            now,
        ),
    )


def local_graph(
    conn: sqlite3.Connection,
    namespace: str,
    around: str,
    depth: int,
) -> Optional[Dict[str, Any]]:
    depth = clamp_int(depth, 0, MAX_GRAPH_DEPTH)
    if depth <= 0 or not around:
        return None

    start = get_node_row(conn, namespace, around)
    if not start or start["suppressed"]:
        return None

    queue = [(start["id"], 0)]
    seen = set()
    node_ids = set()
    edges = []

    while queue and len(edges) < 200:
        node_id, d = queue.pop(0)

        if node_id in seen or d > depth:
            continue

        seen.add(node_id)
        node_ids.add(node_id)

        if d < depth:
            out_rows = conn.execute(
                "SELECT * FROM edges WHERE namespace = ? AND source_id = ?",
                (namespace, node_id),
            ).fetchall()

            for e in out_rows:
                edges.append(dict(e))
                node_ids.add(e["target_id"])
                queue.append((e["target_id"], d + 1))

            in_rows = conn.execute(
                "SELECT * FROM edges WHERE namespace = ? AND target_id = ?",
                (namespace, node_id),
            ).fetchall()

            for e in in_rows:
                edges.append(dict(e))
                node_ids.add(e["source_id"])
                queue.append((e["source_id"], d + 1))

    if not node_ids:
        return {"nodes": [], "edges": []}

    placeholders = ",".join("?" for _ in node_ids)
    node_rows = conn.execute(
        f"SELECT * FROM nodes WHERE id IN ({placeholders})",
        tuple(node_ids),
    ).fetchall()

    nodes = [node_view(conn, row) for row in node_rows]
    name_by_id = {n["id"]: n["name"] for n in nodes}

    public_edges = []
    for e in edges[:200]:
        public_edges.append(
            {
                "source": name_by_id.get(e["source_id"], str(e["source_id"])),
                "target": name_by_id.get(e["target_id"], str(e["target_id"])),
                "relation": e["relation"],
                "weight": clamp(e["weight"], -1.0, 1.0),
                "confidence": clamp(e["confidence"], 0.0, 1.0),
                "evidence": e["evidence"],
                "ts": e["ts"],
            }
        )

    return {
        "nodes": nodes[:100],
        "edges": public_edges,
    }


def top_nodes(conn: sqlite3.Connection, namespace: str, limit: int = 50) -> List[Dict[str, Any]]:
    # FIX #4: prefetch feature sums in one query instead of N+1
    rows = conn.execute(
        """
        SELECT n.*, COALESCE(fs.total, 0.0) AS feature_sum
        FROM nodes n
        LEFT JOIN (
            SELECT node_id, SUM(value) AS total
            FROM features
            WHERE contributes = 1
            GROUP BY node_id
        ) fs ON fs.node_id = n.id
        WHERE n.namespace = ? AND n.suppressed = 0
        LIMIT ?
        """,
        (namespace, MAX_TOP_NODES),
    ).fetchall()

    views = [node_view(conn, row, row["feature_sum"]) for row in rows]
    views.sort(
        key=lambda x: (x["effective_strength"], abs(x["salience"]), x["name"]),
        reverse=True,
    )

    return views[: clamp_int(limit, 1, MAX_RECALL_LIMIT)]


# ---------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------

mcp = FastMCP(
    "CorrectedMemory",
    instructions=(
        "Persistent memory server with bounded retrieval, time-based decay, "
        "bounded reinforcement, forgetting, provenance, and wake/sleep state. "
        "Use 'sleep_start' to enter sleep mode, then call 'dream' for "
        "maintenance and novel synthesis, followed by 'wake'."
    ),
)


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------


@mcp.tool()
def cite(
    name: str,
    # FIX #9: None = keep, "" = clear, otherwise set
    objective: Optional[str] = None,
    flow: Optional[str] = None,
    corpus: Optional[str] = None,
    salience: Optional[float] = None,
    affect: Optional[float] = None,
    verify: Optional[bool] = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Create or reinforce a memory node.

    For existing nodes, pass `None` (omit) to keep a field, `""` to clear it,
    or a non-empty string to set it. `corpus=""` resets to the "ABSENT" sentinel.
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    name = require_text(name, "name", MAX_NAME)

    sal = as_finite_float(salience, "salience")
    aff = as_finite_float(affect, "affect")

    if sal is not None:
        sal = clamp(sal, -1.0, 1.0)
    if aff is not None:
        aff = clamp(aff, -1.0, 1.0)

    with db() as conn:
        state = brain_state(conn, namespace)

        with conn:
            row = get_node_row(conn, namespace, name)

            if row:
                # FIX #9: distinguish "keep" (None) from "clear" ("")
                if objective is None:
                    final_objective = row["objective"]
                else:
                    final_objective = clean_text(objective, MAX_TEXT)

                if flow is None:
                    final_flow = row["flow"]
                else:
                    final_flow = clean_text(flow, MAX_TEXT)

                if corpus is None:
                    final_corpus = row["corpus"]
                else:
                    final_corpus = clean_text(corpus, MAX_TEXT) or "ABSENT"

                final_salience = row["salience"] if sal is None else sal
                final_affect = row["affect"] if aff is None else aff

                raw_gain = (0.05 + 0.10 * abs(final_salience)) * state["plasticity"]

                if final_affect > 0:
                    raw_gain += 0.04 * final_affect
                elif final_affect < 0:
                    raw_gain -= 0.02 * abs(final_affect)

                current_base = clamp(row["base_strength"], 0.0, 1.0)
                gain = raw_gain * (1.0 - current_base)
                new_base = clamp(current_base + gain, 0.0, 1.0)

                verified = row["verified"]
                if verify is not None:
                    verified = 1 if verify else 0

                sal_ts = now_utc() if sal is not None else row["salience_ts"]
                aff_ts = now_utc() if aff is not None else row["affect_ts"]

                conn.execute(
                    """
                    UPDATE nodes
                    SET objective = ?, flow = ?, corpus = ?, verified = ?, suppressed = 0,
                        base_strength = ?, salience = ?, affect = ?,
                        salience_ts = ?, affect_ts = ?, last_accessed = ?
                    WHERE id = ?
                    """,
                    (
                        final_objective,
                        final_flow,
                        final_corpus,
                        verified,
                        new_base,
                        final_salience,
                        final_affect,
                        sal_ts,
                        aff_ts,
                        now_utc(),
                        row["id"],
                    ),
                )

                node_id = row["id"]

                write_journal(
                    conn,
                    namespace,
                    "cite",
                    node_id,
                    {
                        "name": name,
                        "objective": final_objective,
                        "flow": final_flow,
                        "corpus": final_corpus,
                        "salience": final_salience,
                        "affect": final_affect,
                        "verify": verify,
                        "plasticity": state["plasticity"],
                    },
                )

            else:
                final_objective = clean_text(objective, MAX_TEXT)
                final_flow = clean_text(flow, MAX_TEXT)
                final_corpus = clean_text(corpus, MAX_TEXT) or "ABSENT"

                final_salience = 0.0 if sal is None else sal
                final_affect = 0.0 if aff is None else aff

                base_strength = clamp(0.30 + 0.15 * state["plasticity"], 0.0, 1.0)
                verified = 1 if verify else 0
                now = now_utc()

                cur = conn.execute(
                    """
                    INSERT INTO nodes (
                        namespace, canonical, display_name, objective, flow, corpus,
                        verified, suppressed, base_strength, salience, affect,
                        salience_ts, affect_ts, last_accessed, created
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        namespace,
                        canonical_name(name),
                        name,
                        final_objective,
                        final_flow,
                        final_corpus,
                        verified,
                        0,
                        base_strength,
                        final_salience,
                        final_affect,
                        now,
                        now,
                        now,
                        now,
                    ),
                )

                node_id = cur.lastrowid

                write_journal(
                    conn,
                    namespace,
                    "cite",
                    node_id,
                    {
                        "name": name,
                        "objective": final_objective,
                        "flow": final_flow,
                        "corpus": final_corpus,
                        "salience": final_salience,
                        "affect": final_affect,
                        "verify": verify,
                        "plasticity": state["plasticity"],
                    },
                )

        row = get_node_row_by_id(conn, node_id)
        if not row:
            raise ValueError("node not found after cite")

        result = node_view(conn, row)
        result["brain"] = state
        return result


@mcp.tool()
def flat(
    node: str,
    key: str,
    value: float,
    contributes: bool = False,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Store a bounded feature/weight on a node.

    Set contributes=true only for memory-strength features.
    Metadata should use contributes=false.
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    node = require_text(node, "node", MAX_NAME)
    key = require_text(key, "key", 200)

    val = as_finite_float(value, "value", required=True)
    if contributes:
        val = clamp(val, -10.0, 10.0)
    else:
        val = clamp(val, -1_000_000.0, 1_000_000.0)

    with db() as conn:
        state = brain_state(conn, namespace)

        with conn:
            row = ensure_node(
                conn,
                namespace,
                node,
                plasticity=state["plasticity"],
                corpus="FLAT",
            )

            conn.execute(
                """
                INSERT INTO features(node_id, key, value, contributes, ts)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(node_id, key) DO UPDATE SET
                    value = excluded.value,
                    contributes = excluded.contributes,
                    ts = excluded.ts
                """,
                (row["id"], key, val, 1 if contributes else 0, now_utc()),
            )

            touch_node(conn, row["id"])

            write_journal(
                conn,
                namespace,
                "flat",
                row["id"],
                {
                    "key": key,
                    "value": val,
                    "contributes": bool(contributes),
                    "plasticity": state["plasticity"],
                },
            )

            refreshed = get_node_row_by_id(conn, row["id"])
            if not refreshed:
                raise ValueError("node not found after flat")

            result = node_view(conn, refreshed)
            result["brain"] = state
            return result


@mcp.tool()
def admit(
    source: str,
    target: str,
    relation: str,
    weight: float,
    confidence: float = 1.0,
    evidence: str = "",
    objective: str = "",
    allow_cycle: bool = False,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Add or update a causal/labeled edge.

    Special relations:
    - replay: strengthens source/target and edge weight
    - social: raises target salience/affect
    - threat: raises target salience, lowers affect
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    source = require_text(source, "source", MAX_NAME)
    target = require_text(target, "target", MAX_NAME)
    relation = clean_text(relation, 100).lower() or "related_to"

    w = as_finite_float(weight, "weight", required=True)
    c = as_finite_float(confidence, "confidence")
    if c is None:
        c = 1.0

    w = clamp(w, -1.0, 1.0)
    c = clamp(c, 0.0, 1.0)

    with db() as conn:
        state = brain_state(conn, namespace)

        with conn:
            source_row = ensure_node(
                conn,
                namespace,
                source,
                plasticity=state["plasticity"],
                objective=objective,
                corpus="ADMIT",
            )

            target_row = ensure_node(
                conn,
                namespace,
                target,
                plasticity=state["plasticity"],
                corpus="ADMIT",
            )

            if source_row["id"] == target_row["id"]:
                raise ValueError("source and target must be different")

            # FIX #5 (cont'd): use MAX_GRAPH_DEPTH consistently
            if not allow_cycle and would_create_cycle(
                conn,
                namespace,
                source_row["id"],
                target_row["id"],
                max_depth=MAX_GRAPH_DEPTH,
            ):
                raise ValueError("edge would create a cycle")

            if relation == "replay":
                strengthen_node(conn, source_row["id"], 0.015 * state["plasticity"])
                strengthen_node(conn, target_row["id"], 0.030 * state["plasticity"])

                old = conn.execute(
                    """
                    SELECT weight, confidence
                    FROM edges
                    WHERE namespace = ? AND source_id = ? AND target_id = ? AND relation = ?
                    """,
                    (namespace, source_row["id"], target_row["id"], relation),
                ).fetchone()

                old_weight = clamp(old["weight"], 0.0, 1.0) if old else clamp(w, 0.0, 1.0)
                old_conf = clamp(old["confidence"], 0.0, 1.0) if old else 0.0

                new_weight = clamp(old_weight + 0.05 * state["plasticity"], 0.0, 1.0)
                new_conf = clamp(max(old_conf, c), 0.0, 1.0)

                upsert_edge(
                    conn,
                    namespace,
                    source_row["id"],
                    target_row["id"],
                    relation,
                    new_weight,
                    new_conf,
                    evidence,
                )

                final_weight = new_weight
                final_conf = new_conf

            elif relation == "social":
                positive_weight = clamp(w, 0.0, 1.0)
                adjust_emotion(
                    conn,
                    target_row["id"],
                    0.05 * positive_weight * state["plasticity"],
                    0.02 * positive_weight,
                )

                upsert_edge(
                    conn,
                    namespace,
                    source_row["id"],
                    target_row["id"],
                    relation,
                    positive_weight,
                    c,
                    evidence,
                )

                final_weight = positive_weight
                final_conf = c

            elif relation == "threat":
                positive_weight = clamp(w, 0.0, 1.0)
                adjust_emotion(
                    conn,
                    target_row["id"],
                    0.04 * positive_weight * state["plasticity"],
                    -0.03 * positive_weight,
                )

                upsert_edge(
                    conn,
                    namespace,
                    source_row["id"],
                    target_row["id"],
                    relation,
                    positive_weight,
                    c,
                    evidence,
                )

                final_weight = positive_weight
                final_conf = c

            else:
                upsert_edge(
                    conn,
                    namespace,
                    source_row["id"],
                    target_row["id"],
                    relation,
                    w,
                    c,
                    evidence,
                )

                final_weight = w
                final_conf = c

            touch_node(conn, source_row["id"])
            touch_node(conn, target_row["id"])

            write_journal(
                conn,
                namespace,
                "admit",
                source_row["id"],
                {
                    "source": source_row["display_name"],
                    "target": target_row["display_name"],
                    "relation": relation,
                    "weight": final_weight,
                    "confidence": final_conf,
                    "evidence": evidence,
                    "plasticity": state["plasticity"],
                },
            )

        return {
            "source": source_row["display_name"],
            "target": target_row["display_name"],
            "relation": relation,
            "weight": final_weight,
            "confidence": final_conf,
            "brain": state,
        }


@mcp.tool()
def recall(
    query: str = "",
    namespace: str = DEFAULT_NAMESPACE,
    limit: int = 10,
    reinforce: bool = False,
    objective: str = "",
    verify_required: bool = False,
    around: str = "",
    depth: int = 0,
) -> Dict[str, Any]:
    """Retrieve memories.

    Uses FTS when available, otherwise LIKE fallback.
    Ranking combines lexical score, effective strength, and salience.
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    query = clean_text(query, MAX_QUERY)
    objective = clean_text(objective, MAX_TEXT)

    limit = clamp_int(limit, 1, MAX_RECALL_LIMIT)
    depth = clamp_int(depth, 0, MAX_GRAPH_DEPTH)

    query_tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not query_tokens:
        query = ""

    with db() as conn:
        state = brain_state(conn, namespace)

        rows = []
        used_fts = False

        if query and FTS_AVAILABLE:
            try:
                safe_query = fts_escape(query)

                sql = """
                    SELECT n.*, bm.rank AS fts_rank
                    FROM nodes_fts AS bm
                    JOIN nodes AS n ON n.id = bm.rowid
                    WHERE bm MATCH ?
                      AND n.namespace = ?
                      AND n.suppressed = 0
                """
                params: List[Any] = [safe_query, namespace]

                if verify_required:
                    sql += " AND n.verified = 1"

                sql += " ORDER BY bm.rank LIMIT ?"
                params.append(limit * 2)

                rows = conn.execute(sql, params).fetchall()
                used_fts = True

            except sqlite3.OperationalError:
                rows = []
                used_fts = False

        if query and not used_fts:
            tokens = query_tokens[:10]
            if tokens:
                clauses = []
                params = [namespace]

                verify_sql = " AND verified = 1" if verify_required else ""

                for token in tokens:
                    like = f"%{token}%"
                    clauses.append(
                        "(display_name LIKE ? OR objective LIKE ? OR corpus LIKE ?)"
                    )
                    params.extend([like, like, like])

                sql = f"""
                    SELECT *, NULL AS fts_rank
                    FROM nodes
                    WHERE namespace = ?
                      AND suppressed = 0
                      {verify_sql}
                      AND ({' OR '.join(clauses)})
                    LIMIT ?
                """
                params.append(limit * 2)

                rows = conn.execute(sql, params).fetchall()

        if not query:
            sql = """
                SELECT *, NULL AS fts_rank
                FROM nodes
                WHERE namespace = ? AND suppressed = 0
            """
            params = [namespace]

            if verify_required:
                sql += " AND verified = 1"

            sql += " ORDER BY base_strength DESC LIMIT ?"
            params.append(limit * 2)

            rows = conn.execute(sql, params).fetchall()

        # FIX #4: prefetch feature sums for the candidate set
        candidate_ids = [row["id"] for row in rows]
        feature_totals: Dict[int, float] = {}
        if candidate_ids:
            placeholders = ",".join("?" for _ in candidate_ids)
            for r in conn.execute(
                f"""
                SELECT node_id, SUM(value) AS total
                FROM features
                WHERE contributes = 1 AND node_id IN ({placeholders})
                GROUP BY node_id
                """,
                tuple(candidate_ids),
            ):
                feature_totals[r["node_id"]] = safe_float(r["total"], 0.0)

        candidates = []

        objective_cf = objective.casefold()

        for row in rows:
            view = node_view(conn, row, feature_totals.get(row["id"], 0.0))

            if view["suppressed"]:
                continue

            if verify_required and not view["verified"]:
                continue

            rank = row["fts_rank"] if "fts_rank" in row.keys() else None
            if rank is not None:
                # FIX #1: bm25 is negative, more negative = better match.
                # Map to (0, 1) monotonically so better matches get higher scores.
                r = clamp(rank, -1000.0, 1000.0)
                if r < 0:
                    magnitude = -r
                    lexical = magnitude / (1.0 + magnitude)
                else:
                    # Positive bm25 is unusual; treat as weak.
                    lexical = 1.0 / (1.0 + r)
            else:
                lexical = 0.6

            objective_boost = 1.0
            if objective_cf and clean_text(row["objective"], MAX_TEXT).casefold() == objective_cf:
                objective_boost = 1.2

            score = (
                lexical
                * (0.2 + 0.8 * view["effective_strength"])
                * (1.0 + 0.5 * abs(view["salience"]))
                * objective_boost
            )

            view["score"] = clamp(score, 0.0, 1000.0)
            candidates.append(view)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:limit]

        if reinforce and candidates:
            with conn:
                for item in candidates:
                    touch_node(conn, item["id"])

        context = None
        if around:
            context = local_graph(conn, namespace, around, depth)

        return {
            "count": len(candidates),
            "results": candidates,
            "context": context,
            "brain": state,
        }


@mcp.tool()
def forget(
    name: str,
    mode: str = "soft",
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Forget a node.

    mode=soft suppresses it.
    mode=hard deletes it and cascades edges/features.
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    name = require_text(name, "name", MAX_NAME)
    mode = clean_text(mode, 20).lower()

    if mode not in ("soft", "hard"):
        raise ValueError("mode must be 'soft' or 'hard'")

    with db() as conn:
        row = get_node_row(conn, namespace, name)
        if not row:
            raise ValueError("node not found")

        with conn:
            if mode == "soft":
                conn.execute(
                    "UPDATE nodes SET suppressed = 1, last_accessed = ? WHERE id = ?",
                    (now_utc(), row["id"]),
                )

                write_journal(
                    conn,
                    namespace,
                    "forget_soft",
                    row["id"],
                    {"name": row["display_name"]},
                )

            else:
                write_journal(
                    conn,
                    namespace,
                    "forget_hard",
                    row["id"],
                    {"name": row["display_name"]},
                )

                conn.execute("DELETE FROM nodes WHERE id = ?", (row["id"],))

        return {
            "forgotten": name,
            "mode": mode,
            "namespace": namespace,
        }


@mcp.tool()
def verify(
    name: str,
    corpus: str = "",
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Mark a node as verified and optionally set its corpus/source."""
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    name = require_text(name, "name", MAX_NAME)
    corpus = clean_text(corpus, MAX_TEXT)

    with db() as conn:
        state = brain_state(conn, namespace)
        row = get_node_row(conn, namespace, name)

        if not row:
            raise ValueError("node not found")

        with conn:
            new_corpus = corpus or row["corpus"]

            conn.execute(
                """
                UPDATE nodes
                SET verified = 1, corpus = ?, suppressed = 0, last_accessed = ?
                WHERE id = ?
                """,
                (new_corpus, now_utc(), row["id"]),
            )

            write_journal(
                conn,
                namespace,
                "verify",
                row["id"],
                {"name": row["display_name"], "corpus": new_corpus},
            )

        refreshed = get_node_row_by_id(conn, row["id"])
        if not refreshed:
            raise ValueError("node not found after verify")

        result = node_view(conn, refreshed)
        result["brain"] = state
        return result


# FIX #5: namespace config tools
@mcp.tool()
def get_config(
    key: str = "",
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Read namespace config values.

    If `key` is empty, returns all config for the namespace.
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    key = clean_text(key, 200)

    with db() as conn:
        if key:
            value = read_config(conn, namespace, key, "")
            return {"namespace": namespace, "key": key, "value": value}

        rows = conn.execute(
            "SELECT key, value, ts FROM config WHERE namespace = ? ORDER BY key",
            (namespace,),
        ).fetchall()
        return {
            "namespace": namespace,
            "config": [dict(r) for r in rows],
        }


@mcp.tool()
def set_config(
    key: str,
    value: str,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Set a namespace config value (e.g. 'circadian_peak_hour' = '14.5')."""
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    key = require_text(key, "key", 200)
    value = clean_text(value, 500)

    with db() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO config(namespace, key, value, ts)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    ts = excluded.ts
                """,
                (namespace, key, value, now_utc()),
            )
            write_journal(
                conn,
                namespace,
                "set_config",
                None,
                {"key": key, "value": value},
            )

        return {"namespace": namespace, "key": key, "value": value}


# FIX #6: namespace enumeration
@mcp.tool()
def namespaces() -> Dict[str, Any]:
    """List namespaces with node and edge counts."""
    with db() as conn:
        rows = conn.execute(
            """
            SELECT ns.namespace, ns.node_count, COALESCE(e.edge_count, 0) AS edge_count
            FROM (
                SELECT namespace, COUNT(*) AS node_count
                FROM nodes
                GROUP BY namespace
            ) ns
            LEFT JOIN (
                SELECT namespace, COUNT(*) AS edge_count
                FROM edges
                GROUP BY namespace
            ) e ON e.namespace = ns.namespace
            ORDER BY ns.namespace
            LIMIT ?
            """,
            (MAX_NAMESPACES,),
        ).fetchall()

        return {"namespaces": [dict(r) for r in rows]}


@mcp.tool()
def wake(
    quality: float = 0.7,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Record a wake event. Quality describes the preceding sleep.

    If the most recent event is already a `wake`, no new event is recorded.
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    q = as_finite_float(quality, "quality")
    q = clamp(q, 0.0, 1.0) if q is not None else 0.7

    with db() as conn:
        # FIX #7: refuse redundant wake event (only when latest event is wake;
        # a fresh DB with no events is allowed).
        if latest_event_type(conn, namespace) == "wake":
            state = brain_state(conn, namespace)
            state["refused"] = True
            state["message"] = "Already awake. No new wake event recorded."
            return state

        with conn:
            conn.execute(
                "INSERT INTO events(namespace, type, quality, ts) VALUES(?, 'wake', ?, ?)",
                (namespace, q, now_utc()),
            )

            write_journal(
                conn,
                namespace,
                "wake",
                None,
                {"quality": q},
            )

        return brain_state(conn, namespace)


@mcp.tool()
def sleep_start(
    quality: float = 0.7,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Record sleep onset. Quality is expected sleep quality.

    If already asleep (latest event is `sleep_start`), no new event is recorded.
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    q = as_finite_float(quality, "quality")
    q = clamp(q, 0.0, 1.0) if q is not None else 0.7

    with db() as conn:
        # FIX #7: refuse duplicate sleep_start
        if latest_event_type(conn, namespace) == "sleep_start":
            state = brain_state(conn, namespace)
            state["refused"] = True
            state["message"] = "Already asleep. Call dream then wake to continue."
            return state

        with conn:
            conn.execute(
                "INSERT INTO events(namespace, type, quality, ts) VALUES(?, 'sleep_start', ?, ?)",
                (namespace, q, now_utc()),
            )

            write_journal(
                conn,
                namespace,
                "sleep_start",
                None,
                {"quality": q},
            )

        return brain_state(conn, namespace)


# ---------------------------------------------------------------------
# Synthesis mechanisms (dream)
# ---------------------------------------------------------------------


RELATION_COMPOSITIONS: Dict[Tuple[str, str], str] = {
    # Pure transitive relations
    ("is_a", "is_a"): "is_a",
    ("part_of", "part_of"): "part_of",
    ("contains", "contains"): "contains",
    ("causes", "causes"): "causes",
    ("leads_to", "leads_to"): "leads_to",
    ("implies", "implies"): "implies",
    ("before", "before"): "before",
    ("after", "after"): "after",
    ("associated", "associated"): "associated",
    ("related_to", "related_to"): "related_to",
    ("similar_to", "similar_to"): "similar_to",
    # is_a on the left promotes the right-hand relation
    ("is_a", "part_of"): "part_of",
    ("is_a", "causes"): "causes",
    ("is_a", "leads_to"): "leads_to",
    ("is_a", "associated"): "associated",
    ("is_a", "related_to"): "related_to",
    ("causes", "leads_to"): "leads_to",
    # Deliberately NOT composed (ambiguous):
    #   ("part_of", "is_a")      A part_of B, B is_a C
    #   ("contains", "part_of")  inverse relations
    #   ("before", "after")      already ordered
}


def _compose_relations(r1: str, r2: str) -> Optional[str]:
    return RELATION_COMPOSITIONS.get((r1, r2))


def _empty_synth_result() -> Dict[str, Any]:
    return {
        "proposed": 0,
        "created": 0,
        "skipped_exists": 0,
        "skipped_cycle": 0,
        "skipped_invalid": 0,
        "examples": [],
    }


# Relations whose semantics are symmetric: an edge A->B implies B~A.
# These are excluded from cycle checking (they assert relatedness, not
# ordering, so traversing them cannot produce a contradiction) and are
# deduplicated in both directions. Any relation not listed here (including
# user-defined ones) is treated as directed/conservative.
SYMMETRIC_RELATIONS = frozenset({
    "associated",
    "similar_to",
    "related_to",
    "co_occurred",
    "affective",
})


class _GraphSnapshot:
    """In-memory directed-adjacency snapshot of a namespace's edges.

    Replaces per-candidate SQL cycle checks with dict lookups, and serves
    three purposes during synthesis:

    1. Exact cycle checks without per-node SQL: cycle checks traverse only
       directed relations (symmetric relations cannot form contradictions).
    2. Pair+relation dedupe: an edge is a duplicate only if the exact
       (source, target, relation) triple exists, so a weak `associated`
       edge no longer blocks a typed `is_a`/`causes` edge between the same
       nodes — they coexist as separate, equally traversable edges.
    3. Exact dry_run preview: synthesized edges are recorded in the
       snapshot even when the DB is untouched, so later candidates within
       the same dream see them and the reported skip counts are exact.
    """

    def __init__(self, conn: sqlite3.Connection, namespace: str) -> None:
        rows = conn.execute(
            "SELECT source_id, target_id, relation FROM edges WHERE namespace = ?",
            (namespace,),
        ).fetchall()
        self.edges: set = {
            (r["source_id"], r["target_id"], r["relation"]) for r in rows
        }
        # Directed adjacency only: symmetric relations are excluded because
        # following them cannot return to the start as a contradiction.
        self.adj: Dict[int, set] = {}
        for s, t, rel in self.edges:
            if rel not in SYMMETRIC_RELATIONS:
                self.adj.setdefault(s, set()).add(t)
        # Edge writes are deferred until _run_synthesis finishes so every
        # mechanism's SQL candidate query sees the same dream-start baseline
        # in both dry_run and real mode (see _try_upsert).
        self.pending: List[tuple] = []

    def has_edge(self, source_id: int, target_id: int, relation: str) -> bool:
        return (source_id, target_id, relation) in self.edges

    def would_cycle(self, source_id: int, target_id: int, max_depth: int) -> bool:
        """True if adding a directed edge source_id -> target_id closes a loop.

        DFS from target_id over directed adjacency, bounded by max_depth —
        same reachability semantics as the SQL-based would_create_cycle,
        minus one query per visited node.
        """
        if source_id == target_id:
            return True
        seen: set = set()
        stack = [(target_id, 0)]
        while stack:
            node, depth = stack.pop()
            if node == source_id:
                return True
            if depth >= max_depth or node in seen:
                continue
            seen.add(node)
            for nxt in self.adj.get(node, ()):
                stack.append((nxt, depth + 1))
        return False

    def add(self, source_id: int, target_id: int, relation: str) -> None:
        """Record a (to-be-)created edge so subsequent candidates see it."""
        self.edges.add((source_id, target_id, relation))
        if relation not in SYMMETRIC_RELATIONS:
            self.adj.setdefault(source_id, set()).add(target_id)


def _try_upsert(
    conn: sqlite3.Connection,
    namespace: str,
    source_id: int,
    target_id: int,
    relation: str,
    weight: float,
    confidence: float,
    evidence: str,
    snapshot: "_GraphSnapshot",
    dry_run: bool,
) -> str:
    """Return one of: created, skipped_exists, skipped_cycle, skipped_invalid.

    Dedupe is per (source, target, relation): a weak `associated` edge
    between two nodes does not block a typed `is_a` edge — both can exist.
    Symmetric relations additionally treat the reversed pair as the same
    edge. Cycle checks use the in-memory snapshot: symmetric relations are
    excluded from the traversal (they cannot form contradictions), and the
    snapshot is updated even in dry_run so that within one dream, later
    candidates see earlier synthesized edges — skip counts are exact in
    both modes.
    """
    if source_id == target_id:
        return "skipped_invalid"
    if snapshot.has_edge(source_id, target_id, relation):
        return "skipped_exists"
    if relation in SYMMETRIC_RELATIONS and snapshot.has_edge(target_id, source_id, relation):
        return "skipped_exists"
    if relation not in SYMMETRIC_RELATIONS and snapshot.would_cycle(
        source_id, target_id, MAX_GRAPH_DEPTH
    ):
        return "skipped_cycle"
    snapshot.add(source_id, target_id, relation)
    if dry_run:
        return "created"
    # Real run: record for deferred write. Mechanisms after this one still
    # read the dream-start DB for candidate discovery, so candidate pools
    # are identical across dry_run/real; the edge becomes visible to SQL at
    # flush time (end of synthesis) and cascades across future dreams.
    snapshot.pending.append((source_id, target_id, relation, weight, confidence, evidence))
    return "created"


def _feature_vectors(
    conn: sqlite3.Connection,
    namespace: str,
    node_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    if not node_ids:
        return {}
    placeholders = ",".join("?" for _ in node_ids)
    rows = conn.execute(
        f"""
        SELECT f.node_id, f.key, f.value
        FROM features f
        JOIN nodes n ON n.id = f.node_id
        WHERE n.namespace = ? AND f.contributes = 1
          AND f.node_id IN ({placeholders})
        ORDER BY f.node_id, f.key
        """,
        (namespace, *node_ids),
    ).fetchall()
    vectors: Dict[int, Dict[str, float]] = {}
    for r in rows:
        vectors.setdefault(r["node_id"], {})[r["key"]] = safe_float(r["value"], 0.0)
    return vectors


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    common = v1.keys() & v2.keys()
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    n1 = math.sqrt(sum(v * v for v in v1.values()))
    n2 = math.sqrt(sum(v * v for v in v2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (n1 * n2)))


def _sample_by_weight(items, weights, k, rng, temperature):
    """Sample up to k items without replacement, weights = softmax(|w|/T).

    temperature > 1 → flatter (more exploration)
    temperature < 1 → greedier (favours high-weight candidates)
    """
    if not items:
        return []
    if len(items) <= k:
        return list(items)
    t = max(0.01, temperature)
    abs_w = [abs(w) for w in weights]
    max_w = max(abs_w) if abs_w else 0.0
    exp_w = [math.exp((w - max_w) / t) for w in abs_w]
    total = sum(exp_w)
    if total <= 0:
        return list(items[:k])
    probs = [w / total for w in exp_w]
    indices = list(range(len(items)))
    result = []
    for _ in range(min(k, len(items))):
        r = rng.random()
        cum = 0.0
        chosen = indices[-1]
        for idx, p in zip(indices, probs):
            cum += p
            if r <= cum:
                chosen = idx
                break
        result.append(items[chosen])
        pos = indices.index(chosen)
        indices.pop(pos)
        probs.pop(pos)
        tot = sum(probs)
        if tot <= 0:
            break
        probs = [p / tot for p in probs]
    return result


def _synth_transitive(conn, namespace, budget, rng, temperature, snapshot, dry_run):
    result = _empty_synth_result()
    if budget <= 0:
        return result
    rows = conn.execute(
        """
        SELECT id, source_id, target_id, relation, weight, confidence
        FROM edges
        WHERE namespace = ? AND ABS(weight) >= 0.1
        ORDER BY ABS(weight) DESC, id
        LIMIT ?
        """,
        (namespace, MAX_SYNTHESIS_CANDIDATES),
    ).fetchall()
    by_source: Dict[int, List[sqlite3.Row]] = {}
    for r in rows:
        by_source.setdefault(r["source_id"], []).append(r)

    seen: set = set()
    candidates: List[Dict[str, Any]] = []
    for e1 in rows:
        for e2 in by_source.get(e1["target_id"], []):
            if e1["source_id"] == e2["target_id"]:
                continue
            composed = _compose_relations(e1["relation"], e2["relation"])
            if not composed:
                continue
            key = (e1["source_id"], e2["target_id"], composed)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "source_id": e1["source_id"],
                "target_id": e2["target_id"],
                "relation": composed,
                "weight": clamp(e1["weight"] * e2["weight"] * RELATION_ATTENUATION, -1.0, 1.0),
                "confidence": clamp(e1["confidence"] * e2["confidence"] * RELATION_ATTENUATION, 0.0, 1.0),
                "evidence": f"transitive:{e1['relation']}+{e2['relation']}",
            })
    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], snapshot, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "relation": c["relation"], "weight": round(c["weight"], 3),
            })
    return result


def _synth_feature_recombination(conn, namespace, budget, rng, temperature, snapshot, dry_run):
    result = _empty_synth_result()
    if budget <= 0:
        return result
    rows = conn.execute(
        """
        SELECT n.id
        FROM nodes n
        WHERE n.namespace = ? AND n.suppressed = 0
          AND EXISTS (
              SELECT 1 FROM features f
              WHERE f.node_id = n.id AND f.contributes = 1
          )
        ORDER BY n.base_strength DESC, n.id
        LIMIT ?
        """,
        (namespace, MAX_SYNTHESIS_CANDIDATES),
    ).fetchall()
    node_ids = [r["id"] for r in rows]
    vecs = _feature_vectors(conn, namespace, node_ids)

    key_to_nodes: Dict[str, List[int]] = {}
    for nid, vec in vecs.items():
        for k in vec:
            key_to_nodes.setdefault(k, []).append(nid)

    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    for key, members in key_to_nodes.items():
        if len(members) < 2:
            continue
        for i, n1 in enumerate(members):
            for n2 in members[i + 1:]:
                pair = (min(n1, n2), max(n1, n2))
                if pair in seen:
                    continue
                seen.add(pair)
                sim = _cosine(vecs[n1], vecs[n2])
                if sim < 0.5:
                    continue
                candidates.append({
                    "source_id": n1, "target_id": n2,
                    "relation": "associated",
                    "weight": clamp(sim, 0.0, 1.0),
                    "confidence": clamp(sim, 0.0, 1.0),
                    "evidence": f"feature_cosine:{sim:.3f}",
                })
    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], snapshot, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "weight": round(c["weight"], 3),
            })
    return result


def _synth_analogy(conn, namespace, budget, rng, temperature, snapshot, dry_run):
    """Complete the parallelogram: A→B and C→D with similar A~C and B~D
    implies A→D and C→B."""
    result = _empty_synth_result()
    if budget <= 0:
        return result
    rows = conn.execute(
        """
        SELECT id, source_id, target_id, relation, weight, confidence
        FROM edges
        WHERE namespace = ? AND ABS(weight) >= 0.2
        ORDER BY ABS(weight) DESC, id
        LIMIT ?
        """,
        (namespace, MAX_SYNTHESIS_CANDIDATES),
    ).fetchall()

    by_relation: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        by_relation.setdefault(r["relation"], []).append(r)

    eligible_rels = [(rel, es) for rel, es in by_relation.items() if len(es) >= 2]
    if not eligible_rels:
        return result
    if len(eligible_rels) > 5:
        eligible_rels = _sample_by_weight(
            eligible_rels, [len(es) for _, es in eligible_rels], 5, rng, temperature
        )

    all_nids = set()
    for _, es in eligible_rels:
        for e in es:
            all_nids.add(e["source_id"])
            all_nids.add(e["target_id"])
    vecs = _feature_vectors(conn, namespace, list(all_nids))

    def node_sim(a: int, b: int) -> float:
        return _cosine(vecs.get(a, {}), vecs.get(b, {}))

    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    max_gen = budget * 3
    for rel, edges in eligible_rels:
        for i, e1 in enumerate(edges):
            for e2 in edges[i + 1:]:
                if e1["source_id"] == e2["source_id"] or e1["target_id"] == e2["target_id"]:
                    continue
                sim_src = node_sim(e1["source_id"], e2["source_id"])
                sim_tgt = node_sim(e1["target_id"], e2["target_id"])
                if sim_src < 0.4 or sim_tgt < 0.4:
                    continue
                strength = min(sim_src, sim_tgt)
                for s, t in (
                    (e1["source_id"], e2["target_id"]),
                    (e2["source_id"], e1["target_id"]),
                ):
                    key = (s, t, rel)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "source_id": s, "target_id": t,
                        "relation": rel,
                        "weight": clamp(e1["weight"] * e2["weight"] * strength, -1.0, 1.0),
                        "confidence": clamp(0.5 * strength, 0.0, 1.0),
                        "evidence": f"analogy:{rel}:{sim_src:.2f},{sim_tgt:.2f}",
                    })
                    if len(candidates) >= max_gen:
                        break
                if len(candidates) >= max_gen:
                    break
            if len(candidates) >= max_gen:
                break
        if len(candidates) >= max_gen:
            break

    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], snapshot, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "relation": c["relation"],
            })
    return result


def _synth_schema(conn, namespace, budget, rng, temperature, snapshot, dry_run):
    result = _empty_synth_result()
    result["created_nodes"] = 0
    if budget <= 0:
        return result

    rows = conn.execute(
        """
        SELECT f.node_id, f.key, f.value
        FROM features f
        JOIN nodes n ON n.id = f.node_id
        WHERE n.namespace = ? AND f.contributes = 1 AND n.suppressed = 0
        ORDER BY f.node_id, f.key
        """,
        (namespace,),
    ).fetchall()
    key_nodes: Dict[str, List[Tuple[int, float]]] = {}
    for r in rows:
        key_nodes.setdefault(r["key"], []).append(
            (r["node_id"], safe_float(r["value"], 0.0))
        )

    eligible = [(k, m) for k, m in key_nodes.items() if len(m) >= 3]
    if not eligible:
        return result
    eligible.sort(key=lambda x: len(x[1]), reverse=True)
    if len(eligible) > MAX_SCHEMAS_PER_DREAM:
        eligible = _sample_by_weight(
            eligible, [len(m) for _, m in eligible], MAX_SCHEMAS_PER_DREAM, rng, temperature
        )
    result["proposed"] = len(eligible)

    created_edges = 0
    for key, members in eligible:
        if created_edges >= budget:
            break
        schema_name = f"{SCHEMA_NODE_PREFIX}{key}"
        schema_row = get_node_row(conn, namespace, schema_name)

        if schema_row:
            schema_id = schema_row["id"]
        elif dry_run:
            result["created_nodes"] += 1
            result["created"] += min(len(members), budget - created_edges)
            created_edges += min(len(members), budget - created_edges)
            if len(result["examples"]) < 3:
                result["examples"].append({"schema": schema_name, "members": len(members), "dry_run": True})
            continue
        else:
            try:
                schema_row = ensure_node(
                    conn, namespace, schema_name,
                    plasticity=0.5, corpus="SCHEMA",
                )
                schema_id = schema_row["id"]
                result["created_nodes"] += 1
            except Exception:
                continue

        for nid, val in members:
            if created_edges >= budget:
                break
            status = _try_upsert(
                conn, namespace, nid, schema_id, "instance_of",
                clamp(val, -1.0, 1.0), 0.7, f"schema:{key}",
                snapshot, dry_run,
            )
            if status == "created":
                created_edges += 1
                result["created"] += 1
            elif status in result:
                result[status] += 1

        if len(result["examples"]) < 3:
            result["examples"].append({"schema": schema_name, "members": len(members)})

    return result


def _synth_journal_cooccurrence(conn, namespace, budget, rng, temperature, snapshot, dry_run):
    result = _empty_synth_result()
    if budget <= 0:
        return result
    lookback = (
        datetime.now(timezone.utc)
        - timedelta(days=JOURNAL_COOCCURRENCE_LOOKBACK_DAYS)
    ).isoformat()
    rows = conn.execute(
        """
        SELECT node_id, ts
        FROM journal
        WHERE namespace = ? AND node_id IS NOT NULL AND ts >= ?
        ORDER BY ts, id
        LIMIT ?
        """,
        (namespace, lookback, MAX_SYNTHESIS_CANDIDATES * 2),
    ).fetchall()

    entries: List[Tuple[datetime, int]] = []
    for r in rows:
        t = parse_ts(r["ts"])
        if t:
            entries.append((t, r["node_id"]))
    if len(entries) < 2:
        return result

    window_seconds = JOURNAL_COOCCURRENCE_WINDOW_HOURS * 3600.0
    # Canonical pair -> [co-occurrence count, smallest gap in seconds].
    # Counting (rather than first-seen dedupe) lets repeated co-occurrence
    # strengthen one edge instead of repeats being silently discarded.
    # Canonical (min_id, max_id) ordering makes co_occurred edges symmetric:
    # every repeat aggregates onto the same edge regardless of timestamp order.
    pair_stats: Dict[Tuple[int, int], List[float]] = {}
    for i, (t1, n1) in enumerate(entries):
        for j in range(i + 1, len(entries)):
            t2, n2 = entries[j]
            gap_s = (t2 - t1).total_seconds()
            if gap_s > window_seconds:
                break
            if n1 == n2:
                continue
            pair = (min(n1, n2), max(n1, n2))
            stats = pair_stats.setdefault(pair, [0.0, gap_s])
            stats[0] += 1.0
            if gap_s < stats[1]:
                stats[1] = gap_s
            if len(pair_stats) >= MAX_SYNTHESIS_CANDIDATES:
                break
        if len(pair_stats) >= MAX_SYNTHESIS_CANDIDATES:
            break
    if not pair_stats:
        return result

    # Soft semantic boost (not a gate): pairs where both nodes carry
    # contributes=1 features are scaled by 0.6 + 0.4*cosine, so similar
    # pairs are rewarded and dissimilar ones damped but never excluded.
    # Featureless pairs keep factor 1.0 — this is the only mechanism that
    # can link featureless nodes, so lack of features must not penalize.
    all_nids = sorted({nid for pair in pair_stats for nid in pair})
    vecs = _feature_vectors(conn, namespace, all_nids)

    candidates: List[Dict[str, Any]] = []
    for (n1, n2), (count, min_gap_s) in pair_stats.items():
        gap_hours = min_gap_s / 3600.0
        temporal = clamp(1.0 / (1.0 + gap_hours), 0.0, 1.0)
        frequency = 1.0 - math.exp(-count)
        semantic = 1.0
        sim: Optional[float] = None
        v1 = vecs.get(n1, {})
        v2 = vecs.get(n2, {})
        if v1 and v2:
            sim = _cosine(v1, v2)
            semantic = 0.6 + 0.4 * sim
        evidence = f"journal_cooccurrence:{int(count)}x:{gap_hours:.2f}h"
        if sim is not None:
            evidence += f":sim={sim:.2f}"
        candidates.append({
            "source_id": n1, "target_id": n2,
            "relation": "co_occurred",
            "weight": clamp(temporal * frequency * semantic, 0.0, 1.0),
            "confidence": clamp(0.5 + 0.1 * min(count, 3.0), 0.0, 0.8),
            "evidence": evidence,
            "count": int(count),
        })

    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], snapshot, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "count": c["count"],
                "gap_hours": c["evidence"].split(":")[2],
            })
    return result


def _synth_emotion(conn, namespace, budget, rng, temperature, snapshot, dry_run):
    result = _empty_synth_result()
    if budget <= 0:
        return result
    rows = conn.execute(
        """
        SELECT *
        FROM nodes
        WHERE namespace = ? AND suppressed = 0
        ORDER BY ABS(salience) DESC, id
        LIMIT ?
        """,
        (namespace, MAX_SYNTHESIS_CANDIDATES),
    ).fetchall()

    eligible: List[Tuple[int, float]] = []
    for r in rows:
        sal = effective_salience(r)
        if abs(sal) >= 0.3:
            eligible.append((r["id"], sal))
    if len(eligible) < 2:
        return result

    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    max_gen = budget * 3
    for i, (n1, s1) in enumerate(eligible):
        for n2, s2 in eligible[i + 1:]:
            pair = (min(n1, n2), max(n1, n2))
            if pair in seen:
                continue
            seen.add(pair)
            candidates.append({
                "source_id": n1, "target_id": n2,
                "relation": "affective",
                "weight": clamp(s1 * s2, -1.0, 1.0),
                "confidence": 0.5,
                "evidence": f"emotion:{s1:+.2f}*{s2:+.2f}",
            })
            if len(candidates) >= max_gen:
                break
        if len(candidates) >= max_gen:
            break

    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], snapshot, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "weight": round(c["weight"], 3),
            })
    return result


def _run_synthesis(
    conn: sqlite3.Connection,
    namespace: str,
    budget: int,
    seed: Optional[int],
    temperature: float,
    dry_run: bool,
) -> Dict[str, Any]:
    rng = random.Random(seed) if seed is not None else random.Random()
    snapshot = _GraphSnapshot(conn, namespace)

    # Priority order: highest-confidence signals first, exploratory ones last.
    mechanisms = [
        ("transitive",             _synth_transitive,             0.25),
        ("feature_recombination",  _synth_feature_recombination,  0.20),
        ("analogy",                _synth_analogy,                0.15),
        ("schema",                 _synth_schema,                 0.15),
        ("journal_cooccurrence",   _synth_journal_cooccurrence,   0.15),
        ("emotion",                _synth_emotion,                0.10),
    ]

    results: Dict[str, Any] = {}
    total_edges = 0
    for name, fn, share in mechanisms:
        mech_budget = min(int(round(budget * share)), budget - total_edges)
        if mech_budget <= 0:
            results[name] = {"skipped": "budget_exhausted"}
            continue
        try:
            r = fn(conn, namespace, mech_budget, rng, temperature, snapshot, dry_run)
            results[name] = r
            total_edges += r.get("created", 0)
        except Exception as e:  # keep one broken mechanism from killing dream
            results[name] = {"error": f"{type(e).__name__}: {e}"[:200]}

    # Deferred writes: flush accepted candidates to the DB in real mode.
    # Doing this only after all mechanisms finished keeps candidate pools
    # identical between dry_run and real (mechanisms never see each other's
    # new edges mid-dream), while cascading inference still accumulates
    # across successive dreams.
    flush_errors = 0
    if not dry_run:
        for source_id, target_id, relation, weight, confidence, evidence in snapshot.pending:
            try:
                upsert_edge(conn, namespace, source_id, target_id, relation, weight, confidence, evidence)
            except Exception:
                flush_errors += 1
    if flush_errors:
        results["_flush"] = {"skipped_invalid": flush_errors}

    return {
        "mechanisms": results,
        "total_edges_created": total_edges,
        "total_edges_skipped_cycle": sum(
            r.get("skipped_cycle", 0) for r in results.values() if isinstance(r, dict)
        ),
        "total_edges_skipped_exists": sum(
            r.get("skipped_exists", 0) for r in results.values() if isinstance(r, dict)
        ),
        "budget": budget,
        "seed": seed,
        "temperature": temperature,
        "dry_run": dry_run,
    }


@mcp.tool()
def dream(
    namespace: str = DEFAULT_NAMESPACE,
    max_synthesis_edges: int = 100,
    seed: Optional[int] = None,
    temperature: float = 1.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Sleep-cycle maintenance and novel synthesis.

    Must be in 'sleep' state (latest event is sleep_start).

    Maintenance: DB optimize, journal/weak-edge pruning, orphan suppression,
    FTS rebuild.

    Synthesis (all bounded by max_synthesis_edges, cycle-checked):
      - transitive:           relation-typed composition (is_a∘is_a, causes∘causes, …)
      - feature_recombination: cosine similarity over contributes=1 features
      - analogy:              parallelogram completion over same-relation edges
      - schema:               creates __schema__::<key> nodes for shared features
      - journal_cooccurrence: temporal adjacency in the journal
      - emotion:              high-|salience| pairs, signed weight

    Args:
        seed:        int → deterministic output; None → fresh sample each call
        temperature: >1 more exploratory, <1 greedier, 1.0 balanced
        dry_run:     if True, do not mutate the DB (maintenance is skipped,
                     synthesis is previewed)
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    max_synthesis_edges = clamp_int(max_synthesis_edges, 1, 500)
    temperature = clamp(temperature if math.isfinite(temperature) else 1.0, 0.01, 10.0)

    with db() as conn:
        state = brain_state(conn, namespace)

        if state["state"] != "asleep":
            return {
                "status": "awake",
                "message": "Brain is not in sleep state. Call sleep_start first.",
                "brain": state,
            }

        results: Dict[str, Any] = {
            "maintenance": {},
            "synthesis": {},
            "brain": state,
            "dry_run": dry_run,
        }

        with conn:
            # 1. Maintenance (skipped in dry_run)
            if dry_run:
                results["maintenance"]["skipped"] = "dry_run"
            else:
                conn.execute("PRAGMA optimize")

                cutoff_journal = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                cur_j = conn.execute(
                    "DELETE FROM journal WHERE namespace = ? AND ts < ?",
                    (namespace, cutoff_journal),
                )
                results["maintenance"]["journal_pruned"] = cur_j.rowcount

                cutoff_edges = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
                cur_e = conn.execute(
                    """
                    DELETE FROM edges
                    WHERE namespace = ? AND ABS(weight) < 0.05 AND ts < ?
                    """,
                    (namespace, cutoff_edges),
                )
                results["maintenance"]["weak_edges_pruned"] = cur_e.rowcount

                cutoff_orphans = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                cur_o = conn.execute(
                    """
                    UPDATE nodes SET suppressed = 1
                    WHERE namespace = ?
                      AND id NOT IN (
                          SELECT source_id FROM edges
                          UNION SELECT target_id FROM edges
                      )
                      AND base_strength < 0.1
                      AND last_accessed < ?
                    """,
                    (namespace, cutoff_orphans),
                )
                results["maintenance"]["orphans_suppressed"] = cur_o.rowcount

                if FTS_AVAILABLE:
                    try:
                        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
                        results["maintenance"]["fts_rebuilt"] = True
                    except sqlite3.OperationalError:
                        results["maintenance"]["fts_rebuilt"] = False

            # 2. Synthesis
            results["synthesis"] = _run_synthesis(
                conn,
                namespace,
                max_synthesis_edges,
                seed,
                temperature,
                dry_run,
            )

            # 3. Hub strengthening (unchanged, skipped in dry_run)
            if dry_run:
                results["synthesis"]["hubs_strengthened"] = 0
            else:
                hub_sql = """
                    SELECT id, degree FROM (
                        SELECT n.id AS id,
                               (SELECT COUNT(*) FROM edges WHERE source_id = n.id) +
                               (SELECT COUNT(*) FROM edges WHERE target_id = n.id) AS degree
                        FROM nodes n
                        WHERE n.namespace = ? AND n.suppressed = 0
                    )
                    WHERE degree >= 5
                    ORDER BY degree DESC
                    LIMIT 20
                """
                hubs = conn.execute(hub_sql, (namespace,)).fetchall()
                strengthened = 0
                for h in hubs:
                    boost = 0.01 * min(h["degree"] / 10.0, 1.0)
                    strengthen_node(conn, h["id"], boost)
                    strengthened += 1
                results["synthesis"]["hubs_strengthened"] = strengthened

            if not dry_run:
                write_journal(
                    conn,
                    namespace,
                    "dream",
                    None,
                    {
                        "maintenance": results["maintenance"],
                        "synthesis": {
                            "total_edges_created": results["synthesis"].get("total_edges_created", 0),
                            "budget": max_synthesis_edges,
                            "seed": seed,
                            "temperature": temperature,
                            "mechanisms": {
                                name: {
                                    k: v for k, v in r.items()
                                    if k in ("created", "created_nodes", "proposed",
                                             "skipped_cycle", "skipped_exists")
                                }
                                for name, r in results["synthesis"].get("mechanisms", {}).items()
                                if isinstance(r, dict)
                            },
                        },
                    },
                )

        results["brain"] = brain_state(conn, namespace)
        return results
@mcp.tool()
def rebuild_fts() -> Dict[str, Any]:
    """Rebuild the FTS index."""
    global FTS_AVAILABLE

    with db() as conn:
        FTS_AVAILABLE = setup_fts(conn)
        conn.commit()

    return {"fts_available": FTS_AVAILABLE}


@mcp.tool()
def prune_journal(
    older_than_days: int = 30,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Delete old journal entries."""
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    days = clamp_int(older_than_days, 1, 3650)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with db() as conn:
        with conn:
            cur = conn.execute(
                "DELETE FROM journal WHERE namespace = ? AND ts < ?",
                (namespace, cutoff.isoformat()),
            )

        return {
            "deleted": cur.rowcount,
            "older_than_days": days,
            "namespace": namespace,
        }

# ---------------------------------------------------------------------
# Restored semantic graph layer
# ---------------------------------------------------------------------


def _is_suppressed(conn, node_id: int) -> bool:
    row = conn.execute(
        "SELECT suppressed FROM nodes WHERE id = ?",
        (node_id,),
    ).fetchone()
    return bool(row and row["suppressed"])


def _name_map(conn, node_ids) -> Dict[str, Any]:
    if not node_ids:
        return {}

    placeholders = ",".join("?" for _ in node_ids)
    rows = conn.execute(
        f"SELECT id, display_name FROM nodes WHERE id IN ({placeholders})",
        tuple(node_ids),
    ).fetchall()

    return {row["id"]: row["display_name"] for row in rows}


def _public_edge(edge_row, name_by_id: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": edge_row["id"],
        "source": name_by_id.get(edge_row["source_id"], str(edge_row["source_id"])),
        "target": name_by_id.get(edge_row["target_id"], str(edge_row["target_id"])),
        "relation": edge_row["relation"],
        "weight": clamp(edge_row["weight"], -1.0, 1.0),
        "confidence": clamp(edge_row["confidence"], 0.0, 1.0),
        "evidence": edge_row["evidence"],
        "ts": edge_row["ts"],
    }


def traverse_semantic(
    conn,
    namespace: str,
    start_name: str,
    direction: str,
    depth: int,
    include_suppressed: bool = False,
) -> Dict[str, Any]:
    direction = clean_text(direction, 10).lower()
    if direction not in ("in", "out", "both"):
        direction = "both"

    depth = clamp_int(depth, 0, MAX_GRAPH_DEPTH)
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    start = get_node_row(conn, namespace, start_name)

    if not start:
        raise ValueError("start node not found")

    if start["suppressed"] and not include_suppressed:
        return {
            "root": start["display_name"],
            "direction": direction,
            "depth": depth,
            "nodes": [],
            "edges": [],
        }

    queue = [(start["id"], 0)]
    seen = {start["id"]}
    node_ids = {start["id"]}
    edge_rows = []
    edge_ids = set()
    max_edges = 200

    while queue:
        node_id, d = queue.pop(0)

        if d >= depth or len(edge_rows) >= max_edges:
            continue

        if direction in ("out", "both"):
            out_rows = conn.execute(
                """
                SELECT *
                FROM edges
                WHERE namespace = ? AND source_id = ?
                ORDER BY ABS(weight) DESC
                LIMIT 30
                """,
                (namespace, node_id),
            ).fetchall()

            for e in out_rows:
                if len(edge_rows) >= max_edges:
                    break
                if e["id"] in edge_ids:
                    continue

                neighbor_id = e["target_id"]

                if not include_suppressed and _is_suppressed(conn, neighbor_id):
                    continue

                edge_ids.add(e["id"])
                edge_rows.append(e)
                node_ids.add(neighbor_id)

                if neighbor_id not in seen:
                    seen.add(neighbor_id)
                    queue.append((neighbor_id, d + 1))

        if direction in ("in", "both"):
            in_rows = conn.execute(
                """
                SELECT *
                FROM edges
                WHERE namespace = ? AND target_id = ?
                ORDER BY ABS(weight) DESC
                LIMIT 30
                """,
                (namespace, node_id),
            ).fetchall()

            for e in in_rows:
                if len(edge_rows) >= max_edges:
                    break
                if e["id"] in edge_ids:
                    continue

                neighbor_id = e["source_id"]

                if not include_suppressed and _is_suppressed(conn, neighbor_id):
                    continue

                edge_ids.add(e["id"])
                edge_rows.append(e)
                node_ids.add(neighbor_id)

                if neighbor_id not in seen:
                    seen.add(neighbor_id)
                    queue.append((neighbor_id, d + 1))

    if not node_ids:
        return {
            "root": start["display_name"],
            "direction": direction,
            "depth": depth,
            "nodes": [],
            "edges": [],
        }

    placeholders = ",".join("?" for _ in node_ids)
    node_rows = conn.execute(
        f"SELECT * FROM nodes WHERE id IN ({placeholders})",
        tuple(node_ids),
    ).fetchall()

    nodes = []
    included_node_ids = set()

    for row in node_rows:
        if row["suppressed"] and not include_suppressed:
            continue

        included_node_ids.add(row["id"])
        nodes.append(node_view(conn, row))

    name_by_id = {n["id"]: n["name"] for n in nodes}

    edges = []
    for e in edge_rows:
        if e["source_id"] in included_node_ids and e["target_id"] in included_node_ids:
            edges.append(_public_edge(e, name_by_id))

    return {
        "root": start["display_name"],
        "direction": direction,
        "depth": depth,
        "nodes": nodes[:100],
        "edges": edges[:200],
    }


def find_paths(
    conn,
    namespace: str,
    source_name: str,
    target_name: str,
    max_depth: int = 3,
    limit_paths: int = 20,
    include_suppressed: bool = False,
) -> Dict[str, Any]:
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    max_depth = clamp_int(max_depth, 1, MAX_GRAPH_DEPTH)
    limit_paths = clamp_int(limit_paths, 1, 50)

    source_row = get_node_row(conn, namespace, source_name)
    target_row = get_node_row(conn, namespace, target_name)

    if not source_row or not target_row:
        raise ValueError("source or target node not found")

    if (source_row["suppressed"] or target_row["suppressed"]) and not include_suppressed:
        return {
            "source": source_row["display_name"],
            "target": target_row["display_name"],
            "count": 0,
            "paths": [],
        }

    if source_row["id"] == target_row["id"]:
        view = node_view(conn, source_row)
        return {
            "source": source_row["display_name"],
            "target": target_row["display_name"],
            "count": 1,
            "paths": [
                {
                    "nodes": [view],
                    "edges": [],
                    "score": 1.0,
                }
            ],
        }

    stack = [(source_row["id"], [source_row["id"]], [])]
    found = []
    iterations = 0

    while stack and len(found) < limit_paths and iterations < 5000:
        iterations += 1

        node_id, node_path, edge_path = stack.pop()

        if len(node_path) - 1 >= max_depth:
            continue

        rows = conn.execute(
            """
            SELECT *
            FROM edges
            WHERE namespace = ? AND source_id = ?
            ORDER BY ABS(weight) DESC
            LIMIT 30
            """,
            (namespace, node_id),
        ).fetchall()

        for e in rows:
            next_id = e["target_id"]

            if next_id in node_path:
                continue

            if not include_suppressed and _is_suppressed(conn, next_id):
                continue

            new_node_path = node_path + [next_id]
            new_edge_path = edge_path + [e["id"]]

            if next_id == target_row["id"]:
                found.append((new_node_path, new_edge_path))
                if len(found) >= limit_paths:
                    break
            else:
                stack.append((next_id, new_node_path, new_edge_path))

    all_node_ids = {source_row["id"], target_row["id"]}
    all_edge_ids = set()

    for node_path, edge_path in found:
        all_node_ids.update(node_path)
        all_edge_ids.update(edge_path)

    node_placeholders = ",".join("?" for _ in all_node_ids)
    node_rows = conn.execute(
        f"SELECT * FROM nodes WHERE id IN ({node_placeholders})",
        tuple(all_node_ids),
    ).fetchall()

    node_by_id = {}
    for row in node_rows:
        if row["suppressed"] and not include_suppressed:
            continue
        node_by_id[row["id"]] = node_view(conn, row)

    edge_by_id = {}
    if all_edge_ids:
        edge_placeholders = ",".join("?" for _ in all_edge_ids)
        edge_rows = conn.execute(
            f"SELECT * FROM edges WHERE id IN ({edge_placeholders})",
            tuple(all_edge_ids),
        ).fetchall()
        edge_by_id = {row["id"]: row for row in edge_rows}

    name_by_id = {nid: view["name"] for nid, view in node_by_id.items()}

    paths_out = []

    for node_path, edge_path in found:
        path_nodes = [node_by_id[nid] for nid in node_path if nid in node_by_id]
        path_edges = []
        score = 1.0

        for edge_id in edge_path:
            edge_row = edge_by_id.get(edge_id)
            if not edge_row:
                continue

            path_edges.append(_public_edge(edge_row, name_by_id))
            score *= clamp(edge_row["weight"], -1.0, 1.0)
            score *= clamp(edge_row["confidence"], 0.0, 1.0)

        paths_out.append(
            {
                "nodes": path_nodes,
                "edges": path_edges,
                "score": abs(score),
            }
        )

    paths_out.sort(key=lambda x: x["score"], reverse=True)

    return {
        "source": source_row["display_name"],
        "target": target_row["display_name"],
        "count": len(paths_out),
        "paths": paths_out[:limit_paths],
    }


def intervene_semantic(
    conn,
    namespace: str,
    do_node_name: str,
    target_name: str = "",
    depth: int = 3,
) -> Dict[str, Any]:
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    depth = clamp_int(depth, 1, MAX_GRAPH_DEPTH)

    do_row = get_node_row(conn, namespace, do_node_name)
    if not do_row:
        raise ValueError("intervention node not found")

    if do_row["suppressed"]:
        return {
            "do": do_row["display_name"],
            "target": target_name or None,
            "target_activation": None,
            "affected": [],
            "edges": [],
        }

    activations = {do_row["id"]: 1.0}
    edges_used = []
    edge_ids = set()
    queue = [(do_row["id"], 1.0, 0)]
    iterations = 0

    while queue and iterations < 2000:
        iterations += 1

        node_id, activation, d = queue.pop(0)

        if d >= depth or abs(activation) < 0.01:
            continue

        rows = conn.execute(
            """
            SELECT *
            FROM edges
            WHERE namespace = ? AND source_id = ?
            ORDER BY ABS(weight) DESC
            LIMIT 30
            """,
            (namespace, node_id),
        ).fetchall()

        for e in rows:
            child_id = e["target_id"]

            # Intervention cuts incoming feedback into the intervened node.
            if child_id == do_row["id"]:
                continue

            if _is_suppressed(conn, child_id):
                continue

            delta = (
                activation
                * clamp(e["weight"], -1.0, 1.0)
                * clamp(e["confidence"], 0.0, 1.0)
            )

            if abs(delta) < 0.005:
                continue

            old_activation = activations.get(child_id, 0.0)
            new_activation = clamp(old_activation + delta, -1.0, 1.0)

            if abs(new_activation - old_activation) <= 0.001:
                continue

            activations[child_id] = new_activation

            if e["id"] not in edge_ids:
                edge_ids.add(e["id"])
                edges_used.append(e)

            queue.append((child_id, new_activation, d + 1))

    node_ids = set(activations.keys())
    for e in edges_used:
        node_ids.add(e["source_id"])
        node_ids.add(e["target_id"])

    if not node_ids:
        return {
            "do": do_row["display_name"],
            "target": target_name or None,
            "target_activation": None,
            "affected": [],
            "edges": [],
        }

    placeholders = ",".join("?" for _ in node_ids)
    node_rows = conn.execute(
        f"SELECT * FROM nodes WHERE id IN ({placeholders})",
        tuple(node_ids),
    ).fetchall()

    node_by_id = {}
    for row in node_rows:
        if row["suppressed"]:
            continue
        node_by_id[row["id"]] = node_view(conn, row)

    affected = []

    for node_id, activation in activations.items():
        view = node_by_id.get(node_id)
        if not view:
            continue

        affected.append(
            {
                "name": view["name"],
                "activation": clamp(activation, -1.0, 1.0),
                "effective_strength": view["effective_strength"],
            }
        )

    affected.sort(key=lambda x: abs(x["activation"]), reverse=True)

    target_activation = None
    target_display = None

    if target_name:
        target_row = get_node_row(conn, namespace, target_name)
        if target_row:
            target_display = target_row["display_name"]
            target_activation = clamp(activations.get(target_row["id"], 0.0), -1.0, 1.0)

    name_by_id = _name_map(conn, node_ids)
    public_edges = [_public_edge(e, name_by_id) for e in edges_used[:200]]

    return {
        "do": do_row["display_name"],
        "depth": depth,
        "target": target_display,
        "target_activation": target_activation,
        "affected": affected[:100],
        "edges": public_edges,
        "iterations": iterations,
    }


def explain_node(conn, namespace: str, name: str) -> Dict[str, Any]:
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    row = get_node_row(conn, namespace, name)

    if not row:
        raise ValueError("node not found")

    view = node_view(conn, row)

    incoming_rows = conn.execute(
        """
        SELECT e.*, s.display_name AS source_name
        FROM edges e
        JOIN nodes s ON s.id = e.source_id
        WHERE e.namespace = ? AND e.target_id = ?
        ORDER BY ABS(e.weight) DESC
        LIMIT 50
        """,
        (namespace, row["id"]),
    ).fetchall()

    outgoing_rows = conn.execute(
        """
        SELECT e.*, t.display_name AS target_name
        FROM edges e
        JOIN nodes t ON t.id = e.target_id
        WHERE e.namespace = ? AND e.source_id = ?
        ORDER BY ABS(e.weight) DESC
        LIMIT 50
        """,
        (namespace, row["id"]),
    ).fetchall()

    feature_rows = conn.execute(
        """
        SELECT key, value, contributes, ts
        FROM features
        WHERE node_id = ?
        ORDER BY key
        """,
        (row["id"],),
    ).fetchall()

    journal_rows = conn.execute(
        """
        SELECT tool, detail, ts
        FROM journal
        WHERE node_id = ?
        ORDER BY id DESC
        LIMIT 10
        """,
        (row["id"],),
    ).fetchall()

    incoming = []
    for e in incoming_rows:
        incoming.append(
            {
                "source": e["source_name"],
                "relation": e["relation"],
                "weight": clamp(e["weight"], -1.0, 1.0),
                "confidence": clamp(e["confidence"], 0.0, 1.0),
                "evidence": e["evidence"],
                "ts": e["ts"],
            }
        )

    outgoing = []
    for e in outgoing_rows:
        outgoing.append(
            {
                "target": e["target_name"],
                "relation": e["relation"],
                "weight": clamp(e["weight"], -1.0, 1.0),
                "confidence": clamp(e["confidence"], 0.0, 1.0),
                "evidence": e["evidence"],
                "ts": e["ts"],
            }
        )

    features = [dict(r) for r in feature_rows]
    journal = [dict(r) for r in journal_rows]

    return {
        "node": view,
        "incoming": incoming,
        "outgoing": outgoing,
        "features": features,
        "journal": journal,
        "brain": brain_state(conn, namespace),
    }


@mcp.tool()
def ancestors(
    name: str,
    depth: int = 2,
    include_suppressed: bool = False,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Incoming semantic neighborhood: things that point to this node."""
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    name = require_text(name, "name", MAX_NAME)

    with db() as conn:
        return traverse_semantic(
            conn,
            namespace,
            name,
            "in",
            depth,
            include_suppressed,
        )


@mcp.tool()
def descendants(
    name: str,
    depth: int = 2,
    include_suppressed: bool = False,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Outgoing semantic neighborhood: things this node points to."""
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    name = require_text(name, "name", MAX_NAME)

    with db() as conn:
        return traverse_semantic(
            conn,
            namespace,
            name,
            "out",
            depth,
            include_suppressed,
        )


@mcp.tool()
def paths(
    source: str,
    target: str,
    max_depth: int = 3,
    limit_paths: int = 20,
    include_suppressed: bool = False,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Find bounded directed paths from source to target."""
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    source = require_text(source, "source", MAX_NAME)
    target = require_text(target, "target", MAX_NAME)

    with db() as conn:
        return find_paths(
            conn,
            namespace,
            source,
            target,
            max_depth,
            limit_paths,
            include_suppressed,
        )


@mcp.tool()
def intervene(
    do_node: str,
    target: str = "",
    depth: int = 3,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Approximate causal intervention.

    Activates do_node and propagates activation along outgoing edges.
    Incoming edges into do_node are ignored during propagation.
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    do_node = require_text(do_node, "do_node", MAX_NAME)
    target = clean_text(target, MAX_NAME)

    with db() as conn:
        return intervene_semantic(
            conn,
            namespace,
            do_node,
            target,
            depth,
        )


@mcp.tool()
def explain(
    name: str,
    namespace: str = DEFAULT_NAMESPACE,
) -> Dict[str, Any]:
    """Explain a node: its state, incoming causes, outgoing effects, features, and recent journal."""
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    name = require_text(name, "name", MAX_NAME)

    with db() as conn:
        return explain_node(conn, namespace, name)

# ---------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------


# FIX #6: resources now cover all namespaces, not just DEFAULT_NAMESPACE,
# while preserving the original top-level keys for backward compatibility.


def _all_namespaces(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT namespace FROM nodes UNION SELECT namespace FROM events"
    ).fetchall()
    seen = sorted({r["namespace"] for r in rows})
    return seen or [DEFAULT_NAMESPACE]


@mcp.resource("memory://brain")
def resource_brain() -> str:
    with db() as conn:
        ns_list = _all_namespaces(conn)
        payload = {
            "brain": brain_state(conn, DEFAULT_NAMESPACE),
            "namespaces": {ns: brain_state(conn, ns) for ns in ns_list},
        }
        return json.dumps(payload, allow_nan=False, indent=2, default=str)


@mcp.resource("memory://fit")
def resource_fit() -> str:
    with db() as conn:
        ns_list = _all_namespaces(conn)
        payload = {
            "fit": top_nodes(conn, DEFAULT_NAMESPACE, 50),
            "namespaces": {ns: top_nodes(conn, ns, 50) for ns in ns_list},
        }
        return json.dumps(payload, allow_nan=False, indent=2, default=str)


@mcp.resource("memory://metrics")
def resource_metrics() -> str:
    with db() as conn:
        ns_list = _all_namespaces(conn)

        def ns_metrics(ns: str) -> Dict[str, Any]:
            return {
                "nodes": conn.execute(
                    "SELECT COUNT(*) AS c FROM nodes WHERE namespace = ?", (ns,)
                ).fetchone()["c"],
                "edges": conn.execute(
                    "SELECT COUNT(*) AS c FROM edges WHERE namespace = ?", (ns,)
                ).fetchone()["c"],
                "features": conn.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM features f
                    JOIN nodes n ON n.id = f.node_id
                    WHERE n.namespace = ?
                    """,
                    (ns,),
                ).fetchone()["c"],
                "events": conn.execute(
                    "SELECT COUNT(*) AS c FROM events WHERE namespace = ?", (ns,)
                ).fetchone()["c"],
                "journal": conn.execute(
                    "SELECT COUNT(*) AS c FROM journal WHERE namespace = ?", (ns,)
                ).fetchone()["c"],
            }

        default_metrics = ns_metrics(DEFAULT_NAMESPACE)
        default_metrics["namespace"] = DEFAULT_NAMESPACE
        default_metrics["fts_available"] = FTS_AVAILABLE
        default_metrics["db_size_bytes"] = (
            os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        )

        payload = {
            "metrics": default_metrics,
            "namespaces": {ns: ns_metrics(ns) for ns in ns_list},
        }
        return json.dumps(payload, allow_nan=False, indent=2, default=str)


@mcp.resource("memory://journal")
def resource_journal() -> str:
    with db() as conn:
        ns_list = _all_namespaces(conn)

        def ns_journal(ns: str) -> List[Dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT id, tool, node_id, detail, ts
                FROM journal
                WHERE namespace = ?
                ORDER BY id DESC
                LIMIT 100
                """,
                (ns,),
            ).fetchall()
            return [dict(row) for row in rows]

        payload = {
            "journal": ns_journal(DEFAULT_NAMESPACE),
            "namespaces": {ns: ns_journal(ns) for ns in ns_list},
        }
        return json.dumps(payload, allow_nan=False, indent=2, default=str)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


if __name__ == "__main__":
    init_db()
    mcp.run(transport="stdio")