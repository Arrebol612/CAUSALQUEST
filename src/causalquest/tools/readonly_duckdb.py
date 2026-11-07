"""Small read-only DuckDB boundary with explicit query restrictions."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

ALLOWED_RELATIONS = {
    "daily_policy_metrics",
    "events",
    "exposures",
    "metrics",
    "policies",
    "snapshot_metadata",
    "source_files",
}
BANNED_TOKENS = {
    "alter",
    "attach",
    "call",
    "copy",
    "create",
    "delete",
    "detach",
    "drop",
    "export",
    "glob",
    "httpfs",
    "import",
    "insert",
    "install",
    "json_scan",
    "load",
    "mysql_scan",
    "parquet_scan",
    "postgres_scan",
    "pragma",
    "read_blob",
    "read_csv",
    "read_json",
    "read_parquet",
    "read_text",
    "set",
    "sqlite_scan",
    "update",
}


class UnsafeQueryError(ValueError):
    """Raised when SQL crosses the local read-only policy."""


def _load_duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("duckdb is required; install the project dependencies first") from exc
    return duckdb


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def validate_read_only_sql(
    sql: str,
    allowed_relations: set[str] | frozenset[str] | None = None,
) -> str:
    """Reject mutation, extension loading, external scans, and multiple statements."""

    statement = sql.strip()
    if statement.endswith(";"):
        statement = statement[:-1].rstrip()
    if not statement or ";" in statement:
        raise UnsafeQueryError("exactly one SQL statement is required")
    without_comments = re.sub(r"/\*.*?\*/|--[^\n]*", " ", statement, flags=re.DOTALL)
    tokens = set(re.findall(r"[a-z_]+", without_comments.lower()))
    if not re.match(r"^\s*(select|with)\b", without_comments, flags=re.IGNORECASE):
        raise UnsafeQueryError("query must start with SELECT or WITH")
    if re.search(r"\b(from|join)\s+['\"]", without_comments, flags=re.IGNORECASE):
        raise UnsafeQueryError("file replacement scans are not allowed")
    blocked = sorted(tokens.intersection(BANNED_TOKENS))
    if blocked:
        raise UnsafeQueryError(f"blocked SQL token(s): {', '.join(blocked)}")
    cte_names = set(
        re.findall(r"(?:\bwith\b|,)\s*([a-z_][a-z0-9_]*)\s+as\s*\(", without_comments.lower())
    )
    relations = set(re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", without_comments.lower()))
    relation_allowlist = set(allowed_relations or ALLOWED_RELATIONS)
    unknown_relations = sorted(relations.difference(relation_allowlist, cte_names))
    if unknown_relations:
        raise UnsafeQueryError(f"relation is not allowlisted: {', '.join(unknown_relations)}")
    return statement


class ReadOnlyDuckDB:
    """A read-only database handle with bounded JSON-safe results."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        max_rows: int = 200,
        allowed_relations: set[str] | frozenset[str] | None = None,
    ) -> None:
        duckdb = _load_duckdb()
        self.max_rows = max_rows
        self.allowed_relations = set(allowed_relations or ALLOWED_RELATIONS)
        self.connection = duckdb.connect(str(database_path), read_only=True)
        self.connection.execute("SET TimeZone = 'UTC'")

    def close(self) -> None:
        self.connection.close()

    def describe_table(self, table: str) -> list[dict[str, Any]]:
        if table not in self.allowed_relations:
            raise UnsafeQueryError(f"relation is not allowlisted: {table}")
        rows = self.connection.execute(f"DESCRIBE {table}").fetchall()
        return [
            {
                "column_name": row[0],
                "column_type": row[1],
                "nullable": row[2],
            }
            for row in rows
        ]

    def query(self, sql: str, parameters: list[Any] | None = None) -> dict[str, Any]:
        statement = validate_read_only_sql(sql, self.allowed_relations)
        cursor = self.connection.execute(statement, parameters or [])
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchmany(self.max_rows + 1)
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        return {
            "columns": columns,
            "rows": [[_json_value(value) for value in row] for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()
