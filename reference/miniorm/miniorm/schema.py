"""DDL generation and migrations."""

from __future__ import annotations

from miniorm.fields import MiniORMError


def create_table_sql(model: type) -> str:
    """CREATE TABLE for a model, with constraints."""
    parts: list[str] = []
    for name, field in model.fields.items():
        piece = name + " " + field.sql_type
        if field.primary_key:
            piece += " PRIMARY KEY"
            if field.sql_type == "INTEGER":
                piece += " AUTOINCREMENT"
        else:
            if not field.nullable:
                piece += " NOT NULL"
            if field.unique:
                piece += " UNIQUE"
        parts.append(piece)
    return (
        "CREATE TABLE IF NOT EXISTS " + model.table_name + " (\n  "
        + ",\n  ".join(parts)
        + "\n)"
    )


def drop_table_sql(model: type) -> str:
    return "DROP TABLE IF EXISTS " + model.table_name


def migration_sql(old_columns: list[str], new_columns: list[str], table_name: str) -> list[str]:
    """Statements to migrate from one column set to another.

    Added columns become ALTER TABLE ... ADD COLUMN. Removed columns are
    reported as unsupported rather than silently dropped -- SQLite cannot drop a
    column in older versions, and silently ignoring it would lose data.
    """
    removed = [c for c in old_columns if c not in new_columns]
    if removed:
        raise MiniORMError(
            "removing column(s) is not supported: " + ", ".join(removed)
        )
    added = [c for c in new_columns if c not in old_columns]
    return [
        "ALTER TABLE " + table_name + " ADD COLUMN " + c + " TEXT" for c in added
    ]
