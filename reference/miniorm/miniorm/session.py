"""Session: connection, identity map, persistence."""

from __future__ import annotations

import sqlite3
from typing import Any

from miniorm.fields import MiniORMError
from miniorm.query import Query
from miniorm.schema import create_table_sql


class Session:
    """Wraps a sqlite3 connection with an identity map."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._identity: dict[tuple[str, Any], Any] = {}

    # ---- schema ----
    def create_tables(self, models: list[type]) -> None:
        cursor = self.connection.cursor()
        for model in models:
            cursor.execute(create_table_sql(model))
        self.connection.commit()

    # ---- writes ----
    def add(self, instance: Any) -> Any:
        model = type(instance)
        pk = model.primary_key
        columns = [n for n in model.fields if not (n == pk and getattr(instance, n) is None)]
        values = [model.fields[n].to_db(getattr(instance, n)) for n in columns]
        placeholders = ", ".join("?" for _ in columns)
        sql = (
            "INSERT INTO " + model.table_name + " (" + ", ".join(columns) + ") "
            "VALUES (" + placeholders + ")"
        )
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, values)
        except sqlite3.Error as exc:
            raise MiniORMError("insert failed: " + str(exc)) from exc
        if getattr(instance, pk) is None:
            object.__setattr__(instance, pk, cursor.lastrowid)
        self._identity[(model.__name__, getattr(instance, pk))] = instance
        return instance

    def delete(self, instance: Any) -> None:
        model = type(instance)
        pk = model.primary_key
        key = getattr(instance, pk)
        self.connection.cursor().execute(
            "DELETE FROM " + model.table_name + " WHERE " + pk + " = ?", (key,)
        )
        self._identity.pop((model.__name__, key), None)

    # ---- reads ----
    def get(self, model: type, key: Any) -> Any | None:
        cached = self._identity.get((model.__name__, key))
        if cached is not None:
            return cached
        sql = (
            "SELECT " + ", ".join(model.fields) + " FROM " + model.table_name
            + " WHERE " + model.primary_key + " = ?"
        )
        row = self.connection.cursor().execute(sql, (key,)).fetchone()
        if row is None:
            return None
        return self._materialise(model, row)

    def execute(self, query: Query) -> list[Any]:
        sql, params = query.compile()
        rows = self.connection.cursor().execute(sql, params).fetchall()
        return [self._materialise(query.model, row) for row in rows]

    def _materialise(self, model: type, row: tuple) -> Any:
        values = {
            name: model.fields[name].from_db(row[i])
            for i, name in enumerate(model.fields)
        }
        key = values[model.primary_key]
        cached = self._identity.get((model.__name__, key))
        if cached is not None:
            return cached
        instance = model(**values)
        self._identity[(model.__name__, key)] = instance
        return instance

    # ---- transactions ----
    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        """Roll back and evict the identity map, which may hold stale objects."""
        self.connection.rollback()
        self._identity.clear()

    def close(self) -> None:
        self.connection.close()
