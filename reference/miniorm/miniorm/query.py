"""The immutable query builder."""

from __future__ import annotations

from typing import Any

from miniorm.fields import MiniORMError

_OPS = {
    "eq": "=",
    "lt": "<",
    "lte": "<=",
    "gt": ">",
    "gte": ">=",
    "ne": "!=",
}


class Query:
    """Fluent, immutable query over one model."""

    def __init__(self, model: type) -> None:
        self.model = model
        self._where: list[tuple[str, str, Any]] = []
        self._order: tuple[str, bool] | None = None
        self._limit: int | None = None
        self._offset: int | None = None

    # ---- construction ----
    def _clone(self) -> Query:
        q = Query(self.model)
        q._where = list(self._where)
        q._order = self._order
        q._limit = self._limit
        q._offset = self._offset
        return q

    def filter(self, **conditions: Any) -> Query:
        q = self._clone()
        for key, value in conditions.items():
            column, op = _split(key)
            if column not in self.model.fields:
                raise MiniORMError(
                    "unknown column '" + column + "' on " + self.model.__name__
                )
            q._where.append((column, op, value))
        return q

    def order_by(self, column: str, descending: bool = False) -> Query:
        if column not in self.model.fields:
            raise MiniORMError(
                "unknown column '" + column + "' on " + self.model.__name__
            )
        q = self._clone()
        q._order = (column, descending)
        return q

    def limit(self, n: int) -> Query:
        q = self._clone()
        q._limit = n
        return q

    def offset(self, n: int) -> Query:
        q = self._clone()
        q._offset = n
        return q

    # ---- compilation ----
    def compile(self) -> tuple[str, list[Any]]:
        columns = ", ".join(self.model.fields)
        sql = "SELECT " + columns + " FROM " + self.model.table_name
        params: list[Any] = []

        if self._where:
            clauses = []
            for column, op, value in self._where:
                clauses.append(column + " " + _OPS[op] + " ?")
                field = self.model.fields[column]
                params.append(field.to_db(value))
            sql += " WHERE " + " AND ".join(clauses)

        if self._order:
            column, descending = self._order
            sql += " ORDER BY " + column + (" DESC" if descending else " ASC")
        if self._limit is not None:
            sql += " LIMIT " + str(self._limit)
        if self._offset is not None:
            sql += " OFFSET " + str(self._offset)
        return sql, params

    def __repr__(self) -> str:
        return "Query(" + self.model.__name__ + ", " + self.compile()[0] + ")"


def _split(key: str) -> tuple[str, str]:
    """``score__gt`` -> ``("score", "gt")``; ``name`` -> ``("name", "eq")``."""
    if "__" in key:
        column, _, suffix = key.rpartition("__")
        if suffix in _OPS:
            return column, suffix
    return key, "eq"
