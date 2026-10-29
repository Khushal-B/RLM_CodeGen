"""The Table: querying and rendering."""

from __future__ import annotations

from typing import Any, Callable

from tinycsv.types import ColumnType, TinyCSVError, coerce, infer_column_type


class Table:
    """A parsed table of type-coerced rows."""

    def __init__(
        self,
        columns: list[str],
        rows: list[dict[str, Any]],
        column_types: dict[str, ColumnType] | None = None,
    ) -> None:
        self.columns = list(columns)
        self.rows = list(rows)
        self.column_types = dict(column_types or {})

    # ---- introspection ----
    @property
    def row_count(self) -> int:
        return len(self.rows)

    def column_type(self, name: str) -> ColumnType:
        if name not in self.column_types:
            raise TinyCSVError("unknown column: " + name)
        return self.column_types[name]

    # ---- querying ----
    def select(self, columns: list[str]) -> Table:
        for c in columns:
            if c not in self.columns:
                raise TinyCSVError("unknown column: " + c)
        rows = [{c: r.get(c) for c in columns} for r in self.rows]
        types = {c: self.column_types.get(c, ColumnType.STR) for c in columns}
        return Table(columns, rows, types)

    def filter(self, predicate: Callable[[dict[str, Any]], bool]) -> Table:
        return Table(self.columns, [r for r in self.rows if predicate(r)], self.column_types)

    def sort(self, column: str, descending: bool = False) -> Table:
        if column not in self.columns:
            raise TinyCSVError("unknown column: " + column)

        def key(row: dict[str, Any]):
            value = row.get(column)
            # None sorts last in both directions.
            return (1, "") if value is None else (0, value)

        rows = sorted(self.rows, key=key, reverse=descending)
        if descending:
            nones = [r for r in self.rows if r.get(column) is None]
            rest = [r for r in rows if r.get(column) is not None]
            rows = rest + nones
        return Table(self.columns, rows, self.column_types)

    def aggregate(self, column: str) -> dict[str, Any]:
        if column not in self.columns:
            raise TinyCSVError("unknown column: " + column)
        ctype = self.column_types.get(column, ColumnType.STR)
        if ctype not in (ColumnType.INT, ColumnType.FLOAT):
            raise TinyCSVError("cannot aggregate non-numeric column: " + column)
        values = [r[column] for r in self.rows if r.get(column) is not None]
        if not values:
            return {"count": 0, "sum": 0, "min": None, "max": None, "mean": None}
        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    def group_by(self, column: str, value_column: str) -> dict[Any, dict[str, Any]]:
        if column not in self.columns:
            raise TinyCSVError("unknown column: " + column)
        groups: dict[Any, list[dict[str, Any]]] = {}
        for r in self.rows:
            groups.setdefault(r.get(column), []).append(r)
        return {
            key: Table(self.columns, rows, self.column_types).aggregate(value_column)
            for key, rows in groups.items()
        }

    # ---- rendering ----
    def to_csv(self, delimiter: str = ",") -> str:
        lines = [delimiter.join(_quote(c, delimiter) for c in self.columns)]
        for r in self.rows:
            lines.append(
                delimiter.join(_quote(_render(r.get(c)), delimiter) for c in self.columns)
            )
        return "\n".join(lines) + "\n"

    def to_aligned(self) -> str:
        cells = [[c for c in self.columns]]
        for r in self.rows:
            cells.append([_render(r.get(c), none="-") for c in self.columns])
        widths = [max(len(row[i]) for row in cells) for i in range(len(self.columns))]
        return "\n".join(
            " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in cells
        )

    def __repr__(self) -> str:
        return "Table(columns=" + repr(self.columns) + ", rows=" + str(len(self.rows)) + ")"


def _render(value: Any, none: str = "") -> str:
    if value is None:
        return none
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _quote(text: str, delimiter: str) -> str:
    if any(ch in text for ch in (delimiter, '"', "\n", "\r")):
        return '"' + text.replace('"', '""') + '"'
    return text
