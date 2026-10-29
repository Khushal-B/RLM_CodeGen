"""Type inference and coercion."""

from __future__ import annotations

from enum import Enum


class TinyCSVError(Exception):
    """Base class for every error tinycsv raises."""


class ColumnType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"


_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


def _is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def infer_column_type(values: list[str]) -> ColumnType:
    """Infer a column's type from its raw string values."""
    present = [v for v in values if v is not None and str(v).strip() != ""]
    if not present:
        return ColumnType.STR
    if all(_is_int(v) for v in present):
        return ColumnType.INT
    if all(_is_float(v) for v in present):
        return ColumnType.FLOAT
    if all(str(v).strip().lower() in (_TRUE | _FALSE) for v in present):
        return ColumnType.BOOL
    return ColumnType.STR


def coerce(value: str | None, column_type: ColumnType):
    """Coerce a raw value to a column's inferred type."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value)
    try:
        if column_type is ColumnType.INT:
            return int(text)
        if column_type is ColumnType.FLOAT:
            return float(text)
        if column_type is ColumnType.BOOL:
            low = text.strip().lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            raise TinyCSVError("cannot coerce " + text + " to bool")
        return text
    except (ValueError, TypeError) as exc:
        raise TinyCSVError("cannot coerce " + text + " to " + column_type.value) from exc
