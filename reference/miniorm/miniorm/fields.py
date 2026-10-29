"""Field types and value conversion."""

from __future__ import annotations

import datetime as dt
from typing import Any


class MiniORMError(Exception):
    """Base class for every error miniorm raises."""


class Field:
    """A typed column."""

    sql_type = "TEXT"
    python_types: tuple[type, ...] = (str,)

    def __init__(
        self,
        default: Any = None,
        nullable: bool = True,
        primary_key: bool = False,
        unique: bool = False,
    ) -> None:
        self.default = default
        self.primary_key = primary_key
        # A primary key is implicitly NOT NULL.
        self.nullable = False if primary_key else nullable
        self.unique = unique
        self.name: str = ""

    # ---- conversion ----
    def check(self, value: Any) -> None:
        """Raise if ``value`` is not valid for this field, naming the field."""
        if value is None:
            return
        # bool is a subclass of int, so it must be rejected explicitly wherever
        # bool is not itself the declared type.
        if isinstance(value, bool) and bool not in self.python_types:
            ok = False
        else:
            ok = isinstance(value, self.python_types)
        if not ok:
            raise MiniORMError(
                "field '" + (self.name or "?") + "' expects "
                + "/".join(t.__name__ for t in self.python_types)
                + ", got " + type(value).__name__
            )

    def to_db(self, value: Any) -> Any:
        self.check(value)
        return value

    def from_db(self, value: Any) -> Any:
        return value

    def __repr__(self) -> str:
        return type(self).__name__ + "(name=" + repr(self.name) + ")"


class IntegerField(Field):
    sql_type = "INTEGER"
    python_types = (int,)

    def check(self, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise MiniORMError(
                "field '" + (self.name or "?") + "' expects int, got "
                + type(value).__name__
            )


class TextField(Field):
    sql_type = "TEXT"
    python_types = (str,)


class RealField(Field):
    sql_type = "REAL"
    python_types = (int, float)

    def check(self, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MiniORMError(
                "field '" + (self.name or "?") + "' expects float, got "
                + type(value).__name__
            )

    def to_db(self, value: Any) -> Any:
        self.check(value)
        return None if value is None else float(value)


class BooleanField(Field):
    sql_type = "INTEGER"
    python_types = (bool,)

    def check(self, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, bool):
            raise MiniORMError(
                "field '" + (self.name or "?") + "' expects bool, got "
                + type(value).__name__
            )

    def to_db(self, value: Any) -> Any:
        self.check(value)
        return None if value is None else int(value)

    def from_db(self, value: Any) -> Any:
        return None if value is None else bool(value)


class DateTimeField(Field):
    sql_type = "TEXT"
    python_types = (dt.datetime,)

    def check(self, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, dt.datetime):
            raise MiniORMError(
                "field '" + (self.name or "?") + "' expects datetime, got "
                + type(value).__name__
            )

    def to_db(self, value: Any) -> Any:
        self.check(value)
        return None if value is None else value.isoformat()

    def from_db(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dt.datetime):
            return value
        return dt.datetime.fromisoformat(str(value))
