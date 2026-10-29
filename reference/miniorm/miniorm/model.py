"""Model declaration."""

from __future__ import annotations

from typing import Any

from miniorm.fields import Field, MiniORMError


class ModelMeta(type):
    """Collects fields, derives the table name, validates the primary key."""

    def __new__(mcls, name: str, bases: tuple, namespace: dict) -> ModelMeta:
        fields: dict[str, Field] = {}

        # Inherited fields first, in declaration order.
        for base in bases:
            fields.update(getattr(base, "fields", {}) or {})

        for key, value in list(namespace.items()):
            if isinstance(value, Field):
                value.name = key
                fields[key] = value
                namespace.pop(key)

        namespace["fields"] = fields
        cls = super().__new__(mcls, name, bases, namespace)

        # The abstract base itself declares no fields and is exempt.
        if not fields:
            if name != "Model":
                raise MiniORMError(
                    "model '" + name + "' declares no primary key field"
                )
            return cls

        pks = [n for n, f in fields.items() if f.primary_key]
        if len(pks) != 1:
            raise MiniORMError(
                "model '" + name + "' must declare exactly one primary key, found "
                + str(len(pks))
            )
        cls.primary_key = pks[0]

        if "table_name" not in namespace:
            cls.table_name = name.lower() + "s"
        return cls


class Model(metaclass=ModelMeta):
    """Base class for declared models."""

    fields: dict[str, Field] = {}
    table_name: str = ""
    primary_key: str = ""

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - set(self.fields)
        if unknown:
            raise MiniORMError(
                "unknown field(s) for " + type(self).__name__ + ": "
                + ", ".join(sorted(unknown))
            )
        for name, field in self.fields.items():
            if name in kwargs:
                value = kwargs[name]
                field.check(value)
            elif field.default is not None:
                value = field.default
            elif field.nullable or field.primary_key:
                # A primary key may be assigned by the database on insert.
                value = None
            else:
                raise MiniORMError(
                    "field '" + name + "' is required and has no default"
                )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.fields}

    def __eq__(self, other: Any) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return getattr(self, self.primary_key) == getattr(other, other.primary_key)

    def __hash__(self) -> int:
        return hash((type(self).__name__, getattr(self, self.primary_key)))

    def __repr__(self) -> str:
        inner = ", ".join(
            name + "=" + repr(getattr(self, name)) for name in self.fields
        )
        return type(self).__name__ + "(" + inner + ")"
