"""parse_csv: text -> Table."""

from __future__ import annotations

from tinycsv.parser import split_rows
from tinycsv.table import Table
from tinycsv.types import ColumnType, TinyCSVError, coerce, infer_column_type


def parse_csv(text: str, delimiter: str = ",", header: bool = True) -> Table:
    """Parse CSV text into a typed Table."""
    raw = split_rows(text, delimiter)
    if not raw:
        return Table([], [], {})

    if header:
        columns = raw[0]
        body = raw[1:]
    else:
        columns = ["col_" + str(i) for i in range(len(raw[0]))]
        body = raw

    for index, row in enumerate(body):
        if len(row) > len(columns):
            raise TinyCSVError(
                "row " + str(index + 1) + " has " + str(len(row))
                + " fields but header declares " + str(len(columns))
            )

    padded = [list(row) + [""] * (len(columns) - len(row)) for row in body]
    types = {
        name: infer_column_type([row[i] for row in padded])
        for i, name in enumerate(columns)
    }
    rows = [
        {name: coerce(row[i], types[name]) for i, name in enumerate(columns)}
        for row in padded
    ]
    return Table(columns, rows, types)
