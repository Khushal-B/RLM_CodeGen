"""CSV parsing."""

from __future__ import annotations

from tinycsv.types import ColumnType, TinyCSVError, coerce, infer_column_type


def split_rows(text: str, delimiter: str = ",") -> list[list[str]]:
    """Split CSV text into rows of raw fields, honouring quotes."""
    rows: list[list[str]] = []
    field: list[str] = []
    row: list[str] = []
    in_quotes = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                in_quotes = False
                i += 1
                continue
            field.append(ch)
            i += 1
            continue
        if ch == '"':
            in_quotes = True
            i += 1
            continue
        if ch == delimiter:
            row.append("".join(field))
            field = []
            i += 1
            continue
        if ch in "\r\n":
            if ch == "\r" and i + 1 < n and text[i + 1] == "\n":
                i += 1
            row.append("".join(field))
            field = []
            rows.append(row)
            row = []
            i += 1
            continue
        field.append(ch)
        i += 1
    if field or row:
        row.append("".join(field))
        rows.append(row)
    # Drop only a TRAILING empty row (the artifact of a final newline). A blank
    # line in the middle of the file is a real record with empty fields.
    while rows and rows[-1] == [""]:
        rows.pop()
    return rows
