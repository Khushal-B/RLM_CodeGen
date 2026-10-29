# tinycsv

A small library for reading, filtering and summarising CSV data, implemented
using only the Python standard library.

## Overview

`tinycsv` parses delimited text into typed records, lets a caller filter and
aggregate those records, and renders results back out as text. It is deliberately
split across several concerns that depend on one another, so the parsing layer,
the type-inference layer, the query layer and the rendering layer each have a
clear interface.

## Requirements

### Type inference

The library must infer the type of every column from its values. A column is an
integer column if every non-empty value parses as an integer; a float column if
every non-empty value parses as a float and it is not an integer column; a
boolean column if every non-empty value is one of `true`, `false`, `yes`, `no`,
`1`, `0` (case insensitive) and it is not a numeric column; otherwise a string
column. Empty values become `None` regardless of column type.

Callers must be able to ask for the inferred type of a column by name, and to
coerce a raw string value to a column's inferred type.

### Parsing

The library must parse CSV text into a table. Parsing must support:

- a configurable single-character delimiter, defaulting to a comma
- an optional header row, present by default; when absent, columns are named
  `col_0`, `col_1`, and so on
- double-quoted fields, where a doubled quote inside a quoted field is a literal
  quote, and where delimiters and newlines inside quotes are not special
- rows with fewer fields than the header are padded with empty values; rows with
  more fields than the header raise an error identifying the row number

The parsed table exposes its column names, its row count, and its rows as
dictionaries mapping column name to a type-coerced value. Values must be coerced
using the type-inference rules above.

### Querying

Given a table, the library must support:

- selecting a subset of columns by name, returning a new table
- filtering rows by a predicate over the row dictionary, returning a new table
- sorting rows by one column, ascending or descending, with `None` sorting last
- aggregating a numeric column: count, sum, minimum, maximum and mean. Rows whose
  value is `None` are skipped. Aggregating a non-numeric column raises an error.
- grouping rows by the value of one column and aggregating a numeric column
  within each group, returning a mapping from group value to aggregate

### Rendering

The library must render a table back to text in two forms:

- CSV, with the same quoting rules used when parsing, and a header row
- a fixed-width aligned text table, where each column is padded to the width of
  its widest cell (including the header) and columns are separated by ` | `

Rendering `None` produces an empty string in CSV and the literal `-` in the
aligned form.

### Errors

All errors raised by the library must be instances of a single public exception
type, so a caller can catch everything the library raises with one `except`.

## Public API

Everything a user needs must be importable directly from the top-level package:

    from tinycsv import parse_csv, Table, ColumnType, infer_column_type, TinyCSVError
