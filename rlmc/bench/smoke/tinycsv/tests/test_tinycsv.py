"""Upstream-style suite for tinycsv.

Written against the public API described in requirements.md only. The agent
never sees this file; it is copied in at evaluation time, exactly as
NL2Repo-Bench scores against the original upstream suite.
"""

import pytest
from tinycsv import ColumnType, Table, TinyCSVError, infer_column_type, parse_csv

BASIC = "name,age,score,active\nalice,30,91.5,true\nbob,25,78.0,false\ncarol,41,88.25,yes\n"


# --- type inference -------------------------------------------------------

def test_infer_int_column():
    assert infer_column_type(["1", "2", "3"]) == ColumnType.INT


def test_infer_float_column():
    assert infer_column_type(["1.5", "2", "3.25"]) == ColumnType.FLOAT


def test_infer_bool_column():
    assert infer_column_type(["true", "no", "1"]) == ColumnType.BOOL


def test_infer_string_column():
    assert infer_column_type(["alice", "bob"]) == ColumnType.STR


def test_infer_ignores_empty_values():
    assert infer_column_type(["1", "", "3"]) == ColumnType.INT


# --- parsing --------------------------------------------------------------

def test_parse_columns_and_rowcount():
    t = parse_csv(BASIC)
    assert t.columns == ["name", "age", "score", "active"]
    assert t.row_count == 3


def test_parse_coerces_types():
    row = parse_csv(BASIC).rows[0]
    assert row["name"] == "alice"
    assert row["age"] == 30
    assert row["score"] == pytest.approx(91.5)
    assert row["active"] is True


def test_parse_custom_delimiter():
    t = parse_csv("a;b\n1;2\n", delimiter=";")
    assert t.columns == ["a", "b"]
    assert t.rows[0]["b"] == 2


def test_parse_without_header():
    t = parse_csv("1,2\n3,4\n", header=False)
    assert t.columns == ["col_0", "col_1"]
    assert t.row_count == 2


def test_parse_quoted_field_with_delimiter():
    t = parse_csv('name,note\nalice,"hello, world"\n')
    assert t.rows[0]["note"] == "hello, world"


def test_parse_escaped_quote():
    t = parse_csv('name,note\nalice,"say ""hi"""\n')
    assert t.rows[0]["note"] == 'say "hi"'


def test_parse_short_row_is_padded():
    t = parse_csv("a,b,c\n1,2\n")
    assert t.rows[0]["c"] is None


def test_parse_long_row_raises():
    with pytest.raises(TinyCSVError):
        parse_csv("a,b\n1,2,3\n")


def test_empty_value_becomes_none():
    t = parse_csv("a,b\n1,\n")
    assert t.rows[0]["b"] is None


# --- querying -------------------------------------------------------------

def test_select_subset_of_columns():
    t = parse_csv(BASIC).select(["name", "age"])
    assert t.columns == ["name", "age"]
    assert set(t.rows[0]) == {"name", "age"}


def test_filter_rows():
    t = parse_csv(BASIC).filter(lambda r: r["age"] > 26)
    assert t.row_count == 2


def test_sort_ascending_and_descending():
    asc = parse_csv(BASIC).sort("age")
    assert [r["age"] for r in asc.rows] == [25, 30, 41]
    desc = parse_csv(BASIC).sort("age", descending=True)
    assert [r["age"] for r in desc.rows] == [41, 30, 25]


def test_sort_puts_none_last():
    t = parse_csv("a\n2\n\n1\n").sort("a")
    assert t.rows[-1]["a"] is None


def test_aggregate_numeric_column():
    agg = parse_csv(BASIC).aggregate("age")
    assert agg["count"] == 3
    assert agg["sum"] == 96
    assert agg["min"] == 25
    assert agg["max"] == 41
    assert agg["mean"] == pytest.approx(32.0)


def test_aggregate_skips_none():
    agg = parse_csv("a\n1\n\n3\n").aggregate("a")
    assert agg["count"] == 2
    assert agg["sum"] == 4


def test_aggregate_non_numeric_raises():
    with pytest.raises(TinyCSVError):
        parse_csv(BASIC).aggregate("name")


def test_group_by():
    data = "team,pts\nred,10\nblue,4\nred,6\n"
    groups = parse_csv(data).group_by("team", "pts")
    assert groups["red"]["sum"] == 16
    assert groups["blue"]["count"] == 1


# --- rendering ------------------------------------------------------------

def test_render_csv_roundtrips():
    t = parse_csv(BASIC)
    again = parse_csv(t.to_csv())
    assert again.columns == t.columns
    assert again.rows == t.rows


def test_render_csv_quotes_delimiter():
    t = parse_csv('a,b\n1,"x,y"\n')
    assert '"x,y"' in t.to_csv()


def test_render_csv_none_is_empty():
    line = parse_csv("a,b\n1,\n").to_csv().splitlines()[1]
    assert line.endswith(",")


def test_render_aligned_pads_columns():
    out = parse_csv("name,x\nal,1\nbobby,2\n").to_aligned()
    lines = out.splitlines()
    assert " | " in lines[0]
    assert len(lines[1]) == len(lines[2])


def test_render_aligned_none_is_dash():
    assert "-" in parse_csv("a,b\n1,\n").to_aligned()


# --- errors ---------------------------------------------------------------

def test_all_errors_share_one_base_type():
    assert issubclass(TinyCSVError, Exception)
    with pytest.raises(TinyCSVError):
        parse_csv(BASIC).aggregate("nope")


def test_table_is_exported():
    assert isinstance(parse_csv(BASIC), Table)
