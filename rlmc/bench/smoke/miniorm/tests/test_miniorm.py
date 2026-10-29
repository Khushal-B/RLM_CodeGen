"""Upstream-style suite for miniorm."""

import datetime as dt
import sqlite3

import pytest
from miniorm import (
    BooleanField,
    DateTimeField,
    Field,
    IntegerField,
    MiniORMError,
    Model,
    Query,
    RealField,
    Session,
    TextField,
    create_table_sql,
    drop_table_sql,
    migration_sql,
)


class User(Model):
    id = IntegerField(primary_key=True)
    name = TextField()
    score = RealField(default=0.0)
    active = BooleanField(default=True)


# --- fields ---------------------------------------------------------------

def test_field_sql_types():
    assert IntegerField().sql_type == "INTEGER"
    assert TextField().sql_type == "TEXT"
    assert RealField().sql_type == "REAL"
    assert BooleanField().sql_type == "INTEGER"
    assert DateTimeField().sql_type == "TEXT"


def test_field_defaults_and_flags():
    f = TextField(default="x", nullable=False, unique=True)
    assert f.default == "x"
    assert f.nullable is False
    assert f.unique is True


def test_primary_key_is_not_nullable():
    assert IntegerField(primary_key=True).nullable is False


def test_boolean_roundtrip():
    f = BooleanField()
    assert f.to_db(True) == 1
    assert f.from_db(0) is False


def test_datetime_roundtrip():
    f = DateTimeField()
    when = dt.datetime(2026, 5, 4, 3, 2, 1)
    assert f.from_db(f.to_db(when)) == when


def test_wrong_type_raises_naming_field():
    f = IntegerField()
    f.name = "score"
    with pytest.raises(MiniORMError) as exc:
        f.to_db("not an int")
    assert "score" in str(exc.value)


def test_field_is_public_base():
    assert issubclass(IntegerField, Field)


# --- models ---------------------------------------------------------------

def test_fields_collected_in_order():
    assert list(User.fields) == ["id", "name", "score", "active"]


def test_table_name_derived():
    assert User.table_name == "users"


def test_explicit_table_name_wins():
    class Person(Model):
        table_name = "people"
        id = IntegerField(primary_key=True)

    assert Person.table_name == "people"


def test_missing_primary_key_raises():
    with pytest.raises(MiniORMError):
        class NoPK(Model):
            name = TextField()


def test_two_primary_keys_raise():
    with pytest.raises(MiniORMError):
        class TwoPK(Model):
            a = IntegerField(primary_key=True)
            b = IntegerField(primary_key=True)


def test_instance_takes_keyword_args():
    u = User(id=1, name="alice")
    assert u.name == "alice"


def test_unknown_kwarg_raises():
    with pytest.raises(MiniORMError):
        User(id=1, name="a", nope=1)


def test_defaults_applied():
    u = User(id=1, name="a")
    assert u.score == 0.0
    assert u.active is True


def test_to_dict():
    u = User(id=1, name="a")
    assert u.to_dict()["name"] == "a"


def test_equality_by_primary_key():
    assert User(id=1, name="a") == User(id=1, name="b")


# --- schema ---------------------------------------------------------------

def test_create_table_sql_shape():
    sql = create_table_sql(User).upper()
    assert "CREATE TABLE" in sql
    assert "USERS" in sql
    assert "PRIMARY KEY" in sql


def test_create_table_includes_columns():
    sql = create_table_sql(User)
    for col in ("id", "name", "score", "active"):
        assert col in sql


def test_not_null_emitted():
    class Strict(Model):
        id = IntegerField(primary_key=True)
        code = TextField(nullable=False)

    assert "NOT NULL" in create_table_sql(Strict).upper()


def test_unique_emitted():
    class Uniq(Model):
        id = IntegerField(primary_key=True)
        email = TextField(unique=True)

    assert "UNIQUE" in create_table_sql(Uniq).upper()


def test_drop_table_sql():
    assert "DROP TABLE" in drop_table_sql(User).upper()


def test_migration_adds_column():
    stmts = migration_sql(["id", "name"], ["id", "name", "email"], "users")
    assert any("ADD COLUMN" in s.upper() and "email" in s for s in stmts)


def test_migration_reports_removal_as_unsupported():
    with pytest.raises(MiniORMError):
        migration_sql(["id", "name"], ["id"], "users")


# --- query builder --------------------------------------------------------

def test_query_is_immutable():
    q = Query(User)
    q2 = q.filter(name="a")
    assert q is not q2


def test_filter_equality_compiles():
    sql, params = Query(User).filter(name="alice").compile()
    assert "WHERE" in sql.upper()
    assert "?" in sql
    assert params == ["alice"]


def test_comparison_operators():
    sql, params = Query(User).filter(score__gt=5).compile()
    assert ">" in sql
    assert params == [5]


def test_multiple_filters_and_order():
    sql, params = Query(User).filter(name="a").filter(score__lte=9).compile()
    assert params == ["a", 9]


def test_order_by_desc():
    sql, _ = Query(User).order_by("score", descending=True).compile()
    assert "DESC" in sql.upper()


def test_limit_and_offset():
    sql, _ = Query(User).limit(5).offset(10).compile()
    assert "LIMIT" in sql.upper()
    assert "OFFSET" in sql.upper()


def test_unknown_column_raises():
    with pytest.raises(MiniORMError) as exc:
        Query(User).filter(nope=1).compile()
    assert "nope" in str(exc.value)


# --- session --------------------------------------------------------------

def _session():
    s = Session(sqlite3.connect(":memory:"))
    s.create_tables([User])
    return s


def test_create_tables_and_add():
    s = _session()
    s.add(User(name="alice"))
    s.commit()
    assert s.execute(Query(User))[0].name == "alice"


def test_add_populates_primary_key():
    s = _session()
    u = User(name="alice")
    s.add(u)
    assert u.id is not None


def test_get_by_primary_key():
    s = _session()
    u = User(name="alice")
    s.add(u)
    s.commit()
    assert s.get(User, u.id).name == "alice"


def test_identity_map_returns_same_object():
    s = _session()
    u = User(name="alice")
    s.add(u)
    s.commit()
    assert s.get(User, u.id) is s.get(User, u.id)


def test_execute_returns_model_instances():
    s = _session()
    s.add(User(name="alice", score=9.0))
    s.commit()
    rows = s.execute(Query(User).filter(score__gte=5))
    assert isinstance(rows[0], User)


def test_delete_removes_row():
    s = _session()
    u = User(name="alice")
    s.add(u)
    s.commit()
    s.delete(u)
    s.commit()
    assert s.execute(Query(User)) == []


def test_rollback_discards_and_evicts():
    s = _session()
    u = User(name="alice")
    s.add(u)
    s.rollback()
    assert s.execute(Query(User)) == []


def test_boolean_roundtrips_through_db():
    s = _session()
    u = User(name="a", active=False)
    s.add(u)
    s.commit()
    assert s.get(User, u.id).active is False


def test_errors_share_one_base_type():
    assert issubclass(MiniORMError, Exception)
