# miniorm

A small object-relational mapper over `sqlite3`, with schema definition,
migrations, a query builder and a session/identity map. Standard library only.

## Overview

`miniorm` lets a caller declare model classes with typed fields, generate the SQL
to create their tables, build queries fluently, and read and write model
instances through a session that caches identity.

This is a genuinely layered system: the field/type layer is used by the model
layer, which is used by the schema generator and the query builder, which are
both used by the session. Getting the interfaces between those layers right is
most of the work.

## Requirements

### Fields

The library provides field types: integer, text, real, boolean and datetime.
Every field has an optional default, a nullable flag (default true), a primary
key flag (default false) and a unique flag (default false). A primary key field
is implicitly not nullable.

Each field must be able to report its SQL column type, convert a Python value to
the value stored in SQLite, and convert a stored value back to Python. Booleans
are stored as integers 0/1; datetimes are stored as ISO-8601 strings.

Assigning a value of the wrong type to a field raises an error naming the field.

### Models

A model is declared by subclassing a base class and assigning fields as class
attributes. Declaring a model must:

- collect its fields in declaration order
- derive its table name as the lowercased class name with a trailing `s`, unless
  the class sets an explicit table name
- require exactly one primary key field; zero or more than one raises an error at
  class definition time

Model instances are constructed with keyword arguments matching field names.
Unknown keyword arguments raise an error. Missing arguments take the field
default, or `None` when the field is nullable, or raise when the field is
required and has no default.

Instances expose their field values as a dictionary, and two instances of the
same model with equal primary keys are equal.

### Schema

Given a model, the library generates a `CREATE TABLE` statement including column
types, `PRIMARY KEY`, `NOT NULL` and `UNIQUE` constraints, and a `DROP TABLE`
statement. It must also generate the statements needed to migrate from one
version of a model to another: added columns become `ALTER TABLE ... ADD COLUMN`,
and removed columns are reported as unsupported rather than silently ignored.

### Query builder

A query is built fluently and is immutable — each call returns a new query.
Supported operations: filtering by equality and by the comparison operators
less-than, less-than-or-equal, greater-than, greater-than-or-equal and not-equal;
ordering by a column ascending or descending; limit and offset.

A query compiles to a SQL string with `?` placeholders and a matching ordered
list of parameters. Filtering on a column the model does not declare raises an
error naming the column.

### Session

A session wraps a `sqlite3` connection and provides:

- creating all tables for a set of models
- adding an instance, which inserts it and populates its primary key when the key
  is an auto-assigned integer
- getting an instance by primary key, returning the same Python object on a
  second lookup of the same key (an identity map)
- executing a query and returning model instances
- deleting an instance
- committing and rolling back, where rollback also evicts the identity map

## Errors

Every error raised by the library derives from one public exception type.

## Public API

    from miniorm import (
        Model, Field, IntegerField, TextField, RealField, BooleanField, DateTimeField,
        Query, Session, create_table_sql, drop_table_sql, migration_sql, MiniORMError,
    )
