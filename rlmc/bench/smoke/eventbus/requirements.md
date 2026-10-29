# eventbus

An in-process publish/subscribe event bus with middleware, retries and a
recorded audit trail. Standard library only.

## Overview

`eventbus` routes typed events from publishers to subscribers. Handlers can be
registered for exact topics or wildcard patterns, wrapped in middleware, retried
on failure, and every dispatch is recorded so a caller can inspect what happened.

The library has four clearly separable concerns — the event model, topic
matching, the middleware chain, and the bus itself — which depend on one another
in that order.

## Requirements

### Event model

An event carries a topic, a payload, a monotonically increasing sequence number
assigned at publish time, and a timestamp. Events are immutable: attempting to
mutate one raises an error. Two events are equal when their topic, payload and
sequence number are equal.

An event can be converted to a plain dictionary and reconstructed from one, and
the round trip must preserve all four fields.

### Topic matching

Topics are dot-separated, e.g. `user.created` or `order.line.added`. Subscribers
register a pattern which may contain wildcards:

- `*` matches exactly one segment
- `#` matches one or more trailing segments, and may only appear as the last
  segment of a pattern
- a pattern with no wildcards matches only that exact topic

So `user.*` matches `user.created` but not `user.profile.updated`; `user.#`
matches both. A pattern using `#` anywhere other than the final position raises
an error.

The matcher must also report the specificity of a pattern, so that more specific
subscriptions can be dispatched first: an exact pattern is more specific than one
containing `*`, which is more specific than one containing `#`. Between two
patterns of the same wildcard class, the one with more segments is more specific.

### Middleware

Middleware wraps handler invocation. Each middleware receives the event and a
callable representing the rest of the chain, and may inspect the event, short
circuit by not calling the rest of the chain, or transform the result.

Middleware runs in registration order — the first registered is the outermost.
A middleware that raises propagates the error to the bus.

The library must ship one built-in middleware that collects the topic and
duration of every call it wraps.

### The bus

The bus must support:

- subscribing a handler to a pattern, returning a subscription handle that can be
  used to unsubscribe
- publishing an event to a topic, which assigns the next sequence number and
  invokes every matching handler, most specific pattern first
- retrying a handler that raises, up to a configurable maximum number of attempts
  (default 1, meaning no retry); a handler that fails every attempt is recorded
  as failed and does not prevent other handlers from running
- publishing returns a result object reporting how many handlers matched, how
  many succeeded, how many failed, and the sequence number of the event
- an audit trail of every publish, retrievable in order, and clearable

Subscribing the same handler to the same pattern twice registers it twice.
Unsubscribing a handle that was already removed is a no-op, not an error.

### Errors

Every error the library raises must derive from one public exception type.

## Public API

    from eventbus import (
        Event, EventBus, Subscription, PublishResult,
        matches, specificity, TimingMiddleware, EventBusError,
    )
