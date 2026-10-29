"""Upstream-style suite for eventbus."""

import pytest
from eventbus import (
    Event,
    EventBus,
    EventBusError,
    PublishResult,
    Subscription,
    TimingMiddleware,
    matches,
    specificity,
)

# --- event model ----------------------------------------------------------

def test_event_has_topic_and_payload():
    e = Event(topic="user.created", payload={"id": 1}, sequence=1)
    assert e.topic == "user.created"
    assert e.payload == {"id": 1}
    assert e.sequence == 1


def test_event_has_timestamp():
    e = Event(topic="a", payload=None, sequence=1)
    assert isinstance(e.timestamp, float)


def test_event_is_immutable():
    e = Event(topic="a", payload=None, sequence=1)
    with pytest.raises(Exception):
        e.topic = "b"


def test_event_equality():
    a = Event(topic="a", payload=1, sequence=3)
    b = Event(topic="a", payload=1, sequence=3)
    assert a == b


def test_event_dict_roundtrip():
    e = Event(topic="a.b", payload={"x": 1}, sequence=7)
    again = Event.from_dict(e.to_dict())
    assert again.topic == e.topic
    assert again.payload == e.payload
    assert again.sequence == e.sequence
    assert again.timestamp == e.timestamp


# --- topic matching -------------------------------------------------------

def test_exact_match():
    assert matches("user.created", "user.created")
    assert not matches("user.created", "user.deleted")


def test_star_matches_one_segment():
    assert matches("user.*", "user.created")
    assert not matches("user.*", "user.profile.updated")


def test_hash_matches_trailing_segments():
    assert matches("user.#", "user.created")
    assert matches("user.#", "user.profile.updated")


def test_hash_must_be_last():
    with pytest.raises(EventBusError):
        matches("user.#.created", "user.a.created")


def test_specificity_ordering():
    assert specificity("user.created") > specificity("user.*")
    assert specificity("user.*") > specificity("user.#")


def test_specificity_prefers_more_segments():
    assert specificity("a.b.c") > specificity("a.b")


# --- middleware -----------------------------------------------------------

def test_middleware_runs_in_registration_order():
    order = []

    def outer(event, nxt):
        order.append("outer-in")
        r = nxt(event)
        order.append("outer-out")
        return r

    def inner(event, nxt):
        order.append("inner")
        return nxt(event)

    bus = EventBus()
    bus.use(outer)
    bus.use(inner)
    bus.subscribe("a", lambda e: order.append("handler"))
    bus.publish("a", None)
    assert order == ["outer-in", "inner", "handler", "outer-out"]


def test_middleware_can_short_circuit():
    calls = []
    bus = EventBus()
    bus.use(lambda event, nxt: None)
    bus.subscribe("a", lambda e: calls.append(e))
    bus.publish("a", None)
    assert calls == []


def test_timing_middleware_records_calls():
    tm = TimingMiddleware()
    bus = EventBus()
    bus.use(tm)
    bus.subscribe("a", lambda e: None)
    bus.publish("a", None)
    assert len(tm.records) == 1
    assert tm.records[0]["topic"] == "a"
    assert tm.records[0]["duration"] >= 0


# --- the bus --------------------------------------------------------------

def test_subscribe_returns_handle():
    bus = EventBus()
    handle = bus.subscribe("a", lambda e: None)
    assert isinstance(handle, Subscription)


def test_publish_invokes_matching_handler():
    seen = []
    bus = EventBus()
    bus.subscribe("user.*", seen.append)
    bus.publish("user.created", {"id": 1})
    assert len(seen) == 1
    assert seen[0].payload == {"id": 1}


def test_publish_skips_non_matching():
    seen = []
    bus = EventBus()
    bus.subscribe("order.*", seen.append)
    bus.publish("user.created", None)
    assert seen == []


def test_sequence_numbers_increase():
    bus = EventBus()
    first = bus.publish("a", None)
    second = bus.publish("a", None)
    assert second.sequence > first.sequence


def test_publish_returns_result():
    bus = EventBus()
    bus.subscribe("a", lambda e: None)
    r = bus.publish("a", None)
    assert isinstance(r, PublishResult)
    assert r.matched == 1
    assert r.succeeded == 1
    assert r.failed == 0


def test_more_specific_pattern_runs_first():
    order = []
    bus = EventBus()
    bus.subscribe("user.#", lambda e: order.append("hash"))
    bus.subscribe("user.created", lambda e: order.append("exact"))
    bus.publish("user.created", None)
    assert order == ["exact", "hash"]


def test_unsubscribe_stops_delivery():
    seen = []
    bus = EventBus()
    handle = bus.subscribe("a", seen.append)
    bus.unsubscribe(handle)
    bus.publish("a", None)
    assert seen == []


def test_unsubscribe_twice_is_noop():
    bus = EventBus()
    handle = bus.subscribe("a", lambda e: None)
    bus.unsubscribe(handle)
    bus.unsubscribe(handle)


def test_same_handler_twice_registers_twice():
    seen = []
    bus = EventBus()
    bus.subscribe("a", seen.append)
    bus.subscribe("a", seen.append)
    bus.publish("a", None)
    assert len(seen) == 2


def test_failing_handler_is_recorded_not_raised():
    bus = EventBus()

    def boom(event):
        raise ValueError("nope")

    bus.subscribe("a", boom)
    r = bus.publish("a", None)
    assert r.failed == 1
    assert r.succeeded == 0


def test_failure_does_not_block_other_handlers():
    seen = []
    bus = EventBus()
    bus.subscribe("a", lambda e: (_ for _ in ()).throw(ValueError()))
    bus.subscribe("a", seen.append)
    bus.publish("a", None)
    assert len(seen) == 1


def test_retry_reattempts_failing_handler():
    attempts = []

    def flaky(event):
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("later")

    bus = EventBus(max_attempts=3)
    bus.subscribe("a", flaky)
    r = bus.publish("a", None)
    assert len(attempts) == 3
    assert r.succeeded == 1


def test_default_is_no_retry():
    attempts = []
    bus = EventBus()
    bus.subscribe("a", lambda e: (attempts.append(1), (_ for _ in ()).throw(ValueError()))[1])
    bus.publish("a", None)
    assert len(attempts) == 1


def test_audit_trail_records_publishes():
    bus = EventBus()
    bus.subscribe("a", lambda e: None)
    bus.publish("a", 1)
    bus.publish("a", 2)
    assert len(bus.audit()) == 2


def test_audit_trail_is_ordered():
    bus = EventBus()
    bus.publish("a", None)
    bus.publish("b", None)
    assert [e.topic for e in bus.audit()] == ["a", "b"]


def test_audit_can_be_cleared():
    bus = EventBus()
    bus.publish("a", None)
    bus.clear_audit()
    assert bus.audit() == []


def test_errors_share_one_base_type():
    assert issubclass(EventBusError, Exception)
