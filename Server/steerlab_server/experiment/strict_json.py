"""JSON that every reader accepts: a number that is not finite becomes null.

Python's ``json`` writes NaN and infinity as bare words. No strict reader
takes them (the Mac's does not, and neither does an engine's request parser),
and Python's own strict mode raises on them. A lens record can hold such a
number: a qualification stores every check's measured values whether it
passed or not, and a check that could not be scored records NaN.

What leaves this machine, or crosses to the Mac, goes through :func:`strict`.
An engine that is sent such a number anyway refuses it by name
(:func:`first_non_finite`), and reads its own stored records through
:func:`strict` when it compares them with what was sent. This module is the
one place that says which numbers those are.

It lives here because an import's answer needs it too, and the client depends
on this package, never the reverse.
"""
import math


def _not_finite(item):
    return isinstance(item, float) and not math.isfinite(item)


def first_non_finite(value, where='the value'):
    """Where ``value`` first holds a number that is not finite, as a path from
    ``where`` (``where.key[2].name``), or None when it holds none."""
    if _not_finite(value):
        return where
    if isinstance(value, dict):
        for name, item in value.items():
            found = first_non_finite(item, f'{where}.{name}')
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            found = first_non_finite(item, f'{where}[{position}]')
            if found:
                return found
    return None


def strict(value):
    """``(value, count)``: ``value`` with every number that is not finite
    replaced by None, and how many there were. The input is not changed."""
    count = 0

    def walk(item):
        nonlocal count
        if _not_finite(item):
            count += 1
            return None
        if isinstance(item, dict):
            return {key: walk(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [walk(entry) for entry in item]
        return item

    return walk(value), count
