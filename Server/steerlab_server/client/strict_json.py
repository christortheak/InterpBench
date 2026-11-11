"""JSON that every reader accepts: a number that is not finite becomes null.

Python's ``json`` writes NaN and infinity as bare words. No strict reader
takes them (the Mac's does not, and neither does an engine's request parser),
and Python's own strict mode raises on them. A lens record can hold such a
number: a qualification stores every check's measured values whether it
passed or not, and a check that could not be scored records NaN.

What leaves this machine, or crosses to the Mac, goes through :func:`strict`.
"""
import math


def strict(value):
    """``(value, count)``: ``value`` with every number that is not finite
    replaced by None, and how many there were. The input is not changed."""
    count = 0

    def walk(item):
        nonlocal count
        if isinstance(item, float) and not math.isfinite(item):
            count += 1
            return None
        if isinstance(item, dict):
            return {key: walk(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [walk(entry) for entry in item]
        return item

    return walk(value), count
