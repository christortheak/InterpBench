"""The cross-platform client's own modules (portability program, Phase 2).

``steerlab_server.client_cli`` (Phase 1b) authors the LOCAL workspace and talks
to nothing. This package holds the pieces that talk to a **runner** — an engine
serving ``steerlab_server.api``, over HTTP — starting with
:mod:`steerlab_server.client.runner`.

Deliberately a package beside ``client_cli`` rather than a rewrite of it:
Phase 1b's module is a tested contract (its verb table, its envelope, its
import graph), and the runner adapter is additive to all three. Nothing was
moved here.

**Nothing in this package is imported at ``client_cli`` import time.** The
light-install guarantee (``PORTABILITY-CONTRACTS.md`` §7) is measured by an
out-of-process assertion that importing ``client_cli`` and running ``--help``
pulls nothing third-party at all; the ``runner`` family imports this package
lazily, inside the verb, exactly as every engine module is imported there.
"""
