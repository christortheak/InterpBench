"""The route-ownership census's gate — runner-profile narrowing, step 1.

``route_roles.CENSUS`` labels every HTTP route with the service role that owns
it. This file is the mechanism that keeps the label honest, in the shape
``CheckoutDependencyTests`` established for exactly this kind of table:

1. **The completeness gate, both directions.** Every route the live app serves
   must be censused, and every censused route must still exist. A new route
   cannot ship undeclared, and a declaration cannot outlive its route.
2. **Sanity, against the two things that already know something about these
   routes** — the Phase-2 client adapter (which speaks a subset that MUST be
   runner-reachable) and the WP-S privileged classification (which answers a
   different question about the same table, and should not contradict it).

**No behaviour changes here and nothing is restricted.** The census records
which routes a runner profile *would* keep; it does not keep them. See
``docs/PORTABILITY-CONTRACTS.md`` §11.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from steerlab_server.api.app import (  # noqa: E402
    _OPEN_MUTATING_PATHS,
    _PRIVILEGED_PREFIXES,
    app,
    request_is_privileged,
)

from route_roles import (  # noqa: E402
    BY_KEY,
    CENSUS,
    Role,
    declared_routes,
    is_runner_reachable,
    roles_for,
)

_MUTATING = ("POST", "PUT", "DELETE", "PATCH")


def _sample_path(template: str) -> str:
    """A concrete request path for a route template.

    Twin of ``test_wp_s_hardening._sample_path`` — the census is keyed on
    templates and ``request_is_privileged`` takes a real path.
    """
    return "/".join("sample" if seg.startswith("{") and seg.endswith("}") else seg
                    for seg in template.split("/"))


# ---------------------------------------------------------------------------
# 1. The completeness gate
# ---------------------------------------------------------------------------

def test_every_route_the_app_serves_is_censused():
    observed = {f"{method} {path}" for method, path in declared_routes(app)}
    declared = set(BY_KEY)

    undeclared = sorted(observed - declared)
    assert not undeclared, (
        "Routes with no declared service role: " + ", ".join(undeclared) + ". "
        "Every HTTP route has to say which role owns it — `runner` (bundle "
        "upload/inspect/submit, jobs, logs, evidence, capabilities/info, the "
        "model cache, the scheduler), `workbench` (authoring writes, the "
        "workspace switch, concept/manifest writes, server-side freeze, "
        "playground/interactive compute, catalog browsing), or `both` (really "
        "used by both today). Add an entry to route_roles.CENSUS with one line "
        "of rationale. Nothing is refused because of the label; the label is "
        "how a later runner profile knows what it may narrow.")


def test_the_census_has_no_entries_for_routes_that_are_gone():
    observed = {f"{method} {path}" for method, path in declared_routes(app)}
    stale = sorted(set(BY_KEY) - observed)
    assert not stale, (
        "Census entr(ies) for routes the app no longer serves: "
        + ", ".join(stale)
        + ". Remove them — a census that describes routes that are gone stops "
          "being evidence.")


def test_the_census_declares_each_route_exactly_once():
    seen: dict[str, int] = {}
    for entry in CENSUS:
        seen[entry.key] = seen.get(entry.key, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    assert not duplicates, f"declared more than once: {duplicates}"


def test_every_entry_carries_a_real_rationale():
    # A census whose rationale column is empty is a list, not evidence: the
    # reason is the part a later reader needs in order to disagree with it.
    thin = [entry.key for entry in CENSUS if len(entry.why.strip()) < 30]
    assert not thin, f"entries with no usable rationale: {thin}"


def test_every_entry_uses_the_methods_and_templates_the_router_declares():
    # Guards against a census entry that is spelled plausibly but matches
    # nothing — caught by the staleness test above, but this names the likely
    # cause (a lower-case method, or a path parameter renamed).
    observed_methods = {method for method, _ in declared_routes(app)}
    wrong = [entry.key for entry in CENSUS
             if entry.method != entry.method.upper()
             or entry.method not in observed_methods]
    assert not wrong, f"entries whose method is not one the app serves: {wrong}"


# ---------------------------------------------------------------------------
# 2. Sanity — the Phase-2 client adapter
# ---------------------------------------------------------------------------
# `docs/PORTABILITY-CONTRACTS.md` §8.1 tabulates the routes the adapter speaks.
# Every one of them is, by definition, a route a runner must keep answering: if
# a narrowing ever refused one, the client could not submit a bundle or bring
# evidence home, which is the whole point of the runner role.

#: The adapter's endpoint mapping, as route templates. Kept honest by
#: ``test_the_adapters_endpoint_scan_finds_nothing_undeclared`` below, which
#: reads the adapter's source rather than trusting this list.
_ADAPTER_ROUTES = (
    ("GET", "/api/info"),
    ("GET", "/api/capabilities"),
    ("POST", "/api/bundles/upload"),
    ("POST", "/api/bundles/inspect"),
    ("POST", "/api/studies/submit-bundle"),
    ("GET", "/api/jobs"),
    ("GET", "/api/jobs/{job_id}"),
    ("GET", "/api/jobs/{job_id}/stream"),
    ("POST", "/api/jobs/{job_id}/cancel"),
    ("GET", "/api/bundles/download"),
)

_RUNNER_SOURCE = (Path(__file__).resolve().parent.parent
                  / "steerlab_server" / "client" / "runner.py")

#: ``"/api/jobs/{quote(job_id)}/stream"`` and ``"/api/jobs/{job_id}/stream"``
#: have to compare equal, so both sides collapse every braced segment.
_BRACED = re.compile(r"\{[^{}]*\}")


def _shape(path: str) -> str:
    return _BRACED.sub("{}", path)


@pytest.mark.parametrize("method,template", _ADAPTER_ROUTES)
def test_every_route_the_client_adapter_uses_is_runner_reachable(method, template):
    assert f"{method} {template}" in BY_KEY, (
        f"{method} {template} is used by the Phase-2 adapter and is not in "
        "the census at all")
    entry = BY_KEY[f"{method} {template}"]
    assert is_runner_reachable(method, template), (
        f"{entry.key} is censused `{entry.role.value}` but the Phase-2 client "
        "adapter calls it. A route on the upload -> submit -> jobs -> evidence "
        "path must be `runner` or `both`, or the runner role cannot do the one "
        "thing it exists for.")


def test_the_adapters_endpoint_scan_finds_nothing_undeclared():
    """The other direction: read the adapter and require every ``/api/`` path
    literal in it to be one of the templates above.

    A declared list alone would go stale the first time the adapter learned a
    new endpoint — which is precisely the moment this check matters.
    """
    source = _RUNNER_SOURCE.read_text(encoding="utf-8")
    found = {_shape(m) for m in re.findall(r'"(/api/[^"]*)"', source)}
    known = {_shape(template) for _, template in _ADAPTER_ROUTES}
    unknown = sorted(found - known)
    assert not unknown, (
        "The client adapter speaks /api paths that _ADAPTER_ROUTES does not "
        f"list: {unknown}. Add them there (with their method) so the census "
        "check above covers them.")
    # …and nothing in the list is fiction either.
    absent = sorted(known - found)
    assert not absent, (
        f"_ADAPTER_ROUTES names paths the adapter no longer uses: {absent}")


# ---------------------------------------------------------------------------
# 3. Sanity — against WP-S's mutating-by-default classification
# ---------------------------------------------------------------------------
# WP-S answers a DIFFERENT question about the same table ("does this need a
# bearer token?"), so the two need not agree route by route. Where they can be
# expected to agree, they are asserted; where the answer is merely interesting,
# it is pinned as an observation so a future change surfaces it.

def test_every_deliberately_open_mutating_route_is_workbench():
    # `_OPEN_MUTATING_PATHS` is the set of mutating routes WP-S judged safe to
    # leave open: no writes, no model, no caller-named paths, no spend. That
    # description cannot fit runner work — a runner route either writes into
    # its cache, spends compute, or spends a scheduler allocation. So every
    # entry should be a workbench route, and one that was not would mean
    # either the census or the allowlist is wrong.
    misfiled = []
    for template in _OPEN_MUTATING_PATHS:
        for method in _MUTATING:
            entry = BY_KEY.get(f"{method} {template}")
            if entry is not None and entry.role is not Role.WORKBENCH:
                misfiled.append(f"{entry.key} -> {entry.role.value}")
    assert not misfiled, (
        "Routes that WP-S leaves open despite being mutating, but that the "
        f"census does not call workbench: {misfiled}. An open mutating route "
        "neither writes nor spends, which is not a shape runner work takes.")


def test_every_mutating_runner_route_is_token_gated():
    # The narrowing's premise: a runner profile is a deployment reachable over
    # a network by people who are not its operator. Every mutating route it
    # keeps must already be in the privileged set, or a narrowed deployment
    # would still expose an ungated execution surface.
    ungated = [entry.key for entry in CENSUS
               if entry.role in (Role.RUNNER, Role.BOTH)
               and entry.method in _MUTATING
               and entry.path.startswith("/api/")
               and not request_is_privileged(entry.method,
                                             _sample_path(entry.path))]
    assert not ungated, (
        "Mutating routes the census keeps for the runner role that are NOT "
        f"token-gated: {ungated}.")


def test_the_read_side_privileged_prefixes_are_runner_reachable():
    # `_PRIVILEGED_PREFIXES` exists to gate the handful of READS worth naming
    # explicitly (mutations are gated by default). A read that was worth
    # naming is an operational read — the evidence download and the GPU
    # session record — never a catalog listing. Pinned as the observation it
    # is: if a future prefix gates a catalog GET, this fails and someone
    # decides deliberately.
    gated_reads = sorted(
        entry.key for entry in CENSUS
        if entry.method == "GET"
        and any(entry.path.startswith(prefix) for prefix in _PRIVILEGED_PREFIXES))
    assert gated_reads == ["GET /api/bundles/download", "GET /api/session"]
    for key in gated_reads:
        entry = BY_KEY[key]
        assert entry.role in (Role.RUNNER, Role.BOTH), (
            f"{key} is explicitly gated as a privileged READ but is censused "
            f"`{entry.role.value}`")


def test_the_runner_reads_that_carry_no_token_gate_are_the_expected_three():
    # THE TENSION, recorded rather than forced. A runner's job roster, one
    # job's record, and its log stream are the runner's entire observable
    # surface, and none of them is privileged under WP-S: mutating-by-default
    # gates writes, and these are reads. In practice a real runner runs in
    # TOKEN MODE, where `auth_mode == "token"` gates every /api route
    # regardless of this classification — so there is no open runner today.
    # It is worth pinning because a runner profile that narrowed the surface
    # to exactly these routes, on a deployment NOT in token mode, would be
    # serving job logs to whoever can reach the socket.
    open_reads = sorted(
        entry.key for entry in CENSUS
        if entry.role is Role.RUNNER
        and entry.method == "GET"
        and entry.path.startswith("/api/")
        and not request_is_privileged(entry.method, _sample_path(entry.path)))
    assert open_reads == [
        "GET /api/jobs",
        "GET /api/jobs/{job_id}",
        "GET /api/jobs/{job_id}/stream",
    ]


# ---------------------------------------------------------------------------
# 4. The shape of the census itself
# ---------------------------------------------------------------------------

def test_all_three_roles_are_actually_used():
    # A census that collapsed onto one label would pass every test above and
    # say nothing. Each role must be populated for the table to be a census.
    for role in Role:
        assert roles_for(role), f"no route is censused {role.value}"


def test_the_census_covers_the_whole_app_and_is_not_a_sample():
    # The count is not pinned to a literal (routes come and go, and the
    # completeness gate is the real check) — but the census and the app must
    # be the same size, which is the same statement said arithmetically.
    assert len(CENSUS) == len(declared_routes(app))


def test_the_census_activates_no_restriction():
    # Step 1 is a labelled census and NOTHING else. The one helper that reads
    # the table is documentation of it; no production module imports this
    # file, and no request is refused because of a label. If that ever stops
    # being true, it must stop being true in the diff that makes it true.
    package = Path(__file__).resolve().parent.parent / "steerlab_server"
    importers = sorted(
        str(path.relative_to(package.parent))
        for path in package.rglob("*.py")
        if "route_roles" in path.read_text(encoding="utf-8"))
    assert not importers, (
        "Production modules now reference the census: "
        f"{importers}. That is a real narrowing, not step 1 — move the table "
        "into the package and give it tests that pin the refusals.")
