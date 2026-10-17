# memento

A memory-engineered MCP server: a SQLite-backed semantic graph with nodes,
features, typed edges, and a wake/sleep lifecycle. The `dream` tool runs
sleep-cycle maintenance and synthesizes novel edges via six mechanisms
(transitive composition, feature recombination, analogy, schema abstraction,
journal co-occurrence, and emotion-weighted association), all budget-shared,
cycle-checked, and seed/temperature/dry_run controllable.

## Files

- `semcp_0.1.py` — the server. Install with `pip install mcp`, run with
  `MCP_MEMORY_DB=<path> python semcp_0.1.py` (stdio transport).
- `TODO.md` — design spec for the dream synthesis mechanisms (implemented),
  including per-mechanism weights, behaviour notes, and known limitations.
- `test_dream_synthesis.py` — functional test suite. Stubs `mcp` if absent,
  drives the real module against a temp DB, and verifies all six mechanisms,
  determinism, dry_run isolation, cycle safety, and journal provenance:

  ```
  python3 test_dream_synthesis.py
  ```

## Notable design choice

Journal co-occurrence is frequency-weighted with a soft semantic boost
rather than a hard semantic gate — repeated co-occurrence strengthens a
single edge (`1−e^−count`), confidence scales with count, and pairs with
features get a `0.6 + 0.4·cosine` factor. Featureless pairs are exempt so
the mechanism keeps its unique ability to link nodes no other mechanism
can reach. See TODO.md §8–9 for rationale.
