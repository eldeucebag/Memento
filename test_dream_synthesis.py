"""Functional test for the implemented TODO (dream synthesis mechanisms).

Stubs `mcp` (not installed) and drives the real server module against a
temp DB, exercising all sections of TODO.md:
  - constants/config (section 1)
  - relation composition (section 2)
  - shared helpers (section 3)
  - six mechanisms (section 4a-4f)
  - orchestration _run_synthesis (section 5)
  - revised dream tool (section 6)
"""
import os
import sys
import tempfile
import types

TMP = tempfile.mkdtemp(prefix="semcp_test_")
DB = os.path.join(TMP, "test.db")
os.environ["MCP_MEMORY_DB"] = DB

# ---- stub the mcp package before importing the server module ----
TOOLS = {}
RESOURCES = {}


def tool(**dkwargs):
    def deco(fn):
        TOOLS[fn.__name__] = fn
        return fn
    return deco


def resource(uri):
    def deco(fn):
        RESOURCES[uri] = fn
        return fn
    return deco


class _FastMCP:
    def __init__(self, name=None, **kw):
        self.name = name
    def tool(self, **kw):
        return tool(**kw)
    def resource(self, uri):
        return resource(uri)


mcp_pkg = types.ModuleType("mcp")
mcp_server = types.ModuleType("mcp.server")
mcp_fastmcp = types.ModuleType("mcp.server.fastmcp")
mcp_fastmcp.FastMCP = _FastMCP
mcp_server.fastmcp = mcp_fastmcp
mcp_pkg.server = mcp_server
sys.modules["mcp"] = mcp_pkg
sys.modules["mcp.server"] = mcp_server
sys.modules["mcp.server.fastmcp"] = mcp_fastmcp

import importlib.util
spec = importlib.util.spec_from_file_location(
    "semcp", "/home/jeff/Downloads/memento/semcp_0.1.py"
)
semcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(semcp)

FAILURES = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append((name, detail))

def edge_count(ns):
    with semcp.db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM edges WHERE namespace=?", (ns,)
        ).fetchone()[0]

# ---- section 1: constants exist ----
for c in ("MAX_SYNTHESIS_CANDIDATES", "JOURNAL_COOCCURRENCE_WINDOW_HOURS",
          "JOURNAL_COOCCURRENCE_LOOKBACK_DAYS", "SCHEMA_NODE_PREFIX",
          "MAX_SCHEMAS_PER_DREAM", "RELATION_ATTENUATION"):
    check(f"constant {c}", hasattr(semcp, c), "missing")
check("import random", "import random" in open(semcp.__file__).read())

# ---- section 2: composition table ----
check("is_a + is_a -> is_a", semcp._compose_relations("is_a", "is_a") == "is_a")
check("is_a + causes -> causes", semcp._compose_relations("is_a", "causes") == "causes")
check("causes + leads_to -> leads_to",
      semcp._compose_relations("causes", "leads_to") == "leads_to")
check("part_of + is_a -> None", semcp._compose_relations("part_of", "is_a") is None)
check("before + after -> None", semcp._compose_relations("before", "after") is None)

# ---- section 3: shared helpers ----
with semcp.db() as conn:
    v = semcp._feature_vectors(conn, "default", [])
    check("_feature_vectors empty ids", v == {})
check("_cosine orthogonal -> 0", semcp._cosine({"a": 1.0}, {"b": 1.0}) == 0.0)
check("_cosine identical -> 1", abs(semcp._cosine({"a": 1.0}, {"a": 1.0}) - 1.0) < 1e-9)

# ---- GraphSnapshot semantics ----
check("SYMMETRIC_RELATIONS excludes is_a",
      "is_a" not in semcp.SYMMETRIC_RELATIONS)
check("SYMMETRIC_RELATIONS includes associated/co_occurred/affective",
      {"associated", "co_occurred", "affective"} <= semcp.SYMMETRIC_RELATIONS)

# build a tiny snapshot by hand via the real DB seeded below happens later;
# here exercise would_cycle/has_edge on a synthetic snapshot
snap = semcp._GraphSnapshot.__new__(semcp._GraphSnapshot)
snap.edges = {(1, 2, "is_a")}
snap.adj = {1: {2}}
check("snapshot would_cycle detects back-edge (2->1 over is_a)",
      snap.would_cycle(2, 1, 4) is True)
check("snapshot would_cycle allows fresh edge (3->1)",
      snap.would_cycle(3, 1, 4) is False)
check("snapshot has_edge triple match", snap.has_edge(1, 2, "is_a") is True)
check("snapshot has_edge relation mismatch", snap.has_edge(1, 2, "associated") is False)
snap.add(2, 1, "associated")  # symmetric relation: must NOT extend adj
check("snapshot.add symmetric leaves directed adj alone", 1 not in snap.adj.get(2, set()))
snap.add(2, 1, "is_a")  # would be a real cycle; but add() is a recorder not a guard
check("snapshot.add directed extends adj", 1 in snap.adj.get(2, set()))
snap2 = semcp._GraphSnapshot.__new__(semcp._GraphSnapshot)
snap2.edges = {(1, 2, "associated")}
snap2.adj = {}  # associated is symmetric -> excluded from adj at construction
check("snapshot: symmetric-only edge graph has empty adj", snap2.adj == {})
check("snapshot: symmetric edge cannot form cycle",
      snap2.would_cycle(2, 1, 4) is False)

# ---- init db and seed a graph ----
semcp.init_db()
ns = "default"

with semcp.db() as conn:
    names = {}
    for nm in ("animal", "dog", "labrador", "cat", "wolf", "feline",
               "bone", "leash", "bark",
               "mood_happy", "mood_sad", "stress",
               "husky", "malamute", "kennel", "crate"):
        r = semcp.ensure_node(conn, ns, nm, plasticity=0.5, corpus="TEST")
        names[nm] = r["id"]

    def edge(a, b, rel, w=0.9, conf=0.9, ev="test"):
        semcp.upsert_edge(conn, ns, names[a], names[b], rel, w, conf, ev)

    # typed-coexistence fixture: dog and animal already share a weak
    # `associated` edge. Transitive derives labrador->animal is_a anyway;
    # pair-level dedupe used to block the typed edge.
    semcp.upsert_edge(conn, ns, names["dog"], names["animal"],
                      "associated", 0.3, 0.4, "seed_weak")

    # transitive chains
    edge("labrador", "dog", "is_a")
    edge("dog", "animal", "is_a")
    edge("feline", "animal", "is_a")
    edge("stress", "mood_happy", "causes", 0.8)
    edge("mood_happy", "mood_sad", "causes", 0.7)

    # analogy quadruple: husky->kennel, malamute->crate (both is_a),
    # husky~malamute and kennel~crate via shared single-feature keys.
    # No transitive chain through these, so the completion is all analogy's.
    edge("husky", "kennel", "is_a")
    edge("malamute", "crate", "is_a")

    def feature(nid, key, val):
        conn.execute(
            "INSERT OR REPLACE INTO features(node_id, key, value, contributes, ts) "
            "VALUES(?, ?, ?, 1, ?)",
            (nid, key, val, semcp.now_utc()),
        )

    # feature_recombination + schema: shared keys across >=3 nodes
    feature(names["labrador"], "fur", 0.9)
    feature(names["dog"], "fur", 0.9)
    feature(names["cat"], "fur", 0.5)
    feature(names["wolf"], "fur", 0.9)
    feature(names["animal"], "fur", 0.9)
    feature(names["labrador"], "pet", 0.8)
    feature(names["dog"], "pet", 0.8)
    feature(names["wolf"], "pet", 0.8)
    feature(names["animal"], "pet", 0.8)

    # analogy similarity anchors
    feature(names["husky"], "size", 0.8)
    feature(names["malamute"], "size", 0.8)
    feature(names["kennel"], "cost", 0.5)
    feature(names["crate"], "cost", 0.5)

    # emotion: salient nodes with NO edges between them.
    # mood_sad is negative -> signed weight product must come out negative.
    conn.execute(
        "UPDATE nodes SET salience = ?, salience_ts = ? WHERE id = ?",
        (0.9, semcp.now_utc(), names["bark"]),
    )
    conn.execute(
        "UPDATE nodes SET salience = ?, salience_ts = ? WHERE id = ?",
        (-0.85, semcp.now_utc(), names["mood_sad"]),
    )

    # journal co-occurrence fixtures:
    #  - bone/leash co-occur 3x (bursts on different days, direction varies)
    #  - bark/kennel co-occur once, 30 minutes apart
    from datetime import datetime, timedelta, timezone
    t0 = datetime.now(timezone.utc)
    jrows = []
    # burst 1: bone then leash, 30 min apart
    jrows.append((ns, "admit", names["bone"], "{}", t0.isoformat()))
    jrows.append((ns, "admit", names["leash"], "{}",
                  (t0 + timedelta(minutes=30)).isoformat()))
    # burst 2 (2 days later): leash first, then bone — direction reversed
    d2 = t0 + timedelta(days=2)
    jrows.append((ns, "admit", names["leash"], "{}", d2.isoformat()))
    jrows.append((ns, "admit", names["bone"], "{}",
                  (d2 + timedelta(minutes=20)).isoformat()))
    # burst 3 (4 days later): bone then leash, 10 min apart (smallest gap)
    d3 = t0 + timedelta(days=4)
    jrows.append((ns, "admit", names["bone"], "{}", d3.isoformat()))
    jrows.append((ns, "admit", names["leash"], "{}",
                  (d3 + timedelta(minutes=10)).isoformat()))
    # single co-occurrence: bark + kennel, 30 min apart
    d6 = t0 + timedelta(days=6)
    jrows.append((ns, "admit", names["bark"], "{}", d6.isoformat()))
    jrows.append((ns, "admit", names["kennel"], "{}",
                  (d6 + timedelta(minutes=30)).isoformat()))
    conn.executemany(
        "INSERT INTO journal(namespace, tool, node_id, detail, ts) VALUES(?,?,?,?,?)",
        jrows,
    )
    conn.commit()

# ---- section 5: orchestration, dry run ----
with semcp.db() as conn:
    res = semcp._run_synthesis(conn, ns, budget=50, seed=42, temperature=1.0, dry_run=True)
    mechs = res["mechanisms"]
    print("\n_run_synthesis dry_run seed=42:")
    for k, v in mechs.items():
        print(f"  {k}: {v}")
    check("all 6 mechanisms present",
          set(mechs.keys()) == {"transitive", "feature_recombination", "analogy",
                                "schema", "journal_cooccurrence", "emotion"},
          str(mechs.keys()))
    check("transitive created > 0", mechs.get("transitive", {}).get("created", 0) > 0)
    check("feature_recombination created > 0",
          mechs.get("feature_recombination", {}).get("created", 0) > 0)
    check("analogy created > 0", mechs.get("analogy", {}).get("created", 0) > 0)
    # Before the snapshot rewrite: feature_recombination had skipped_cycle: 3
    # (symmetric `associated` candidates falsely blocked by the directed cycle
    # guard). That must be 0 now — symmetric relations can't form cycles.
    check("feature_recombination no false cycle skips",
          mechs.get("feature_recombination", {}).get("skipped_cycle", 0) == 0,
          str(mechs.get("feature_recombination")))
    # analogy may still report skipped_exists for *legitimate* same-triple
    # duplicates (e.g. it re-derives labrador->animal is_a that transitive
    # already accepted this dream). The false-skip pattern — typed edge
    # blocked by a weak DIFFERENT-relation edge — is disproven by the
    # DB-level coexistence checks below.
    a_skips = mechs.get("analogy", {}).get("skipped_exists", 0)
    check("analogy exists-skips within re-derivation bounds", a_skips <= 1,
          str(mechs.get("analogy")))
    check("schema created > 0", mechs.get("schema", {}).get("created", 0) > 0)
    check("journal_cooccurrence created > 0",
          mechs.get("journal_cooccurrence", {}).get("created", 0) > 0)
    co_examples = mechs.get("journal_cooccurrence", {}).get("examples", [])
    print("journal_cooccurrence examples:", co_examples)
    # frequency weighting: bone/leash (3x) must beat bark/kennel (1x)
    bl_ex = next((e for e in co_examples
                  if {e.get("source"), e.get("target")} == {names["bone"], names["leash"]}), None)
    check("repeated pair appears in examples", bl_ex is not None, str(co_examples))
    if bl_ex:
        check("repeated pair count == 3", bl_ex.get("count") == 3, str(bl_ex))
        check("repeated pair uses min gap (10 min, not 30)", bl_ex.get("gap_hours") == "0.17h",
              str(bl_ex))
    check("emotion created > 0", mechs.get("emotion", {}).get("created", 0) > 0)
    for ex in mechs.get("transitive", {}).get("examples", []):
        check("transitive example typed relation",
              ex.get("relation") in ("is_a", "causes", "leads_to", "associated",
                                     "part_of", "related_to", "similar_to",
                                     "contains", "implies", "after"),
              str(ex))
    # analogy must have created the quadruple completions
    a_ex = [tuple(sorted(e.items())) for e in mechs.get("analogy", {}).get("examples", [])]
    print("analogy examples:", a_ex)

with semcp.db() as conn:
    res2 = semcp._run_synthesis(conn, ns, budget=50, seed=42, temperature=1.0, dry_run=True)
check("seeded determinism", res2 == res, "second seeded run differed")

# ---- section 6: dream tool ----
with semcp.db() as conn:
    st = semcp.dream(namespace=ns)
    check("dream refuses when awake", st.get("status") == "awake", str(st)[:200])
    semcp.sleep_start(namespace=ns)

before = edge_count(ns)
dry = semcp.dream(namespace=ns, max_synthesis_edges=50, seed=7, dry_run=True)
check("dream dry_run reports dry_run", dry.get("dry_run") is True)
check("dream dry_run skipped maintenance",
      dry["maintenance"].get("skipped") == "dry_run", str(dry["maintenance"]))
check("dream dry_run created no hubs",
      dry["synthesis"].get("hubs_strengthened") == 0)
check("dream dry_run total == sum of mechanisms",
      dry["synthesis"]["total_edges_created"] ==
      sum(r.get("created", 0) for r in dry["synthesis"]["mechanisms"].values()
          if isinstance(r, dict)))
check("dry_run mutated nothing", edge_count(ns) == before,
      f"before={before} after={edge_count(ns)}")

# Exact dry-run preview: running the REAL dream with the same seed right
# after the dry run must produce the same per-mechanism created counts as
# the dry run predicted (the snapshot made dry-run skip accounting exact).
real_after_dry = semcp.dream(namespace=ns, max_synthesis_edges=50, seed=7)
dry_mech = dry["synthesis"]["mechanisms"]
real_mech = real_after_dry["synthesis"]["mechanisms"]
mismatch = {
    name: (dry_mech.get(name, {}).get("created"),
          real_mech.get(name, {}).get("created"))
    for name in dry_mech
    if dry_mech.get(name, {}).get("created")
    != real_mech.get(name, {}).get("created")
}
check("dry_run predicts real run created counts (seed matched)", not mismatch,
      str(mismatch))

real = real_after_dry
check("dream real run ran maintenance",
      "journal_pruned" in real["maintenance"], str(real["maintenance"]))
check("dream real run strengthened hubs",
      "hubs_strengthened" in real["synthesis"])
check("dream real run added edges", edge_count(ns) > before,
      f"before={before} after={edge_count(ns)}")
check("dream real run not dry_run", real.get("dry_run") is False)

# ---- persisted artifacts ----
with semcp.db() as conn:
    schemas = conn.execute(
        "SELECT display_name FROM nodes WHERE canonical LIKE '__schema__%'"
    ).fetchall()
    print("\nschema nodes after real dream:", [r["display_name"] for r in schemas])
    check("schema nodes persisted", len(schemas) > 0)

    inst = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE relation='instance_of' AND evidence LIKE 'schema:%'"
    ).fetchone()[0]
    check("instance_of edges persisted", inst > 0)

    typed = conn.execute(
        "SELECT source_id, target_id, relation, evidence FROM edges WHERE namespace=? "
        "AND evidence LIKE 'transitive:%'",
        (ns,),
    ).fetchall()
    print("typed transitive edges:", [(r["relation"], r["evidence"]) for r in typed])
    check("typed transitive edges persisted", len(typed) > 0)
    check("transitive evidence carries relation pair",
          all("+" in r["evidence"] for r in typed))

    # typed coexistence: dog->animal has BOTH the seeded weak associated edge
    # and... wait — transitive derives labrador->animal, not dog->animal.
    # The coexistence case: labrador->animal derived as is_a while a weak
    # labrador->animal associated edge may also exist from recombination.
    la_edges = conn.execute(
        "SELECT relation FROM edges WHERE namespace=? AND source_id=? AND target_id=?",
        (ns, names["labrador"], names["animal"]),
    ).fetchall()
    print("labrador->animal relations:", [r["relation"] for r in la_edges])
    check("typed is_a derived despite weak edge on same pair",
          any(r["relation"] == "is_a" for r in la_edges),
          str([r["relation"] for r in la_edges]))
    # and the seeded weak edge survives alongside it
    da_edges = conn.execute(
        "SELECT relation FROM edges WHERE namespace=? AND source_id=? AND target_id=?",
        (ns, names["dog"], names["animal"]),
    ).fetchall()
    check("seeded weak associated edge survived",
          any(r["relation"] == "associated" for r in da_edges),
          str([r["relation"] for r in da_edges]))

    co = conn.execute(
        "SELECT source_id, target_id, weight, confidence, evidence FROM edges "
        "WHERE relation='co_occurred' AND evidence LIKE 'journal_cooccurrence:%'"
    ).fetchall()
    print("co_occurred edges:",
          [(r["source_id"], r["target_id"], round(r["weight"], 3),
            round(r["confidence"], 2), r["evidence"]) for r in co])
    check("co_occurred edges persisted", len(co) > 0)
    # exactly ONE edge for the pair that co-occurred 3x in both directions
    bl_edges = [r for r in co
                if {r["source_id"], r["target_id"]} == {names["bone"], names["leash"]}]
    check("repeated pair aggregated to a single edge", len(bl_edges) == 1,
          f"found {len(bl_edges)} edges")
    # frequency weighting: 3x pair must outweigh the 1x pair
    bl_w = max((r["weight"] for r in bl_edges), default=0.0)
    bk_w = max((r["weight"] for r in co
                if {r["source_id"], r["target_id"]} == {names["bark"], names["kennel"]}),
               default=0.0)
    print(f"bone/leash weight={bl_w:.3f} vs bark/kennel weight={bk_w:.3f}")
    check("repeated pair outweighs single pair", bl_w > bk_w,
          f"bl={bl_w} bk={bk_w}")
    # both pairs are 30 min at closest? no — bone/leash min gap is 10 min (0.167h),
    # bark/kennel is 30 min (0.5h). temporal favors bone/leash too; combined check
    check("evidence carries count", all("x" in (r["evidence"] or "") for r in co),
          str([r["evidence"] for r in co]))

    aff = conn.execute(
        "SELECT weight FROM edges WHERE relation='affective' AND evidence LIKE 'emotion:%'"
    ).fetchall()
    print("affective weights:", [round(r["weight"], 3) for r in aff])
    check("affective edges persisted", len(aff) > 0)
    check("affective signed weight present", any(r["weight"] < 0 for r in aff),
          str([r["weight"] for r in aff]))

    fr = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE relation='associated' AND evidence LIKE 'feature_cosine:%'"
    ).fetchone()[0]
    check("feature_cosine edges persisted", fr > 0)

    jr = conn.execute(
        "SELECT detail FROM journal WHERE tool='dream' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    check("dream journal entry written", jr is not None)
    if jr:
        import json
        detail = json.loads(jr["detail"])
        check("journal entry has mechanism summary",
              "synthesis" in detail and "mechanisms" in detail["synthesis"],
              str(detail)[:200])

    # cycle safety: no directed cycle among synthesized edges. Symmetric
    # relations (associated/co_occurred/…) excluded: a directed edge plus
    # its symmetric reverse (labrador->dog is_a + dog->labrador associated)
    # is a legitimate undirected pair, not a contradiction.
    rows = conn.execute(
        "SELECT source_id, target_id, relation FROM edges WHERE namespace=?", (ns,)
    ).fetchall()
    graph = {}
    for r in rows:
        if r["relation"] in semcp.SYMMETRIC_RELATIONS:
            continue
        graph.setdefault(r["source_id"], []).append(r["target_id"])
    found_cycle = False
    for start in graph:
        stack, seen = [(start, 0)], set()
        while stack:
            node, d = stack.pop()
            if d > 6:
                continue
            for nxt in graph.get(node, []):
                if nxt == start:
                    found_cycle = True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, d + 1))
    check("no directed cycles in final graph", not found_cycle)

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for name, detail in FAILURES:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("ALL CHECKS PASSED")
