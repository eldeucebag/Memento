##. TODO:

# Implementing all seven synthesis mechanisms

Below is the complete set of additions. The existing `dream` tool is replaced; everything else in the file is unchanged except for the new imports/constants at the top.

The mechanisms share one budget, run in priority order, respect the cycle check, and all honour `seed` / `temperature` / `dry_run`.

---

## 1. New imports and constants

Add `import random` alongside the existing imports, then append these constants near the other config:

```python
import random

# ... after the existing config block ...

# --- Synthesis config ---
MAX_SYNTHESIS_CANDIDATES = 500
JOURNAL_COOCCURRENCE_WINDOW_HOURS = float(
    os.environ.get("MCP_MEMORY_JOURNAL_WINDOW_HOURS", "2.0")
)
JOURNAL_COOCCURRENCE_LOOKBACK_DAYS = 30
SCHEMA_NODE_PREFIX = "__schema__::"
MAX_SCHEMAS_PER_DREAM = 10
RELATION_ATTENUATION = 0.7   # used for typed transitive inference
```

---

## 2. Relation composition table

Only *algebraically valid* relation pairs compose; the result is a **typed** edge rather than `"inferred"`.

```python
RELATION_COMPOSITIONS: Dict[Tuple[str, str], str] = {
    # Pure transitive relations
    ("is_a", "is_a"): "is_a",
    ("part_of", "part_of"): "part_of",
    ("contains", "contains"): "contains",
    ("causes", "causes"): "causes",
    ("leads_to", "leads_to"): "leads_to",
    ("implies", "implies"): "implies",
    ("before", "before"): "before",
    ("after", "after"): "after",
    ("associated", "associated"): "associated",
    ("related_to", "related_to"): "related_to",
    ("similar_to", "similar_to"): "similar_to",
    # is_a on the left promotes the right-hand relation
    ("is_a", "part_of"): "part_of",
    ("is_a", "causes"): "causes",
    ("is_a", "leads_to"): "leads_to",
    ("is_a", "associated"): "associated",
    ("is_a", "related_to"): "related_to",
    ("causes", "leads_to"): "leads_to",
    # Deliberately NOT composed (ambiguous):
    #   ("part_of", "is_a")      A part_of B, B is_a C
    #   ("contains", "part_of")  inverse relations
    #   ("before", "after")      already ordered
}


def _compose_relations(r1: str, r2: str) -> Optional[str]:
    return RELATION_COMPOSITIONS.get((r1, r2))
```

---

## 3. Shared helpers

```python
def _empty_synth_result() -> Dict[str, Any]:
    return {
        "proposed": 0,
        "created": 0,
        "skipped_exists": 0,
        "skipped_cycle": 0,
        "skipped_invalid": 0,
        "examples": [],
    }


def _fetch_existing_pairs(conn: sqlite3.Connection, namespace: str) -> set:
    rows = conn.execute(
        "SELECT source_id, target_id FROM edges WHERE namespace = ?",
        (namespace,),
    ).fetchall()
    return {(r["source_id"], r["target_id"]) for r in rows}


def _try_upsert(
    conn: sqlite3.Connection,
    namespace: str,
    source_id: int,
    target_id: int,
    relation: str,
    weight: float,
    confidence: float,
    evidence: str,
    existing_pairs: set,
    dry_run: bool,
) -> str:
    """Return one of: created, skipped_exists, skipped_cycle, skipped_invalid.

    Note: in dry_run the real graph is not mutated, so `would_create_cycle`
    sees the *original* graph. Cycle skips for chains that would only form
    from other synthesized edges may be under-reported.
    """
    if source_id == target_id:
        return "skipped_invalid"
    if (source_id, target_id) in existing_pairs:
        return "skipped_exists"
    if would_create_cycle(conn, namespace, source_id, target_id, MAX_GRAPH_DEPTH):
        return "skipped_cycle"
    existing_pairs.add((source_id, target_id))
    if dry_run:
        return "created"
    try:
        upsert_edge(conn, namespace, source_id, target_id, relation, weight, confidence, evidence)
    except Exception:
        return "skipped_invalid"
    return "created"


def _feature_vectors(
    conn: sqlite3.Connection,
    namespace: str,
    node_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    if not node_ids:
        return {}
    placeholders = ",".join("?" for _ in node_ids)
    rows = conn.execute(
        f"""
        SELECT f.node_id, f.key, f.value
        FROM features f
        JOIN nodes n ON n.id = f.node_id
        WHERE n.namespace = ? AND f.contributes = 1
          AND f.node_id IN ({placeholders})
        """,
        (namespace, *node_ids),
    ).fetchall()
    vectors: Dict[int, Dict[str, float]] = {}
    for r in rows:
        vectors.setdefault(r["node_id"], {})[r["key"]] = safe_float(r["value"], 0.0)
    return vectors


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    common = v1.keys() & v2.keys()
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    n1 = math.sqrt(sum(v * v for v in v1.values()))
    n2 = math.sqrt(sum(v * v for v in v2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (n1 * n2)))


def _sample_by_weight(items, weights, k, rng, temperature):
    """Sample up to k items without replacement, weights = softmax(|w|/T).

    temperature > 1 → flatter (more exploration)
    temperature < 1 → greedier (favours high-weight candidates)
    """
    if not items:
        return []
    if len(items) <= k:
        return list(items)
    t = max(0.01, temperature)
    abs_w = [abs(w) for w in weights]
    max_w = max(abs_w) if abs_w else 0.0
    exp_w = [math.exp((w - max_w) / t) for w in abs_w]
    total = sum(exp_w)
    if total <= 0:
        return list(items[:k])
    probs = [w / total for w in exp_w]
    indices = list(range(len(items)))
    result = []
    for _ in range(min(k, len(items))):
        r = rng.random()
        cum = 0.0
        chosen = indices[-1]
        for idx, p in zip(indices, probs):
            cum += p
            if r <= cum:
                chosen = idx
                break
        result.append(items[chosen])
        pos = indices.index(chosen)
        indices.pop(pos)
        probs.pop(pos)
        tot = sum(probs)
        if tot <= 0:
            break
        probs = [p / tot for p in probs]
    return result
```

---

## 4. Six synthesis mechanisms

### 4a. Relation-typed transitive composition

```python
def _synth_transitive(conn, namespace, budget, rng, temperature, existing_pairs, dry_run):
    result = _empty_synth_result()
    if budget <= 0:
        return result
    rows = conn.execute(
        """
        SELECT id, source_id, target_id, relation, weight, confidence
        FROM edges
        WHERE namespace = ? AND ABS(weight) >= 0.1
        ORDER BY ABS(weight) DESC
        LIMIT ?
        """,
        (namespace, MAX_SYNTHESIS_CANDIDATES),
    ).fetchall()
    by_source: Dict[int, List[sqlite3.Row]] = {}
    for r in rows:
        by_source.setdefault(r["source_id"], []).append(r)

    seen: set = set()
    candidates: List[Dict[str, Any]] = []
    for e1 in rows:
        for e2 in by_source.get(e1["target_id"], []):
            if e1["source_id"] == e2["target_id"]:
                continue
            composed = _compose_relations(e1["relation"], e2["relation"])
            if not composed:
                continue
            key = (e1["source_id"], e2["target_id"], composed)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "source_id": e1["source_id"],
                "target_id": e2["target_id"],
                "relation": composed,
                "weight": clamp(e1["weight"] * e2["weight"] * RELATION_ATTENUATION, -1.0, 1.0),
                "confidence": clamp(e1["confidence"] * e2["confidence"] * RELATION_ATTENUATION, 0.0, 1.0),
                "evidence": f"transitive:{e1['relation']}+{e2['relation']}",
            })
    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], existing_pairs, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "relation": c["relation"], "weight": round(c["weight"], 3),
            })
    return result
```

### 4b. Feature recombination

```python
def _synth_feature_recombination(conn, namespace, budget, rng, temperature, existing_pairs, dry_run):
    result = _empty_synth_result()
    if budget <= 0:
        return result
    rows = conn.execute(
        """
        SELECT n.id
        FROM nodes n
        WHERE n.namespace = ? AND n.suppressed = 0
          AND EXISTS (
              SELECT 1 FROM features f
              WHERE f.node_id = n.id AND f.contributes = 1
          )
        ORDER BY n.base_strength DESC
        LIMIT ?
        """,
        (namespace, MAX_SYNTHESIS_CANDIDATES),
    ).fetchall()
    node_ids = [r["id"] for r in rows]
    vecs = _feature_vectors(conn, namespace, node_ids)

    key_to_nodes: Dict[str, List[int]] = {}
    for nid, vec in vecs.items():
        for k in vec:
            key_to_nodes.setdefault(k, []).append(nid)

    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    for key, members in key_to_nodes.items():
        if len(members) < 2:
            continue
        for i, n1 in enumerate(members):
            for n2 in members[i + 1:]:
                pair = (min(n1, n2), max(n1, n2))
                if pair in seen:
                    continue
                seen.add(pair)
                sim = _cosine(vecs[n1], vecs[n2])
                if sim < 0.5:
                    continue
                candidates.append({
                    "source_id": n1, "target_id": n2,
                    "relation": "associated",
                    "weight": clamp(sim, 0.0, 1.0),
                    "confidence": clamp(sim, 0.0, 1.0),
                    "evidence": f"feature_cosine:{sim:.3f}",
                })
    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], existing_pairs, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "weight": round(c["weight"], 3),
            })
    return result
```

### 4c. Semantic analogy

```python
def _synth_analogy(conn, namespace, budget, rng, temperature, existing_pairs, dry_run):
    """Complete the parallelogram: A→B and C→D with similar A~C and B~D
    implies A→D and C→B."""
    result = _empty_synth_result()
    if budget <= 0:
        return result
    rows = conn.execute(
        """
        SELECT id, source_id, target_id, relation, weight, confidence
        FROM edges
        WHERE namespace = ? AND ABS(weight) >= 0.2
        ORDER BY ABS(weight) DESC
        LIMIT ?
        """,
        (namespace, MAX_SYNTHESIS_CANDIDATES),
    ).fetchall()

    by_relation: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        by_relation.setdefault(r["relation"], []).append(r)

    eligible_rels = [(rel, es) for rel, es in by_relation.items() if len(es) >= 2]
    if not eligible_rels:
        return result
    if len(eligible_rels) > 5:
        eligible_rels = _sample_by_weight(
            eligible_rels, [len(es) for _, es in eligible_rels], 5, rng, temperature
        )

    all_nids = set()
    for _, es in eligible_rels:
        for e in es:
            all_nids.add(e["source_id"])
            all_nids.add(e["target_id"])
    vecs = _feature_vectors(conn, namespace, list(all_nids))

    def node_sim(a: int, b: int) -> float:
        return _cosine(vecs.get(a, {}), vecs.get(b, {}))

    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    max_gen = budget * 3
    for rel, edges in eligible_rels:
        for i, e1 in enumerate(edges):
            for e2 in edges[i + 1:]:
                if e1["source_id"] == e2["source_id"] or e1["target_id"] == e2["target_id"]:
                    continue
                sim_src = node_sim(e1["source_id"], e2["source_id"])
                sim_tgt = node_sim(e1["target_id"], e2["target_id"])
                if sim_src < 0.4 or sim_tgt < 0.4:
                    continue
                strength = min(sim_src, sim_tgt)
                for s, t in (
                    (e1["source_id"], e2["target_id"]),
                    (e2["source_id"], e1["target_id"]),
                ):
                    key = (s, t, rel)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "source_id": s, "target_id": t,
                        "relation": rel,
                        "weight": clamp(e1["weight"] * e2["weight"] * strength, -1.0, 1.0),
                        "confidence": clamp(0.5 * strength, 0.0, 1.0),
                        "evidence": f"analogy:{rel}:{sim_src:.2f},{sim_tgt:.2f}",
                    })
                    if len(candidates) >= max_gen:
                        break
                if len(candidates) >= max_gen:
                    break
            if len(candidates) >= max_gen:
                break
        if len(candidates) >= max_gen:
            break

    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], existing_pairs, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "relation": c["relation"],
            })
    return result
```

### 4d. Abstraction / schema

```python
def _synth_schema(conn, namespace, budget, rng, temperature, existing_pairs, dry_run):
    result = _empty_synth_result()
    result["created_nodes"] = 0
    if budget <= 0:
        return result

    rows = conn.execute(
        """
        SELECT f.node_id, f.key, f.value
        FROM features f
        JOIN nodes n ON n.id = f.node_id
        WHERE n.namespace = ? AND f.contributes = 1 AND n.suppressed = 0
        """,
        (namespace,),
    ).fetchall()
    key_nodes: Dict[str, List[Tuple[int, float]]] = {}
    for r in rows:
        key_nodes.setdefault(r["key"], []).append(
            (r["node_id"], safe_float(r["value"], 0.0))
        )

    eligible = [(k, m) for k, m in key_nodes.items() if len(m) >= 3]
    if not eligible:
        return result
    eligible.sort(key=lambda x: len(x[1]), reverse=True)
    if len(eligible) > MAX_SCHEMAS_PER_DREAM:
        eligible = _sample_by_weight(
            eligible, [len(m) for _, m in eligible], MAX_SCHEMAS_PER_DREAM, rng, temperature
        )
    result["proposed"] = len(eligible)

    created_edges = 0
    for key, members in eligible:
        if created_edges >= budget:
            break
        schema_name = f"{SCHEMA_NODE_PREFIX}{key}"
        schema_row = get_node_row(conn, namespace, schema_name)

        if schema_row:
            schema_id = schema_row["id"]
        elif dry_run:
            result["created_nodes"] += 1
            result["created"] += min(len(members), budget - created_edges)
            created_edges += min(len(members), budget - created_edges)
            if len(result["examples"]) < 3:
                result["examples"].append({"schema": schema_name, "members": len(members), "dry_run": True})
            continue
        else:
            try:
                schema_row = ensure_node(
                    conn, namespace, schema_name,
                    plasticity=0.5, corpus="SCHEMA",
                )
                schema_id = schema_row["id"]
                result["created_nodes"] += 1
            except Exception:
                continue

        for nid, val in members:
            if created_edges >= budget:
                break
            status = _try_upsert(
                conn, namespace, nid, schema_id, "instance_of",
                clamp(val, -1.0, 1.0), 0.7, f"schema:{key}",
                existing_pairs, dry_run,
            )
            if status == "created":
                created_edges += 1
                result["created"] += 1
            elif status in result:
                result[status] += 1

        if len(result["examples"]) < 3:
            result["examples"].append({"schema": schema_name, "members": len(members)})

    return result
```

### 4e. Journal co-occurrence

```python
def _synth_journal_cooccurrence(conn, namespace, budget, rng, temperature, existing_pairs, dry_run):
    result = _empty_synth_result()
    if budget <= 0:
        return result
    lookback = (
        datetime.now(timezone.utc)
        - timedelta(days=JOURNAL_COOCCURRENCE_LOOKBACK_DAYS)
    ).isoformat()
    rows = conn.execute(
        """
        SELECT node_id, ts
        FROM journal
        WHERE namespace = ? AND node_id IS NOT NULL AND ts >= ?
        ORDER BY ts
        LIMIT ?
        """,
        (namespace, lookback, MAX_SYNTHESIS_CANDIDATES * 2),
    ).fetchall()

    entries: List[Tuple[datetime, int]] = []
    for r in rows:
        t = parse_ts(r["ts"])
        if t:
            entries.append((t, r["node_id"]))
    if len(entries) < 2:
        return result

    window_seconds = JOURNAL_COOCCURRENCE_WINDOW_HOURS * 3600.0
    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    max_gen = budget * 3
    for i, (t1, n1) in enumerate(entries):
        for j in range(i + 1, len(entries)):
            t2, n2 = entries[j]
            gap_s = (t2 - t1).total_seconds()
            if gap_s > window_seconds:
                break
            if n1 == n2:
                continue
            pair = (n1, n2)
            if pair in seen:
                continue
            seen.add(pair)
            gap_hours = gap_s / 3600.0
            candidates.append({
                "source_id": n1, "target_id": n2,
                "relation": "co_occurred",
                "weight": clamp(1.0 / (1.0 + gap_hours), 0.0, 1.0),
                "confidence": 0.6,
                "evidence": f"journal_cooccurrence:{gap_hours:.2f}h",
            })
            if len(candidates) >= max_gen:
                break
        if len(candidates) >= max_gen:
            break

    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], existing_pairs, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "gap_hours": c["evidence"].split(":")[-1],
            })
    return result
```

### 4f. Emotion-weighted association

```python
def _synth_emotion(conn, namespace, budget, rng, temperature, existing_pairs, dry_run):
    result = _empty_synth_result()
    if budget <= 0:
        return result
    rows = conn.execute(
        """
        SELECT *
        FROM nodes
        WHERE namespace = ? AND suppressed = 0
        ORDER BY ABS(salience) DESC
        LIMIT ?
        """,
        (namespace, MAX_SYNTHESIS_CANDIDATES),
    ).fetchall()

    eligible: List[Tuple[int, float]] = []
    for r in rows:
        sal = effective_salience(r)
        if abs(sal) >= 0.3:
            eligible.append((r["id"], sal))
    if len(eligible) < 2:
        return result

    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    max_gen = budget * 3
    for i, (n1, s1) in enumerate(eligible):
        for n2, s2 in eligible[i + 1:]:
            pair = (min(n1, n2), max(n1, n2))
            if pair in seen:
                continue
            seen.add(pair)
            candidates.append({
                "source_id": n1, "target_id": n2,
                "relation": "affective",
                "weight": clamp(s1 * s2, -1.0, 1.0),
                "confidence": 0.5,
                "evidence": f"emotion:{s1:+.2f}*{s2:+.2f}",
            })
            if len(candidates) >= max_gen:
                break
        if len(candidates) >= max_gen:
            break

    result["proposed"] = len(candidates)
    if len(candidates) > budget:
        candidates = _sample_by_weight(
            candidates, [c["weight"] for c in candidates], budget, rng, temperature
        )
    for c in candidates:
        status = _try_upsert(
            conn, namespace, c["source_id"], c["target_id"], c["relation"],
            c["weight"], c["confidence"], c["evidence"], existing_pairs, dry_run,
        )
        if status in result:
            result[status] += 1
        if status == "created" and len(result["examples"]) < 3:
            result["examples"].append({
                "source": c["source_id"], "target": c["target_id"],
                "weight": round(c["weight"], 3),
            })
    return result
```

---

## 5. Orchestration

```python
def _run_synthesis(
    conn: sqlite3.Connection,
    namespace: str,
    budget: int,
    seed: Optional[int],
    temperature: float,
    dry_run: bool,
) -> Dict[str, Any]:
    rng = random.Random(seed) if seed is not None else random.Random()
    existing_pairs = _fetch_existing_pairs(conn, namespace)

    # Priority order: highest-confidence signals first, exploratory ones last.
    mechanisms = [
        ("transitive",             _synth_transitive,             0.25),
        ("feature_recombination",  _synth_feature_recombination,  0.20),
        ("analogy",                _synth_analogy,                0.15),
        ("schema",                 _synth_schema,                 0.15),
        ("journal_cooccurrence",   _synth_journal_cooccurrence,   0.15),
        ("emotion",                _synth_emotion,                0.10),
    ]

    results: Dict[str, Any] = {}
    total_edges = 0
    for name, fn, share in mechanisms:
        mech_budget = min(int(round(budget * share)), budget - total_edges)
        if mech_budget <= 0:
            results[name] = {"skipped": "budget_exhausted"}
            continue
        try:
            r = fn(conn, namespace, mech_budget, rng, temperature, existing_pairs, dry_run)
            results[name] = r
            total_edges += r.get("created", 0)
        except Exception as e:  # keep one broken mechanism from killing dream
            results[name] = {"error": f"{type(e).__name__}: {e}"[:200]}

    return {
        "mechanisms": results,
        "total_edges_created": total_edges,
        "total_edges_skipped_cycle": sum(
            r.get("skipped_cycle", 0) for r in results.values() if isinstance(r, dict)
        ),
        "total_edges_skipped_exists": sum(
            r.get("skipped_exists", 0) for r in results.values() if isinstance(r, dict)
        ),
        "budget": budget,
        "seed": seed,
        "temperature": temperature,
        "dry_run": dry_run,
    }
```

---

## 6. Revised `dream` tool

Replace the existing `dream` with:

```python
@mcp.tool()
def dream(
    namespace: str = DEFAULT_NAMESPACE,
    max_synthesis_edges: int = 100,
    seed: Optional[int] = None,
    temperature: float = 1.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Sleep-cycle maintenance and novel synthesis.

    Must be in 'sleep' state (latest event is sleep_start).

    Maintenance: DB optimize, journal/weak-edge pruning, orphan suppression,
    FTS rebuild.

    Synthesis (all bounded by max_synthesis_edges, cycle-checked):
      - transitive:           relation-typed composition (is_a∘is_a, causes∘causes, …)
      - feature_recombination: cosine similarity over contributes=1 features
      - analogy:              parallelogram completion over same-relation edges
      - schema:               creates __schema__::<key> nodes for shared features
      - journal_cooccurrence: temporal adjacency in the journal
      - emotion:              high-|salience| pairs, signed weight

    Args:
        seed:        int → deterministic output; None → fresh sample each call
        temperature: >1 more exploratory, <1 greedier, 1.0 balanced
        dry_run:     if True, do not mutate the DB (maintenance is skipped,
                     synthesis is previewed)
    """
    namespace = clean_text(namespace, MAX_NAME) or DEFAULT_NAMESPACE
    max_synthesis_edges = clamp_int(max_synthesis_edges, 1, 500)
    temperature = clamp(temperature if math.isfinite(temperature) else 1.0, 0.01, 10.0)

    with db() as conn:
        state = brain_state(conn, namespace)

        if state["state"] != "asleep":
            return {
                "status": "awake",
                "message": "Brain is not in sleep state. Call sleep_start first.",
                "brain": state,
            }

        results: Dict[str, Any] = {
            "maintenance": {},
            "synthesis": {},
            "brain": state,
            "dry_run": dry_run,
        }

        with conn:
            # 1. Maintenance (skipped in dry_run)
            if dry_run:
                results["maintenance"]["skipped"] = "dry_run"
            else:
                conn.execute("PRAGMA optimize")

                cutoff_journal = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                cur_j = conn.execute(
                    "DELETE FROM journal WHERE namespace = ? AND ts < ?",
                    (namespace, cutoff_journal),
                )
                results["maintenance"]["journal_pruned"] = cur_j.rowcount

                cutoff_edges = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
                cur_e = conn.execute(
                    """
                    DELETE FROM edges
                    WHERE namespace = ? AND ABS(weight) < 0.05 AND ts < ?
                    """,
                    (namespace, cutoff_edges),
                )
                results["maintenance"]["weak_edges_pruned"] = cur_e.rowcount

                cutoff_orphans = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                cur_o = conn.execute(
                    """
                    UPDATE nodes SET suppressed = 1
                    WHERE namespace = ?
                      AND id NOT IN (
                          SELECT source_id FROM edges
                          UNION SELECT target_id FROM edges
                      )
                      AND base_strength < 0.1
                      AND last_accessed < ?
                    """,
                    (namespace, cutoff_orphans),
                )
                results["maintenance"]["orphans_suppressed"] = cur_o.rowcount

                if FTS_AVAILABLE:
                    try:
                        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
                        results["maintenance"]["fts_rebuilt"] = True
                    except sqlite3.OperationalError:
                        results["maintenance"]["fts_rebuilt"] = False

            # 2. Synthesis
            results["synthesis"] = _run_synthesis(
                conn,
                namespace,
                max_synthesis_edges,
                seed,
                temperature,
                dry_run,
            )

            # 3. Hub strengthening (unchanged, skipped in dry_run)
            if dry_run:
                results["synthesis"]["hubs_strengthened"] = 0
            else:
                hub_sql = """
                    SELECT id, degree FROM (
                        SELECT n.id AS id,
                               (SELECT COUNT(*) FROM edges WHERE source_id = n.id) +
                               (SELECT COUNT(*) FROM edges WHERE target_id = n.id) AS degree
                        FROM nodes n
                        WHERE n.namespace = ? AND n.suppressed = 0
                    )
                    WHERE degree >= 5
                    ORDER BY degree DESC
                    LIMIT 20
                """
                hubs = conn.execute(hub_sql, (namespace,)).fetchall()
                strengthened = 0
                for h in hubs:
                    boost = 0.01 * min(h["degree"] / 10.0, 1.0)
                    strengthen_node(conn, h["id"], boost)
                    strengthened += 1
                results["synthesis"]["hubs_strengthened"] = strengthened

            if not dry_run:
                write_journal(
                    conn,
                    namespace,
                    "dream",
                    None,
                    {
                        "maintenance": results["maintenance"],
                        "synthesis": {
                            "total_edges_created": results["synthesis"].get("total_edges_created", 0),
                            "budget": max_synthesis_edges,
                            "seed": seed,
                            "temperature": temperature,
                            "mechanisms": {
                                name: {
                                    k: v for k, v in r.items()
                                    if k in ("created", "created_nodes", "proposed",
                                             "skipped_cycle", "skipped_exists")
                                }
                                for name, r in results["synthesis"].get("mechanisms", {}).items()
                                if isinstance(r, dict)
                            },
                        },
                    },
                )

        results["brain"] = brain_state(conn, namespace)
        return results
```

---

## 7. What each mechanism actually contributes

| Mechanism | Signal source | Relation | Typical weight | Where it appears in `dream` output |
|---|---|---|---|---|
| `transitive` | Edge chains | Typed (`is_a`, `causes`, …) | `w1·w2·0.7` | `synthesis.mechanisms.transitive` |
| `feature_recombination` | `features.contributes=1` cosine | `associated` | cosine ∈ [0.5, 1.0] | `synthesis.mechanisms.feature_recombination` |
| `analogy` | Same-relation edge pairs with similar endpoints | Original relation | `w1·w2·min(sim)` | `synthesis.mechanisms.analogy` |
| `schema` | Feature keys shared by ≥3 nodes | `instance_of` + new `__schema__::<key>` node | feature value | `synthesis.mechanisms.schema` |
| `journal_cooccurrence` | Journal entries within 2h (configurable) | `co_occurred` | `(1/(1+gap_h))·(1−e^−count)·(0.6+0.4·cos)`, conf 0.5–0.8 | `synthesis.mechanisms.journal_cooccurrence` |
| `emotion` | `|effective_salience| ≥ 0.3` pairs | `affective` | `s1·s2` (signed) | `synthesis.mechanisms.emotion` |

---

## 8. Behaviour notes

- **Budget**: each mechanism gets a fraction of `max_synthesis_edges` (25/20/15/15/15/10%). Unused budget flows to later mechanisms. Every proposed edge also passes `would_create_cycle`, so a mechanism may propose more than it creates.
- **Stochastic sampling**: `seed=None` gives fresh randomness each call; `seed=42` gives a reproducible dream. `temperature` scales the softmax over |weight|. Because the mechanisms sample when over-budget, a seeded dream is deterministic only if the graph and budget are unchanged.
- **`dry_run=True`**: no maintenance, no edges, no schema nodes. Synthesis still runs and reports counts / examples — useful for calibrating thresholds before a real dream. Cycle checks in dry-run only see the original graph, so a chain of synthesized edges that would form a cycle may not be flagged (documented in `_try_upsert`).
- **`evidence`** on each new edge carries the mechanism (`transitive:…`, `feature_cosine:…`, `analogy:…`, `schema:…`, `journal_cooccurrence:…`, `emotion:…`) so `explain` shows provenance.
- **N+1 recall/top_nodes fixes from before are preserved** — nothing here regresses performance.
- **Schema nodes** use the `__schema__::` prefix and `corpus="SCHEMA"`. If a user already has a node with the exact canonical name, the existing node is reused. `get_node_row` is case-folded, so `__schema__::mood` and `__Schema__::Mood` collide intentionally.

---

## 9. Known limitations (honest)

- **Analogy needs features.** Two edge-pairs with no `contributes=1` features on their endpoints get similarity 0 and are never proposed. Text-based similarity (Jaccard over `display_name` tokens) would broaden coverage but is not implemented.
- **Schema quality depends on feature naming.** `mood`, `happy`, `valence` produce three different schema nodes for related concepts. A feature taxonomy or stemming pass would help.
- **Journal co-occurrence is temporal, not semantic — now frequency-weighted with a soft semantic boost.** Two unrelated nodes mentioned 30 minutes apart still get a `co_occurred` edge, but repeated co-occurrence now strengthens the edge (`1−e^−count`) instead of repeats being discarded, confidence scales with count (0.5–0.8), and pairs whose endpoints both carry features get a cosine factor `0.6+0.4·sim`. Featureless pairs keep factor 1.0 — a hard semantic gate would have excluded exactly the novel links this mechanism exists to catch, and duplicated `feature_recombination`'s signal. Pairs are canonicalized to `(min_id, max_id)` so repeats aggregate onto one edge regardless of timestamp order. Evidence: `journal_cooccurrence:<count>x:<gap>h[:sim=<cos>]`.
- **Emotion mechanism favours |salience|.** Nodes with strong `affect` but low `salience` (or vice versa) are underweighted. A combined score `sqrt(|salience|·|affect|)` is a reasonable extension.
- **Dry-run cycle preview is approximate** (see §8).
- **Sampling is softmax over a bounded candidate pool.** Mechanisms with candidate pools larger than `MAX_SYNTHESIS_CANDIDATES` may miss high-value pairs that happen to fall outside the top-N by their sort key.

