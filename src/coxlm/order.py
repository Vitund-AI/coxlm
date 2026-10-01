"""The ``order`` primitive: rank/sequence a set of items under a criterion.

No new head -- a composition of the existing primitives, two modes:

  score (default, O(n)):   one isolated ordinal `score` question per item, "what
                           position does this item occupy?", over positions 1..n.
                           All n share the cached state, one forward pass. Aggregate
                           by expected position; confidence = product of the mass
                           each item put on the position it was assigned. Cheap route
                           to a single ordering.
  pairwise (O(n^2)):       one isolated yesno per pair, "does i come before j?". The
                           primary structured output: the P(i<j) matrix is thresholded
                           into a partial order (a DAG), and what that graph implies is
                           read off exactly -- each item's feasible position range, the
                           concurrency levels (items runnable in parallel), the flexible
                           pairs, and (n<=9) the count and best of the valid orderings.
                           A total order is the wrong deliverable when the truth is
                           partial; the graph is a Makefile inferred from prose.

Items are always shown in random order in the prompt, so presentation order is
never a carrier -- the model has to read the items. Truth is free by construction:
any procedural text, shuffled, is a labelled example.

Coherence, read off the graph: a 0.5 marginal is ambiguous -- flexible, or unsure --
and one marginal can't tell them apart, but the graph can. A model that knows a pair
is free leaves the edge out and stays acyclic; a confused model makes cycles in the
same triples. So "0.5 and consistent" reads flexible, "0.5 and inconsistent" reads
unsure. That is why ``consistency`` stays in the response.
"""
from __future__ import annotations

import random
from functools import lru_cache

from .schema import questions, score, yesno


def _ordinal(k: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
            7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}.get(k, f"position {k}")


def _state(items, context, rng):
    shown = list(range(len(items)))
    rng.shuffle(shown)  # presentation order is never a carrier
    body = "Items (listed in no particular order):\n" + "\n".join(f"- {items[i]}" for i in shown)
    return (context.strip() + "\n\n" + body) if context and context.strip() else body


# --- partial-order graph machinery (pairwise mode) ---------------------------
# All operate on a node set 0..n-1 and an edge dict {(u, v): P(u before v)}.

def _find_cycle(n, edges):
    """Return one directed cycle as a node list [v, ..., u] (edge u->v closes it), or None."""
    adj = {i: [] for i in range(n)}
    for (u, v) in edges:
        adj[u].append(v)
    color = [0] * n  # 0 white, 1 gray (on stack), 2 black
    parent = {}

    def dfs(start):
        stack = [(start, iter(adj[start]))]
        color[start] = 1
        while stack:
            u, it = stack[-1]
            for v in it:
                if color[v] == 0:
                    color[v] = 1
                    parent[v] = u
                    stack.append((v, iter(adj[v])))
                    break
                if color[v] == 1:  # back edge u->v: walk parents u..v
                    path, x = [u], u
                    while x != v:
                        x = parent[x]
                        path.append(x)
                    path.reverse()
                    return path
            else:
                color[u] = 2
                stack.pop()
        return None

    for s in range(n):
        if color[s] == 0:
            c = dfs(s)
            if c:
                return c
    return None


def _break_cycles(n, edge_w):
    """Drop the edge nearest 0.5 in each cycle until acyclic. Returns (kept dict, count)."""
    edges = dict(edge_w)
    repaired = 0
    while True:
        cyc = _find_cycle(n, edges)
        if not cyc:
            return edges, repaired
        ring = [(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]
        drop = min(ring, key=lambda e: abs(edges[e] - 0.5))
        del edges[drop]
        repaired += 1


def _reachability(n, edges):
    """reach[a] = set of nodes reachable from a via >=1 edge (transitive closure)."""
    adj = {i: [] for i in range(n)}
    for (u, v) in edges:
        adj[u].append(v)
    reach = {}
    for s in range(n):
        seen, stack = set(), list(adj[s])
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x])
        reach[s] = seen
    return reach


def _transitive_reduction(n, edges, reach):
    """Keep only direct precedences: drop (u,v) if some intermediate w has u=>w=>v."""
    return [(u, v) for (u, v) in edges
            if not any(w != u and w != v and w in reach[u] and v in reach[w] for w in range(n))]


def _toposort(n, edges, strength):
    """Topological order of the DAG, ties broken by soft-Borda strength (desc)."""
    import heapq
    indeg = [0] * n
    adj = {i: [] for i in range(n)}
    for (u, v) in edges:
        adj[u].append(v)
        indeg[v] += 1
    heap = [(-strength[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _, u = heapq.heappop(heap)
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (-strength[v], v))
    return out


def _levels(n, edges, topo, strength):
    """Longest-path layering: nodes in one level have no path between them (concurrent)."""
    preds = {v: [] for v in range(n)}
    for (u, v) in edges:
        preds[v].append(u)
    level = [0] * n
    for v in topo:
        level[v] = max((level[u] + 1 for u in preds[v]), default=0)
    hi = max(level, default=-1)
    return [sorted((i for i in range(n) if level[i] == L), key=lambda i: -strength[i])
            for L in range(hi + 1)]


def _preds_mask(n, edges):
    mask = [0] * n
    for (u, v) in edges:
        mask[v] |= (1 << u)
    return mask


def _count_linear_extensions(n, edges):
    """Number of orderings consistent with the DAG (subset DP); None if n>9."""
    if n > 9:
        return None
    preds, full = _preds_mask(n, edges), (1 << n) - 1

    @lru_cache(maxsize=None)
    def f(placed):
        if placed == full:
            return 1
        total = 0
        for v in range(n):
            if not (placed >> v) & 1 and (preds[v] & ~placed) == 0:
                total += f(placed | (1 << v))
        return total

    return f(0)


def _ranked_orders(n, P, k=3):
    """Top-k total orders as a normalized distribution over all n! orders.

    Score an ordering by the product of its pairwise probabilities (the independent-
    pair / Babington-Smith likelihood that every pair is in the stated direction),
    then normalize over every permutation so the numbers are shares of the whole
    ordering space -- "the probability of each path through the items". None for
    n>8, where n! is too large to enumerate exactly.
    """
    if n > 8:
        return None
    import itertools
    scored, Z = [], 0.0
    for perm in itertools.permutations(range(n)):
        s = 1.0
        for a in range(n):
            pa = perm[a]
            for b in range(a + 1, n):
                s *= P[(pa, perm[b])]
        Z += s
        scored.append((s, perm))
    if Z <= 0:
        return None
    scored.sort(key=lambda x: -x[0])
    return [[list(perm), round(s / Z, 4)] for s, perm in scored[:k]]


def order_items(model, items, mode="score", instructions="", context="", seed=0, epsilon=0.15):
    n = len(items)
    if n < 2:
        raise ValueError("order needs at least two items")
    if n > 255:
        raise ValueError("at most 255 items (head cardinality)")
    instr = instructions or "Put these items in the correct order."
    rng = random.Random(seed)
    state = _state(items, context, rng)

    if mode == "score":
        positions = [_ordinal(k) for k in range(1, n + 1)]
        fields = {f"item_{i}": score(positions, instructions=f'{instr} Which position does this item occupy: "{items[i]}"?')
                  for i in range(n)}
        [ans] = model.decide([state], questions(**fields))
        exp = {i: ans[f"item_{i}"].score for i in range(n)}          # 1-based expected position
        dist = {i: ans[f"item_{i}"].probabilities for i in range(n)}
        argmax = {i: max(range(n), key=lambda k: list(dist[i].values())[k]) for i in range(n)}
        order = sorted(range(n), key=lambda i: (exp[i], argmax[i]))
        assigned = {i: rank for rank, i in enumerate(order)}          # 0-based slot
        conf = 1.0
        positions_out = []
        for i in range(n):
            probs = list(dist[i].values())
            conf *= probs[assigned[i]]
            positions_out.append({"item": i, "text": items[i], "expected": round(exp[i], 3),
                                  "dist": [round(p, 3) for p in probs]})
        return {"order": order, "ordered_items": [items[i] for i in order],
                "positions": positions_out, "confidence": round(conf, 3), "mode": "score"}

    if mode == "pairwise":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        fields = {f"p_{i}_{j}": yesno(f'{instr} Does "{items[i]}" precede "{items[j]}"?')
                  for i, j in pairs}
        [ans] = model.decide([state], questions(**fields))
        P = {}
        for i, j in pairs:
            p = ans[f"p_{i}_{j}"].p_yes
            P[(i, j)] = p; P[(j, i)] = 1.0 - p
        strength = {i: sum(P[(i, k)] for k in range(n) if k != i) for i in range(n)}
        cyc = tot = 0
        cyclic_pairs = set()  # unordered pairs that sit in at least one cyclic triple
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len({i, j, k}) < 3:
                        continue
                    tot += 1
                    if P[(i, j)] > 0.5 and P[(j, k)] > 0.5 and P[(k, i)] > 0.5:
                        cyc += 1
                        cyclic_pairs |= {frozenset((i, j)), frozenset((j, k)), frozenset((k, i))}
        matrix = [[round(P[(i, j)], 3) if i != j else None for j in range(n)] for i in range(n)]

        # Threshold the marginals into a partial order, then read off what it implies.
        eps = float(epsilon)
        edge_w = {(i, j): P[(i, j)] for i in range(n) for j in range(n)
                  if i != j and P[(i, j)] >= 1 - eps}
        # A mid-band pair (neither direction confident) is FLEXIBLE only if it is coherent
        # -- not in any cyclic triple. A mid-band pair inside a cycle is UNRESOLVED: the
        # model is confused about it, not indifferent. A single 0.5 marginal can't tell
        # these apart; the triples can.
        flexible, unresolved = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if eps < P[(i, j)] < 1 - eps:
                    row = [i, j, round(P[(i, j)], 3)]
                    (unresolved if frozenset((i, j)) in cyclic_pairs else flexible).append(row)
        edges, repaired = _break_cycles(n, edge_w)          # a coherent model yields a DAG
        reach = _reachability(n, edges)
        reduced = _transitive_reduction(n, edges, reach)    # direct precedences only
        n_anc = [sum(1 for a in range(n) if i in reach[a]) for i in range(n)]
        ranges = [[1 + n_anc[i], n - len(reach[i])] for i in range(n)]  # feasible position span, 1-indexed
        order = _toposort(n, edges, strength)               # argmax order, respects the constraints
        levels = _levels(n, edges, order, strength)         # concurrency groups
        conf = 1.0
        for a in range(len(order) - 1):
            conf *= P[(order[a], order[a + 1])]
        # Probability-weighted position (1-indexed): beats everyone -> 1, loses to all -> n.
        # Continuous, so items sharing a hard feasible span still separate by confidence.
        expected = [round(n - strength[i], 3) for i in range(n)]
        return {
            "order": order, "ordered_items": [items[i] for i in order],
            "graph": {"edges": [[u, v, round(edges[(u, v)], 3)] for (u, v) in reduced], "reduced": True},
            "ranges": ranges, "expected": expected, "levels": levels,
            "flexible_pairs": flexible, "unresolved_pairs": unresolved,
            "linear_extensions": _count_linear_extensions(n, edges),
            "top_orders": _ranked_orders(n, P),
            "consistency": round(1 - cyc / max(tot, 1), 3), "repaired_edges": repaired,
            "confidence": round(conf, 3), "pairs": matrix, "epsilon": eps, "mode": "pairwise",
        }

    raise ValueError(f"unknown order mode {mode!r}")
