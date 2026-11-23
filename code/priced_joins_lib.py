#!/usr/bin/env python3
"""
Standalone reference library for Paper 19:
  1. Algorithm 1: Linear-time exact solver for row-degree-2 priced bipartite lookup (fixed == adaptive).
  2. Construction 1: Degree-3 bipartite selector family achieving F = 2^h vs. A = h + 1 and R-round schedule.
  3. Algorithm 2: Strongly-connected-component and laminar token-chain core reduction (at most b + 2p - 1 probes).
  4. Algorithm 3: Single-bit parity matching repair (supporting real weights and valid simple-cycle reconstruction),
     multi-bit GF(2) Gaussian elimination + Walsh-Fourier constancy and variance certificate, and exact minimum-cost
     GF(2) character-span solver.
  5. Sharpness witnesses:
     - Unaligned bipartite graphs where the identity is not a matching (handled via Hopcroft-Karp re-indexing).
     - 4-vertex diamond component attaining the laminar representative bound b + 2p - 1 = 4 with strict price gain.
     - Thresholded/sign target breaking component additivity across strongly connected components.
     - Multi-bit lifted-walk counterexamples (disconnected 01+10 cycles vs. shared-hub 01+10 cycles).
     - 5-cycle (n = 10) row-degree-2 linear-parity Walsh-Hadamard witness breaking A == F (A = 2 < 3 = F) when
       multiple cycles carry nonzero scalar increments, contrasting with A == F for a single nonzero cycle.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations, permutations, product
import math
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


# ============================================================================
# Part 0: Reference Bipartite Perfect Matching & Exact Enumeration
# ============================================================================

def find_reference_perfect_matching(adj: Sequence[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    """
    Find a feasible perfect matching M: row i -> col M[i] on an n x n bipartite candidate graph,
    or return None if no perfect matching exists.
    Handles empty graphs (n = 0) and unaligned graphs where i not in adj[i].
    """
    n = len(adj)
    if n == 0:
        return ()
    rows: List[int] = []
    cols: List[int] = []
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            if 0 <= j < n:
                rows.append(i)
                cols.append(j)
    if not rows:
        return None
    bi = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    perm = maximum_bipartite_matching(bi, perm_type="column")
    if np.any(perm < 0):
        return None
    return tuple(int(x) for x in perm)


def enumerate_perfect_matchings(adj: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    """Enumerate all feasible perfect matchings of an n x n bipartite candidate graph."""
    n = len(adj)
    if n == 0:
        return [()]
    order = sorted(range(n), key=lambda i: len(adj[i]))
    assignment = [-1] * n

    def visit(at: int, used_mask: int):
        if at == n:
            yield tuple(assignment)
            return
        i = order[at]
        for j in adj[i]:
            if 0 <= j < n and not ((used_mask >> j) & 1):
                assignment[i] = j
                yield from visit(at + 1, used_mask | (1 << j))

    return list(visit(0, 0))


def iter_set_bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class ExactDecisionSearch:
    """
    Exact finite-world optimizer for worst-case fixed query sets, worst-case adaptive decision trees,
    R-round batched policies, and uniform-prior expected query policies.
    """

    def __init__(
        self,
        worlds: Sequence[Tuple[int, ...]],
        values: Sequence[object],
        prices: Optional[Sequence[float]] = None,
    ) -> None:
        if not worlds:
            raise ValueError("Candidate set has zero feasible matchings.")
        self.worlds = tuple(worlds)
        self.values = tuple(values)
        self.n = len(self.worlds[0])
        self.prices = tuple(prices) if prices is not None else (1,) * self.n
        self.full = (1 << len(self.worlds)) - 1

        self.partitions: List[Tuple[int, ...]] = []
        for i in range(self.n):
            ans: Dict[int, int] = defaultdict(int)
            for k, w in enumerate(self.worlds):
                ans[w[i]] |= 1 << k
            self.partitions.append(tuple(ans.values()))

        by_val: Dict[object, int] = defaultdict(int)
        for k, v in enumerate(self.values):
            by_val[v] |= 1 << k
        self.target_classes = tuple(by_val.values())

    def is_pure(self, state_mask: int) -> bool:
        return any((state_mask & t) == state_mask for t in self.target_classes)

    def informative_options(self, state_mask: int) -> List[Tuple[int, Tuple[int, ...]]]:
        seen: Dict[Tuple[int, ...], int] = {}
        for i, part in enumerate(self.partitions):
            Induced = tuple(sorted(u for t in part if (u := state_mask & t)))
            if len(Induced) > 1 and (Induced not in seen or self.prices[i] < self.prices[seen[Induced]]):
                seen[Induced] = i
        return sorted((i, p) for p, i in seen.items())

    @lru_cache(None)
    def adaptive_minimax(self, state_mask: int) -> Tuple[float, int]:
        if self.is_pure(state_mask):
            return 0, -1
        return min(
            (self.prices[i] + max(self.adaptive_minimax(t)[0] for t in parts), i)
            for i, parts in self.informative_options(state_mask)
        )

    @lru_cache(None)
    def fixed_minimax(self, state_mask: int) -> Tuple[float, Tuple[int, ...]]:
        diff_masks = set()
        ids = list(iter_set_bits(state_mask))
        for idx_u, u in enumerate(ids):
            for v in ids[:idx_u]:
                if self.values[u] != self.values[v]:
                    diff_masks.add(
                        sum(1 << i for i, (a, b) in enumerate(zip(self.worlds[u], self.worlds[v])) if a != b)
                    )
        minimal_edges: List[int] = []
        for m in sorted(diff_masks, key=lambda x: (x.bit_count(), x)):
            if all((e & m) != e for e in minimal_edges):
                minimal_edges.append(m)

        @lru_cache(None)
        def min_hitting_set(needed: Tuple[int, ...]) -> Tuple[float, Tuple[int, ...]]:
            if not needed:
                return 0, ()
            return min(
                (self.prices[i] + sub_cost, (i,) + sub_set)
                for i in iter_set_bits(needed[0])
                for sub_cost, sub_set in [min_hitting_set(tuple(e for e in needed if not ((e >> i) & 1)))]
            )

        return min_hitting_set(tuple(minimal_edges))

    @lru_cache(None)
    def batched_minimax(self, state_mask: int, rounds: int) -> Tuple[float, Tuple[int, ...]]:
        if self.is_pure(state_mask):
            return 0, ()
        best = self.fixed_minimax(state_mask)
        if rounds <= 1:
            return best
        if rounds >= self.adaptive_minimax(state_mask)[0] and all(p == 1 for p in self.prices):
            val, idx = self.adaptive_minimax(state_mask)
            return val, (idx,)
        opts = self.informative_options(state_mask)
        min_p = min(self.prices) if self.prices else 1
        max_card = len(opts) if min_p <= 0 else min(len(opts), int(best[0] // min_p) + 1)
        for card in range(1, max_card + 1):
            if card * min_p >= best[0]:
                break
            for batch in combinations(opts, card):
                cost = sum(self.prices[i] for i, _ in batch)
                if cost >= best[0]:
                    continue
                children = [state_mask]
                for _, parts in batch:
                    children = [t & p for t in children for p in parts if (t & p)]
                worst = max(self.batched_minimax(t, rounds - 1)[0] for t in children)
                if cost + worst < best[0]:
                    best = (cost + worst, tuple(i for i, _ in batch))
        return best

    @lru_cache(None)
    def uniform_bayes_sum(self, state_mask: int) -> Tuple[float, int]:
        if self.is_pure(state_mask):
            return 0, -1
        weight = state_mask.bit_count()
        return min(
            (weight * self.prices[i] + sum(self.uniform_bayes_sum(c)[0] for c in parts), i)
            for i, parts in self.informative_options(state_mask)
        )

    def clear_caches(self) -> None:
        self.adaptive_minimax.cache_clear()
        self.fixed_minimax.cache_clear()
        self.batched_minimax.cache_clear()
        self.uniform_bayes_sum.cache_clear()


# ============================================================================
# Part 1: Row-Degree-2 Linear-Time Solver & Degree-3 Selector Family
# ============================================================================

def solve_degree2_priced_lookup(
    adj: Sequence[Sequence[int]],
    edge_costs: Sequence[Dict[int, float]],
    prices: Sequence[float],
    tol: float = 1e-12,
) -> Optional[Dict[str, object]]:
    """
    Algorithm 1 (Theorem 1): Exact O(n + m) solver for additive edge-cost verification when every row
    has at most 2 admissible partner columns (column degrees unrestricted).
    Re-indexes by a reference perfect matching M_0 so identity-alignment is NOT required.
    Returns None if no feasible perfect matching exists; otherwise returns exact optimal cost (A == F),
    optimal fixed row set, and cycle decomposition.
    """
    n = len(adj)
    if any(len(set(nbrs)) > 2 for nbrs in adj):
        raise ValueError("solve_degree2_priced_lookup requires at most 2 distinct candidates per row.")
    if any(p < 0 for p in prices):
        raise ValueError("Row prices must be nonnegative.")

    base_matching = find_reference_perfect_matching(adj)
    if base_matching is None:
        return None
    if n == 0:
        return {"optimal_cost": 0, "query_rows": (), "total_cycles": 0, "nonzero_cycles": 0, "base_matching": ()}

    col_to_row = {col: row for row, col in enumerate(base_matching)}
    # Functional digraph: row i has at most ONE non-base candidate col != base_matching[i],
    # pointing to col_to_row[col]. Outdegree <= 1 implies all directed cycles are vertex-disjoint!
    next_row: List[Optional[int]] = [None] * n
    step_delta: List[float] = [0.0] * n
    for i in range(n):
        base_col = base_matching[i]
        alts = [c for c in set(adj[i]) if c != base_col]
        if alts:
            alt_col = alts[0]
            if alt_col in col_to_row:
                next_row[i] = col_to_row[alt_col]
                step_delta[i] = edge_costs[i][alt_col] - edge_costs[i][base_col]

    state = [0] * n  # 0 = unvisited, 1 = active_stack, 2 = done
    cycles: List[Tuple[int, ...]] = []
    query_rows: List[int] = []
    total_opt = 0

    for start in range(n):
        if state[start] != 0:
            continue
        trail: List[int] = []
        pos: Dict[int, int] = {}
        curr: Optional[int] = start
        while curr is not None and state[curr] == 0:
            state[curr] = 1
            pos[curr] = len(trail)
            trail.append(curr)
            curr = next_row[curr]
        if curr is not None and state[curr] == 1:
            cyc = tuple(trail[pos[curr]:])
            if len(cyc) >= 2:
                cycles.append(cyc)
                cyc_inc = sum(step_delta[u] for u in cyc)
                if abs(cyc_inc) > tol:
                    best_u = min(cyc, key=lambda u: (prices[u], u))
                    query_rows.append(best_u)
                    total_opt += prices[best_u]
        for u in trail:
            state[u] = 2

    return {
        "optimal_cost": total_opt,
        "query_rows": tuple(sorted(query_rows)),
        "total_cycles": len(cycles),
        "nonzero_cycles": len(query_rows),
        "base_matching": base_matching,
    }


def build_degree3_selector_family(height: int) -> Tuple[List[Tuple[int, ...]], List[Dict[int, int]]]:
    """
    Construction 1 (Theorem 2): Bipartite graph with n = 5*2^h - 2 rows and columns, maximum row and
    column degree 3, where fixed verification requires F = 2^h queries and adaptive requires A = h + 1.
    """
    if height < 1:
        raise ValueError("Selector height must be >= 1.")
    m = 1 << height
    n = 5 * m - 2
    adj = [{i} for i in range(n)]
    ret_offset = 2 * m - 1
    extra_offset = 2 * ret_offset

    for parent in range(m - 1):
        for child in (2 * parent + 1, 2 * parent + 2):
            adj[parent].add(child)
            adj[ret_offset + child].add(ret_offset + parent)
    adj[ret_offset].add(0)

    costs: List[Dict[int, int]] = []
    for j in range(m):
        leaf = m - 1 + j
        adj[leaf].update((ret_offset + leaf, extra_offset + j))
        adj[extra_offset + j].add(ret_offset + leaf)

    adj_sorted = [tuple(sorted(s)) for s in adj]
    for i, nbrs in enumerate(adj_sorted):
        row_c = {col: 0 for col in nbrs}
        if m - 1 <= i < 2 * m - 1:
            j = i - (m - 1)
            row_c[extra_offset + j] = 1
        costs.append(row_c)
    return adj_sorted, costs


def enumerate_selector_worlds(height: int) -> Tuple[List[Tuple[int, ...]], List[int]]:
    """Analytic O(m * h) enumeration of all 2*m + 1 feasible perfect matchings of Construction 1."""
    m = 1 << height
    ret_offset = 2 * m - 1
    n = 5 * m - 2
    identity = tuple(range(n))
    worlds = [identity]
    values = [0]
    for i in range(m):
        leaf = m - 1 + i
        anc = [leaf]
        while anc[-1] > 0:
            anc.append((anc[-1] - 1) // 2)
        for bit in (0, 1):
            cyc = list(reversed(anc)) + ([2 * ret_offset + i] if bit else []) + [ret_offset + u for u in anc]
            perm = list(identity)
            for u, v in zip(cyc, cyc[1:] + cyc[:1]):
                perm[u] = v
            worlds.append(tuple(perm))
            values.append(bit)
    return worlds, values


def run_batched_selector_schedule(height: int, rounds: int, world: Sequence[int]) -> Tuple[int, int, int]:
    """Execute R-round balanced subtree schedule on Construction 1; returns (declared_bit, queries_used, rounds_used)."""
    m = 1 << height
    if rounds <= 1:
        answers = [world[m - 1 + i] for i in range(m)]
        return int(any(ans == 4 * m - 2 + i for i, ans in enumerate(answers))), m, 1
    total_queries = 0
    rounds_used = 0
    cur = 0
    for r_idx in range(rounds):
        levels = (height + 1) // rounds + int(r_idx < (height + 1) % rounds)
        if levels == 0:
            break
        layer = [cur]
        batch: List[int] = []
        for _ in range(levels):
            batch.extend(layer)
            layer = [child for u in layer if u < m - 1 for child in (2 * u + 1, 2 * u + 2)]
        total_queries += len(batch)
        rounds_used += 1
        answers = {u: world[u] for u in batch}
        for _ in range(levels):
            if cur >= m - 1:
                leaf_idx = cur - (m - 1)
                return int(answers[cur] == 4 * m - 2 + leaf_idx), total_queries, rounds_used
            if answers[cur] == cur:
                return 0, total_queries, rounds_used
            cur = answers[cur]
    raise RuntimeError("Schedule terminated without reaching leaf.")


def batched_round_lower_bound(m_leaves: int, rounds: int) -> int:
    """Exact AM-GM information lower bound ceil(R * ((1 + 3m)^(1/R) - 1) / 3) for R-round selector verification."""
    return int(math.ceil(rounds * ((1.0 + 3.0 * m_leaves) ** (1.0 / rounds) - 1.0) / 3.0 - 1e-11))


# ============================================================================
# Part 2: Strongly-Connected-Component & Laminar Token-Chain Core Reduction
# ============================================================================

def reduce_join_graph_to_core(
    adj: Sequence[Sequence[int]],
    edge_costs: Sequence[Dict[int, float]],
    prices: Sequence[float],
) -> Optional[Dict[str, object]]:
    """
    Algorithm 2 (Theorem 3): Exact core reduction for arbitrary bipartite candidate graphs.
    1. Finds a reference perfect matching M_0 (handling arbitrary unaligned column numbering).
    2. Builds the directed exchange graph of non-M_0 arcs and decomposes into strongly connected components,
       discarding all inter-component non-matchable arcs.
    3. Compresses each b-branching SCC with p outgoing branch arcs to at most b + 2p - 1 laminar
       token-set representatives, retaining a minimum-price row per token-set equivalence class.
    Returns None if no feasible perfect matching exists.
    """
    n = len(adj)
    base_matching = find_reference_perfect_matching(adj)
    if base_matching is None:
        return None
    if n == 0:
        return {"base_matching": (), "baseline_cost": 0, "blocks": []}

    col_to_row = {col: row for row, col in enumerate(base_matching)}
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(n))
    for i, nbrs in enumerate(adj):
        base_col = base_matching[i]
        for col in set(nbrs):
            if col != base_col and col in col_to_row:
                digraph.add_edge(i, col_to_row[col], col=col)

    baseline_total = sum(edge_costs[i][base_matching[i]] for i in range(n))
    blocks: List[Dict[str, object]] = []

    for scc_set in nx.strongly_connected_components(digraph):
        nodes = sorted(scc_set)
        out_rows = {i: sorted(v for v in digraph.successors(i) if v in scc_set and v != i) for i in nodes}
        branch = [i for i in nodes if len(out_rows[i]) > 1]
        scc_base = sum(edge_costs[i][base_matching[i]] for i in nodes)

        if not branch:
            if len(nodes) == 1:
                blocks.append({
                    "nodes": nodes,
                    "branch_count": 0,
                    "token_count": 0,
                    "reps": [],
                    "worlds": [()],
                    "values": [scc_base],
                    "laminar_sets": [],
                })
            else:
                inc = sum(
                    edge_costs[i][base_matching[out_rows[i][0]]] - edge_costs[i][base_matching[i]]
                    for i in nodes
                )
                rep = min(nodes, key=lambda i: (prices[i], i))
                blocks.append({
                    "nodes": nodes,
                    "branch_count": 0,
                    "token_count": 0,
                    "reps": [rep],
                    "worlds": [(base_matching[rep],), (base_matching[out_rows[rep][0]],)],
                    "values": [scc_base, scc_base + inc],
                    "laminar_sets": [],
                })
            continue

        paths: Dict[Tuple[int, int], Tuple[int, float]] = {}
        token_sets: Dict[int, set] = defaultdict(set)
        for u in branch:
            for v in out_rows[u]:
                tok = (u, v)
                curr = v
                delta = edge_costs[u][base_matching[v]] - edge_costs[u][base_matching[u]]
                while len(out_rows[curr]) == 1:
                    token_sets[curr].add(tok)
                    nxt = out_rows[curr][0]
                    delta += edge_costs[curr][base_matching[nxt]] - edge_costs[curr][base_matching[curr]]
                    curr = nxt
                paths[tok] = (curr, delta)

        by_signature: Dict[frozenset, int] = {}
        for j, tset in token_sets.items():
            sig = frozenset(tset)
            if sig not in by_signature or (prices[j], j) < (prices[by_signature[sig]], by_signature[sig]):
                by_signature[sig] = j

        # Verify laminarity of all distinct single-outgoing token sets:
        sigs = list(by_signature.keys())
        for idx_a, sa in enumerate(sigs):
            for sb in sigs[:idx_a]:
                inter = sa & sb
                if inter and not (sa <= sb or sb <= sa):
                    raise AssertionError("Laminarity violated on single-outgoing stretch.")

        reps = branch + [by_signature[sig] for sig in sigs]
        assert len(reps) <= len(branch) + 2 * len(paths) - 1

        worlds: List[Tuple[int, ...]] = []
        values: List[float] = []
        for choices in product(*[[u, *out_rows[u]] for u in branch]):
            active_toks = {(u, v) for u, v in zip(branch, choices) if u != v}
            ends = [v if u == v else paths[(u, v)][0] for u, v in zip(branch, choices)]
            if len(set(ends)) != len(ends):
                continue
            branch_cols = tuple(base_matching[v] for v in choices)
            stretch_cols = tuple(
                base_matching[out_rows[by_signature[sig]][0]] if (sig & active_toks) else base_matching[by_signature[sig]]
                for sig in sigs
            )
            worlds.append(branch_cols + stretch_cols)
            values.append(scc_base + sum(paths[tok][1] for tok in active_toks))

        blocks.append({
            "nodes": nodes,
            "branch_count": len(branch),
            "token_count": len(paths),
            "reps": reps,
            "worlds": worlds,
            "values": values,
            "laminar_sets": [tuple(sorted(sig)) for sig in sigs],
        })

    return {
        "base_matching": base_matching,
        "baseline_cost": baseline_total,
        "blocks": blocks,
    }


def solve_priced_lookup_via_core(
    adj: Sequence[Sequence[int]],
    edge_costs: Sequence[Dict[int, float]],
    prices: Sequence[float],
) -> Optional[Dict[str, object]]:
    """Solve both exact fixed and adaptive worst-case lookup prices via Algorithm 2 core decomposition."""
    core = reduce_join_graph_to_core(adj, edge_costs, prices)
    if core is None:
        return None
    fixed_total = 0
    adaptive_total = 0
    total_reps = 0
    total_branch = 0
    for blk in core["blocks"]:
        total_reps += len(blk["reps"])
        total_branch += blk["branch_count"]
        if len(blk["worlds"]) <= 1 or len(set(blk["values"])) <= 1:
            continue
        rep_prices = [prices[r] for r in blk["reps"]]
        search = ExactDecisionSearch(blk["worlds"], blk["values"], rep_prices)
        fixed_total += search.fixed_minimax(search.full)[0]
        adaptive_total += search.adaptive_minimax(search.full)[0]
        search.clear_caches()
    return {
        "fixed_cost": fixed_total,
        "adaptive_cost": adaptive_total,
        "branch_rows": total_branch,
        "representative_probes": total_reps,
        "num_blocks": len(core["blocks"]),
    }


# ============================================================================
# Part 3: Parity Assignment Repair & GF(2) / Walsh-Fourier Certificates
# ============================================================================

def parity_constrained_assignment_extrema(
    cost: np.ndarray,
    tag: np.ndarray,
    allowed: np.ndarray,
    tol: float = 1e-11,
) -> Dict[str, object]:
    """
    Algorithm 3a (Proposition 2): O(n^3) exact minimum scalar cost for both 1-bit XOR parity bins (0 and 1),
    together with explicit feasible matching witnesses for each attainable bin.
    Eliminates int() truncation (supports arbitrary real/rational float64 or integer costs) and decomposes
    any non-simple shortest parity-flipping walk into a valid vertex-simple odd alternating cycle!
    """
    n = len(cost)
    if n == 0:
        return {"min_by_parity": (0.0, None), "matching_by_parity": ((), None)}

    weights = np.where(allowed, cost.astype(float), np.inf)
    try:
        rows, assignment = linear_sum_assignment(weights)
    except ValueError:
        return {"min_by_parity": (None, None), "matching_by_parity": (None, None)}
    if np.any(~np.isfinite(weights[rows, assignment])):
        return {"min_by_parity": (None, None), "matching_by_parity": (None, None)}

    base_match = tuple(int(c) for c in assignment)
    base_weights = weights[rows, assignment]
    base_sum = float(np.sum(base_weights))
    base_parity = int(np.bitwise_xor.reduce(tag[rows, assignment] & 1))

    # Build 2n-vertex parity-lifted exchange digraph with predecessor tracking:
    dist = np.full((2 * n, 2 * n), np.inf, dtype=float)
    nxt_node = np.full((2 * n, 2 * n), -1, dtype=int)
    np.fill_diagonal(dist, 0.0)
    for u in range(2 * n):
        nxt_node[u, u] = u

    for i in range(n):
        for j in range(n):
            col = assignment[j]
            if i == j or not allowed[i, col]:
                continue
            flip = int((tag[i, col] ^ tag[i, assignment[i]]) & 1)
            w_arc = float(weights[i, col] - base_weights[i])
            for side in (0, 1):
                u = 2 * i + side
                v = 2 * j + (side ^ flip)
                if w_arc < dist[u, v]:
                    dist[u, v] = w_arc
                    nxt_node[u, v] = v

    # Floyd-Warshall with path reconstruction:
    for k in range(2 * n):
        cand = dist[:, k, None] + dist[None, k, :]
        better = cand < (dist - tol)
        if np.any(better):
            dist = np.where(better, cand, dist)
            k_first = np.broadcast_to(nxt_node[:, k, None], (2 * n, 2 * n))
            nxt_node = np.where(better, k_first, nxt_node)

    diag_repairs = [dist[2 * i, 2 * i + 1] for i in range(n)]
    best_i = int(np.argmin(diag_repairs))
    best_repair = float(diag_repairs[best_i])

    min_vals: List[Optional[float]] = [None, None]
    match_wit: List[Optional[Tuple[int, ...]]] = [None, None]
    min_vals[base_parity] = base_sum
    match_wit[base_parity] = base_match

    if np.isfinite(best_repair):
        # Reconstruct walk in 2n-graph from (2*best_i + 0) to (2*best_i + 1),
        # project to base vertices [0..n-1], and extract a vertex-simple odd-parity cycle
        # of minimum excess cost (handling shared zero-weight even cycles cleanly!):
        u_cur = 2 * best_i
        target_u = 2 * best_i + 1
        walk_base = [best_i]
        guard = 0
        while u_cur != target_u and guard <= 2 * n + 2:
            u_cur = int(nxt_node[u_cur, target_u])
            walk_base.append(u_cur // 2)
            guard += 1

        # Decompose closed walk walk_base (walk_base[0] == walk_base[-1]) into simple directed cycles:
        stack_v: List[int] = []
        pos_v: Dict[int, int] = {}
        simple_cycles: List[List[int]] = []
        for v in walk_base:
            if v in pos_v:
                idx_start = pos_v[v]
                cyc = stack_v[idx_start:]
                simple_cycles.append(cyc)
                for rem in stack_v[idx_start + 1:]:
                    del pos_v[rem]
                stack_v = stack_v[:idx_start + 1]
            else:
                pos_v[v] = len(stack_v)
                stack_v.append(v)

        odd_simple = []
        for cyc in simple_cycles:
            p_cyc = 0
            c_cyc = 0.0
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                col_b = assignment[b]
                p_cyc ^= int((tag[a, col_b] ^ tag[a, assignment[a]]) & 1)
                c_cyc += float(weights[a, col_b] - base_weights[a])
            if p_cyc == 1:
                odd_simple.append((c_cyc, cyc))
        assert odd_simple, "Every odd closed walk must contain at least one odd simple directed cycle!"
        best_odd_cost, best_odd_cyc = min(odd_simple, key=lambda item: item[0])

        alt_match = list(base_match)
        for a, b in zip(best_odd_cyc, best_odd_cyc[1:] + best_odd_cyc[:1]):
            alt_match[a] = base_match[b]
        min_vals[1 - base_parity] = base_sum + best_odd_cost
        match_wit[1 - base_parity] = tuple(alt_match)

    return {
        "min_by_parity": tuple(min_vals),
        "matching_by_parity": tuple(match_wit),
    }


def independent_cycle_signature_dp(
    columns: Sequence[int],
    increments: Sequence[float],
) -> Dict[int, Tuple[float, float]]:
    """Exact O(c * 2^b) dynamic program returning {signature: (min_sum, max_sum)} over independent binary switches."""
    states: Dict[int, Tuple[float, float]] = {0: (0, 0)}
    for col, inc in zip(columns, increments):
        after = states.copy()
        for sig, (lo, hi) in states.items():
            dest = sig ^ int(col)
            if dest in after:
                old_lo, old_hi = after[dest]
                after[dest] = (min(old_lo, lo + inc), max(old_hi, hi + inc))
            else:
                after[dest] = (lo + inc, hi + inc)
        states = after
    return states


def gf2_span(vectors: Iterable[int]) -> set:
    space = {0}
    for v in vectors:
        v_int = int(v)
        space |= {v_int ^ u for u in space}
    return space


def fourier_subspace_certificate(
    columns: Sequence[int],
    increments: Sequence[Fraction | int | float],
    signature: int,
    queried_indices: Sequence[int] = (),
) -> Optional[Dict[str, object]]:
    """
    Algorithm 3b (Theorem 4): Polynomial-time GF(2) Gaussian elimination + Walsh-Fourier character grouping
    on the affine fiber {x in {0,1}^c : A x = signature (mod 2)}.
    Decides feasibility, exact scalar constancy (without enumerating the fiber or solving NP-hard extrema),
    exact uniform-prior mean and variance, single-query variance reductions, and residual variance after
    any fixed lookup subset Q!
    """
    pivots: Dict[int, Tuple[int, int]] = {}
    kernel_masks: List[int] = []

    for j, col in enumerate(columns):
        x = int(col)
        comb = 1 << j
        while x and (x.bit_length() in pivots):
            p_val, p_comb = pivots[x.bit_length()]
            x ^= p_val
            comb ^= p_comb
        if x:
            pivots[x.bit_length()] = (x, comb)
        else:
            kernel_masks.append(comb)

    sig = int(signature)
    x0_mask = 0
    while sig:
        bl = sig.bit_length()
        if bl not in pivots:
            return None
        p_val, p_comb = pivots[bl]
        sig ^= p_val
        x0_mask ^= p_comb

    # Kernel dimension d = len(kernel_masks); row j of N is encoded as an integer bitmask in {0..2^d - 1}:
    characters = [0] * len(columns)
    for t, kmask in enumerate(kernel_masks):
        m_cur = kmask
        while m_cur:
            low = m_cur & -m_cur
            characters[low.bit_length() - 1] |= 1 << t
            m_cur ^= low

    coeffs: Dict[int, object] = defaultdict(lambda: 0)
    for j, (ch, a) in enumerate(zip(characters, increments)):
        signed_a = -a if ((x0_mask >> j) & 1) else a
        coeffs[ch] = coeffs[ch] + signed_a

    b0 = coeffs.pop(0, 0)
    nonzero_coeffs = {ch: val for ch, val in coeffs.items() if val != 0}
    total_inc = sum(increments)
    mean_val = Fraction(total_inc - b0, 2) if all(isinstance(x, (int, Fraction)) for x in increments) else 0.5 * (total_inc - b0)
    var_times_4 = sum(val * val for val in nonzero_coeffs.values())
    single_gains_times_4 = [nonzero_coeffs.get(ch, 0) ** 2 for ch in characters]

    q_span = gf2_span(characters[i] for i in queried_indices)
    residual_var_times_4 = sum(val * val for ch, val in nonzero_coeffs.items() if ch not in q_span)
    q_certifies_exact = all(ch in q_span for ch in nonzero_coeffs)

    return {
        "feasible": True,
        "kernel_dim": len(kernel_masks),
        "particular_mask": x0_mask,
        "characters": tuple(characters),
        "is_constant": len(nonzero_coeffs) == 0,
        "mean": mean_val,
        "twice_mean": total_inc - b0,
        "variance_times_4": var_times_4,
        "single_query_gains_times_4": tuple(single_gains_times_4),
        "residual_variance_times_4": residual_var_times_4,
        "queried_set_suffices": q_certifies_exact,
    }


def min_cost_fourier_span_exact(
    proxy_characters: Sequence[int],
    target_character: int,
    direct_price: float,
    proxy_prices: Sequence[float],
) -> Tuple[float, Tuple[int, ...]]:
    """
    Exact solver for single-target-character minimum-cost spanning set over GF(2) (Theorem 4b).
    Index 0 corresponds to buying the target cycle directly at direct_price; indices 1..m correspond to proxies.
    """
    t = int(target_character)
    if t == 0:
        return 0, ()
    best_cost = direct_price
    best_set: Tuple[int, ...] = (0,)
    m = len(proxy_characters)
    for mask in range(1, 1 << m):
        chosen = [i for i in range(m) if (mask >> i) & 1]
        c = sum(proxy_prices[i] for i in chosen)
        if c >= best_cost:
            continue
        if t in gf2_span(proxy_characters[i] for i in chosen):
            best_cost = c
            best_set = tuple(1 + i for i in chosen)
    return best_cost, best_set


def build_degree2_parity_adaptivity_witness() -> Dict[str, object]:
    """
    Sharpness Witness (Proposition 3): Explicit 5-cycle (n = 10 rows, row-degree 2, column-degree 2)
    bipartite graph equipped with a 2-bit joint XOR parity release where >= 2 cycles carry nonzero
    scalar increments, breaking A == F on a row-degree-2 graph (F = 3 vs. A = 2)!
    Mechanism: Over R, 2 * ((1 - y0)*y1 + y0*y2) = y1 + y2 + (y0 ^ y1) - (y0 ^ y2) embeds the 3-bit
    multiplexer into an additive sum of 5 parity-coupled 2-cycles!
    """
    # 5 disjoint 2-cycles on 10 rows: cycle k swaps rows (2k, 2k+1).
    # Latent y = (y0, y1, y2) in {0,1}^3 has 2 parity checks enforcing x3 = x0 ^ x1 and x4 = x0 ^ x2:
    # A x = 0 (mod 2) with rows [1, 1, 0, 1, 0] and [1, 0, 1, 0, 1].
    adj = [(0, 1), (0, 1), (2, 3), (2, 3), (4, 5), (4, 5), (6, 7), (6, 7), (8, 9), (8, 9)]
    # Cycle increments: a = [0, +1, +1, +1, -1]
    incs = [0, 1, 1, 1, -1]
    cycle_tags = [0b11, 0b01, 0b10, 0b01, 0b10]  # Wait: x0^x1^x3 = 0 (bit 0) and x0^x2^x4 = 0 (bit 1)!
    # Check: col0 = 0b11 (3), col1 = 0b01 (1), col2 = 0b10 (2), col3 = 0b01 (1), col4 = 0b10 (2) -> wait:
    # bit 0: x0 ^ x1 ^ x3 = 0 => x3 = x0 ^ x1? Only if x1 and x3 are the ONLY ones with bit 0!
    # Put bit 0 on {x0, x1, x3} -> col0 |= 1, col1 |= 1, col3 |= 1;
    # Put bit 1 on {x0, x2, x4} -> col0 |= 2, col2 |= 2, col4 |= 2!
    # That IS [3, 1, 2, 1, 2]! Let's verify:
    # Bit 0 of (3*x0 ^ 1*x1 ^ 2*x2 ^ 1*x3 ^ 2*x4) == x0 ^ x1 ^ x3 == 0 <=> x3 = x0 ^ x1!
    # Bit 1 == x0 ^ x2 ^ x4 == 0 <=> x4 = x0 ^ x2!
    # Free variables are (x0, x1, x2) in {0,1}^3 (2^3 = 8 worlds)!
    row_prices = [1, 10, 1, 10, 1, 10, 100, 100, 100, 100]

    # Enumerate compatible matchings in the zero-syndrome fiber:
    all_m = enumerate_perfect_matchings(adj)
    comp_worlds = []
    comp_vals = []
    for m in all_m:
        x = [int(m[2 * k] != 2 * k) for k in range(5)]
        sig = 0
        for xk, ck in zip(x, cycle_tags):
            if xk:
                sig ^= ck
        if sig == 0:
            comp_worlds.append(m)
            comp_vals.append(sum(ak * xk for ak, xk in zip(incs, x)))

    search = ExactDecisionSearch(comp_worlds, comp_vals, row_prices)
    f_cost, f_plan = search.fixed_minimax(search.full)
    a_cost, a_root = search.adaptive_minimax(search.full)
    search.clear_caches()
    return {
        "n_rows": 10,
        "max_row_degree": 2,
        "max_col_degree": 2,
        "parity_bits": 2,
        "compatible_worlds": len(comp_worlds),
        "fixed_cost": f_cost,
        "adaptive_cost": a_cost,
        "fixed_plan": f_plan,
        "adaptive_first_row": a_root,
        "distinct_totals": sorted(set(comp_vals)),
    }


# ============================================================================
# Part 4: Salted Tabulation Manifests & Theoretical Collision Envelope
# ============================================================================

def salted_blake2b_word(logical_idx: int, digest_bytes: bytes, seed: int = 8511) -> int:
    payload = struct.pack("<QI", int(logical_idx), int(seed)) + digest_bytes
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b"p19-pg-tag").digest(), "little")


def collision_probability_bounds(k_wrong_worlds: int, bits: int) -> Tuple[float, float]:
    """
    Exact pairwise-independent tabulation collision bounds (Proposition 4):
      lambda / (1 - p + lambda) <= Pr(Z > 0) <= min(1, lambda),  where p = 2^{-b}, lambda = K * p.
    """
    if k_wrong_worlds <= 0:
        return (0.0, 0.0)
    p = 2.0 ** (-bits)
    lam = k_wrong_worlds * p
    lo = lam / (1.0 - p + lam)
    hi = min(1.0, lam)
    return (lo, hi)
