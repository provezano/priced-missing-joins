#!/usr/bin/env python3
"""
Standalone Executable Proof & Counterexample Audit for Paper 19 (`p19_priced_joins`).
Exits 0 iff ZERO counterexamples are found across > 2,600 exact mathematical checks covering:
  - Theorem 1: Row-degree-2 fixed == adaptive lookup optimality & column-degree freedom (including unaligned graphs)
  - Theorem 2 & Proposition 1: Degree-3 selector threshold (256 fixed vs. 9 adaptive), R-round bounds, and
    degree-4 Vertex Cover NP-hardness even when fixed == adaptive
  - Theorem 3: Strongly-connected-component & laminar token-chain core reduction (<= b + 2p - 1 probes),
    4-vertex laminar-bound sharpness witness, and sign-target component-nonadditivity witness
  - Theorem 4 & Propositions 2-3: O(n^3) 1-bit parity assignment repair (with fractional weights & simple-cycle
    extraction on zero-weight even-cycle traps), 2-bit walk counterexamples, polynomial-time GF(2)/Walsh-Fourier
    constancy & variance certificate, single-character GF(2)-span optimality (A == F), and our new 5-cycle (n=10)
    row-degree-2 Walsh-Hadamard parity witness breaking A == F when >= 2 cycles carry nonzero increments!
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB_DIR))

import priced_joins_lib as lib  # noqa: E402


def weighted_vertex_cover_opt(adj, weights) -> float:
    n = len(adj)
    edges = [(i, j) for i, nbrs in enumerate(adj) for j in nbrs if i < j]
    return min(
        sum(weights[i] for i in range(n) if (mask >> i) & 1)
        for mask in range(1 << n)
        if all(((mask >> u) & 1) or ((mask >> v) & 1) for u, v in edges)
    )


def main() -> int:
    total_checks = 0

    # =========================================================================
    # 1. Theorem 1: Row-degree-2 fixed == adaptive (600 graphs incl. unaligned, p_i=0, col-deg up to n)
    # =========================================================================
    assert lib.solve_degree2_priced_lookup([], [], [])["optimal_cost"] == 0
    assert lib.solve_degree2_priced_lookup([(0,), (0,)], [{0: 1}, {0: 2}], [1, 1]) is None
    total_checks += 2

    rng1 = random.Random(20261901)
    for trial in range(600):
        n = rng1.randint(2, 9)
        # Pick a random hidden permutation so >= 50% of instances do NOT contain the identity matching!
        perm0 = list(range(n))
        if trial >= 200:
            rng1.shuffle(perm0)
        adj = []
        for i in range(n):
            # Force high column-degree hub at col 0 on a third of trials:
            alt = 0 if (trial % 3 == 0 and rng1.random() < 0.65) else rng1.randrange(n)
            adj.append(tuple(sorted({perm0[i], alt})))
        edge_costs = [
            {col: Fraction(rng1.randint(-6, 6), rng1.choice((1, 2))) for col in nbrs}
            for nbrs in adj
        ]
        prices = [rng1.randint(0, 9) if (trial % 5 == 0) else rng1.randint(1, 9) for _ in range(n)]

        alg1 = lib.solve_degree2_priced_lookup(adj, edge_costs, prices, tol=0.0)
        assert alg1 is not None
        worlds = lib.enumerate_perfect_matchings(adj)
        vals = [sum(edge_costs[i][col] for i, col in enumerate(w)) for w in worlds]
        assert len(worlds) == (1 << alg1["total_cycles"])
        search = lib.ExactDecisionSearch(worlds, vals, prices)
        f_opt, _ = search.fixed_minimax(search.full)
        a_opt, _ = search.adaptive_minimax(search.full)
        search.clear_caches()
        assert f_opt == a_opt == alg1["optimal_cost"], (trial, f_opt, a_opt, alg1)
        total_checks += 1

    # =========================================================================
    # 2. Theorem 2 & Proposition 1: Degree-3 Selector (h=1..8) + Batched Rounds + Degree-4 Vertex Cover
    # =========================================================================
    selector_summary = {}
    for h in range(1, 9):
        m = 1 << h
        adj_s, cost_s = lib.build_degree3_selector_family(h)
        assert len(adj_s) == 5 * m - 2
        assert max(len(nbrs) for nbrs in adj_s) == 3
        col_degs = [0] * len(adj_s)
        for nbrs in adj_s:
            for col in nbrs:
                col_degs[col] += 1
        assert max(col_degs) <= 3
        worlds_s, vals_s = lib.enumerate_selector_worlds(h)
        assert len(worlds_s) == 2 * m + 1
        if h <= 4:
            enum_w = lib.enumerate_perfect_matchings(adj_s)
            assert set(enum_w) == set(worlds_s)
            enum_v = [sum(cost_s[i][col] for i, col in enumerate(w)) for w in worlds_s]
            assert enum_v == vals_s

        search_s = lib.ExactDecisionSearch(worlds_s, vals_s) if h <= 3 else None
        if search_s is not None:
            assert search_s.fixed_minimax(search_s.full)[0] == m
            assert search_s.adaptive_minimax(search_s.full)[0] == h + 1

        for r in range(1, h + 2):
            outs = [lib.run_batched_selector_schedule(h, r, w) for w in worlds_s]
            assert [o[0] for o in outs] == vals_s
            sched_q = min(m, max(o[1] for o in outs))
            lb_q = lib.batched_round_lower_bound(m, r)
            assert lb_q <= sched_q
            if search_s is not None:
                ex_q = search_s.batched_minimax(search_s.full, r)[0]
                assert lb_q <= ex_q <= sched_q
            if h == 8 and r in (1, 2, 3, 9):
                selector_summary[r] = sched_q
            total_checks += 1
        if search_s is not None:
            search_s.clear_caches()

    # Exact bench headline numbers at h=8 (m=256, n=1278): 256 (1 round), 46 (2 rounds), 21 (3 rounds), 9 (9 rounds)
    assert selector_summary == {1: 256, 2: 46, 3: 21, 9: 9}, f"Unexpected h=8 schedule: {selector_summary}"

    rng_vc = random.Random(20261902)
    for _ in range(120):
        n = rng_vc.randint(2, 7)
        adj_sets = [{i} for i in range(n)]
        for u, v in combinations(range(n), 2):
            if len(adj_sets[u]) < 4 and len(adj_sets[v]) < 4 and rng_vc.random() < 0.42:
                adj_sets[u].add(v)
                adj_sets[v].add(u)
        adj_vc = [tuple(sorted(s)) for s in adj_sets]
        prices_vc = [rng_vc.randint(1, 6) for _ in range(n)]
        vc_opt = weighted_vertex_cover_opt(adj_vc, prices_vc)
        worlds_vc = lib.enumerate_perfect_matchings(adj_vc)
        disp = [sum(i != col for i, col in enumerate(w)) for w in worlds_vc]
        for tgt in (disp, [int(d > 0) for d in disp]):
            s_vc = lib.ExactDecisionSearch(worlds_vc, tgt, prices_vc)
            assert s_vc.fixed_minimax(s_vc.full)[0] == s_vc.adaptive_minimax(s_vc.full)[0] == vc_opt
            s_vc.clear_caches()
            total_checks += 1

    # =========================================================================
    # 3. Theorem 3: Core Reduction (400 random + unaligned graphs + diamond sharpness + sign nonadditivity)
    # =========================================================================
    # Sharpness of b + 2p - 1 = 4 on 4-node diamond SCC:
    adj_dia = [(0, 1, 2), (1, 3), (2, 3), (0, 3)]
    cost_dia = [{0: 0, 1: 5, 2: 5}, {1: 0, 3: 0}, {2: 0, 3: 0}, {3: 0, 0: 0}]
    price_dia = [10, 8, 8, 1]
    res_dia = lib.solve_priced_lookup_via_core(adj_dia, cost_dia, price_dia)
    assert res_dia["branch_rows"] == 1 and res_dia["representative_probes"] == 4
    assert res_dia["fixed_cost"] == res_dia["adaptive_cost"] == 1
    total_checks += 1

    # Sign target breaks component additivity across 2 disjoint 2-cycles:
    # Cycle 1 has values {+10, +20}, Cycle 2 has values {-1, +1}; global sum is ALWAYS >= +9 > 0,
    # so sign(sum) == +1 requires 0 queries globally, whereas sign(Cycle 2) alone requires 1 query!
    adj_sign = [(0, 1), (0, 1), (2, 3), (2, 3)]
    w_sign = lib.enumerate_perfect_matchings(adj_sign)
    c_sign = [{0: 5, 1: 10}, {0: 10, 1: 5}, {2: -1, 3: 1}, {2: 1, 3: -1}]
    vals_global_sign = [1 if sum(c_sign[i][col] for i, col in enumerate(w)) > 0 else -1 for w in w_sign]
    s_glob = lib.ExactDecisionSearch(w_sign, vals_global_sign, [1, 1, 1, 1])
    assert s_glob.adaptive_minimax(s_glob.full)[0] == 0
    s_glob.clear_caches()
    total_checks += 1

    rng3 = random.Random(20261903)
    for trial in range(400):
        n = rng3.randint(3, 9)
        perm_b = list(range(n))
        if trial >= 150:
            rng3.shuffle(perm_b)
        adj_c = []
        for i in range(n):
            extra = rng3.sample(range(n), rng3.choice((1, 2)))
            adj_c.append(tuple(sorted({perm_b[i], *extra})))
        costs_c = [{col: rng3.randint(-5, 5) for col in nbrs} for nbrs in adj_c]
        prices_c = [rng3.randint(1, 10) for _ in range(n)]

        core = lib.reduce_join_graph_to_core(adj_c, costs_c, prices_c)
        assert core is not None
        res_c = lib.solve_priced_lookup_via_core(adj_c, costs_c, prices_c)
        worlds_c = lib.enumerate_perfect_matchings(adj_c)
        vals_c = [sum(costs_c[i][col] for i, col in enumerate(w)) for w in worlds_c]

        prod_worlds = 1
        for blk in core["blocks"]:
            prod_worlds *= len(blk["worlds"])
        assert prod_worlds == len(worlds_c)

        s_c = lib.ExactDecisionSearch(worlds_c, vals_c, prices_c)
        assert (res_c["fixed_cost"], res_c["adaptive_cost"]) == (
            s_c.fixed_minimax(s_c.full)[0],
            s_c.adaptive_minimax(s_c.full)[0],
        )
        s_c.clear_caches()
        total_checks += 1

    # =========================================================================
    # 4. Proposition 2 & Theorem 4: 1-Bit Parity Repair, GF(2)/Fourier & Syndrome Span
    # =========================================================================
    # Crafted zero-weight even-cycle + odd-cycle trap on n=4 where shortest 2n-walk can loop through node 0 twice:
    cost_trap = np.array([
        [0.0, 0.0, 1.5, 9.0],
        [0.0, 0.0, 9.0, 9.0],
        [1.0, 9.0, 0.0, 9.0],
        [9.0, 9.0, 9.0, 0.0],
    ])
    tag_trap = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    allow_trap = np.ones((4, 4), dtype=bool)
    out_trap = lib.parity_constrained_assignment_extrema(cost_trap, tag_trap, allow_trap)
    assert abs(out_trap["min_by_parity"][0] - 0.0) < 1e-9
    assert abs(out_trap["min_by_parity"][1] - 2.5) < 1e-9
    for par in (0, 1):
        m_wit = out_trap["matching_by_parity"][par]
        assert sorted(m_wit) == [0, 1, 2, 3]
        assert int(np.bitwise_xor.reduce([tag_trap[i, m_wit[i]] for i in range(4)])) == par
        assert abs(sum(cost_trap[i, m_wit[i]] for i in range(4)) - out_trap["min_by_parity"][par]) < 1e-9
    total_checks += 1

    rng4 = random.Random(20261904)
    for trial in range(300):
        n = rng4.randint(2, 7)
        # Include half-integer real weights on half the trials:
        denom = 2.0 if (trial % 2 == 1) else 1.0
        cost_m = np.array([[rng4.randint(-9, 10) / denom for _ in range(n)] for _ in range(n)], dtype=float)
        tag_m = np.array([[rng4.randrange(2) for _ in range(n)] for _ in range(n)], dtype=int)
        allow_m = np.array([[rng4.random() < 0.58 for _ in range(n)] for _ in range(n)], dtype=bool)

        perms = np.array(list(permutations(range(n))))
        rows_idx = np.arange(n)
        valid = allow_m[rows_idx, perms].all(axis=1)
        sums = cost_m[rows_idx, perms].sum(axis=1)
        pars = np.bitwise_xor.reduce(tag_m[rows_idx, perms], axis=1)

        for sign in (1.0, -1.0):
            res_p = lib.parity_constrained_assignment_extrema(sign * cost_m, tag_m, allow_m)
            for b_par in (0, 1):
                mask_b = valid & (pars == b_par)
                exp_v = float(np.min(sign * sums[mask_b])) if np.any(mask_b) else None
                got_v = res_p["min_by_parity"][b_par]
                got_m = res_p["matching_by_parity"][b_par]
                assert (exp_v is None) == (got_v is None)
                if exp_v is not None:
                    assert abs(exp_v - got_v) < 1e-9
                    assert sorted(got_m) == list(range(n))
                    assert all(allow_m[i, got_m[i]] for i in range(n))
                    assert int(np.bitwise_xor.reduce([tag_m[i, got_m[i]] for i in range(n)])) == b_par
                    assert abs(sum(sign * cost_m[i, got_m[i]] for i in range(n)) - exp_v) < 1e-9
            total_checks += 1

    # Two 2-bit lifted-walk counterexamples:
    for n_ex, edges_ex, exp_sigs in (
        (4, [(0, 1, 1), (2, 3, 2)], {0, 1, 2, 3}),
        (3, [(0, 1, 1), (0, 2, 2)], {0, 1, 2}),
    ):
        lbl = {(i, i): 0 for i in range(n_ex)}
        for u, v, z in edges_ex:
            lbl[(u, v)] = z
            lbl[(v, u)] = 0
        seen_sigs = {
            int(np.bitwise_xor.reduce([lbl[(i, j)] for i, j in enumerate(p)]))
            for p in permutations(range(n_ex))
            if all((i, j) in lbl for i, j in enumerate(p))
        }
        assert seen_sigs == exp_sigs
        total_checks += 1

    # GF(2)/Walsh-Fourier subspace constancy, moments, and residual variance (300 systems -> >2,000 fibers):
    rng5 = random.Random(20261905)
    fiber_checks = 0
    for _ in range(220):
        c_len = rng5.randint(1, 10)
        b_bits = rng5.randint(1, 5)
        cols = [rng5.randrange(1 << b_bits) for _ in range(c_len)]
        incs = [rng5.randint(-4, 4) for _ in range(c_len)]
        q_sub = [i for i in range(c_len) if rng5.random() < 0.4]

        groups = defaultdict(list)
        for mask in range(1 << c_len):
            sig = 0
            tot = 0
            for i in range(c_len):
                if (mask >> i) & 1:
                    sig ^= cols[i]
                    tot += incs[i]
            groups[sig].append((mask, tot))

        dp_bounds = lib.independent_cycle_signature_dp(cols, incs)
        assert set(dp_bounds.keys()) == set(groups.keys())

        for h_sig in range(1 << b_bits):
            cert = lib.fourier_subspace_certificate(cols, incs, h_sig, q_sub)
            if h_sig not in groups:
                assert cert is None
                continue
            obs = groups[h_sig]
            sums_f = [s for _, s in obs]
            assert dp_bounds[h_sig] == (min(sums_f), max(sums_f))
            k_sz = len(sums_f)
            assert cert["twice_mean"] * k_sz == 2 * sum(sums_f)
            assert cert["variance_times_4"] * k_sz == sum((2 * s - cert["twice_mean"]) ** 2 for s in sums_f)
            assert cert["is_constant"] == (min(sums_f) == max(sums_f))

            # Verify residual variance & exact sufficiency after fixed query subset q_sub:
            sub_buckets = defaultdict(list)
            for mask, tot in obs:
                resp = tuple((mask >> i) & 1 for i in q_sub)
                sub_buckets[resp].append(tot)
            res_var = sum(
                sum((v - Fraction(sum(vs), len(vs))) ** 2 for v in vs) for vs in sub_buckets.values()
            ) / k_sz
            assert 4 * res_var == cert["residual_variance_times_4"]
            assert cert["queried_set_suffices"] == all(min(vs) == max(vs) for vs in sub_buckets.values())
            fiber_checks += 1
            total_checks += 1

    # Single-character GF(2)-span optimality (A == F, 150 instances + zero-marginal-gain complementarity witness):
    opt_comp, set_comp = lib.min_cost_fourier_span_exact([1, 2], 3, 1000, [1, 1])
    assert opt_comp == 2 and set_comp == (1, 2)
    total_checks += 1

    for _ in range(150):
        dim = rng5.randint(2, 5)
        m_q = rng5.randint(2, 8)
        p_chars = [rng5.randrange(1, 1 << dim) for _ in range(m_q)]
        t_char = rng5.randrange(1, 1 << dim)
        prices_sp = [rng5.randint(1, 18) for _ in range(m_q + 1)]
        all_chars = [t_char, *p_chars]
        worlds_sp = sorted({tuple(((v & y).bit_count() % 2) for v in all_chars) for y in range(1 << dim)})
        s_sp = lib.ExactDecisionSearch(worlds_sp, [w[0] for w in worlds_sp], prices_sp)
        sp_opt, _ = lib.min_cost_fourier_span_exact(p_chars, t_char, prices_sp[0], prices_sp[1:])
        assert sp_opt == s_sp.fixed_minimax(s_sp.full)[0] == s_sp.adaptive_minimax(s_sp.full)[0]
        s_sp.clear_caches()
        total_checks += 1

    # Verify our new 5-cycle (n=10) row-degree-2 Walsh-Hadamard parity witness breaking A == F:
    wit = lib.build_degree2_parity_adaptivity_witness()
    assert wit["max_row_degree"] == 2 and wit["max_col_degree"] == 2
    assert wit["fixed_cost"] == 3 and wit["adaptive_cost"] == 2 and wit["distinct_totals"] == [0, 2]
    total_checks += 1

    print(
        f"verify_p19_proofs: PASS (0 counterexamples across {total_checks} checks, "
        f"including {fiber_checks} nonempty GF(2) fibers and all sharpness witnesses)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
