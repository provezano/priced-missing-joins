#!/usr/bin/env python3
"""
Deterministic quantitative benchmark runner for Paper 19 (`p19_priced_joins_clean_2026-11-22`).
Generates:
  1. exp1_degree_threshold_and_rounds.csv
  2. exp2_core_reduction_and_chains.csv
  3. exp3_sqlite_manifests_and_sakila.csv
  4. exp4_gf2_fourier_and_pricing.csv
  5. P19_RESULTS.json
  6. SHA256SUMS.txt
"""
from __future__ import annotations

import bz2
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
from itertools import combinations, permutations
import json
import math
import os
from pathlib import Path
import random
from statistics import median
import struct
import subprocess
import tempfile
import time
from typing import Dict, List
import zlib

import numpy as np

import priced_joins_lib as lib

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
GE_V26 = Path(__file__).resolve().parents[1] / "data" / "sparse_joins"
CHINOOK_DB = Path(__file__).resolve().parents[1] / "data" / "sqlite_dbs" / "Chinook_Sqlite.sqlite"
NORTHWIND_DB = Path(__file__).resolve().parents[1] / "data" / "sqlite_dbs" / "northwind.db"
SAKILA_DB = Path(__file__).resolve().parents[1] / "data" / "sqlite_dbs" / "sakila_master.db"


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stretch_digraph_candidates(adj: List[ tuple ], depth: int) -> List[tuple]:
    graph = [{i} for i in range(len(adj))]
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            if i == j:
                continue
            prev = i
            for _ in range(depth):
                nxt = len(graph)
                graph.append({nxt})
                graph[prev].add(nxt)
                prev = nxt
            graph[prev].add(j)
    return [tuple(sorted(s)) for s in graph]


def run_exp1_degree_threshold() -> Dict[str, object]:
    rows: List[Dict[str, object]] = []

    # Part A: 200 row-degree-2 graphs (including unaligned column numbering & high column degree)
    rng = random.Random(20261911)
    deg2_matches = 0
    for trial in range(200):
        n = rng.randint(3, 9)
        perm0 = list(range(n))
        rng.shuffle(perm0)
        adj = [tuple(sorted({perm0[i], rng.randrange(n)})) for i in range(n)]
        costs = [{col: rng.randint(-4, 4) for col in nbrs} for nbrs in adj]
        prices = [rng.randint(1, 8) for _ in range(n)]
        alg1 = lib.solve_degree2_priced_lookup(adj, costs, prices)
        worlds = lib.enumerate_perfect_matchings(adj)
        vals = [sum(costs[i][col] for i, col in enumerate(w)) for w in worlds]
        s = lib.ExactDecisionSearch(worlds, vals, prices)
        f_opt = s.fixed_minimax(s.full)[0]
        a_opt = s.adaptive_minimax(s.full)[0]
        s.clear_caches()
        assert f_opt == a_opt == alg1["optimal_cost"]
        deg2_matches += 1

    # Part B: Degree-3 selector family h=1..8 across round budgets R=1..h+1
    selector_rows = []
    for h in range(1, 9):
        m = 1 << h
        adj_s, _ = lib.build_degree3_selector_family(h)
        worlds_s, vals_s = lib.enumerate_selector_worlds(h)
        s_ex = lib.ExactDecisionSearch(worlds_s, vals_s) if h <= 3 else None
        for r in range(1, h + 2):
            outs = [lib.run_batched_selector_schedule(h, r, w) for w in worlds_s]
            sched_q = min(m, max(o[1] for o in outs))
            lb_q = lib.batched_round_lower_bound(m, r)
            ex_q = s_ex.batched_minimax(s_ex.full, r)[0] if s_ex is not None else ""
            rec = {
                "height": h,
                "leaves_m": m,
                "n_rows": len(adj_s),
                "worlds": len(worlds_s),
                "rounds_R": r,
                "info_lower_bound": lb_q,
                "scheduled_queries": sched_q,
                "exact_minimax_queries": ex_q,
                "fixed_over_scheduled_ratio": round(m / sched_q, 4),
            }
            rows.append(rec)
            selector_rows.append(rec)
        if s_ex is not None:
            s_ex.clear_caches()

    write_csv(OUT_DIR / "exp1_degree_threshold_and_rounds.csv", rows)
    h8 = {r["rounds_R"]: r["scheduled_queries"] for r in selector_rows if r["height"] == 8}
    return {
        "degree2_random_unaligned_instances": 200,
        "degree2_fixed_equals_adaptive_rate": deg2_matches / 200.0,
        "degree3_h8_n1278_queries_by_rounds": {
            "R1_fixed": h8[1],
            "R2_two_batches": h8[2],
            "R3_three_batches": h8[3],
            "R9_sequential_adaptive": h8[9],
            "fixed_to_adaptive_ratio": round(h8[1] / h8[9], 4),
        },
    }


def run_exp2_core_reduction() -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    rng = random.Random(20261912)
    exact_matches = 0
    for n in range(3, 10):
        for rep in range(32):
            perm0 = list(range(n))
            if rep % 2 == 1:
                rng.shuffle(perm0)
            adj = [tuple(sorted({perm0[i], *rng.sample(range(n), rng.choice((1, 2)))})) for i in range(n)]
            costs = [{col: rng.randint(-4, 4) for col in nbrs} for nbrs in adj]
            prices = [rng.randint(1, 9) for _ in range(n)]
            res = lib.solve_priced_lookup_via_core(adj, costs, prices)
            worlds = lib.enumerate_perfect_matchings(adj)
            vals = [sum(costs[i][col] for i, col in enumerate(w)) for w in worlds]
            s = lib.ExactDecisionSearch(worlds, vals, prices)
            f_dir = s.fixed_minimax(s.full)[0]
            a_dir = s.adaptive_minimax(s.full)[0]
            s.clear_caches()
            assert (res["fixed_cost"], res["adaptive_cost"]) == (f_dir, a_dir)
            exact_matches += 1
            rows.append({
                "family": "random_bipartite",
                "n_rows": n,
                "stretch_depth": 0,
                "worlds": len(worlds),
                "branch_rows": res["branch_rows"],
                "representative_probes": res["representative_probes"],
                "fixed_cost": res["fixed_cost"],
                "adaptive_cost": res["adaptive_cost"],
                "solve_ms": 0.0,
            })

    chain_summary = {}
    base_sel, _ = lib.build_degree3_selector_family(2)
    for depth in (0, 1, 8, 64, 256):
        adj_st = stretch_digraph_candidates(base_sel, depth)
        costs_st = [{col: int(i in (3, 4, 5, 6) and col == max(nbrs)) for col in nbrs} for i, nbrs in enumerate(adj_st)]
        prices_st = [1 + (i % 9) for i in range(len(adj_st))]
        ticks = []
        res_st = None
        for _ in range(3):
            t0 = time.perf_counter()
            res_st = lib.solve_priced_lookup_via_core(adj_st, costs_st, prices_st)
            ticks.append(1000.0 * (time.perf_counter() - t0))
        med_ms = round(median(ticks), 3)
        rows.append({
            "family": "stretched_selector_h2",
            "n_rows": len(adj_st),
            "stretch_depth": depth,
            "worlds": 9,
            "branch_rows": res_st["branch_rows"],
            "representative_probes": res_st["representative_probes"],
            "fixed_cost": res_st["fixed_cost"],
            "adaptive_cost": res_st["adaptive_cost"],
            "solve_ms": med_ms,
        })
        chain_summary[str(depth)] = {
            "n_rows": len(adj_st),
            "branch_rows": res_st["branch_rows"],
            "representative_probes": res_st["representative_probes"],
            "fixed_cost": res_st["fixed_cost"],
            "adaptive_cost": res_st["adaptive_cost"],
            "median_ms": med_ms,
        }

    write_csv(OUT_DIR / "exp2_core_reduction_and_chains.csv", rows)
    return {
        "random_instances": 224,
        "exact_price_agreement_rate": exact_matches / 224.0,
        "stretched_selector_scaling": chain_summary,
    }


def trace_sqlite_database(exe: Path, db_path: Path, table1: str, table2: str) -> Tuple[np.ndarray, int, str, str]:
    proc = subprocess.run([str(exe), str(db_path), table1, table2], check=True, capture_output=True, text=True)
    hits = np.array([int(row[1]) for row in csv.reader(proc.stdout.splitlines())], dtype=np.int64)
    raw_bytes = db_path.read_bytes()
    page_size, rem = divmod(len(raw_bytes), len(hits))
    assert rem == 0
    sha = hashlib.sha256(raw_bytes).hexdigest()
    return hits, page_size, proc.stderr.strip(), sha


def extract_distinct_pages(db_path: Path, page_size: int, n_pages: int, seed: int):
    order = list(range(n_pages))
    rng = random.Random(seed)
    rng.shuffle(order)
    seen = set()
    entries = []
    with db_path.open("rb", buffering=0) as f:
        for idx in order:
            buf = os.pread(f.fileno(), page_size, idx * page_size)
            digest = hashlib.sha256(buf).digest()
            if digest in seen:
                continue
            seen.add(digest)
            delta_comp = len(bz2.compress(buf, compresslevel=1)) - len(zlib.compress(buf, level=1))
            entries.append((idx, digest, delta_comp))
    return entries, rng


def fixed_priority_order(search: lib.ExactDecisionSearch) -> Tuple[int, Tuple[int, ...]]:
    n = search.n

    @lru_cache_local
    def groups(mask: int):
        if mask == 0:
            return (search.full,)
        bit = mask & -mask
        i = bit.bit_length() - 1
        return tuple(c & p for c in groups(mask ^ bit) if not search.is_pure(c) for p in search.partitions[i] if (c & p))

    @lru_cache_local
    def solve(mask: int):
        active = sum(c.bit_count() for c in groups(mask) if not search.is_pure(c))
        if active == 0:
            return 0, ()
        return min((active + v, (i,) + rest) for i in range(n) if not ((mask >> i) & 1) for v, rest in [solve(mask | (1 << i))])

    return solve(0)


from functools import lru_cache as lru_cache_local  # noqa: E402


def physical_pread_verify(
    fd: int,
    page_size: int,
    indices: List[int],
    digest_to_col: Dict[bytes, int],
    search: lib.ExactDecisionSearch,
    mode: str,
    prio_order: Tuple[int, ...],
) -> Tuple[int, int]:
    used = 0
    state = search.full
    while not search.is_pure(state):
        if mode == "priority":
            row_i = prio_order[used]
        elif mode == "bayes":
            row_i = search.uniform_bayes_sum(state)[1]
        else:
            row_i = search.adaptive_minimax(state)[1]
        buf = os.pread(fd, page_size, page_size * indices[row_i])
        if len(buf) != page_size:
            raise IOError("Incomplete SQLite page pread.")
        ans_col = digest_to_col[hashlib.sha256(buf).digest()]
        # Narrow compatible world bitmask:
        state = next(
            p & state for p in search.partitions[row_i]
            if search.worlds[next(lib.iter_set_bits(p))][row_i] == ans_col
        )
        assert state != 0
        used += 1
    first_w = next(lib.iter_set_bits(state))
    return used, int(search.values[first_w])


def salted_v26_compat(idx: int, digest: bytes, seed: int = 8511) -> int:
    payload = struct.pack("<QI", idx, seed) + digest
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b"sqlite_bench-pg").digest(), "little")


def run_exp3_sqlite_manifests() -> Dict[str, object]:
    """
    Deterministic SQLite VFS physical page-read benchmark across Chinook, Northwind (161 6-page panels),
    and Sakila (24 disjoint 6-page and 24 disjoint 8-page panels x 8 seeds = 192 panel-seed pairs each).
    """
    rows_out: List[Dict[str, object]] = []
    db_meta: Dict[str, Dict[str, object]] = {}

    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "sqlite_vfs_trace"
        subprocess.run(
            [os.getenv("CC", "cc"), "-O2", "-std=c11", str(GE_V26 / "trace.c"), "-lsqlite3", "-o", str(exe)],
            check=True,
        )

        # 1. Chinook + Northwind physical pread verification under row-distributed vs. joint salted checksums
        perms6 = tuple(permutations(range(6)))
        agg_cn = defaultdict(lambda: {"panels": 0, "settled_on_receipt": 0, "preads": 0, "total_bytes": 0})
        mixed1_totals = {"fixed_worst_sum": 0, "adaptive_worst_sum": 0, "priority_preads": 0, "bayes_preads": 0, "strict_adv_panels": 0}

        for source_name, db_path, t1, t2 in [
            ("Chinook", CHINOOK_DB, "Track", "InvoiceLine"),
            ("Northwind", NORTHWIND_DB, "Order Details", "Orders"),
        ]:
            hits, page_sz, trace_info, sha = trace_sqlite_database(exe, db_path, t1, t2)
            entries, rng = extract_distinct_pages(db_path, page_sz, len(hits), seed=18411)
            n_panels = min(120, len(entries) // 6)
            db_meta[source_name] = {
                "pages": int(len(hits)),
                "distinct_pages": len(entries),
                "page_bytes": page_sz,
                "panels": n_panels,
                "total_vfs_reads": int(np.sum(hits)),
                "sha256": sha,
                "trace": trace_info,
            }

            with db_path.open("rb", buffering=0) as f_db:
                #Note: separate rng streams matching storage.py and fingerprint.py deterministically
                rng_storage = random.Random(18411)
                order_tmp = list(range(len(hits)))
                rng_storage.shuffle(order_tmp)

                for k in range(n_panels):
                    batch = entries[6 * k: 6 * k + 6]
                    inv_st = batch.copy()
                    rng_storage.shuffle(inv_st)
                    idx = [r[0] for r in batch]
                    lookup_st = {r[1]: j for j, r in enumerate(inv_st)}
                    gold_st = tuple(lookup_st[r[1]] for r in batch)
                    dig_int = [int.from_bytes(r[1], "little") for r in inv_st]
                    loss_st = [r[2] for r in inv_st]
                    w_hits = [int(hits[i]) for i in idx]

                    # Mixed 1-bit per-row hint configuration:
                    allowed_m1 = []
                    for i_r, r_item in enumerate(batch):
                        sh = (idx[i_r] % 29) * 8
                        h_bit = (int.from_bytes(r_item[1], "little") >> sh) & 1
                        allowed_m1.append(tuple(j for j, d_val in enumerate(dig_int) if ((d_val >> sh) & 1) == h_bit))
                    ws_m1 = lib.enumerate_perfect_matchings(allowed_m1)
                    vals_m1 = [sum(w_hits[i] * loss_st[j] for i, j in enumerate(pi)) for pi in ws_m1]
                    s_m1 = lib.ExactDecisionSearch(ws_m1, vals_m1)
                    f_w = s_m1.fixed_minimax(s_m1.full)[0]
                    a_w = s_m1.adaptive_minimax(s_m1.full)[0]
                    _, po_m1 = fixed_priority_order(s_m1)
                    pr_q, pr_v = physical_pread_verify(f_db.fileno(), page_sz, idx, lookup_st, s_m1, "priority", po_m1)
                    ba_q, ba_v = physical_pread_verify(f_db.fileno(), page_sz, idx, lookup_st, s_m1, "bayes", po_m1)
                    assert pr_v == ba_v == vals_m1[ws_m1.index(gold_st)]
                    s_m1.clear_caches()
                    mixed1_totals["fixed_worst_sum"] += int(f_w)
                    mixed1_totals["adaptive_worst_sum"] += int(a_w)
                    mixed1_totals["priority_preads"] += pr_q
                    mixed1_totals["bayes_preads"] += ba_q
                    mixed1_totals["strict_adv_panels"] += int(a_w < f_w)

                # Now evaluate 6/12/24-bit row vs. salted joint checksums on the exact same 161 panels:
                for k in range(n_panels):
                    batch = entries[6 * k: 6 * k + 6]
                    inv = batch.copy()
                    rng.shuffle(inv)
                    idx = [r[0] for r in batch]
                    lookup = {r[1]: j for j, r in enumerate(inv)}
                    gold = tuple(lookup[r[1]] for r in batch)
                    ds = [r[1] for r in inv]
                    w_hits = [int(hits[i]) for i in idx]
                    loss = [r[2] for r in inv]
                    h_mat = [[salted_v26_compat(i_pg, d_pg, 8511) for d_pg in ds] for i_pg in idx]

                    for b_bits in (6, 12, 24):
                        payload_bytes = (b_bits + 7) // 8
                        # (a) Row-distributed bits (b//6 per row):
                        w_row = b_bits // 6
                        mask_r = (1 << w_row) - 1
                        adj_r = [
                            tuple(j for j in range(6) if (h_mat[row_i][j] & mask_r) == (h_mat[row_i][gold[row_i]] & mask_r))
                            for row_i in range(6)
                        ]
                        ws_r = lib.enumerate_perfect_matchings(adj_r)
                        # (b) Joint salted XOR checksum:
                        gold_xor = 0
                        for row_i in range(6):
                            gold_xor ^= h_mat[row_i][gold[row_i]]
                        mask_j = (1 << b_bits) - 1
                        target_sig = gold_xor & mask_j
                        ws_j = [
                            p for p in perms6
                            if (
                                (h_mat[0][p[0]] ^ h_mat[1][p[1]] ^ h_mat[2][p[2]]
                                 ^ h_mat[3][p[3]] ^ h_mat[4][p[4]] ^ h_mat[5][p[5]]) & mask_j
                            ) == target_sig
                        ]

                        for pol_name, ws_pol in [("row_distributed", ws_r), ("joint_salted_xor", ws_j)]:
                            vals_pol = [sum(w_hits[i] * loss[j] for i, j in enumerate(p)) for p in ws_pol]
                            s_pol = lib.ExactDecisionSearch(ws_pol, vals_pol)
                            settled = int(s_pol.is_pure(s_pol.full))
                            _, po_pol = fixed_priority_order(s_pol)
                            reads_used, out_val = physical_pread_verify(
                                f_db.fileno(), page_sz, idx, lookup, s_pol, "bayes", po_pol
                            )
                            assert out_val == vals_pol[ws_pol.index(gold)]
                            s_pol.clear_caches()
                            tot_b = payload_bytes + page_sz * reads_used
                            key = f"{pol_name}_{b_bits}b"
                            agg_cn[key]["panels"] += 1
                            agg_cn[key]["settled_on_receipt"] += settled
                            agg_cn[key]["preads"] += reads_used
                            agg_cn[key]["total_bytes"] += tot_b
                            rows_out.append({
                                "archive": source_name,
                                "panel_size": 6,
                                "panel_id": k,
                                "seed": 8511,
                                "bits": b_bits,
                                "policy": pol_name,
                                "compatible_worlds": len(ws_pol),
                                "ambiguous_after_receipt": 1 - settled,
                                "physical_preads": reads_used,
                                "payload_plus_read_bytes": tot_b,
                            })

        # 2. Sakila SQLite database replication (n=6 and n=8 disjoint panels x 8 seeds 41601..41608)
        hits_sak, page_sak, trace_sak, sha_sak = trace_sqlite_database(exe, SAKILA_DB, "rental", "inventory")
        entries_sak, rng_sak = extract_distinct_pages(SAKILA_DB, page_sak, len(hits_sak), seed=61631)
        db_meta["Sakila"] = {
            "pages": int(len(hits_sak)),
            "distinct_pages": len(entries_sak),
            "page_bytes": page_sak,
            "total_vfs_reads": int(np.sum(hits_sak)),
            "sha256": sha_sak,
            "trace": trace_sak,
        }

        sakila_summary = {}
        offset = 0
        for n_sz in (6, 8):
            perms_arr = np.array(list(permutations(range(n_sz))))
            idx_arr = np.arange(n_sz)
            sak_counts = defaultdict(lambda: {"informative_pairs": 0, "ambiguous_pairs": 0, "total_colliders": 0})

            for panel_id in range(24):
                batch = entries_sak[offset: offset + n_sz]
                offset += n_sz
                inv = batch.copy()
                rng_sak.shuffle(inv)
                w_vec = hits_sak[[e[0] for e in batch]]
                a_vec = np.array([e[2] for e in inv], dtype=np.int64)
                gold_idx = np.array([inv.index(v) for v in batch], dtype=np.int64)
                vals_all = a_vec[perms_arr] @ w_vec
                truth_val = int(a_vec[gold_idx] @ w_vec)
                wrong_mask = vals_all != truth_val
                is_informative = bool(np.any(wrong_mask))

                for seed in range(41601, 41609):
                    tbl = np.array([[salted_v26_compat(i_pg, d_pg, seed) for _, d_pg, _ in inv] for i_pg, _, _ in batch], dtype=np.uint64)
                    diff = tbl[idx_arr, perms_arr] ^ tbl[idx_arr, gold_idx]
                    joint = np.bitwise_xor.reduce(diff, axis=1)
                    for b_bits in (6, 12, 24):
                        row_masks = np.array([(1 << (b_bits // n_sz + int(i < b_bits % n_sz))) - 1 for i in idx_arr], dtype=np.uint64)
                        ok_row = ~np.any(diff & row_masks, axis=1)
                        ok_joint = (joint & np.uint64((1 << b_bits) - 1)) == 0
                        for pol_name, ok_vec in [("row_distributed", ok_row), ("joint_salted_xor", ok_joint)]:
                            colliders = int(np.count_nonzero(wrong_mask & ok_vec))
                            ambig = int(colliders > 0)
                            key = f"n{n_sz}_{pol_name}_{b_bits}b"
                            if is_informative:
                                sak_counts[key]["informative_pairs"] += 1
                                sak_counts[key]["ambiguous_pairs"] += ambig
                                sak_counts[key]["total_colliders"] += colliders
                            if seed == 41601:
                                rows_out.append({
                                    "archive": "Sakila",
                                    "panel_size": n_sz,
                                    "panel_id": panel_id,
                                    "seed": seed,
                                    "bits": b_bits,
                                    "policy": pol_name,
                                    "compatible_worlds": int(np.count_nonzero(ok_vec)),
                                    "ambiguous_after_receipt": ambig,
                                    "physical_preads": -1,
                                    "payload_plus_read_bytes": (b_bits + 7) // 8,
                                })
            sakila_summary[f"n{n_sz}"] = dict(sak_counts)

    write_csv(OUT_DIR / "exp3_sqlite_manifests_and_sakila.csv", rows_out)
    return {
        "databases": db_meta,
        "chinook_northwind_mixed1_161_panels": mixed1_totals,
        "chinook_northwind_161_panels_by_manifest": dict(agg_cn),
        "sakila_8seed_replication": sakila_summary,
    }


def run_exp4_gf2_fourier() -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(20261914)

    for n in (8, 16, 32, 64, 128):
        cost = rng.integers(-50, 51, size=(n, n)).astype(float)
        tag = rng.integers(0, 2, size=(n, n))
        allow = np.ones((n, n), dtype=bool)
        ticks = []
        res_p = None
        for _ in range(3):
            t0 = time.perf_counter()
            res_p = lib.parity_constrained_assignment_extrema(cost, tag, allow)
            ticks.append(1000.0 * (time.perf_counter() - t0))
        rows.append({
            "method": "parity_1bit_o_n3",
            "n_or_cycles": n,
            "bits": 1,
            "metric_primary": res_p["min_by_parity"][0],
            "metric_secondary": res_p["min_by_parity"][1],
            "median_ms": round(median(ticks), 3),
        })

    for c_len, b_bits in ((40, 4), (100, 8), (200, 12), (1000, 16)):
        cols = rng.integers(0, 1 << b_bits, size=c_len).tolist()
        incs = rng.integers(-5, 6, size=c_len).tolist()
        if c_len <= 200:
            ticks_dp = []
            dp_res = None
            for _ in range(3):
                t0 = time.perf_counter()
                dp_res = lib.independent_cycle_signature_dp(cols, incs)
                ticks_dp.append(1000.0 * (time.perf_counter() - t0))
            rows.append({
                "method": "signature_extrema_dp_2b",
                "n_or_cycles": c_len,
                "bits": b_bits,
                "metric_primary": dp_res[0][0],
                "metric_secondary": dp_res[0][1],
                "median_ms": round(median(ticks_dp), 3),
            })
        ticks_f = []
        cert = None
        for _ in range(3):
            t0 = time.perf_counter()
            cert = lib.fourier_subspace_certificate(cols, incs, 0)
            ticks_f.append(1000.0 * (time.perf_counter() - t0))
        rows.append({
            "method": "gf2_fourier_constancy_poly",
            "n_or_cycles": c_len,
            "bits": b_bits,
            "metric_primary": float(cert["twice_mean"]),
            "metric_secondary": float(cert["variance_times_4"]),
            "median_ms": round(median(ticks_f), 3),
        })

    write_csv(OUT_DIR / "exp4_gf2_fourier_and_pricing.csv", rows)
    wit = lib.build_degree2_parity_adaptivity_witness()
    return {
        "scaling_rows": rows,
        "degree2_parity_adaptivity_witness": wit,
    }


def write_sha256_manifest(directory: Path, filenames: List[str]) -> None:
    lines = []
    for fname in sorted(filenames):
        digest = hashlib.sha256((directory / fname).read_bytes()).hexdigest()
        lines.append(f"{digest}  {fname}")
    (directory / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    exp1 = run_exp1_degree_threshold()
    exp2 = run_exp2_core_reduction()
    exp3 = run_exp3_sqlite_manifests()
    exp4 = run_exp4_gf2_fourier()

    summary = {
        "study": "p19_priced_joins_clean_2026-11-22",
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "exp1_degree_threshold": exp1,
        "exp2_core_reduction": exp2,
        "exp3_sqlite_manifests": exp3,
        "exp4_gf2_fourier": exp4,
    }
    (OUT_DIR / "P19_RESULTS.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    tracked = [
        "priced_joins_lib.py",
        "run_p19_experiments.py",
        "exp1_degree_threshold_and_rounds.csv",
        "exp2_core_reduction_and_chains.csv",
        "exp3_sqlite_manifests_and_sakila.csv",
        "exp4_gf2_fourier_and_pricing.csv",
        "P19_RESULTS.json",
        "PRIORITY_AND_PROOF_REPORT.md",
    ]
    write_sha256_manifest(OUT_DIR, tracked)
    print(f"run_p19_experiments: DONE in {summary['elapsed_seconds']}s -> {OUT_DIR / 'P19_RESULTS.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
