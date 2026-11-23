#!/usr/bin/env python3
"""
Build the three publication vector figures for Paper 19 (`p19_priced_missing_joins`):
  1. figures/fig1_degree_dichotomy_schematic.pdf (Top: equal-aspect Degree-2 Functional Digraph vs Degree-3 Selector Tree vs Degree-2 Parity Multiplexer; Bottom: R-round query scaling)
  2. figures/fig2_laminar_core_and_rounds.pdf    (2x2 grid: SCC + laminar chain schematic, stretched selector compression to 28 probes, runtime scaling, random graph A==F vs A<F)
  3. figures/fig3_gf2_fourier_and_sqlite.pdf     (2x2 grid: GF(2) constancy vs DP runtime, Chinook/Northwind physical preads, Sakila n=6 & n=8 colliders)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperfig as pf  # noqa: E402

FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"


def build_fig1(summary: dict) -> None:
    pf.apply_style()
    fig = plt.figure(figsize=(6.8, 4.65))
    gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[1.18, 1.15], hspace=0.46)
    gs_top = gs_outer[0].subgridspec(1, 3, wspace=0.22)
    gs_bot = gs_outer[1].subgridspec(1, 2, wspace=0.34)

    # Panel (a): Row-Degree <= 2 Functional Exchange Digraph (A = F in P)
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.set_aspect("equal")
    ax_a.set_xlim(-0.35, 2.55)
    ax_a.set_ylim(-1.55, 1.30)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.axis("off")
    ax_a.set_title("(a) Row Degree $\\leq 2$ ($b = 0$)", fontsize=7.8, fontweight="bold", pad=3)

    # Dead feeder node -> disjoint 2-cycles
    pts_a = {"f": (0.12, 0.72), "u1": (1.08, 0.72), "u2": (2.05, 0.72), "v1": (1.08, -0.35), "v2": (2.05, -0.35)}
    ax_a.add_patch(mpatches.FancyArrowPatch((0.32, 0.72), (0.88, 0.72), arrowstyle="-|>,head_length=4.0,head_width=2.4", linestyle=":", color="#94a3b8", linewidth=1.1, zorder=2))
    # Cycle 1 (nonzero delta)
    ax_a.add_patch(mpatches.FancyArrowPatch((1.26, 0.80), (1.87, 0.80), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>,head_length=4.0,head_width=2.4", color=pf.PAL["ours"], linewidth=1.5, zorder=3))
    ax_a.add_patch(mpatches.FancyArrowPatch((1.87, 0.64), (1.26, 0.64), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>,head_length=4.0,head_width=2.4", color=pf.PAL["ours"], linewidth=1.5, zorder=3))
    # Cycle 2 (zero delta)
    ax_a.add_patch(mpatches.FancyArrowPatch((1.26, -0.27), (1.87, -0.27), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>,head_length=4.0,head_width=2.4", color="#94a3b8", linewidth=1.2, zorder=3))
    ax_a.add_patch(mpatches.FancyArrowPatch((1.87, -0.43), (1.26, -0.43), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>,head_length=4.0,head_width=2.4", color="#94a3b8", linewidth=1.2, zorder=3))

    for k, (px, py) in pts_a.items():
        col = pf.PAL["ours"] if k in ("u1", "u2") else "#64748b"
        fc = "#eff6ff" if k == "u1" else "white"
        ax_a.add_patch(mpatches.Circle((px, py), radius=0.17, facecolor=fc, edgecolor=col, linewidth=1.35, zorder=5))
    ax_a.text(0.12, 0.72, "$r_0$", fontsize=6.8, ha="center", va="center", zorder=6)
    ax_a.text(1.08, 0.72, "$r_1$", fontsize=6.8, fontweight="bold", ha="center", va="center", zorder=6)
    ax_a.text(2.05, 0.72, "$r_2$", fontsize=6.8, ha="center", va="center", zorder=6)
    ax_a.text(1.08, -0.35, "$r_3$", fontsize=6.8, ha="center", va="center", zorder=6)
    ax_a.text(2.05, -0.35, "$r_4$", fontsize=6.8, ha="center", va="center", zorder=6)
    ax_a.text(1.56, 1.07, "$\\Delta(C_1)\\neq 0$", fontsize=6.1, color=pf.PAL["ours"], ha="center", va="bottom", fontweight="bold")
    ax_a.text(1.56, -0.74, "$\\Delta(C_2)=0$ (skip)", fontsize=5.9, color="#64748b", ha="center", va="top")
    ax_a.text(
        1.10, -1.32,
        r"$A = F = \sum_{\Delta\neq 0}\min p_i$ [in P]",
        fontsize=6.2, fontweight="bold", color=pf.PAL["ours"], ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.30,rounding_size=0.18", facecolor="#f8fafc", edgecolor=pf.PAL["ours"], linewidth=1.0),
    )

    # Panel (b): Row & Col Degree 3 Selector Routing Tree (F/A = Theta(n / log n))
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_b.set_aspect("equal")
    ax_b.set_xlim(-0.35, 2.55)
    ax_b.set_ylim(-1.55, 1.30)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.axis("off")
    ax_b.set_title("(b) Degree $3$ Routing Tree", fontsize=7.8, fontweight="bold", pad=3)

    root_xy = (1.10, 0.88)
    mid_xy = [(0.52, 0.22), (1.68, 0.22)]
    leaf_xy = [(0.16, -0.52), (0.86, -0.52), (1.34, -0.52), (2.04, -0.52)]
    for mx, my in mid_xy:
        ax_b.add_patch(mpatches.FancyArrowPatch(root_xy, (mx, my + 0.14), arrowstyle="-|>,head_length=3.6,head_width=2.2", color=pf.PAL["ours"], linewidth=1.35, zorder=2))
    for idx, (lx, ly) in enumerate(leaf_xy):
        px, py = mid_xy[idx // 2]
        col = pf.PAL["ours"] if idx == 1 else "#94a3b8"
        lw = 1.45 if idx == 1 else 1.0
        ax_b.add_patch(mpatches.FancyArrowPatch((px, py - 0.14), (lx, ly + 0.14), arrowstyle="-|>,head_length=3.4,head_width=2.1", color=col, linewidth=lw, zorder=2))
        fc = "#fef2f2" if idx == 1 else "white"
        ec = pf.PAL["comp1"] if idx == 1 else "#64748b"
        ax_b.add_patch(mpatches.Circle((lx, ly), radius=0.15, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=5))
        ax_b.text(lx, ly, f"$v_{idx+1}$", fontsize=6.2, ha="center", va="center", zorder=6)

    ax_b.add_patch(mpatches.Circle(root_xy, radius=0.16, facecolor="#eff6ff", edgecolor=pf.PAL["ours"], linewidth=1.35, zorder=5))
    ax_b.text(root_xy[0], root_xy[1], "$s$", fontsize=7.0, fontweight="bold", ha="center", va="center", zorder=6)
    for idx, (mx, my) in enumerate(mid_xy, 1):
        ax_b.add_patch(mpatches.Circle((mx, my), radius=0.15, facecolor="#eff6ff" if idx == 1 else "white", edgecolor=pf.PAL["ours"], linewidth=1.25, zorder=5))
        ax_b.text(mx, my, f"$q_{idx}$", fontsize=6.4, ha="center", va="center", zorder=6)
    ax_b.text(
        1.10, -1.32,
        r"$F = 2^h$ vs. $A = h + 1$",
        fontsize=6.2, fontweight="bold", color=pf.PAL["comp1"], ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.30,rounding_size=0.18", facecolor="#fff7ed", edgecolor=pf.PAL["comp1"], linewidth=1.0),
    )

    # Panel (c): Proposition 3 — Degree 2 + 2-Bit XOR Multiplexer (A = 2 < F = 3)
    ax_c = fig.add_subplot(gs_top[0, 2])
    ax_c.set_aspect("equal")
    ax_c.set_xlim(-0.35, 2.55)
    ax_c.set_ylim(-1.55, 1.30)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.axis("off")
    ax_c.set_title("(c) Degree $2$ + $2$-Bit Checksum", fontsize=7.6, fontweight="bold", pad=3)

    cyc_xs = [0.15, 0.62, 1.10, 1.58, 2.05]
    cyc_lbl = ["$y_0$", "$y_1$", "$y_2$", "$x_3$", "$x_4$"]
    for cx, cl in zip(cyc_xs, cyc_lbl):
        fc = "#ecfdf5" if cl == "$y_0$" else "white"
        ec = pf.PAL["ours_alt"] if cl in ("$y_0$", "$y_1$", "$y_2$") else "#64748b"
        ax_c.add_patch(mpatches.FancyBboxPatch((cx - 0.16, 0.18), 0.32, 0.62, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=4))
        ax_c.text(cx, 0.49, cl, fontsize=6.4, fontweight="bold", ha="center", va="center", zorder=5)
    ax_c.text(1.10, 0.98, "5 disjoint 2-cycles ($n=10$)", fontsize=6.0, color="#334155", ha="center", va="bottom")
    ax_c.text(1.10, -0.35, "$x_3 = y_0 \\oplus y_1,\\; x_4 = y_0 \\oplus y_2$\nMultiplexer $(1-y_0)y_1 + y_0 y_2$", fontsize=5.9, color=pf.PAL["ours_alt"], ha="center", va="center")
    ax_c.text(
        1.10, -1.32,
        r"$A = 2 < F = 3$ ($\geq 2$ characters)",
        fontsize=6.1, fontweight="bold", color=pf.PAL["ours_alt"], ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.30,rounding_size=0.18", facecolor="#ecfdf5", edgecolor=pf.PAL["ours_alt"], linewidth=1.0),
    )

    # Bottom-left (d): R-round query curve at h = 8 (n = 1,278, m = 256)
    ax_d = fig.add_subplot(gs_bot[0, 0])
    rows_h8 = []
    with (DATA_DIR / "exp1_degree_threshold_and_rounds.csv").open() as fh:
        for r in csv.DictReader(fh):
            if int(r["height"]) == 8:
                rows_h8.append(r)
    rounds = [int(r["rounds_R"]) for r in rows_h8]
    sched = [int(r["scheduled_queries"]) for r in rows_h8]
    lb = [int(r["info_lower_bound"]) for r in rows_h8]
    ax_d.plot(rounds, sched, marker="o", linewidth=1.8, color=pf.PAL["ours"], label="Balanced batch schedule")
    ax_d.plot(rounds, lb, marker="s", linestyle="--", linewidth=1.4, color=pf.PAL["comp1"], label="Theorem 2 lower bound")
    ax_d.set_yscale("log")
    ax_d.set_xticks(rounds)
    ax_d.set_ylim(2.0, 1250)
    ax_d.set_xlabel("Permitted batch rounds $R$", fontsize=8.0)
    ax_d.set_ylabel("Worst-case queries (log)", fontsize=8.0)
    ax_d.set_title("(d) Batched Rounds ($h=8, n=1{,}278$)", fontsize=8.0, fontweight="bold", pad=4)
    ax_d.legend(loc="upper right", fontsize=6.0, frameon=True, facecolor="white", framealpha=0.95)
    ax_d.annotate("Fixed: 256", xy=(1, 256), xytext=(2.45, 92.0), fontsize=6.2, fontweight="bold", color=pf.PAL["ours"],
                  arrowprops=dict(arrowstyle="->", color=pf.PAL["ours"], lw=0.9))
    ax_d.annotate("Adaptive: 9 ($28.4\\times$)", xy=(9, 9), xytext=(4.35, 34.0), fontsize=6.2, fontweight="bold", color=pf.PAL["ours_alt"],
                  arrowprops=dict(arrowstyle="->", color=pf.PAL["ours_alt"], lw=0.9))

    # Bottom-right (e): Fixed-to-Adaptive ratio across tree heights h = 1..8
    ax_e = fig.add_subplot(gs_bot[0, 1])
    hs = list(range(1, 9))
    fixed_h = [2**h for h in hs]
    adap_h = [h + 1 for h in hs]
    ratios = [f / a for f, a in zip(fixed_h, adap_h)]
    ax_e.plot(hs, ratios, marker="o", linewidth=1.8, color=pf.PAL["ours"], label="$F / A = 2^h / (h + 1)$")
    ax_e.axhline(1.0, linestyle=":", color="#64748b", linewidth=1.2, label="Row-degree $\\leq 2$ ($F / A = 1$)")
    ax_e.set_xticks(hs)
    ax_e.set_ylim(0, 38)
    ax_e.set_xlabel("Selector routing height $h$ ($n = 5\\cdot 2^h - 2$)", fontsize=8.0)
    ax_e.set_ylabel("Ratio $F / A$", fontsize=8.0)
    ax_e.set_title("(e) Degree-3 Separation Scaling", fontsize=8.0, fontweight="bold", pad=4)
    ax_e.legend(loc="upper left", fontsize=6.1, frameon=True, facecolor="white", framealpha=0.95)
    ax_e.annotate("28.4$\\times$ at $h=8$", xy=(8, ratios[-1]), xytext=(4.2, 13.5), fontsize=6.4, fontweight="bold", color=pf.PAL["ours"],
                  arrowprops=dict(arrowstyle="->", color=pf.PAL["ours"], lw=0.9))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pf.finalize_figure(
        fig, FIG_DIR / "fig1_degree_dichotomy_schematic.pdf",
        script_path=Path(__file__), input_paths=[DATA_DIR / "P19_RESULTS.json"],
        preview_png=FIG_DIR / "fig1_degree_dichotomy_schematic.png", strict_overlap=True,
    )


def build_fig2(summary: dict) -> None:
    pf.apply_style()
    fig = plt.figure(figsize=(6.8, 4.60))
    gs = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.34)

    # (a) Laminar Token-Chain Compression Schematic (4-node sharpness diamond b=1, p=2 -> 4)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_aspect("equal")
    ax_a.set_xlim(-0.35, 3.05)
    ax_a.set_ylim(-1.25, 1.25)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.axis("off")
    ax_a.set_title("(a) Laminar Token-Chain Core ($b + 2q - 1$)", fontsize=7.9, fontweight="bold", pad=3)

    b_xy = (0.18, 0.0)
    t1_xy = (1.28, 0.62)
    t2_xy = (1.28, -0.62)
    m_xy = (2.42, 0.0)
    ax_a.add_patch(mpatches.FancyArrowPatch((0.38, 0.10), (1.08, 0.54), arrowstyle="-|>,head_length=4.0,head_width=2.4", color=pf.PAL["ours"], linewidth=1.45, zorder=3))
    ax_a.add_patch(mpatches.FancyArrowPatch((0.38, -0.10), (1.08, -0.54), arrowstyle="-|>,head_length=4.0,head_width=2.4", color=pf.PAL["comp1"], linewidth=1.45, zorder=3))
    ax_a.add_patch(mpatches.FancyArrowPatch((1.48, 0.54), (2.22, 0.12), arrowstyle="-|>,head_length=4.0,head_width=2.4", color=pf.PAL["ours"], linewidth=1.45, zorder=3))
    ax_a.add_patch(mpatches.FancyArrowPatch((1.48, -0.54), (2.22, -0.12), arrowstyle="-|>,head_length=4.0,head_width=2.4", color=pf.PAL["comp1"], linewidth=1.45, zorder=3))
    ax_a.add_patch(mpatches.FancyArrowPatch((2.42, 0.22), (0.18, 0.22), connectionstyle="arc3,rad=0.58", arrowstyle="-|>,head_length=4.0,head_width=2.4", linestyle="--", color=pf.PAL["ours_alt"], linewidth=1.35, zorder=2))

    for (px, py), lbl, fc, ec in [
        (b_xy, "$u_b$", "#fef2f2", pf.PAL["invalid"]),
        (t1_xy, "$x_1$", "#eff6ff", pf.PAL["ours"]),
        (t2_xy, "$x_2$", "#fff7ed", pf.PAL["comp1"]),
        (m_xy, "$x_m$", "#ecfdf5", pf.PAL["ours_alt"]),
    ]:
        ax_a.add_patch(mpatches.Circle((px, py), radius=0.20, facecolor=fc, edgecolor=ec, linewidth=1.35, zorder=5))
        ax_a.text(px, py, lbl, fontsize=7.0, fontweight="bold", ha="center", va="center", zorder=6)

    ax_a.text(0.18, -0.38, "price $10$", fontsize=6.0, color=pf.PAL["invalid"], ha="center", va="top")
    ax_a.text(1.28, 0.96, "$T=\\{\\tau_1\\}$", fontsize=6.0, color=pf.PAL["ours"], ha="center", va="bottom")
    ax_a.text(1.28, -0.96, "$T=\\{\\tau_2\\}$", fontsize=6.0, color=pf.PAL["comp1"], ha="center", va="top")
    ax_a.text(2.42, -0.38, "$T=\\{\\tau_1,\\tau_2\\}$\nprice $1$ [Opt]", fontsize=6.0, color=pf.PAL["ours_alt"], ha="center", va="top", fontweight="bold")

    # (b) Stretched Selector Row Count vs Compressed Representative Probes
    ax_b = fig.add_subplot(gs[0, 1])
    sc = summary["exp2_core_reduction"]["stretched_selector_scaling"]
    depths = ["0", "1", "8", "64", "256"]
    n_rows = [sc[d]["n_rows"] for d in depths]
    rep_p = [sc[d]["representative_probes"] for d in depths]
    b_rows = [sc[d]["branch_rows"] for d in depths]
    xs = np.arange(len(depths))
    ax_b.plot(xs, n_rows, marker="s", linestyle="--", linewidth=1.6, color=pf.PAL["comp1"], label="Raw rows $n$")
    ax_b.plot(xs, rep_p, marker="o", linewidth=1.8, color=pf.PAL["ours"], label="Compressed probes ($\\leq b+2q-1$)")
    ax_b.plot(xs, b_rows, marker="^", linestyle=":", linewidth=1.4, color=pf.PAL["ours_alt"], label="Branching rows $b$")
    ax_b.set_yscale("log")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([f"$s={d}$" for d in depths], fontsize=7.2)
    ax_b.set_ylim(3.5, 220000)
    ax_b.set_xlabel("Chain stretch depth $s$", fontsize=8.0)
    ax_b.set_ylabel("Row / probe count (log)", fontsize=8.0)
    ax_b.set_title("(b) Core Compression ($6{,}418 \\to 28$)", fontsize=8.0, fontweight="bold", pad=4)
    ax_b.legend(loc="upper left", fontsize=5.9, frameon=True, facecolor="white", framealpha=0.95)
    ax_b.annotate("$229\\times$ fewer\nat $n=6{,}418$", xy=(4, 28), xytext=(2.65, 82), fontsize=6.2, fontweight="bold", color=pf.PAL["ours"],
                  arrowprops=dict(arrowstyle="->", color=pf.PAL["ours"], lw=0.9))

    # (c) Compression + Exact Minimax Solve Runtime (ms) across n = 18 .. 6,418
    ax_c = fig.add_subplot(gs[1, 0])
    ms_vals = [sc[d]["median_ms"] for d in depths]
    ax_c.plot(n_rows, ms_vals, marker="o", linewidth=1.8, color=pf.PAL["ours"], label="Alg. 1 + Core DP")
    ax_c.set_xscale("log")
    ax_c.set_ylim(0, 29)
    ax_c.set_xlabel("Candidate graph rows $n$ (log)", fontsize=8.0)
    ax_c.set_ylabel("Median solve time (ms)", fontsize=8.0)
    ax_c.set_title("(c) End-to-End Exact Pricing Time", fontsize=8.0, fontweight="bold", pad=4)
    ax_c.legend(loc="upper left", fontsize=6.5, frameon=True, facecolor="white", framealpha=0.92)
    ax_c.annotate("23.1 ms at $n=6{,}418$", xy=(6418, ms_vals[-1]), xytext=(140, 18.5), fontsize=6.5, fontweight="bold", color=pf.PAL["ours"],
                  arrowprops=dict(arrowstyle="->", color=pf.PAL["ours"], lw=0.9))

    # (d) Exact Verification Across 200 Degree-2 + 224 General Unaligned Random Graphs
    ax_d = fig.add_subplot(gs[1, 1])
    cats = ["Deg-$2$ ($n\\leq 9$)\n$A = F$ rate", "General ($224$)\nCore exactness", "SQLite ($161$)\n$A < F$ panels"]
    vals = [100.0, 100.0, 100.0 * 6.0 / 161.0]
    cols = [pf.PAL["ours"], pf.PAL["ours_alt"], pf.PAL["comp1"]]
    bars = ax_d.bar(np.arange(3), vals, width=0.52, color=cols, edgecolor="#1e293b", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax_d.text(b.get_x() + b.get_width() / 2, v + 3.0, f"{v:.1f}%", fontsize=6.8, fontweight="bold", ha="center", va="bottom")
    ax_d.set_xticks(np.arange(3))
    ax_d.set_xticklabels(cats, fontsize=6.8)
    ax_d.set_ylim(0, 118)
    ax_d.set_ylabel("Verified percentage (%)", fontsize=8.0)
    ax_d.set_title("(d) Exactness & Empirical Sparsity", fontsize=8.0, fontweight="bold", pad=4)

    pf.finalize_figure(
        fig, FIG_DIR / "fig2_laminar_core_and_rounds.pdf",
        script_path=Path(__file__), input_paths=[DATA_DIR / "P19_RESULTS.json"],
        preview_png=FIG_DIR / "fig2_laminar_core_and_rounds.png", strict_overlap=True,
    )


def build_fig3(summary: dict) -> None:
    pf.apply_style()
    fig = plt.figure(figsize=(6.8, 4.60))
    gs = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.34)

    # (a) GF(2) / Walsh-Fourier Constancy Certificate vs 2^b Extrema DP Runtime
    ax_a = fig.add_subplot(gs[0, 0])
    rows4 = summary["exp4_gf2_fourier"]["scaling_rows"]
    dp_b = [r["bits"] for r in rows4 if r["method"] == "signature_extrema_dp_2b"]
    dp_ms = [r["median_ms"] for r in rows4 if r["method"] == "signature_extrema_dp_2b"]
    gf_b = [r["bits"] for r in rows4 if r["method"] == "gf2_fourier_constancy_poly"]
    gf_ms = [r["median_ms"] for r in rows4 if r["method"] == "gf2_fourier_constancy_poly"]
    ax_a.plot(dp_b, dp_ms, marker="s", linestyle="--", linewidth=1.6, color=pf.PAL["comp1"], label="Exact extrema DP ($2^b$)")
    ax_a.plot(gf_b, gf_ms, marker="o", linewidth=1.8, color=pf.PAL["ours"], label="Alg. 2 $\\mathbb{F}_2$ constancy")
    ax_a.set_yscale("log")
    ax_a.set_xticks([4, 8, 12, 16])
    ax_a.set_ylim(0.018, 2800)
    ax_a.set_xlabel("Checksum width $b$ (bits)", fontsize=8.0)
    ax_a.set_ylabel("Median runtime (ms, log)", fontsize=8.0)
    ax_a.set_title("(a) $\\mathbb{F}_2$ Uniqueness vs. Extrema DP", fontsize=8.0, fontweight="bold", pad=4)
    ax_a.legend(loc="upper left", fontsize=6.0, frameon=True, facecolor="white", framealpha=0.95)
    ax_a.annotate("$503\\times$ faster\nat $b=12$", xy=(12, gf_ms[2]), xytext=(11.5, 9.5), fontsize=6.3, fontweight="bold", color=pf.PAL["ours"],
                  arrowprops=dict(arrowstyle="->", color=pf.PAL["ours"], lw=0.9))

    # (b) Chinook + Northwind (161 panels): Physical SQLite VFS preads by Manifest Design
    ax_b = fig.add_subplot(gs[0, 1])
    cn = summary["exp3_sqlite_manifests"]["chinook_northwind_161_panels_by_manifest"]
    bits_list = [6, 12, 24]
    row_preads = [cn[f"row_distributed_{b}b"]["preads"] for b in bits_list]
    xor_preads = [cn[f"joint_salted_xor_{b}b"]["preads"] for b in bits_list]
    x = np.arange(len(bits_list))
    w = 0.34
    b1 = ax_b.bar(x - w / 2, row_preads, width=w, color=pf.PAL["comp1"], label="Row-distributed bits")
    b2 = ax_b.bar(x + w / 2, xor_preads, width=w, color=pf.PAL["ours"], label="Salted joint XOR")
    for br in list(b1) + list(b2):
        h = br.get_height()
        ax_b.text(br.get_x() + br.get_width() / 2, h + 7, f"{int(h)}", fontsize=6.5, fontweight="bold", ha="center", va="bottom")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{b} bits" for b in bits_list], fontsize=7.5)
    ax_b.set_ylim(0, 345)
    ax_b.set_xlabel("Manifest payload budget", fontsize=8.0)
    ax_b.set_ylabel("Physical SQLite page reads", fontsize=8.0)
    ax_b.set_title("(b) Chinook & Northwind (161 Panels)", fontsize=7.9, fontweight="bold", pad=4)
    ax_b.legend(loc="upper right", fontsize=6.3, frameon=True, facecolor="white", framealpha=0.92)

    # (c) Sakila 8-Seed Replication (n = 6, 168 pairs): Remaining Ambiguous Panels
    ax_c = fig.add_subplot(gs[1, 0])
    sak6 = summary["exp3_sqlite_manifests"]["sakila_8seed_replication"]["n6"]
    s6_row = [sak6[f"n6_row_distributed_{b}b"]["ambiguous_pairs"] for b in bits_list]
    s6_xor = [sak6[f"n6_joint_salted_xor_{b}b"]["ambiguous_pairs"] for b in bits_list]
    b3 = ax_c.bar(x - w / 2, s6_row, width=w, color=pf.PAL["comp1"], label="Row-distributed")
    b4 = ax_c.bar(x + w / 2, s6_xor, width=w, color=pf.PAL["ours"], label="Salted joint XOR")
    for br in list(b3) + list(b4):
        h = br.get_height()
        ax_c.text(br.get_x() + br.get_width() / 2, h + 4, f"{int(h)}", fontsize=6.5, fontweight="bold", ha="center", va="bottom")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{b} bits" for b in bits_list], fontsize=7.5)
    ax_c.set_ylim(0, 212)
    ax_c.set_xlabel("Manifest payload budget", fontsize=8.0)
    ax_c.set_ylabel("Unresolved pairs (/ 168)", fontsize=8.0)
    ax_c.set_title("(c) Sakila Replication ($n = 6$, 168 Pairs)", fontsize=7.8, fontweight="bold", pad=4)
    ax_c.legend(loc="upper right", fontsize=6.3, frameon=True, facecolor="white", framealpha=0.92)

    # (d) Sakila Larger Panels (n = 8, 184 pairs): Total Colliding World Assignments (log)
    ax_d = fig.add_subplot(gs[1, 1])
    sak8 = summary["exp3_sqlite_manifests"]["sakila_8seed_replication"]["n8"]
    s8_row = [max(1, sak8[f"n8_row_distributed_{b}b"]["total_colliders"]) for b in bits_list]
    s8_xor = [max(1, sak8[f"n8_joint_salted_xor_{b}b"]["total_colliders"]) for b in bits_list]
    ax_d.plot(x, s8_row, marker="s", linestyle="--", linewidth=1.6, color=pf.PAL["comp1"], label="Row-distributed")
    ax_d.plot(x, s8_xor, marker="o", linewidth=1.8, color=pf.PAL["ours"], label="Salted joint XOR")
    ax_d.set_yscale("log")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"{b} bits" for b in bits_list], fontsize=7.5)
    ax_d.set_ylim(0.5, 900000)
    ax_d.set_xlabel("Manifest payload budget", fontsize=8.0)
    ax_d.set_ylabel("Colliding worlds (log)", fontsize=8.0)
    ax_d.set_title("(d) Sakila Colliders ($n = 8$, 184 Pairs)", fontsize=7.8, fontweight="bold", pad=4)
    ax_d.legend(loc="upper right", fontsize=6.3, frameon=True, facecolor="white", framealpha=0.92)
    ax_d.annotate("96 vs. 1\nat 24 bits", xy=(2, 1), xytext=(0.75, 4.5), fontsize=6.4, fontweight="bold", color=pf.PAL["ours"],
                  arrowprops=dict(arrowstyle="->", color=pf.PAL["ours"], lw=0.9))

    pf.finalize_figure(
        fig, FIG_DIR / "fig3_gf2_fourier_and_sqlite.pdf",
        script_path=Path(__file__), input_paths=[DATA_DIR / "P19_RESULTS.json"],
        preview_png=FIG_DIR / "fig3_gf2_fourier_and_sqlite.png", strict_overlap=True,
    )


def main() -> None:
    summary = json.loads((DATA_DIR / "P19_RESULTS.json").read_text(encoding="utf-8"))
    build_fig1(summary)
    build_fig2(summary)
    build_fig3(summary)
    print("P19_FIGURES_BUILT_OK")


if __name__ == "__main__":
    main()
