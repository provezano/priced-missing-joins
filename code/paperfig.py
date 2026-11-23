#!/usr/bin/env python3
"""Shared publication figure builder, CVD-safe palette, overlap checker, and provenance sidecar writer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

STYLE_PATH = Path(__file__).resolve().parent / "paperstyle.mplstyle"

# CVD-safe (deuteranopia/protanopia + grayscale tested) house palette; pair with hatches/markers
PAL = {
    "ours": "#1f6feb",       # primary blue
    "ours_alt": "#0a6c74",   # deep teal for secondary ours variant
    "comp1": "#eb6834",      # burnt orange (strongest competitor)
    "comp2": "#576574",      # dark slate
    "comp3": "#8395a7",      # medium cool grey
    "comp4": "#b2bec3",      # light slate grey
    "warn": "#d97706",       # amber (diagnostic / unstratified warning)
    "invalid": "#a61e22",    # crimson with diagonal hatch for invalid FWER baselines
    "rule": "#2d3436",       # threshold / guard line
}

HATCHES = {
    "ours": "",
    "ours_alt": "//",
    "comp1": "",
    "comp2": "..",
    "comp3": "",
    "comp4": "",
    "warn": "//",
    "invalid": "xx",
}


def apply_style() -> None:
    plt.style.use(str(STYLE_PATH))


def check_artist_overlaps(fig: plt.Figure, tol_px: float = 1.5) -> list[str]:
    """Detect undesired bounding-box collisions among text labels, tick labels, titles, and legends."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues: list[str] = []

    # Collect visible text items per axes + figure-level texts/legends
    text_boxes: list[tuple[str, Bbox]] = []
    for ax_idx, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        if ax.get_title():
            tb = ax.title.get_window_extent(renderer).shrunk_to_aspect(0.98)
            text_boxes.append((f"ax{ax_idx}:title({ax.get_title()[:24]})", tb))
        if ax.get_xlabel():
            xb = ax.xaxis.label.get_window_extent(renderer)
            text_boxes.append((f"ax{ax_idx}:xlabel({ax.get_xlabel()[:24]})", xb))
        if ax.get_ylabel():
            yb = ax.yaxis.label.get_window_extent(renderer)
            text_boxes.append((f"ax{ax_idx}:ylabel({ax.get_ylabel()[:24]})", yb))
        xlo, xhi = sorted(ax.get_xlim())
        ylo, yhi = sorted(ax.get_ylim())
        for loc, t in zip(ax.get_xticks(), ax.get_xticklabels()):
            if xlo - 1e-9 <= loc <= xhi + 1e-9 and t.get_visible() and t.get_text().strip():
                text_boxes.append((f"ax{ax_idx}:xtick({t.get_text()[:18]})", t.get_window_extent(renderer)))
        for loc, t in zip(ax.get_yticks(), ax.get_yticklabels()):
            if ylo - 1e-9 <= loc <= yhi + 1e-9 and t.get_visible() and t.get_text().strip():
                text_boxes.append((f"ax{ax_idx}:ytick({t.get_text()[:18]})", t.get_window_extent(renderer)))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                text_boxes.append((f"ax{ax_idx}:annot({t.get_text()[:18]})", t.get_window_extent(renderer)))
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            lb = leg.get_window_extent(renderer)
            text_boxes.append((f"ax{ax_idx}:legend", lb))
            # Also check if legend collides with data patches/bars or lines inside the same axes!
            for p_i, patch in enumerate(ax.patches):
                if patch.get_visible() and patch.get_width() > 0 and patch.get_height() > 0:
                    pb = patch.get_window_extent(renderer)
                    inter = Bbox.intersection(lb, pb)
                    if inter is not None and inter.width > tol_px and inter.height > tol_px:
                        issues.append(f"ax{ax_idx}:legend overlaps data bar/patch #{p_i}")

    for leg_i, leg in enumerate(fig.legends):
        if leg.get_visible():
            text_boxes.append((f"fig:legend{leg_i}", leg.get_window_extent(renderer)))

    # Pairwise text-vs-text / legend collision check
    n = len(text_boxes)
    for i in range(n):
        name_i, b_i = text_boxes[i]
        for j in range(i + 1, n):
            name_j, b_j = text_boxes[j]
            inter = Bbox.intersection(b_i, b_j)
            if inter is not None and inter.width > tol_px and inter.height > tol_px:
                issues.append(f"TEXT_OVERLAP: {name_i} <-> {name_j} ({inter.width:.1f}x{inter.height:.1f} px)")

    return issues


def finalize_figure(
    fig: plt.Figure,
    pdf_path: Path,
    *,
    script_path: Path,
    input_paths: Sequence[Path] = (),
    preview_png: Path | None = None,
    strict_overlap: bool = True,
) -> dict:
    """Validate artist bounding boxes, write deterministic vector PDF (no Type 3 fonts), and emit provenance sidecar."""
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    issues = check_artist_overlaps(fig)
    if strict_overlap and issues:
        raise RuntimeError(f"Figure {pdf_path.name} failed overlap check:\n  " + "\n  ".join(issues))

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"CreationDate": None, "Producer": "paperfig"},
    )
    if preview_png is not None:
        Path(preview_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_png, format="png", dpi=240, bbox_inches="tight", pad_inches=0.04)

    plt.close(fig)

    prov = {
        "figure": pdf_path.name,
        "figure_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "script": str(script_path.name),
        "script_sha256": hashlib.sha256(Path(script_path).read_bytes()).hexdigest() if Path(script_path).exists() else "",
        "inputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in input_paths
            if Path(p).exists()
        },
        "matplotlib_version": matplotlib.__version__,
        "pdf_fonttype": int(plt.rcParams.get("pdf.fonttype", 0)),
        "overlap_issues": issues,
    }
    prov_path = pdf_path.with_suffix(".pdf.prov.json")
    prov_path.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    return prov
