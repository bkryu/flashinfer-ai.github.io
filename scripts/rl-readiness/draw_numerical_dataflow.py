"""CPU-only, reproducible illustrations of GEMM numerical dataflow.

The FP32 examples illustrate nonassociativity, not undocumented MMA internals.
The N/K partition is schematic; class-regions.svg contains the actual policy map.
"""
import argparse
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "svg.hashsalt": "flashinfer-numerical-dataflow", "savefig.facecolor": "white",
})
INK = "#24312b"
MUTED = "#617069"
BLUE = "#2563eb"
BLUE_FILL = "#e5edfc"
AMBER = "#a56300"
AMBER_FILL = "#fff0d7"
GREEN = "#365c20"
GREEN_FILL = "#edf5e8"
RED = "#b43232"


def fp32(x):
    return struct.unpack("!f", struct.pack("!f", x))[0]


def add(a, b):
    # Binary64 exactly adds the binary32 operands used in this small example;
    # packing then performs the required binary32 round-to-nearest-even.
    return fp32(a + b)


def fold(values):
    acc = fp32(0)
    for value in values:
        acc = add(acc, value)
    return acc


def bf16_bits(x):
    bits = struct.unpack("!I", struct.pack("!f", x))[0]
    assert bits & 0xffff == 0  # Both illustrative final outputs are exact BF16.
    return f"0x{bits >> 16:04x}"


def box(ax, x, y, w, h, text, fill="white", edge="#c9d0c5", size=12,
        color=INK, weight="normal", mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.06",
                               facecolor=fill, edgecolor=edge, linewidth=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=color, weight=weight,
            fontfamily="DejaVu Sans Mono" if mono else "DejaVu Sans",
            linespacing=1.45)


def arrow(ax, a, b, color="#899682", lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                linewidth=lw, color=color, shrinkA=3, shrinkB=3))


def setup(height, title, subtitle):
    fig, ax = plt.subplots(figsize=(13, height))
    ax.set(xlim=(0, 13), ylim=(0, height))
    ax.axis("off")
    ax.text(.35, height - .38, title, fontsize=19, weight="bold", color=INK)
    ax.text(.35, height - .82, subtitle, fontsize=12.5, color=MUTED)
    fig.subplots_adjust(left=.02, right=.99, top=.99, bottom=.02)
    return fig, ax


def save(fig, out, name):
    for ext in ("svg", "png"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=170, pad_inches=.18,
                    metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def tensor_core_figure(out):
    fig, ax = setup(8.2, "Tensor Core instructions are numerical primitives",
                    "A shape-dependent instruction choice can change a token's output bits.")
    box(ax, .45, 6.05, 5.5, .82, "Small M  →  warp-level mma.sync", BLUE_FILL, BLUE,
        size=14, color=BLUE, weight="bold")
    box(ax, 7.05, 6.05, 5.5, .82, "Large M  →  tcgen05.mma", AMBER_FILL, AMBER,
        size=14, color=AMBER, weight="bold")
    ax.text(6.5, 5.58, "Same BF16 inputs and FP32 accumulators do not imply the same reduction order.",
            ha="center", fontsize=12, color=MUTED)
    ax.text(.45, 4.98, "Why grouping matters: an illustrative FP32 example, x = 2²⁵",
            fontsize=13, weight="bold", color=INK)
    # Two possible scalar reduction trees. These are deliberately not attributed
    # to either instruction family's undocumented internal accumulation order.
    for offset, right in ((.45, False), (7.05, True)):
        xs = [offset + .7, offset + 2.7, offset + 4.7]
        for x, value in zip(xs, ("x", "−x", "1")):
            box(ax, x - .44, 4.04, .88, .48, value, "#f0f2ef", size=14, mono=True)
        first = xs[0] + 1 if not right else xs[1] + 1
        second = xs[2] if not right else xs[0]
        first_value = "0" if not right else "−x"
        for index in ((0, 1) if not right else (1, 2)):
            arrow(ax, (xs[index], 4.04), (first, 3.53))
        box(ax, first - .66, 2.91, 1.32, .58, f"+ → {first_value}",
            BLUE_FILL if not right else AMBER_FILL, BLUE if not right else AMBER,
            size=13, mono=True)
        midpoint = offset + 2.7
        arrow(ax, (first, 2.91), (midpoint, 2.38))
        arrow(ax, (second, 4.04), (midpoint + (.2 if not right else -.2), 2.38))
        box(ax, midpoint - .55, 1.78, 1.1, .55, "+ → 1" if not right else "+ → 0",
            size=13, mono=True)
        text = "(x + (−x)) + 1" if not right else "x + ((−x) + 1)"
        ax.text(midpoint, 1.31, text, ha="center", fontsize=13, color=INK,
                fontfamily="DejaVu Sans Mono")
        ax.text(midpoint, .88, "BF16 bits: 0x3f80" if not right else "BF16 bits: 0x0000",
                ha="center", fontsize=12.5, weight="bold", color=BLUE if not right else AMBER)
    ax.text(6.5, .28, "Schematic grouping only; these trees do not describe either instruction's internal implementation.",
            ha="center", fontsize=11, color=MUTED)
    save(fig, out, "tensor-core-numerics")


def split_k_figure(out):
    fig, ax = setup(8.4, "Split-K changes where accumulation restarts",
                    "The same dot-product terms can produce different bits, even with an ordered fixup.")
    ax.text(.45, 6.91, "Illustrative FP32 accumulation: x = 2²⁵; terms along K are [x, 1, −x, 1].",
            fontsize=12.5, color=MUTED)
    ax.text(.45, 6.35, "Large M: one uninterrupted accumulator", fontsize=14, color=BLUE, weight="bold")
    xs = (1.15, 3.7, 6.25, 8.8)
    for i, (x, op, state) in enumerate(zip(xs, ("+ x", "+ 1", "+ (−x)", "+ 1"),
                                         ("x", "x", "0", "1"))):
        box(ax, x - .65, 5.13, 1.3, .67, f"{op}\n= {state}", BLUE_FILL, BLUE,
            size=12.5, mono=True)
        if i:
            arrow(ax, (xs[i - 1] + .65, 5.465), (x - .65, 5.465))
    ax.text(.45, 4.66, "Start at +0; the first +1 is rounded away.", fontsize=11.5, color=MUTED)
    box(ax, 10.15, 5.1, 2.1, .73, "1  ·  0x3f80\nBF16 output", BLUE_FILL, BLUE, size=12, color=BLUE)
    arrow(ax, (9.45, 5.465), (10.15, 5.465))
    ax.text(.45, 4.01, "Small M: two independently accumulated chunks", fontsize=14, color=AMBER, weight="bold")
    box(ax, .55, 2.71, 4.35, .77, "chunk 0: [x, 1]\n+0 → x → x", AMBER_FILL, AMBER, size=12.5, mono=True)
    box(ax, 6.15, 2.71, 4.35, .77, "chunk 1: [−x, 1]\n+0 → −x → −x", AMBER_FILL, AMBER, size=12.5, mono=True)
    for x in (2.72, 8.32):
        arrow(ax, (x, 2.71), (5.52, 2.1))
    box(ax, 3.4, 1.39, 4.25, .65, "fixup: (+0 + x) + (−x) = 0", size=12, mono=True)
    arrow(ax, (7.65, 1.715), (10.15, 1.715))
    box(ax, 10.15, 1.35, 2.1, .73, "0  ·  0x0000\nBF16 output", AMBER_FILL, AMBER, size=12, color=AMBER)
    box(ax, .55, .32, 11.7, .57,
        "Both paths are deterministic. Choosing the reduction by M breaks batch invariance.",
        GREEN_FILL, "#cbdac0", size=12, color=GREEN)
    save(fig, out, "split-k-numerics")


def class_partition_figure(out):
    fig, ax = setup(8.0, "Choose the numerical class from the fixed weight shape",
                    "Schematic N/K partition for one format and architecture; M is not a class-selection axis.")
    # No dimensional tick values: this is intentionally a conceptual region map,
    # not a simplified replacement for the alignment-sensitive source policy.
    x0, y0, w, h = .95, 1.32, 6.35, 5.0
    ax.add_patch(Polygon(((x0, y0), (x0+w, y0), (x0+w, y0+h), (x0, y0+h)),
                         facecolor=BLUE_FILL, edgecolor="#c1cbb9"))
    boundary = ((x0, y0+1.4), (x0+3.55, y0+4.0), (x0+3.55, y0+h))
    ax.add_patch(Polygon(((x0, y0+1.4), (x0, y0+h), (x0+3.55, y0+h),
                          (x0+3.55, y0+4.0)), facecolor=AMBER_FILL, edgecolor="none"))
    ax.plot([p[0] for p in boundary], [p[1] for p in boundary], color=AMBER, lw=1.5, ls="--")
    arrow(ax, (x0-.1, y0), (x0+w+.25, y0), color=INK)
    arrow(ax, (x0, y0-.05), (x0, y0+h+.25), color=INK)
    ax.text(x0+w/2, .85, "N · output columns →", ha="center", fontsize=12, color=INK)
    ax.text(.35, y0+h/2, "K · reduction length →", ha="center", va="center", rotation=90, fontsize=12, color=INK)
    ax.text(2.45, 5.56, "Ordered chunk fold", ha="center", fontsize=13.5, color=AMBER, weight="bold")
    ax.text(2.45, 5.13, "C(N, K, format) > 1", ha="center", fontsize=12, color=AMBER)
    ax.text(2.45, 4.72, "narrow N, long K", ha="center", fontsize=11.5, color=MUTED)
    ax.text(4.85, 2.45, "Uninterrupted K fold", ha="center", fontsize=13.5, color=BLUE, weight="bold")
    ax.text(4.85, 2.02, "C = 1", ha="center", fontsize=12, color=BLUE)
    ax.scatter([3.55, 5.9], [5.88, 3.55], c=[AMBER, BLUE], s=75, zorder=4)
    arrow(ax, (3.65, 5.88), (8.05, 5.4), color=AMBER)
    arrow(ax, (6.0, 3.55), (8.05, 3.7), color=BLUE)
    box(ax, 8.1, 4.8, 4.0, 1.15, "Split-class weight\nSame chunks + ordered fold at every M", AMBER_FILL, AMBER, size=12, color=AMBER)
    box(ax, 8.1, 3.05, 4.0, 1.15, "Unsplit-class weight\nSame uninterrupted fold at every M", BLUE_FILL, BLUE, size=12, color=BLUE)
    ax.text(10.1, 2.52, "Decode  →  middle  →  prefill", ha="center", fontsize=12, color=INK)
    ax.text(10.1, 2.08, "Tune tiles, scheduling, and placement", ha="center", fontsize=11.5, color=MUTED)
    ax.text(10.1, 1.7, "within the selected numerical class.", ha="center", fontsize=11.5, color=MUTED)
    box(ax, .95, .14, 11.15, .5,
        "Schematic boundary. The exact policy also enforces granule alignment, chunk length, and traffic limits.",
        GREEN_FILL, "#cbdac0", size=11, color=GREEN)
    save(fig, out, "numerical-class-partition")


def class_configurations_figure(out):
    fig, ax = setup(8.1, "A batch-invariant heuristic stays within one numerical class",
                    "For the same weight shape, M may change the configuration while the dataflow stays fixed.")
    for x, title, primitive, fill, edge, prefix in (
        (.45, "Numerical class A", "tcgen05.mma", BLUE_FILL, BLUE, "A"),
        (7.05, "Numerical class B", "warp-level mma.sync", AMBER_FILL, AMBER, "B"),
    ):
        box(ax, x, 2.3, 5.5, 4.3, "", fill, edge)
        mid = x + 2.75
        ax.text(mid, 6.18, title, ha="center", fontsize=15, weight="bold", color=edge)
        ax.text(mid, 5.76, primitive, ha="center", fontsize=13, color=edge,
                fontfamily="DejaVu Sans Mono")
        ax.text(mid, 5.29, f"Shared canonical dataflow {prefix}", ha="center",
                fontsize=12, color=INK)
        for i, (config, variation) in enumerate((("1", "smaller tile"), ("2", "larger tile"), ("3", "other raster"))):
            left = x + .26 + i * 1.69
            box(ax, left, 3.88, 1.59, .87, f"Config {prefix}{config}\n{variation}",
                "white", edge, size=11.5)
        arrow(ax, (x + 1.0, 3.62), (x + 4.5, 3.62), color=GREEN, lw=1.6)
        arrow(ax, (x + 4.5, 3.41), (x + 1.0, 3.41), color=GREEN, lw=1.6)
        ax.text(mid, 2.92, "Allowed: choose within the class", ha="center",
                fontsize=12.5, color=GREEN, weight="bold")
        ax.text(mid, 2.55, "Same input → exact same output bits", ha="center",
                fontsize=11.5, color=INK)
    # Class boundaries are fixed for this weight as M changes. Different weights
    # may select different classes, as the N/K partition drawing explains.
    # One continuous elbow per class, from the frame's outer bottom edge to
    # the warning box's outer side edge (box() adds .06 padding on each side).
    # A single path prevents gaps or extra arrowheads at the corner.
    for points in (((3.2, 2.24), (3.2, 1.43), (3.99, 1.43)),
                   ((9.8, 2.24), (9.8, 1.43), (9.01, 1.43))):
        path = MplPath(points, [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO])
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=14,
                                    linewidth=1.5, color=RED, shrinkA=0, shrinkB=0,
                                    zorder=3))
    box(ax, 4.05, 1.07, 4.9, .72, "Across classes: bits can change", "#fceeee", RED,
        size=12.5, color=RED, weight="bold")
    ax.text(6.5, .7, "Not allowed as an M-dependent choice for a fixed weight shape.",
            ha="center", fontsize=12, color=RED)
    ax.text(6.5, .22, "Illustrative classes. Tile and scheduling variants must be admitted by each class's contract and validation.",
            ha="center", fontsize=10.5, color=MUTED)
    save(fig, out, "numerical-class-configurations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    x = fp32(2 ** 25)
    assert add(add(x, -x), 1) == 1
    assert add(x, add(-x, 1)) == 0
    terms = (x, 1., -x, 1.)
    unsplit = fold(terms)
    chunked = fold((fold(terms[:2]), fold(terms[2:])))
    assert unsplit == 1 and chunked == 0
    assert bf16_bits(unsplit) == "0x3f80" and bf16_bits(chunked) == "0x0000"
    tensor_core_figure(args.out)
    split_k_figure(args.out)
    class_partition_figure(args.out)
    class_configurations_figure(args.out)
    print(f"Wrote four SVG/PNG diagrams to {args.out}; FP32 witnesses verified.")


if __name__ == "__main__":
    main()
