"""Render the blog's performance presentation from frozen analysis, without CUDA.

Usage (matplotlib/numpy required):
  python plot_blog_performance.py --analysis analysis-final --out <fresh-directory>

Inputs are read-only summary.json and latency.csv. Output includes plots, exact
display ratios, reciprocal split-K ratios, and input/source checksums. No timings
are recomputed. The archived study's paired_speedup retains its original meaning.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from backend_metadata import BATCH_INVARIANT

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "svg.hashsalt": "flashinfer-blog-performance", "savefig.facecolor": "white",
    "font.size": 11,
})
BI_COLOR = "#367b45"
NON_BI_COLOR = "#85928c"
INK = "#24312b"
MUTED = "#617069"
REGIMES = ("decode", "middle", "prefill")
REGIME_TITLES = ("Decode · M ≤ 16", "Middle · 16 < M < 1024", "Prefill · M ≥ 1024")
MODES = ("untuned", "autotuned")
DISPLAY_EXCLUSIONS = {"flashinfer_cutile"}
PAIRS = {
    "bf16": [
        ("PyTorch Inductor/Triton", "torch_inductor_bi_triton", "bf16-untuned-v2", "bf16-autotuned", ""),
        ("vLLM Triton", "vllm_triton_bi", "bf16-extra-untuned", "bf16-extra-autotuned", " †"),
        ("DeepGEMM BF16", "deepgemm_bf16_bi", "bf16-deepgemm-untuned", "bf16-deepgemm-autotuned", " †"),
        ("FI CUTLASS", "flashinfer_cutlass", "bf16-untuned-v2", "bf16-autotuned", ""),
        ("FI cuDNN", "flashinfer_cudnn", "bf16-libraries-untuned", "bf16-libraries-autotuned", ""),
        ("FI CuTe DSL", "flashinfer_bf16_cute_dsl", "bf16-extra-untuned", "bf16-extra-autotuned", " *"),
    ],
    "nvfp4": [
        ("vLLM BI W4A4", "vllm_cutlass_bi_w4a4", "nvfp4-primary", "nvfp4-autotuned", " †"),
        ("FI CuTe DSL", "flashinfer_cute_dsl", "nvfp4-primary", "nvfp4-autotuned", ""),
    ],
}
NAMES = {
    "flashinfer_bi": "FI batch invariant",
    "flashinfer_cutlass": "FI CUTLASS [non-BI]",
    "flashinfer_bf16_cute_dsl": "FI CuTe DSL [non-BI; decode only]",
    "flashinfer_cute_dsl": "FI CuTe DSL [non-BI]",
    "flashinfer_cudnn": "FI cuDNN [non-BI]",
    "torch_inductor_bi_triton": "PyTorch Inductor/Triton [BI]",
    "vllm_triton_bi": "vLLM Triton [BI]",
    "deepgemm_bf16_bi": "DeepGEMM BF16 [BI]",
    "vllm_cutlass_bi_w4a4": "vLLM BI W4A4",
}
COLORS = {
    "flashinfer_bi": "#367b45", "flashinfer_cutlass": "#718079",
    "flashinfer_bf16_cute_dsl": "#bd7a23", "flashinfer_cute_dsl": "#718079",
    "flashinfer_cudnn": "#b34e51",
    "torch_inductor_bi_triton": "#2676af", "vllm_triton_bi": "#9869aa",
    "deepgemm_bf16_bi": "#8d5a37", "vllm_cutlass_bi_w4a4": "#9869aa",
}


def save(fig, out, name):
    for ext in ("svg", "png"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=180, pad_inches=.18,
                    metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def overview(data, out, dtype, display_rows):
    pairs = PAIRS[dtype]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5 if dtype == "bf16" else 6.4),
                             sharey=True, squeeze=False)
    y = np.arange(len(pairs))
    labels = [f"{name} [{'BI' if BATCH_INVARIANT[backend] else 'non-BI'}]{note}"
              for name, backend, _, _, note in pairs]
    for row, mode in enumerate(MODES):
        for col, regime in enumerate(REGIMES):
            axis = axes[row, col]
            for i, (_, backend, untuned, tuned, _) in enumerate(pairs):
                dataset = tuned if mode == "autotuned" else untuned
                stats = data[dataset]["speedups"][backend][regime]
                value = stats["geomean_speedup"]
                display_rows.append({"kind": "competitor", "dtype": dtype, "study": mode,
                    "dataset": dataset, "backend": backend,
                    "batch_invariant": "Yes" if BATCH_INVARIANT[backend] else "No",
                    "regime": regime, "ratio": value, "cases": stats["cases"],
                    "definition": "comparator latency / FI batch invariant latency"})
                if value is None:
                    axis.text(.57, i, "N/A", va="center", fontsize=11, color=MUTED)
                    continue
                axis.barh(i, value, height=.64, color=BI_COLOR if BATCH_INVARIANT[backend] else NON_BI_COLOR)
                axis.text(value * 1.06, i, f"{value:.2f}×", va="center", fontsize=11, color=INK,
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": .5, "alpha": .95})
            axis.axvline(1, color="#343e37", linestyle="--", lw=1.1)
            axis.set_xscale("log", base=2)
            xmax = 32 if dtype == "bf16" and mode == "untuned" else 8 if dtype == "bf16" else 4
            ticks = [.5, 1, 2, 4, 8, 16, 32]
            axis.set_xlim(.5, xmax)
            axis.set_xticks([x for x in ticks if x <= xmax], [f"{x:g}×" for x in ticks if x <= xmax])
            axis.set_yticks(y, labels)
            axis.set_ylim(len(pairs) - .5, -.5)
            axis.set_title(f"{'Untuned' if row == 0 else 'Autotuned'} · {REGIME_TITLES[col]}",
                           fontsize=12, pad=12, weight="bold")
            axis.grid(axis="x", alpha=.18)
            axis.set_axisbelow(True)
            if dtype == "bf16":
                axis.axhline(2.5, color="#cfd7cc", linewidth=.8)
            for tick, (_, backend, _, _, _) in zip(axis.get_yticklabels(), pairs):
                tick.set_fontweight("bold" if BATCH_INVARIANT[backend] else "normal")
                tick.set_color(BI_COLOR if BATCH_INVARIANT[backend] else MUTED)
    fig.suptitle(f"{dtype.upper()} GEMM · NVIDIA Vera Rubin NVL4", fontsize=17, weight="bold", y=.985)
    fig.text(.59, .935 if dtype == "bf16" else .91,
             "Comparator latency / FI batch invariant latency · >1× favors FI batch invariant",
             ha="center", fontsize=12, color=INK)
    legend = fig.legend(handles=[Patch(color=BI_COLOR, label="Batch invariant (BI)"),
                                 Patch(color=NON_BI_COLOR, label="Non-batch-invariant (non-BI)"),
                                 Line2D([0], [0], color="#343e37", linestyle="--", label="FI batch invariant · 1×")],
                        loc="lower center", bbox_to_anchor=(.59, .073 if dtype == "bf16" else .10),
                        ncol=3, frameon=False, fontsize=10.5)
    for tick in (legend.get_texts()[0], legend.get_texts()[2]):
        tick.set_fontweight("bold")
    note = ("* FI BF16 CuTe DSL: M ≤ 16 only.  † vLLM and DeepGEMM retain their upstream table/heuristic in both rows.\n"
            "PyTorch untuned uses an unprofiled deterministic selector; autotuned uses Triton profiling. BI labels apply to the validated matrix."
            if dtype == "bf16" else
            "† vLLM BI W4A4 retains the same fixed configuration in both rows. Prequantized GEMM only; BF16 output.")
    fig.text(.02, .018, note, fontsize=10, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=.255, right=.97, top=.875 if dtype == "bf16" else .81,
                        bottom=.17 if dtype == "bf16" else .24, wspace=.28, hspace=.35 if dtype == "bf16" else .65)
    save(fig, out, f"{dtype}-overview")


def splitk_figures(data, rows, out, display_rows):
    cells = []
    values_by_dataset = {}
    for dtype in ("bf16", "nvfp4"):
        for mode in MODES:
            name = f"{dtype}-splitk-ablation-{mode}"
            selected = {}
            for record in rows:
                if record["dataset"] != name or record["backend"] != "flashinfer_unsplit_ablation":
                    continue
                policy = float(record["paired_fi_median_us"])
                unsplit = float(record["median_us"])
                retained = policy / unsplit
                assert math.isclose(retained * float(record["paired_speedup"]), 1, rel_tol=1e-12)
                n, k, m = (int(record[x]) for x in ("N", "K", "M"))
                selected[n, k, m] = retained
                cells.append({"dtype": dtype, "study": mode, "dataset": name, "N": n, "K": k, "M": m,
                    "policy_latency_us": policy, "forced_unsplit_latency_us": unsplit,
                    "retained_performance_without_splitk": retained,
                    "performance_loss_percent": 100 * (1 - retained)})
            values_by_dataset[name] = selected
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for row, mode in enumerate(MODES):
        for col, regime in enumerate(REGIMES):
            axis = axes[row, col]
            for i, dtype in enumerate(("bf16", "nvfp4")):
                name = f"{dtype}-splitk-ablation-{mode}"
                stats = data[name]["speedups"]["flashinfer_unsplit_ablation"][regime]
                retained = 1 / stats["geomean_speedup"]
                values = [v for (_, _, m), v in values_by_dataset[name].items()
                          if (m <= 16 if regime == "decode" else 16 < m < 1024 if regime == "middle" else m >= 1024)]
                assert math.isclose(math.exp(statistics.mean(map(math.log, values))), retained, rel_tol=1e-12)
                axis.barh(i, retained, height=.54, color="#bb5b4e" if retained < 1 else BI_COLOR)
                axis.text(retained + .035, i, f"{retained:.2f}×", va="center", fontsize=12, weight="bold",
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": .5, "alpha": .95})
                display_rows.append({"kind": "splitk_retention", "dtype": dtype, "study": mode,
                    "dataset": name, "backend": "flashinfer_unsplit_ablation", "batch_invariant": "Yes",
                    "regime": regime, "ratio": retained, "cases": stats["cases"],
                    "definition": "policy-class latency / forced-unsplit latency"})
            axis.axvline(1, linestyle="--", color="#343e37", lw=1)
            axis.set_xlim(0, 1.65)
            axis.set_ylim(1.65, -.65)
            axis.set_yticks([0, 1], ["BF16 [BI]", "NVFP4 [BI]"], weight="bold")
            axis.set_xticks([0, .5, 1, 1.5], ["0", "0.5×", "1×", "1.5×"])
            axis.set_title(f"{'Untuned' if row == 0 else 'Autotuned'} · {('Decode · M=1,16', 'Middle · M=128', 'Prefill · M=1024,8192')[col]}",
                           fontsize=11.5, weight="bold", pad=10)
            axis.grid(axis="x", alpha=.18)
            axis.set_axisbelow(True)
    fig.suptitle("Hypothetical no-split policy: performance retained", fontsize=17, weight="bold", y=.985)
    fig.text(.55, .91, "Unsplit throughput / policy-class throughput = policy latency / unsplit latency",
             ha="center", fontsize=12)
    fig.text(.55, .11, "0.5× = half the performance · 1× = no change · >1× = faster without splitting",
             ha="center", fontsize=11.5)
    fig.text(.03, .03, "Measured ablation of a hypothetical policy on eight split-region shapes; both classes are independently batch invariant.\n"
             "Same operands, MMA primitive, and epilogue; each class uses its own default or autotuned configuration.",
             fontsize=10, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=.16, right=.97, top=.80, bottom=.24, wspace=.18, hspace=.65)
    save(fig, out, "splitk-retained-performance")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.6))
    limit = max(3, math.ceil(max(abs(math.log2(cell["retained_performance_without_splitk"])) for cell in cells)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for row, mode in enumerate(MODES):
        for col, dtype in enumerate(("bf16", "nvfp4")):
            axis = axes[row, col]
            name = f"{dtype}-splitk-ablation-{mode}"
            manifest = data[name]["manifest"]
            shapes = [tuple(shape) for shape in manifest["shapes_nk"]]
            ms = manifest["tokens"]
            values = np.array([[values_by_dataset[name][n, k, m] for m in ms] for n, k in shapes])
            image = axis.imshow(np.log2(values), cmap="RdYlGn", norm=norm, aspect="auto")
            labels = [f"{n} × {k // 1024}K · C={manifest['ablation_classes'][f'N={n},K={k}']['policy_chunks']}" for n, k in shapes]
            axis.set_yticks(range(len(shapes)), labels, fontsize=9.5)
            axis.set_xticks(range(len(ms)), [str(m) for m in ms])
            axis.set_xlabel("Token rows M")
            axis.set_ylabel("N × K · policy chunks")
            axis.set_title(f"{dtype.upper()} · {'Untuned' if row == 0 else 'Autotuned'}", fontsize=13, weight="bold")
            for i in range(len(shapes)):
                for j in range(len(ms)):
                    axis.text(j, i, f"{values[i, j]:.2f}×", ha="center", va="center", fontsize=10,
                              color="white" if abs(math.log2(values[i, j])) > 2 else INK)
    fig.subplots_adjust(left=.16, right=.85, top=.86, bottom=.13, wspace=.64, hspace=.40)
    cax = fig.add_axes([.88, .20, .02, .60])
    bar = fig.colorbar(image, cax=cax, ticks=list(range(-limit, limit + 1)))
    bar.ax.set_yticklabels([f"{2 ** value:g}×" for value in range(-limit, limit + 1)])
    bar.set_label("Performance retained without split-K")
    fig.suptitle("Hypothetical no-split policy: per-shape tradeoffs", fontsize=17, weight="bold", y=.975)
    fig.text(.5, .922, "Policy latency / forced-unsplit latency · <1× means a loss · >1× means an improvement",
             ha="center", fontsize=12)
    fig.text(.03, .035, "Measured kernels for a hypothetical policy. Same primitive and epilogue; each reduction class is independently batch invariant.\n"
             "All measured shape/M pairs are shown. Values and absolute latencies are exported in splitk-retention-cells.csv.",
             fontsize=10.5, color=MUTED, linespacing=1.5)
    save(fig, out, "splitk-ablation")
    write_csv(out / "splitk-retention-cells.csv", cells)


def latency_curves(data, rows, out):
    by_dataset = defaultdict(dict)
    for row in rows:
        by_dataset[row["dataset"]][row["backend"], int(row["M"]), int(row["N"]), int(row["K"])] = float(row["median_us"])
    for name, result in data.items():
        if "splitk-ablation" in name:
            continue
        manifest = result["manifest"]
        medians = by_dataset[name]
        shapes = [tuple(shape) for shape in manifest["shapes_nk"]]
        backends = ["flashinfer_bi", *[b for b in result["speedups"] if b not in DISPLAY_EXCLUSIONS]]
        fig, axes = plt.subplots(math.ceil(len(shapes) / 2), 2,
                                 figsize=(12.8, 3.05 * math.ceil(len(shapes) / 2)), squeeze=False)
        for axis, (n, k) in zip(axes.flat, shapes):
            for backend in backends:
                ms = sorted(m for m in manifest["tokens"] if (backend, m, n, k) in medians)
                axis.plot(ms, [medians[backend, m, n, k] for m in ms], marker="o", markersize=3,
                          label=NAMES[backend], color=COLORS[backend], linewidth=1.7)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.axvline(16, linestyle=":", color="#aaa", linewidth=.8)
            axis.axvline(1024, linestyle=":", color="#aaa", linewidth=.8)
            axis.set_title(f"N={n:,}, K={k:,}", fontsize=11)
            axis.set_xlabel("Token rows M")
            axis.set_ylabel("GPU latency (µs)")
            axis.grid(alpha=.18, which="both")
        for axis in list(axes.flat)[len(shapes):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5,
                            frameon=False, bbox_to_anchor=(.5, .005))
        for text, backend in zip(legend.get_texts(), backends):
            text.set_fontweight("bold" if BATCH_INVARIANT[backend] else "normal")
        mode = "Autotuned" if manifest["fi_autotuned"] else "Untuned"
        fig.suptitle(f"{manifest['dtype'].upper()} GEMM · NVIDIA Vera Rubin NVL4\n{mode} study · cold L2 · CUPTI · BF16 output",
                     fontsize=14)
        fig.tight_layout(rect=(0, .065, 1, .94))
        save(fig, out, name)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Choose a fresh output directory; previous figures are retained")
    data = json.loads((args.analysis / "summary.json").read_text())
    with (args.analysis / "latency.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.out.mkdir(parents=True)
    display_rows = []
    for dtype in ("bf16", "nvfp4"):
        overview(data, args.out, dtype, display_rows)
    splitk_figures(data, rows, args.out, display_rows)
    latency_curves(data, rows, args.out)
    write_csv(args.out / "display-ratios.csv", display_rows)
    manifest = {
        "scope": "presentation-only rendering of frozen measurements", "gpu_launches": 0,
        "source_analysis": str(args.analysis.resolve()),
        "input_sha256": {name: hashlib.sha256((args.analysis / name).read_bytes()).hexdigest()
                         for name in ("summary.json", "latency.csv")},
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "backend_metadata_sha256": hashlib.sha256(Path(__file__).with_name("backend_metadata.py").read_bytes()).hexdigest(),
        "overview_layout": "untuned row, autotuned row; decode/middle/prefill columns",
        "overview_ratio": "comparator latency / FI batch invariant latency; >1 favors FI",
        "splitk_ratio": "policy-class latency / forced-unsplit latency; <1 means performance loss without splitting",
        "BI_labels": "bold and explicitly marked BI; other comparator labels marked non-BI",
        "display_exclusions": sorted(DISPLAY_EXCLUSIONS),
        "exclusion_scope": "blog presentation only; all original backend measurements remain in the frozen analysis and archive",
        "splitk_scope": "hypothetical no-split policy evaluated using measured kernels",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(__file__, args.out / "plot_script.py")
    shutil.copy2(Path(__file__).with_name("backend_metadata.py"), args.out / "backend_metadata.py")
    print(json.dumps({"out": str(args.out), "figure_pairs": len(list(args.out.glob('*.svg'))),
                      "aggregate_display_ratios": len(display_rows), "gpu_launches": 0}))


if __name__ == "__main__":
    main()
