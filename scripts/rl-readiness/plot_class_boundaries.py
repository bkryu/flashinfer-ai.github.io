"""Draw boundary-only N/K maps from saved class-policy evaluations, without CUDA.

Example from the study directory (with matplotlib/numpy on PYTHONPATH):
  python plot_class_boundaries.py --data class-map-v2 --out class-map-boundaries-v3

Input CSV values are preserved. Only plot presentation changes; the old annotated
figures and their original evaluation/generation code remain in the data directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "svg.hashsalt": "flashinfer-class-boundaries", "savefig.facecolor": "white",
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory containing classes.csv and manifest.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="Fresh output directory; previous figures are retained")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Choose a fresh output directory; previous figure versions are retained")
    manifest = json.loads((args.data / "manifest.json").read_text())
    ns, ks = manifest["N_grid"], manifest["K_grid"]
    with (args.data / "classes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {(row["format"], int(row["N"]), int(row["K"])): int(row["chunks"])
              for row in rows if row["format"] in ("BF16", "NVFP4")}
    assert len(counts) == 2 * len(ns) * len(ks)
    assert set(counts.values()) <= {1, 2, 4, 8, 16, 32, 64, 128}
    args.out.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    fig.subplots_adjust(left=.07, right=.84, top=.82, bottom=.25, wspace=.14)
    cmap = plt.get_cmap("viridis", 8)
    norm = BoundaryNorm(np.arange(-.5, 8.5), 8)
    for axis, name in zip(axes, ("BF16", "NVFP4")):
        values = np.array([[int(counts[name, n, k]).bit_length() - 1 for n in ns] for k in ks])
        # Same saved N/K grid and chunk counts as the prior annotated plot.
        mesh = axis.pcolormesh(ns, ks, values, cmap=cmap, norm=norm,
                               shading="nearest", rasterized=True)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xlabel("Output dimension N", fontsize=11)
        axis.set_title(name, fontsize=13, weight="bold", pad=10)
        axis.set_xticks([32, 128, 512, 2048, 8192, 32768], ["32", "128", "512", "2K", "8K", "32K"])
        axis.set_yticks([256, 1024, 4096, 16384, 65536, 262144], ["256", "1K", "4K", "16K", "64K", "256K"])
        axis.set_xlim(32, 32768)
        axis.set_ylim(256, 524288)
        axis.grid(alpha=.12)
    axes[0].set_ylabel("Reduction dimension K", fontsize=11)
    cax = fig.add_axes([.865, .25, .018, .57])
    colorbar = fig.colorbar(mesh, cax=cax, ticks=range(8), label="Rule-selected chunk count C")
    colorbar.ax.set_yticklabels(["1 (unsplit)", "2", "4", "8", "16", "32", "64", "128"])
    fig.suptitle("Analytical N/K class-selection rule · SM107", fontsize=16, weight="bold", y=.98)
    fig.text(.5, .916, "Fixed weight dimensions and format select C; token count M is not an input.",
             ha="center", fontsize=12, color="#617069")
    for x, label in ((.07, "1 · Add parallelism when N is narrow"),
                     (.37, "2 · Limit partial-result traffic"),
                     (.67, "3 · Keep chunks aligned and long")):
        fig.add_artist(FancyBboxPatch((x, .105), .268, .055,
                                      boxstyle="round,pad=.006", transform=fig.transFigure,
                                      facecolor="#edf5e8", edgecolor="#cbdac0", linewidth=.8))
        fig.text(x + .134, .1325, label, ha="center", va="center", fontsize=10, color="#365c20")
    fig.text(.07, .052,
             "NVFP4: unsplit classes use 128-wide MMA steps; split classes use 64-wide steps. Alignment also shapes the boundaries.",
             fontsize=9.5, color="#617069")
    for ext in ("svg", "png"):
        path = args.out / f"class-regions.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=170, pad_inches=.18,
                    metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    provenance = {
        "kind": "boundary-only rendering of existing policy evaluations",
        "gpu_launches": 0, "policy_records": len(rows), "plotted_records": len(counts),
        "source_data_directory": str(args.data.resolve()),
        "classes_csv_sha256": hashlib.sha256((args.data / "classes.csv").read_bytes()).hexdigest(),
        "source_manifest": manifest,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes": "No benchmark markers; no policy values or measurements changed. Original annotated figures retained in source data directory.",
    }
    (args.out / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copy2(__file__, args.out / "plot_script.py")
    print(json.dumps({"out": str(args.out), "policy_records": len(rows),
                      "plotted_records": len(counts), "gpu_launches": 0}))


if __name__ == "__main__":
    main()
