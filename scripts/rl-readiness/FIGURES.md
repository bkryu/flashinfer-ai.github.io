# Reproduce the current blog figures on CPU

Requires Python, matplotlib, and numpy. No CUDA, FlashInfer installation, GPU, or new measurements are needed.

From this extracted directory, with each output directory initially absent:

    python plot_blog_performance.py --analysis analysis --out performance
    python plot_class_boundaries.py --data class-map --out boundaries
    python draw_batch_invariance.py --out batch-diagrams
    python draw_numerical_dataflow.py --out dataflow-diagrams

The performance renderer writes BF16/NVFP4 overviews with separate untuned/autotuned rows, split-K retention bars and heatmap, ten full-latency figures, display-ratios.csv, and splitk-retention-cells.csv. BI names are bold and explicitly labeled; non-BI paths are labeled separately. Ratios in the main comparisons are comparator latency / FI batch invariant latency. Split-K retention is policy latency / forced-unsplit latency, so values below one show the performance lost when removing splitting. The original latency.csv paired_speedup column retains its original reciprocal definition.

analysis/summary.json and analysis/latency.csv are unchanged frozen analysis inputs. class-map/ contains the unchanged saved class-policy evaluations; the new boundary-only rendering has no benchmark-shape markers. Input hashes and source hashes accompany the scripts. The original raw-measurement archive supplies kernel/harness snapshots, samples, correctness checks, and the older canonical-engine diagram source. Earlier figure versions remain in the workspace archives.


The current figures show selected comparisons while retaining all original measurements in the frozen inputs. The split-K figures evaluate a hypothetical no-split policy with actual kernel timings. The numerical-class diagram uses continuous elbow connectors.

The displayed split-K study now uses seven shapes, excluding N=128,K=65536 from both the heatmap and aggregate bars. The full eight-shape sweep remains in the CSV; included_in_displayed_study marks the 140 displayed and 20 retained-but-omitted cells. The same subset is used for all formats and tuning modes.
