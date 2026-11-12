"""Classify the selected backend mode, independently of performance results."""

BATCH_INVARIANT = {
    "flashinfer_bi": True,
    "torch_inductor_bi_triton": True,
    "vllm_triton_bi": True,
    "deepgemm_bf16_bi": True,
    "vllm_cutlass_bi_w4a4": True,
    "flashinfer_unsplit_ablation": True,
    "flashinfer_cutlass": False,
    "flashinfer_bf16_cute_dsl": False,
    "flashinfer_cute_dsl": False,
    "flashinfer_cudnn": False,
    "flashinfer_cutile": False,
}


def batch_invariant_label(backend):
    return "Yes" if BATCH_INVARIANT[backend] else "No"


def render_results(summary, labels):
    lines=["# Measured GEMM results", "",
           "Speedup is baseline latency / FlashInfer invariant latency; above 1 favors FlashInfer invariant.", "",
           "The FlashInfer invariant control is batch invariant (Yes) in every comparison. "
           "The column below describes the comparator's selected mode within the validated study matrix. "
           "Regular modes marked No do not provide the requested batch-invariance contract. "
           "The forced-unsplit ablation is independently batch invariant, with a different numerical class.", ""]
    for name,data in summary.items():
        lines += [f"## {name}", "",
                  "| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for backend,stats in data["speedups"].items():
            cells=[f"{stats[r]['geomean_speedup']:.2f}x" if stats[r]['geomean_speedup'] is not None else "N/A"
                   for r in ("decode","middle","prefill")]
            lines.append("| "+" | ".join([labels[backend],batch_invariant_label(backend),*cells])+" |")
        lines.append("")
    return "\n".join(lines).rstrip()+"\n"
