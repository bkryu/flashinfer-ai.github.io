# Measured GEMM results

Speedup is baseline latency / FlashInfer invariant latency; above 1 favors FlashInfer invariant.

The FlashInfer invariant control is batch invariant (Yes) in every comparison. The column below describes the comparator's selected mode within the validated study matrix. Regular modes marked No do not provide the requested batch-invariance contract. The forced-unsplit ablation is independently batch invariant, with a different numerical class.

## bf16-untuned-v2

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FlashInfer regular CUTLASS | No | 1.69x | 1.46x | 2.51x |
| PyTorch Inductor/Triton invariant option | Yes | 9.03x | 11.49x | 20.89x |

## bf16-autotuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FlashInfer regular CUTLASS | No | 1.15x | 1.21x | 1.29x |
| PyTorch Inductor/Triton invariant option | Yes | 1.31x | 1.60x | 1.59x |

## bf16-extra-untuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FlashInfer regular BF16 CuTe DSL | No | 0.82x | N/A | N/A |
| vLLM persistent Triton invariant | Yes | 6.07x | 4.31x | 3.43x |

## bf16-extra-autotuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FlashInfer regular BF16 CuTe DSL | No | 0.76x | N/A | N/A |
| vLLM persistent Triton invariant | Yes | 6.06x | 4.40x | 3.46x |

## bf16-libraries-untuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FlashInfer cuDNN | No | 0.94x | 0.97x | 1.00x |
| FlashInfer cuTile | No | 3.05x | 2.20x | 2.64x |

## bf16-libraries-autotuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FlashInfer cuDNN | No | 0.94x | 0.99x | 1.00x |
| FlashInfer cuTile | No | 1.82x | 1.71x | 1.39x |

## bf16-deepgemm-untuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| DeepGEMM deterministic BF16 | Yes | 1.21x | 1.18x | 0.98x |

## bf16-deepgemm-autotuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| DeepGEMM deterministic BF16 | Yes | 1.22x | 1.21x | 0.99x |

## nvfp4-primary

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FlashInfer regular CuTe DSL | No | 0.92x | 0.96x | 1.02x |
| vLLM CUTLASS invariant W4A4 | Yes | 2.67x | 2.27x | 1.82x |

## nvfp4-autotuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FlashInfer regular CuTe DSL | No | 0.89x | 0.96x | 1.01x |
| vLLM CUTLASS invariant W4A4 | Yes | 2.73x | 2.30x | 1.84x |

## bf16-splitk-ablation-untuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FI forced unsplit (different numerical class) | Yes | 2.15x | 1.94x | 1.10x |

## bf16-splitk-ablation-autotuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FI forced unsplit (different numerical class) | Yes | 2.15x | 2.34x | 1.11x |

## nvfp4-splitk-ablation-untuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FI forced unsplit (different numerical class) | Yes | 1.26x | 1.46x | 0.72x |

## nvfp4-splitk-ablation-autotuned

| Baseline | Batch Invariant | Decode geometric mean | Middle geometric mean | Prefill geometric mean |
| --- | --- | ---: | ---: | ---: |
| FI forced unsplit (different numerical class) | Yes | 1.45x | 1.59x | 1.08x |
