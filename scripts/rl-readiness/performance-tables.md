# Measured GEMM results

Primary comparison ratios are comparator latency / FI batch invariant latency; above 1 favors FI batch invariant.
The split-K sections instead show performance retained under a hypothetical no-split policy, using measured kernels. Values below 1 show a loss.

Both classes in the split-K study are independently batch invariant, with different numerical contracts.

## bf16-untuned-v2

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI CUTLASS [non-BI] | No | 1.69× | 1.46× | 2.51× |
| PyTorch Inductor/Triton [BI] | Yes | 9.03× | 11.49× | 20.89× |

## bf16-autotuned

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI CUTLASS [non-BI] | No | 1.15× | 1.21× | 1.29× |
| PyTorch Inductor/Triton [BI] | Yes | 1.31× | 1.60× | 1.59× |

## bf16-extra-untuned

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI CuTe DSL [non-BI; decode only] | No | 0.82× | N/A | N/A |
| vLLM Triton [BI] | Yes | 6.07× | 4.31× | 3.43× |

## bf16-extra-autotuned

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI CuTe DSL [non-BI; decode only] | No | 0.76× | N/A | N/A |
| vLLM Triton [BI] | Yes | 6.06× | 4.40× | 3.46× |

## bf16-libraries-untuned

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI cuDNN [non-BI] | No | 0.94× | 0.97× | 1.00× |

## bf16-libraries-autotuned

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI cuDNN [non-BI] | No | 0.94× | 0.99× | 1.00× |

## bf16-deepgemm-untuned

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| DeepGEMM BF16 [BI] | Yes | 1.21× | 1.18× | 0.98× |

## bf16-deepgemm-autotuned

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| DeepGEMM BF16 [BI] | Yes | 1.22× | 1.21× | 0.99× |

## nvfp4-primary

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI CuTe DSL [non-BI] | No | 0.92× | 0.96× | 1.02× |
| vLLM BI W4A4 | Yes | 2.67× | 2.27× | 1.82× |

## nvfp4-autotuned

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI CuTe DSL [non-BI] | No | 0.89× | 0.96× | 1.01× |
| vLLM BI W4A4 | Yes | 2.73× | 2.30× | 1.84× |

## bf16-splitk-ablation-untuned

**Hypothetical no-split policy:** policy-class latency / forced-unsplit latency. Geometric means use the seven displayed shapes.

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI forced unsplit (hypothetical policy) | Yes | 0.54× | 0.59× | 0.91× |

## bf16-splitk-ablation-autotuned

**Hypothetical no-split policy:** policy-class latency / forced-unsplit latency. Geometric means use the seven displayed shapes.

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI forced unsplit (hypothetical policy) | Yes | 0.54× | 0.47× | 0.91× |

## nvfp4-splitk-ablation-untuned

**Hypothetical no-split policy:** policy-class latency / forced-unsplit latency. Geometric means use the seven displayed shapes.

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI forced unsplit (hypothetical policy) | Yes | 0.91× | 0.77× | 1.47× |

## nvfp4-splitk-ablation-autotuned

**Hypothetical no-split policy:** policy-class latency / forced-unsplit latency. Geometric means use the seven displayed shapes.

| Comparator | Batch Invariant | Decode | Middle | Prefill |
| --- | --- | ---: | ---: | ---: |
| FI forced unsplit (hypothetical policy) | Yes | 0.78× | 0.70× | 0.92× |
