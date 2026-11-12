# Reproducing the Vera Rubin NVL4 GEMM study

This payload preserves the measured code snapshots and raw timing samples. It does not contain GPU binaries, vendored dependency clones, or an installed serving engine. All public datasets use NVIDIA Vera Rubin NVL4 (SM107), BF16 output, cold-L2 CUPTI timing, and three rounds of fifty samples. Quantization is excluded from NVFP4 timings.

## Source and environment

Extract the archive into a workspace with `flashinfer/` and `vllm/` source directories beside `blog/`. Use the exact source commits recorded in each dataset's manifest. The primary commits are FlashInfer `3db2f474e4053a6400d3128ba0683cbd4a05f5a4` and vLLM `aeadf9252586e9db0e7658312d562036b45cd217`. Initialize FlashInfer's CUTLASS/CCCL dependencies from that checkout. Place CUTLASS v4.7.1, commit `cb4247394dd82148787aed73e5dc7cef33cbf862`, at `blog/rl-readiness/cutlass-vllm/` for the registration-only vLLM NVFP4 binding.

Each newer dataset includes `pip-freeze.txt`; manifests record compiler, driver, framework, CUPTI, and DSL versions. The measured environment uses Python 3.12, Torch 2.14.1+cu132, CUDA compiler 13.4.92, driver 620.43, Triton 3.8.0, CuTe DSL 4.8.1, CUPTI Python 13.5.0, cuDNN 9.24.0.43, and cuDNN frontend 1.32.2. The FlashInfer checkout is imported directly. Its stale editable distribution version is not the source identity.

Activate the matching Python environment. Set `CUDA_HOME` and its `bin` directory in `PATH`, and set `TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"` because the bundled Triton assembler does not support SM107. Keep `FLASHINFER_WORKSPACE_BASE`, `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, and compiler caches on local scratch. Create the workspace `.gpu-locks/` directory. The measured host's existing `env.sh` provides these settings; `run_study.sh` sources it when present.

The vLLM adapter uses unchanged upstream CUDA source and the `sm_100f` compatibility target, because its pinned CUTLASS lacks SM107 feature guards. The BF16 vLLM adapter loads unchanged AST function bodies and upstream shape tables, replacing platform-property queries only. It does not install or time a full vLLM engine.

DeepGEMM is pinned to `5081f8513af57e61696de4f4f04b0e707ffb503e`, with CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` and DeepJIT `2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43`. Run `bash blog/rl-readiness/build_deepgemm.sh` for an isolated build. It extracts missing Ubuntu arm64 libdw/libelf headers locally and uses the recorded FI CCCL headers, without installing system packages or replacing Torch. `deepgemm-build-manifest.json` records the header packages' hashes and compiler. DeepGEMM's deterministic flag disables its cuBLAS dispatch. Its pinned CUTLASS also fails native SM107 feature guards; `deepgemm_nvcc_compat.py`, selected through the native DeepJIT compiler-path option, rewrites only the JIT target from `sm_107f` to compatible `sm_100f`. Source kernels and heuristics are unchanged. Requested/actual compiler arguments, generated sources, and binary hashes are archived.

## Verify and rerun

Run `sha256sum -c SHA256SUMS` from the archive root to check the payload. All GPU launches must pass through `with_idle_gpu.py`; on the measured host, only physical GPUs 2 and 3 are permitted. The gate acquires an exclusive lock and checks target utilization, memory, and processes three times before launch. It refuses a busy device.

Replay an exact measured snapshot into a fresh directory:

```bash
export VLLM_BATCH_INVARIANT=1 MAX_JOBS=2
export TORCHINDUCTOR_BATCH_INVARIANT=1 TORCHINDUCTOR_DETERMINISTIC=0
export PYTHONDONTWRITEBYTECODE=1
python -B blog/rl-readiness/with_idle_gpu.py --gpu 2 -- \
  python -B blog/rl-readiness/reproduce.py \
  blog/rl-readiness/results/bf16-autotuned \
  --out blog/rl-readiness/results/replay-bf16-autotuned
```

`reproduce.py` checks the measured harness hash and source commits, restores the measured shape/M grid and selection flags, and uses measured helper snapshots. The same command works with any included dataset. It rejects existing sample files. Tuning is performed outside measurement; fresh profiling can choose another equivalent tactic. Archived configuration files preserve what the original study selected.

To collect a fresh pair with the current harness:

```bash
bash blog/rl-readiness/run_study.sh --gpu 2 --study bf16-libraries \
  --out blog/rl-readiness/results/fresh-libraries
```

The runner defaults to **both** modes. Other studies are `bf16-primary`, `bf16-extra`, `bf16-deepgemm`, `nvfp4`, `splitk-bf16`, and `ablation-bf16` / `ablation-nvfp4`. Separate `--mode untuned` or `--mode autotuned` invocations are supported for scheduling. vLLM's BF16 shape table and fixed NVFP4 kernel, and DeepGEMM's native deterministic BF16 heuristic, expose no runtime tuning interface; both studies report their actual unchanged selection mode.

`--splitk-ablation` changes the reduction class by forcing C=1 through FI's existing internal runner, holding the native split region's MMA primitive fixed. Both classes are independently tested for invariance; tuned results must match their own defaults. The ablation uses eight split-region N/K shapes and M=1,16,128,1024,8192, independently of the primary performance matrix. `map_classes.py --out <fresh-directory>` evaluates the actual SM107 class policy on CPU for all six named input formats. `class-map/` includes the grid, exact selected shapes, source hashes, plot, and script snapshot.

## Regenerate plots

Install Matplotlib and its dependencies into a separate plotting environment. The original plotting environment used Matplotlib 3.11.2. Pass all complete publication datasets to `analyze_results.py --out <analysis-directory>`, then run `plot_overview.py <analysis-directory>` and `plot_ablation.py <analysis-directory>`. The supplied `analysis/summary.json` records that dataset list, and `analysis/latency.csv` includes absolute latency, dispersion, effective TFLOP/s, paired invariant controls, and ratios.

Reports and CSVs include the selected backend's `Batch Invariant` Yes/No classification. `backend_metadata.py` supplies those flags; it does not infer a guarantee from numerical coincidence in a regular backend. The local article uses relative PNG links. Run `update_draft_results.py <analysis-directory>` and then `prepare_review.py --analysis <analysis-directory>` to refresh its tables and standalone offline HTML preview (requires markdown-it-py and PyYAML). `prepare_site_branch.py` emits root-relative SVG links for Jekyll and a PNG reading copy for GitHub. SVGs and PNGs are both included in the archive.

Analysis rejects incomplete datasets, unexpected record counts, missing/duplicate rounds, wrong repeat counts, failed invariant checks, and failed accuracy checks. Each ratio uses the FI batch invariant control measured in the same sweep. Aggregates weight each shape/M case equally. The article restricts BF16 CuTe DSL to decode (M <= 16); its middle and prefill cells are unavailable. Early M=32 measurements remain in raw files but are excluded from figures and aggregates. Previous figures are archived on regeneration. Raw samples are never replaced by plot summaries.

`audit_results.py <analysis-directory>` verifies snapshot/helper hashes, source-generated code, correctness records, GPU restrictions, and dispersion. `audit_round_sensitivity.py` checks the effect of omitting round zero without changing any publication data. Individual BF16 first rounds can be slower, and follow-up five-round probes reproduce warming over more than one round in some cases. The largest change in a reported regime aggregate when omitting round zero is 0.51%; this sensitivity result is not a confidence bound on every individual case. Treat near-parity rankings cautiously. The five-round probes and their launch scripts are under `diagnostics/`, separate from publication inputs.

The bundle excludes exploratory, failed, interrupted, and dynamic-PyTorch runs from publication inputs. Those remain in the working directory. No cuBLAS timing data enters the publication artifacts.
