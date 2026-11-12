# Reproducing the RL GEMM Study

The [article](../../_posts/2026-11-17-rl-ready-kernels.md) measures GEMMs on NVIDIA Vera Rubin NVL4 (SM107). Both untuned and autotuned measurements, their exact source snapshots, raw samples, and selected configurations are retained.

## Measurement Bundle

Download the [measurement archive](../../assets/data/rl-ready-kernels/flashinfer-rl-study.tar.gz) and its [SHA-256 checksum](../../assets/data/rl-ready-kernels/SHA256SUMS). From their download directory:

```bash
sha256sum -c SHA256SUMS
tar -xzf flashinfer-rl-study.tar.gz
```

Follow `flashinfer-rl-study/blog/rl-readiness/REPRODUCING.md` for the pinned dependencies, environment setup, and exact-snapshot replay commands. GPU launches use the included idle-device gate. On the measured host, use only physical GPUs 2 and 3 after the gate confirms they are free.

The bundle includes the full measured shape sweeps and all timing samples. The article's hypothetical no-split figures display seven of the eight measured split-region shapes; display-selection flags preserve that distinction.

## Current Figures on CPU

The [figure reproducer](../../assets/data/rl-ready-kernels/figure-reproducer.zip) includes the current drawing and plotting scripts, frozen analysis inputs, manifests, and internal checksums. It does not require CUDA or a GPU. Verify it against [FIGURE_SHA256SUMS](../../assets/data/rl-ready-kernels/FIGURE_SHA256SUMS), then extract it and follow `figure-reproducer/README.md`.

Use this separate figure bundle for the article's current presentation. The measurement archive preserves the original campaign and its original rendering scripts unchanged.

The [display ratios](../../assets/data/rl-ready-kernels/display-ratios.csv) contain the exact aggregates behind the comparison bars. The [split-K cells](../../assets/data/rl-ready-kernels/splitk-retention-cells.csv) retain all 160 measured cells and identify the 140 used in the displayed study. These CSVs share the figure-artifact checksum file.

All published operands, scales, output data types, source commits, measurement boundaries, and selection modes are documented in the article and archive. No cuBLAS performance comparison is included in the article.
