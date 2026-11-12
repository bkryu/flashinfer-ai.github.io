# Branch validation

- Built the updated branch on November 12, 2026 with the official GitHub Pages image, `ghcr.io/actions/jekyll-build-pages:latest`, digest `sha256:0a7bd7594147d19792eedae8e32cf15f00e431e783e9fb3424b037f9abedff15`, and compared it with the previously built upstream `main` baseline.
- The x86 image ran through a local static emulator on the ARM host, with networking disabled. A Ruby runner exited after successful rendering to avoid an emulator shutdown fault. Rendering exited successfully; site templates, dependencies, and layout styles were unchanged.
- Compared generated output: 82 existing files unchanged. Only `index.html`, `posts/index.html`, `feed.xml`, and `sitemap.xml` change to include the new article. No existing files removed.
- All 22 article figures, six artifact download links, and appendix fragment links resolve. The main BF16/NVFP4 results use bar charts with separate untuned/autotuned rows and explicit BI/non-BI labels; the methodology and software versions are in appendices.
- The GitHub reading copy renders the same 22 figures through relative PNG links. The website post uses root-relative SVG links. `scripts/` remains excluded from the published site.
- The numerical-class schematic uses continuous connectors. The class-boundary map has no benchmark-shape markers. cuTile is excluded from displayed comparisons while its original measurements remain archived.
- All 60 selected aggregate display ratios match the frozen analysis; all 160 split-K retained-performance cells match the measured policy/unsplit latencies. The split-K study is explicitly a hypothetical policy evaluated with measured kernels. Other presentation changes do not alter measurements.
- The original measurement archive, summary, latency CSV, and policy-evaluation CSV retain their checksums. The separate CPU figure-reproducer ZIP passes its internal checksums and includes the current plotting sources and unchanged frozen analysis inputs. Its public CSV exports and artifact checksums are verified.
- Staged whitespace validation passes. No new GPU runs were needed for these presentation changes.
