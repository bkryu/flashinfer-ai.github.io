# Branch validation

- Built both upstream `main` and this branch with the official GitHub Pages image, `ghcr.io/actions/jekyll-build-pages:latest`, digest `sha256:0a7bd7594147d19792eedae8e32cf15f00e431e783e9fb3424b037f9abedff15`.
- The x86 image ran through a local static emulator on the ARM host. A Ruby runner exited after successful rendering to avoid an emulator shutdown fault. Both final builds exited successfully; site sources and dependencies were unchanged.
- Compared generated output: 82 existing files unchanged. Only `index.html`, `posts/index.html`, `feed.xml`, and `sitemap.xml` change to include the new article. No existing files removed.
- All 14 figures and both reproduction download links resolve in the built article. The five backend/performance tables include the Batch Invariant column.
- The GitHub reading copy renders the same 14 figures through relative PNG links. The website post uses root-relative SVG links. `scripts/` remains excluded from the published site.
- All previous numerical results are unchanged. CSV values and SVG geometry match the study originals. The refreshed reproduction archive's 913 files passed their internal checksum checks.
- Staged whitespace validation passes. No new GPU runs were needed for these presentation changes.
