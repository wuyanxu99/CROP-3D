# Reproduction Boundary

This artifact supports two levels of reproduction.

## Included

- Regenerating plant-height metric and x-z side-view figures from compact packaged results.
- Recomputing one light-interception ROI from a packaged local canopy PLY and PPFD model.
- Inspecting five-region light-interception summary results.
- Reading cleaned copies of the reconstruction, rendering, and downstream phenotyping scripts.
- Running the one-command reconstruction entry (`quick_start.py`) when COLMAP, CUDA PyTorch, DA3 weights, and 2DGS extensions are installed.

## Not Included

- Full raw camera image/video datasets.
- Full COLMAP databases.
- Full reconstruction point clouds and complete training outputs.
- Large model checkpoints and high-resolution videos.
- A lightweight bundled reconstruction dataset large enough for final paper-quality training.

These are omitted to keep the repository GitHub-friendly and to respect data-size and release constraints. Add final external links in `docs/dataset.md` and `docs/model_weights.md` before public release.
