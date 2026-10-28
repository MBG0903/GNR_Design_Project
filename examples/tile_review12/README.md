# Twelve native-pixel review tiles

Input: the existing dense GenAI draft (401 × 340). No new staircase corrections have been applied.

Cores form a 4-column × 3-row partition, with 32 native pixels of surrounding context. Core sizes are 100/101 × 113/114; contexts are clipped only at image edges. Each tile has matching sketch and draft contexts. `manifest.json` defines global boxes and exact local coordinates.

`grid_preview.png` has blue diagnostic boundaries; those marks are not in the raw tile PNGs. `roundtrip_no_edits.png` matches the original draft pixel-for-pixel.

Do not cut off a curve's front/back or tips when reviewing it. Increase context or inspect the full component. Only core edits are merged; overlapping context is read-only. Coordinated cross-core corrections require full-curve and seam review after merging. Tiling does not itself enforce semantic hierarchy. The rejected strip fitter remains disabled.
