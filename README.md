# Urban Mission Planning Agent — Dataset & Submission Guide

## Overview
Participants must produce road-constrained pixel paths between mission start and goal points on high-resolution aerial TIFF images. The goal is to submit plausible on-road paths (pixel coordinates) for each test image; a supplied evaluator scores submissions by length and off-road violations.

## Dataset layout
- `reference/` — training/reference data
  - `sats/` — satellite TIFFs (native resolution)
  - `maps/` — organizer-provided road masks (reference)
  - `solutions/` — example solution JSONs (50 items)
- `test/` — test inputs
  - `sats/` — test satellite TIFFs (10 items: `test_001.tiff` → `test_010.tiff`)

## Mission input
- Satellite image: single RGB TIFF at native resolution.
- Mission: start and end pixel coordinates (format: `[x, y]`). Origin is top-left; `x` increases to the right, `y` increases downward.

## Submission format
- Submit one JSON record per test sample. The repository includes an example at [example_submission.json](example_submission.json).
- Each entry must be a JSON object with the following schema:

```
{
  "id": "test_001",
  "path": [[x1,y1],[x2,y2],...]
}
```

- Requirements:
  - `id` must match the test file name (e.g., `test_001`).
  - `path` must contain at least two points.
  - All coordinates must be integers and within the image bounds.
  - Paths should remain on-road to avoid violations.

## Example
- See [example_submission.json](example_submission.json) for a minimal, valid example covering `test_001` → `test_010`.

## Scoring / Evaluation
The official evaluator computes:

Score = 1000 - PathLength - 50 × Violations

Where:
- `PathLength` is cumulative Euclidean distance in pixel units along polyline segments.
- `Violations` counts off-road traversals, out-of-bounds points, and other invalid moves. The evaluator rasterizes submitted segments against the official map mask to detect segment-level off-road crossings.

Tips:
- Keep paths compact and follow road centerlines when possible.
- Avoid unnecessary jitter — shorter, valid paths score higher.

## How to generate a submission
1. For each test TIFF in `test/sats/`, compute a path (list of `[x,y]` pixel coordinates).
2. Create one JSON object per test image following the schema above.
3. Combine objects into a single JSON array (or provide one JSON file per test if required by an external leaderboard).

## Reference and contact
- If you have questions open an issue on the project GitHub: https://github.com/SamyakSS83/ump_data
