# Spec Test Datasheet

## Purpose
This use case validates dataset quantity constraints during data profiling.

## Data Profiling Output
The operation `data_profiling_quantity_ge` generates a JSON report at:
`src/local_platform/artifacts/report/data_quantity_profile_report.json`

## Threshold Semantics
- `min_rows`: minimum accepted row count (inclusive)
- `max_rows`: maximum accepted row count (inclusive)
- Validation passes only when row count is within `[min_rows, max_rows]`

## Notes
Version v1 is intentionally limited to quantity checks using Great Expectations.
