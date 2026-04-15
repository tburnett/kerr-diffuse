# SED Poisson Workflow Summary

## Scope
This note summarizes the FermiFit SED-Poisson workflow implemented and validated in the current development cycle.

## What Was Implemented
- Added a per-energy-bin SED Poisson table API where each row contains a `Poisson` object.
- Added convenience access from `FermiFit`.
- Added notebook validation that compares direct likelihood values with the Poisson approximation at representative flux points.

## Main Code Changes
- `like3/sedfuns.py`
  - Added `SED.poisson_table(...)` to return a DataFrame with one Poisson entry per bin.
  - Added `sed_poisson_table(...)` helper with compatibility for both ROI-like and FermiFit workflows.
- `like3/main.py`
  - Added `FermiFit.get_sed_poisson_table(...)`.
  - Initialized the `LikelihoodViews` base in `FermiFit.__init__` so view-based APIs work consistently.
- `like3/plotting/__init__.py`
  - Made optional plotting imports resilient so non-plotting SED paths can import cleanly.
- `like3/plotting/tsmap.py`
  - Fixed relative import for `SkyDir`.
- `like3/plotting/counts.py`
  - Made diffuse import optional and guarded isotropic normalization logic.

## Notebook Outcome
In `fermifit.ipynb`, a dedicated SED-Poisson section now:
- Builds and displays per-bin SED Poisson table columns.
- Plots the selected bin Poisson profile in flux space.
- Compares direct vs Poisson delta log-likelihood at low/peak/high test fluxes.

Representative validation result from the comparison step:
- `Max |difference| = 6.322e-05` for the selected bin.

## Interpretation
The tiny maximum discrepancy indicates the per-bin Poisson approximation is consistent with direct likelihood evaluation for the tested bin and points.

## Related Notebook
A standalone reproducible notebook is provided at:
- `docs/SED_POISSON_WORKFLOW.ipynb`
