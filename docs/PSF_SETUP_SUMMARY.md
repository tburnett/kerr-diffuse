# PSF Setup Summary

**Source:** `files/loc/psf_psf_table.pkl` — one row per (energy, event_type) pair with columns
`energy` (MeV), `event_type` (2–5 → PSF0–PSF3), `r68` (deg), and tabulated `x`/`y` arrays.

**`PSFlist.PSF` object** (`pylib/psf_func.py`): fits a `CubicSpline` to `(x, log(y))` with one
linear extrapolation step at the high-angle end. Calling `psf(angle)` returns `exp(spline(angle))`
— the PSF density in deg⁻². Key attributes: `r68`, `energy`, `event_type`.

**Assignment in `BandList.from_pixel_table`** (`like3/bands.py`):
1. Reads `pixel_table.meta_df` to get the `event_type_code` (2–5) for each band.
2. Calls `PSFlist(event_type=code, table_path='files/loc/psf_psf_table.pkl')` — PSF-partition IRF objects, not FB.
3. Each band is matched to the nearest PSF entry in log-energy space. If the table has fewer
   entries than bands, the highest-energy PSF is cloned for the extras.
4. The selected PSF object is stored as `band.psf`.

## r68 (deg) by partition and energy

| energy (MeV) | PSF0 | PSF1 | PSF2 | PSF3 |
|---:|---:|---:|---:|---:|
| 133 | — | — | 3.424 | 2.505 |
| 237 | — | — | 2.179 | 1.515 |
| 421 | — | 1.878 | 1.359 | 0.901 |
| 749 | — | 1.151 | 0.814 | 0.529 |
| 1333 | 1.479 | 0.703 | 0.482 | 0.320 |
| 2371 | 1.050 | 0.432 | 0.297 | 0.206 |
| 4216 | 0.768 | 0.280 | 0.195 | 0.138 |
| 7498 | 0.571 | 0.194 | 0.135 | 0.094 |
| 13335 | 0.447 | 0.144 | 0.098 | 0.067 |
| 23713 | 0.389 | 0.120 | 0.078 | 0.050 |
| 42169 | 0.359 | 0.113 | 0.069 | 0.041 |
| 74989 | 0.334 | 0.110 | 0.066 | 0.038 |
