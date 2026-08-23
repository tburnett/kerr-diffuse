# pixel_table.py Module Documentation

**File:** `like3/pixel_table.py`  
**Purpose:** Load, inspect, plot, and export Fermi Kerr pixel table data with per-band HEALPix views and residual diagnostics.

## Related Documentation

- Source list API: [SOURCELIST_API_REFERENCE.md](SOURCELIST_API_REFERENCE.md)
- Residual analysis summary: [RESIDUAL_ANALYSIS_SUMMARY.md](RESIDUAL_ANALYSIS_SUMMARY.md)

---

## Module Overview

This module provides a complete workflow for working with sparse HEALPix-based photon count maps from Fermi observations. The data can be loaded from Kerr format (NPZ + pickle pair) or from the module's Kerr-style FITS representation and is organized by PSF type and energy band.

### Main Classes
- `PixelTable` — Load and access pixel table bands
- `PixelTable.Band` — Single band (PSF × energy) with per-pixel model components
- `ResidualPlotter` — Compute and visualize residual diagnostics
- `ResidualPoints` — Collect and cluster high-sigma outliers
- `BinnedStat` — Helper for profile plots
- `KerrDataFile` — FITS export wrapper

### Helper Functions & Classes
- `grouper()` — Connected-component clustering for residual points
- `plot_residuals_for_given_energy()` — Multi-PSF scatter plots
- `histograms_of_residuals_for_given_energy()` — Multi-PSF residual histograms
- `multi_residual_plotter()` — PSF × energy grid of residual histograms
- `residual_scatter()` — Binned residual plot with error bars
- `multi_ait()` — Panel of AIT projections by band

---

## Class Reference

### PixelTable(dict)

Sparse HEALPix-based photon and model count container loaded from Kerr `.npz/.pickle` files or a Kerr-style FITS file.

#### Constructor

```python
PixelTable(root, *, ring=None)
```

**Parameters:**
- `root` (str or Path): Path stem for `.npz` and `.pickle` files, or a FITS filename. For example, `'files/kerr/toby_v4'` loads `toby_v4.npz` and `toby_v4.pickle`, while `'files/kerr/toby_v4.fits'` loads the FITS form directly.
- `ring` (bool or None): Output ordering. For Kerr `.npz/.pickle`, `None` behaves like `False` and preserves NESTED ordering; `True` converts to RING. For FITS input, `None` infers ordering from the FITS `SKYMAP` header.

**Attributes:**
- `self.name` (str): Root filename
- `self.ring` (bool): Ring ordering flag
- `self.diffuse` (ndarray): Flattened diffuse model counts across all bands
- `self.ptsrc` (ndarray): Flattened point-source model counts
- `self.photons` (ndarray): Flattened observed photon counts (int32)
- `self.pix` (ndarray): Flattened NESTED pixel indices (int64)
- `self.extsrc` (ndarray, optional): Flattened extended-source model
- `self.sunmoon` (ndarray, optional): Flattened Sun/Moon scattered light
- `self.meta_df` (DataFrame): Band metadata (event_type, emin, emax, nside, nocc, occupancy)
- `self.totals` (dict): Dictionary with 'diffuse' and 'ptsrc' arrays of per-band sums

**Example:**
```python
from like3.pixel_table import PixelTable
pt = PixelTable('files/kerr/toby_v4')
band = pt(2, 4)  # PSF2, energy_index=4
print(pt.meta_df)
```

#### Methods

**`__call__(psf_index, energy_index) → Band`**

Return a single band by indices (dictionary-style access).

```python
band = pixel_table(2, 7)  # PSF2, energy_index=7
```

**`ring_map(nside=128, component='data', frame='galactic') → ndarray`**

Combine all bands with compatible nside into a single HEALPix RING map.

**Parameters:**
- `nside` (int): Target HEALPix resolution
- `component` (str): Component to map: 'data', 'diffuse', 'ptsrc', 'model', 'resid'
- `frame` (str): Astropy coordinate frame (e.g., 'galactic', 'icrs')

**Returns:** 1D array of length `12 * nside²` in RING order

**Example:**
```python
mp = pixel_table.ring_map(nside=128, component='data')
```

**`ait_plot(component='data', *, nside=128, figsize=(12,6), fig=None, colorbar=True, cmap='viridis', frame='galactic', **kwargs) → AITfigure`**

Render all-sky AIT projection aggregated across all bands.

**Parameters:**
- `component` (str): Component to plot
- `nside`, `figsize`, `fig`, `colorbar`, `cmap`, `frame`, `**kwargs`: Passed to AITfigure

**Returns:** `utilities.skymaps.AITfigure` object (chainable)

**Example:**
```python
pt.ait_plot('resid', nside=64).show()
```

**`zea_plot(center, *, component='data', nside=256, figsize=(8,8), size=5, pixelsize=0.1, fig=None, frame='icrs', proj='ZEA', cmap='viridis', colorbar=True, title=None, **kwargs) → ZEAfigure`**

Render local ZEA projection around a center coordinate, aggregated across bands.

**Parameters:**
- `center` (SkyCoord or tuple): Center position
- `component`, `nside`, `size`: As in Band methods
- Other parameters: Passed to `utilities.skymaps.ZEAfigure`

**Returns:** `utilities.skymaps.ZEAfigure` object (chainable)

---

### PixelTable.Band(HEALPix)

Single event-type/energy slice of the pixel table. Extends `astropy_healpix.HEALPix` with model component arrays and residual computation.

#### Constructor

```python
Band(meta)
```

**Parameters:**
- `meta` (tuple): 5-element tuple `(psf_string, e0, e1, nside, nocc)` from metadata table

**Attributes (initialized at construction):**
- `self.psf` (str): PSF type (e.g., 'psf0', 'psf1', 'psf2', 'psf3')
- `self.e0`, `self.e1` (float): Energy bounds in MeV
- `self.energy` (str): Formatted energy string (e.g., '1.33 GeV')
- `self.key` (tuple): `(psf_index, energy_index)` for dictionary access
- `self.nside` (int): HEALPix nside parameter
- `self.nocc` (int): Number of occupied pixels
- `self.counts` (int): Always 0; reserved for future use

**Sparse array attributes (populated by PixelTable.__init__):**
- `self.pix` (ndarray): Pixel indices in nested order
- `self.photons` (ndarray): Observed photon counts per pixel
- `self.diffuse` (ndarray): Diffuse model counts per pixel
- `self.ptsrc` (ndarray): Point-source model counts per pixel
- `self.extsrc` (ndarray or None): Extended-source model counts per pixel
- `self.sunmoon` (ndarray or None): Sun/Moon scattered light per pixel
- `self.slice` (slice): Range into flattened PixelTable arrays
- `self.totals` (dict): Dictionary with per-band sums ('diffuse', 'ptsrc')

#### Inherited Properties (from HEALPix)

- `self.nside` — HEALPix resolution parameter
- `self.frame` — Coordinate frame (always 'galactic')
- `self.order` — Pixel ordering ('nested' or 'ring')
- HEALPix methods: `healpix_to_skycoord()`, `lonlat_to_healpix()`, `ring_to_nested()`, etc.

#### Methods

**`__repr__() → str`**

Returns a compact string representation.

```python
>>> band
Band(2, 4): psf2@4.22 GeV nside 256 occ 0.999
```

**`_optional_component(name) → ndarray or None`** [Internal]

Safe accessor for optional model components (`extsrc`, `sunmoon`).

**`_model_counts() → ndarray`** [Internal]

Return the full fitted model: `diffuse + ptsrc + extsrc + sunmoon` (with None handling for optional components).

**`_component_values(component) → ndarray`** [Internal]

Resolve a named component ('data', 'model', 'resid', 'sigma', 'diffuse', 'ptsrc', 'extsrc', 'sunmoon') to a per-pixel array.

**`_pixels_in_frame(frame) → ndarray`** [Internal]

Transform pixel indices to a specified coordinate frame if needed.

**`pix_to_ring(*, inplace=False) → ndarray`**

Convert pixel indices from NESTED to RING ordering.

**Parameters:**
- `inplace` (bool): If True, update `self.pix` in place; otherwise return a copy

**`skycoords` [property]**

Return all photon pixel centers as a `SkyCoord` in galactic frame.

**`cone_search(center, radius=5.0) → ndarray`**

Return a boolean mask for pixels within `radius` degrees of `center`.

**Parameters:**
- `center` (SkyCoord): Center position
- `radius` (float): Separation threshold in degrees

**Returns:** Boolean array indexing into `self.pix`

**`ring_map(nside=None, component='data', frame='galactic') → ndarray`**

Generate a HEALPix RING map of a single component, optionally degraded to lower resolution.

**Parameters:**
- `nside` (int or None): Target resolution; if None or > band nside, use band nside
- `component` (str): 'data', 'diffuse', 'ptsrc', 'model', 'resid', 'sigma'
- `frame` (str): Coordinate frame

**Returns:** 1D array of length `12 * nside²` in RING order

**Example:**
```python
data_map = band.ring_map(nside=64, component='data')
resid_map = band.ring_map(component='resid')
```

**`ait_plot(component, *, nside=128, figsize=(12,6), fig=None, colorbar=True, shrink=0.7, cmap='viridis', frame='galactic', log=True, **kwargs) → AITfigure`**

Render all-sky AIT projection for this band.

**Parameters:**
- `component` (str): Component to plot
- `log` (bool): If True, show log10(counts); NaN for zero values
- Other parameters: Standard plot options

**Returns:** `utilities.skymaps.AITfigure` (chainable)

**`zea_plot(component, center, *, nside=256, figsize=(8,8), pixelsize=0.05, size=5, fig=None, cmap='viridis', colorbar=True, title=None, **kwargs) → ZEAfigure`**

Render local Zero Equal Area projection around `center`.

**Parameters:**
- `component` (str): Component to plot
- `center` (SkyCoord): Center coordinate
- `size` (float): Side length in degrees
- Other parameters: Standard plot options

**Returns:** `utilities.skymaps.ZEAfigure` (chainable)

**`get_outliers(sigma_min=4) → DataFrame`**

Extract pixels whose normalized residual exceeds `sigma_min`.

**Returns:** DataFrame with columns:
- `pixel` (int): Nested pixel index
- `data` (float): Observed counts
- `model` (float): Model counts
- `sigma` (float): Normalized residual (data - model) / √model

**Example:**
```python
outliers = band.get_outliers(sigma_min=5)
print(f"Found {len(outliers)} pixels > 5σ")
```

---

### ResidualPlotter

Precompute and visualize residual diagnostics for a single band.

#### Constructor

```python
ResidualPlotter(band, nside=64)
```

**Parameters:**
- `band` (Band): Band object to analyze
- `nside` (int): Resolution for map degradation

**Attributes:**
- `self.band` (Band): Reference to input band
- `self.nside` (int): Effective resolution (min of input and band nside)
- `self.photons` (ndarray): Data map at nside resolution (RING)
- `self.model` (ndarray): Model map (diffuse + ptsrc + extended + sunmoon)
- `self.resid` (ndarray): Residual map (photons - model)
- `self.rnorm` (ndarray): Normalized residuals (resid / √model)

#### Methods

**`residual_adjustment(ylim=np.array([-10,10]), ax=None)`**

Fit a quadratic trend to fractional residuals in log-count space.

**Parameters:**
- `ylim` (array): Y-axis range for diagnostic plot
- `ax` (Axes): Matplotlib axis; if None, only compute coefficients

**Stores:**
- `self.coefficients` (ndarray): 3 polynomial coefficients
- `self.adjusted_model` (ndarray): Bias-corrected model

**`residual_hist(ax=None, rnorm=None, ylim=np.array([-5,5]), legend_fontsize=14)`**

Plot histogram of normalized residuals with overlaid Gaussian fit.

**Parameters:**
- `ax` (Axes): If None, create a new figure
- `rnorm` (ndarray): Custom residual array; uses `self.rnorm` if None
- `ylim` (array): Histogram x-range

**`plots()`**

Render a standard 3-panel diagnostic dashboard:
1. AIT projection of observed data
2. AIT projection of normalized residuals
3. Scatter plot of residuals vs. model with binned averages
4. Residual histogram with Gaussian fit

---

### ResidualPoints

Collect significant residual pixels across multiple PSF bands at a fixed energy.

#### Constructor

```python
ResidualPoints(pixel_table, energy_index, sigma_min=5)
```

**Parameters:**
- `pixel_table` (PixelTable): Source data
- `energy_index` (int): Energy bin (0–11 typical)
- `sigma_min` (float): Minimum significance threshold (default 5)

**Attributes:**
- `self.bands` (list): List of 4 Band objects (PSF0–3 at fixed energy)
- `self.sigma_min` (float): Threshold used
- `self.df` (DataFrame): Merged outlier table from all bands
  - Columns: pixel, data, model, sigma, glon, glat, psf, nside
- `self.skycoord` (SkyCoord): Galactic coordinates of outliers
- `self.cluster_idx` (list of arrays): Cluster membership indices (after `clusterer()`)
- `self.cldf` (DataFrame): Cluster summary table (after `clusterer()`)
- `self.clpoints` (SkyCoord): Representative points per cluster (after `ait_cluster_plot()`)

#### Methods

**`ait_plot()`**

Plot all outlier points on an all-sky AIT projection.

Colors and marker sizes reflect significance.

**`clusterer(radius=1.5, ptmin=2)`**

Group outliers into connected components using spatial separation threshold.

**Parameters:**
- `radius` (float): Maximum separation in degrees for two points to be neighbors
- `ptmin` (int): Minimum cluster size to keep

**Creates:**
- `self.cluster_idx` (list of arrays): Cluster membership
- `self.cldf` (DataFrame): Summary of each cluster
  - Columns: glon, glat, data, model, sigma, n, ids
  - Representative point chosen as highest-model pixel in cluster

**`zea_plot(center, size=5, **kwargs) → ZEAfigure`**

Plot outliers in a local ZEA projection around `center`.

**`ait_cluster_plot(*, figsize=(10,10), title=None, **kwargs)`**

Plot one representative point per cluster on all-sky AIT.

Marker size proportional to cluster size, color to significance.

---

### BinnedStat

Helper class for binned statistics (profile plots).

#### Constructor

```python
BinnedStat(x, y, bins)
```

**Parameters:**
- `x`, `y` (array): Data coordinates
- `bins` (array or int): Bin edges or count (scipy.stats.binned_statistic convention)

**Attributes:**
- `self.mean` (ndarray): Mean y per bin
- `self.std` (ndarray): Standard deviation of y per bin
- `self.count` (ndarray): Number of points per bin
- `self.x` (ndarray): Bin centers
- `self.xerr` (ndarray): Half-bin widths
- `self.bins` (ndarray): Bin edges

#### Example

```python
bstat = BinnedStat(model, residual, bins=np.linspace(0, 5, 20))
ax.errorbar(bstat.x, bstat.mean, xerr=bstat.xerr, yerr=bstat.std)
```

---

### KerrDataFile

Serialize `PixelTable` content to FITS format.

#### Constructor

```python
KerrDataFile(pixel_table, *, order='ring')
```

**Parameters:**
- `pixel_table` (PixelTable): Source data
- `order` (str): Declared pixel ordering in FITS header ('ring' or 'nested')

#### Methods

**`skymap_hdu() → BinTableHDU`**

Create sparse SKYMAP HDU with PIX, CHANNEL, VALUE columns.

**`band_hdu(version=5) → BinTableHDU`**

Create BANDS HDU with NSIDE, E_MIN, E_MAX, EVENT_TYPE columns.

**`writeto(filename, overwrite=True)`**

Write PrimaryHDU + SKYMAP + BANDS to FITS file.

**`readfrom(filename, kerrmodel)` [classmethod]**

Open and summarize a FITS file, return KerrDataFile instance.

**`to_fits(kerr_file, fits_file, *, ring=False, overwrite=True)` [classmethod]**

Convert NPZ/pickle pair to FITS in a single call.

---

## Helper Functions

### `grouper(points, radius) → list`

Group a SkyCoord array into connected clusters using angular separation.

**Parameters:**
- `points` (SkyCoord): Point positions
- `radius` (float): Maximum degrees for two points to be neighbors

**Returns:** List of clusters, each an array of point indices

**Algorithm:** Depth-first traversal to find connected components

---

### `plot_residuals_for_given_energy(pixel_table, energy_index) → Figure`

Scatter plot of photons vs. model in 2×2 grid (one per PSF type).

---

### `histograms_of_residuals_for_given_energy(pixel_table, energy_index) → Figure`

Residual histograms in 2×2 grid with Gaussian overlays.

---

### `multi_residual_plotter(pixel_table, nside=64) → Figure`

Large grid of residual histograms: rows = PSF, columns = energy.

---

### `residual_scatter(model, norm, ax=None, ylim=[-5, 5])`

Binned residual plot: normalized residuals vs. log10(model counts).

---

### `multi_ait(pixel_table, et, component='diffuse') → Figure`

3×4 grid of AIT projections by band for one event type.

---

## Usage Examples

### Load and Inspect

```python
from like3.pixel_table import PixelTable, ResidualPlotter, ResidualPoints

# Load pixel table
pt = PixelTable('files/kerr/toby_v4')
print(f"Loaded {len(pt)} bands, {pt.photons.sum():,d} photons")

# Access a single band
band = pt(2, 4)  # PSF2, 1.33 GeV
print(band)
print(band.skycoords)
```

### Visualize Components

```python
# All-sky map of one component
band.ait_plot('data', nside=64).show()

# Local zoom around a point
from astropy.coordinates import SkyCoord
center = SkyCoord(0, 0, unit='deg', frame='galactic')
band.zea_plot('model', center, size=5).show()
```

### Residual Analysis

```python
# Single-band diagnostics
rp = ResidualPlotter(band, nside=64)
rp.plots()  # Dashboard

# Cross-PSF outlier clustering
rp = ResidualPoints(pt, energy_index=4, sigma_min=5)
rp.clusterer(radius=1.5)
rp.ait_cluster_plot()
print(rp.cldf)
```

### Export to FITS

```python
from like3.pixel_table import KerrDataFile

KerrDataFile.to_fits('files/kerr/toby_v4', 'output.fits', ring=True)
```

---

## Type Hints

The module uses Python 3.10+ union syntax for optional types:

```python
self.extsrc: np.ndarray | None = None
self.totals: dict[str, object] = {}
```

---

## Dependencies

**Internal:**
- `utilities.skymaps` (AITfigure, ZEAfigure)

**External:**
- NumPy, Pandas, Matplotlib
- Astropy (SkyCoord, Angle, fits)
- astropy_healpix (HEALPix)
- scipy.stats (for Gaussian fitting in ResidualPlotter)

---

## Notes

1. **Lazy evaluation:** Most plot methods return chainable objects. Call `.show()` to display.

2. **Memory:** Pixel arrays are stored as views/slices into the parent PixelTable arrays to save memory.

3. **HEALPix ordering:** The module uses NESTED (compact) ordering internally but returns RING maps for compatibility with standard consumers.

4. **Component names:** Recognized by `_component_values()` and `ring_map()`:
   - `'data'` — observed counts
   - `'model'` — full fitted model
   - `'diffuse'` — diffuse component
   - `'ptsrc'` — point-source component
   - `'extsrc'` — extended sources (if present)
   - `'sunmoon'` — Sun/Moon scattering (if present)
   - `'resid'` — data - model
   - `'sigma'` — (data - model) / √model

5. **Optional components:** `extsrc` and `sunmoon` are initialized as `None` and safely handled in `_optional_component()` and `_model_counts()`.
