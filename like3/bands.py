"""Band-level model evaluation utilities for likelihood workflows.

This module defines:
- `Band`: a HEALPix-backed view of one energy bin with methods to evaluate
    model flux, gradients, predictions, simulation, and local map plotting.
- `BandList`: a container of `Band` objects with helpers for per-band counts,
    count gradients, and simple simulation/demo setups.
"""

import numpy as np
import pandas as pd
import copy
from astropy_healpix import HEALPix 
from astropy.coordinates import SkyCoord
from .sourcelist import SourceModel
from collections import namedtuple

# Define a namedtuple type for a key,value pair of lists
Pixels = namedtuple('Pixels', ['key', 'value'])

def create_pixels(keys, values):
    """Create a Pixels namedtuple from lists of keys and values.
    """
    if not isinstance(keys, (list, np.ndarray)) or not isinstance(values, (list, np.ndarray)):
        raise ValueError("Keys and values must be lists or numpy arrays.")
    return Pixels(keys, values)


def _event_type_code(value):
    """Normalize event-type metadata to the standard integer code."""
    if hasattr(value, 'event_type'):
        return int(getattr(value, 'event_type'))

    if isinstance(value, str):
        label = value.strip().upper()
        if label == 'FRONT':
            return 0
        if label == 'BACK':
            return 1
        if label.startswith('PSF'):
            return int(label[3:]) + 2
        if label.isdigit():
            return int(label)

    if isinstance(value, (int, np.integer)):
        return int(value)

    raise ValueError(f'Unsupported event type metadata: {value!r}')


def _clone_psf_with_event_type(psf, event_type):
    """Return a shallow PSF copy whose event_type matches the target band."""
    cloned = copy.copy(psf)
    setattr(cloned, 'event_type', int(event_type))
    if isinstance(cloned, dict):
        cloned['event_type'] = int(event_type)
    return cloned


def _make_sparse_pixel_lookup(pixel_band, exposure_values):
    """Build a callable that returns saved sparse-pixel exposure by pixel id."""
    pix = np.asarray(getattr(pixel_band, 'pix', ()), dtype=int).ravel()
    values = np.asarray(exposure_values, dtype=float).ravel()

    if pix.size != values.size:
        raise ValueError(
            f'Saved pixel exposure for band {getattr(pixel_band, "key", "<unknown>")} has '
            f'{values.size} values for {pix.size} sparse pixels'
        )

    lookup = dict(zip(pix.tolist(), values.tolist()))

    def map_from_sparse_pixels(requested_pix, lookup=lookup):
        requested = np.asarray(requested_pix, dtype=int).ravel()
        return np.array([lookup.get(int(pix), 0.0) for pix in requested], dtype=float)

    return map_from_sparse_pixels



class Band(HEALPix):
    """HEALPix representation of a single energy band for a source model.

    Notes
    -----
    TODO: include full PSF handling per band and use it to convolve source
    model terms during flux and gradient evaluation.
    """

    def __init__(self, band_info, source_model, exposure_map=None, data=None):
        """Initialize one analysis band from metadata and a source model.

        Parameters        
        ----------
        band_info : dict
            Band metadata. Expected keys include `energy`, `nside`, and `psf`
        source_model : SourceModel
            Source model, a list of sources, used to evaluate count fluxes and gradients for this band.
        exposure_map : optional
            Exposure map for the band, if any. 
        data : optional
            Data, a tuple (pixels, counts) for the band, if any.
            If provided, this is used for sparcifationn of model evaluation and gradient calculation.
            It can be set with the `simulate` method if not provided at initialization.
        """
        from typing import Callable
        self.source_model = source_model
        self.exposure_map: Callable = exposure_map  # type: ignore[assignment]
        self.data = data
        self.energy = band_info.get("energy")
        self.nside = band_info.get("nside")
        self.psf = band_info.get("psf")
        self.order = band_info.get("order", "nested")

        # Initialize the HEALPix geometry used by response evaluators.
        super().__init__(self.nside, order=self.order, frame='galactic')
        # set up exposure calculation function for this band based on energy, if not provided by a data-based exposure model
        if self.exposure_map is None:
            self.exposure_map = lambda pix: np.ones_like(pix) * 1e13 * self.energy / 100
        assert callable(self.exposure_map), "exposure_map must be a callable function of pixel indices"

    def __repr__(self):
        event_type = getattr(self.psf, 'event_type', self.psf)
        return f'Band(energy={self.energy:.1f} MeV, et={event_type} nside={self.nside})'

    def response(self, source, pixels=None):
        """Return the response, or evaluation of the PSF, for a given source and pixel set.
        """
        return source.response(self).evaluate( pixels)

    def pixel_counts(self, pixels=None):
        """Evaluate model counts on a set of pixels on the sparse set of illuminated pixels

        Parameters
        ----------
        pixels : array-like, optional
            Pixel indices to evaluate. If None, all illuminated pixels are used.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pixel indices and their corresponding model counts.
        """

        from collections import defaultdict

        # Accumulate contributions from all sources into a sparse pixel map.
        accum = defaultdict(float)
        for src in self.source_model:

            flux = src.model(self.energy)
            k, v = self.response(src, pixels)
            for pix, value in zip(k, v):
                accum[pix] += value * flux

        k = np.fromiter(accum.keys(), dtype=int)
        v = np.fromiter(accum.values(), dtype=float)
        v *= self.exposure_map(k)  # apply exposure scaling to model flux
        return k, v

    def counts(self):
        """Return predicted total counts."""
        return np.sum(self.pixel_counts()[1])
    
    def pixel_gradient(self, data):
        """Evaluate per-pixel count gradients for the currently free model parameters.

        Parameters
        ----------
        data : tuple (pixels, counts)
            Pixel indices and corresponding counts; only the pixel index array is
            used to evaluate responses.

        Returns
        -------
        g : np.ndarray
            Gradient matrix with shape `(n_selected_pixels, n_free_parameters)`.
        """

        keys, _ = data 
        g = []
        
        for src in self.source_model:
            # Restrict to currently free parameters before projecting to pixels.
            grad = src.model.gradient(self.energy)[src.model.free]
            _, v = src.response(self).evaluate(keys)
            g.append(v[:, None] * grad[None, :])
        g = np.hstack(g)  # stack before scaling to avoid list * array error
        g *= self.exposure_map(keys)[:, None]  # apply exposure scaling to gradients
        return g

    def simulate(self, random_state=None, total_counts=None,):
        """Simulate pixel counts for this band.

        If `random_state` is provided, Poisson fluctuations are applied/
        Only non-zero pixels are returned.

        Parameters
        ----------
        random_state : int or np.random.Generator, optional
            Random seed/state for reproducible Poisson sampling.
        total_counts : float, optional
            Total expected counts to distribute proportionally to model weights.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pixel indices and counts, with Poisson noise if `random_state` is provided,
            and only non-zero pixels returned.
        """
        k, counts = self.pixel_counts()

        if total_counts is not None:
            # Normalize the model shape to the requested total counts.
            counts = total_counts * counts / counts.sum()
        
        if random_state is not None:
            # Apply Poisson noise when a seed or Generator is provided.
            rng = np.random.default_rng(random_state)
            counts = rng.poisson(counts)
        else:
            counts = counts.astype(int)
        
        # return only non-zero pixels to avoid unnecessary computation in likelihood evaluation
        select = counts > 0
        return k[select], counts[select]

    def loglike(self, skydir=None):
        """Evaluate the Poisson log-likelihood for the band's stored data.

        Parameters
        ----------
        skydir : SkyCoord or None, optional
            Trial sky position to apply before evaluating the likelihood.
            When provided, this is forwarded to
            `self.source_model.setposition(skydir)` and therefore mutates the
            currently selected source model position. The position is not
            restored by this method.

        Returns
        -------
        float
            The summed Poisson log-likelihood,
            `sum(counts * log(model) - model)`, evaluated only on the pixels
            stored in `self.data`.

        Notes
        -----
        This method assumes `self.data` is a `(pixels, counts)` tuple. The
        model is evaluated sparsely on those same pixel indices via
        `pixel_counts(data_pix)`, so pixels not present in `self.data` do not
        contribute to the result.
        """

        if skydir is not None:
            self.source_model.setposition(skydir)
            
        data_pix, counts = self.data if self.data is not None else (np.nan, np.nan)

        _, model = self.pixel_counts(data_pix)

        return np.sum(counts * np.log(model) - model)

    def TSmap(self, skydir_grid):
        """Evaluate a TS map on a grid of trial sky positions.

        Parameters
        ----------
        skydir_grid : array-like of SkyCoord
            Grid of trial sky positions to evaluate the TS map on.

        Returns
        -------
        np.ndarray
            TS values evaluated at each position in `skydir_grid`.
        """
        ts_values = []
        for skydir in skydir_grid:
            ll_null = self.loglike()  # log-likelihood under null hypothesis (no source)
            ll_alt = self.loglike(skydir)  # log-likelihood under alternative hypothesis (source at skydir)
            ts = 2 * (ll_alt - ll_null)  # TS is defined as twice the log-likelihood ratio
            ts_values.append(ts)
        return np.array(ts_values)
    
    def plot_pixel_map(self, center, *, data=None, fig=None, label=None, log=True, **kwargs):
        """Plot per-pixel values for this band in a local ZEA projection.

        Parameters
        ----------
        center : tuple or SkyCoord
            Plot center in sky coordinates.
        data : tuple[np.ndarray, np.ndarray] or dict, optional
            Pixel/value data to display. If omitted, uses `self.pixel_counts()`.
        fig : matplotlib.figure.Figure, optional
            Existing figure target.
        label : str, optional
            Colorbar label.
        log : bool, optional
            If true, plot `log10` values.
        **kwargs
            Forwarded to `utilities.skymaps.ZEAfigure`.
        """
        from utilities.skymaps import ZEAfigure
        from matplotlib import colors
        
        pixmap = np.zeros(self.npix)
        if isinstance(data, dict):
            k = np.array(list(data.keys()))
            v = np.array(list(data.values()))
        else:
            k, v = data if data is not None else self.pixel_counts()
        if self.order == 'nested':
            k = self.nested_to_ring(np.asarray(k, dtype=int))
        pixmap[k] = v
        # Mask empty pixels so they do not dominate the color scale.
        pixmap[pixmap == 0] = np.nan

        # PSF width sets both field size and resolution for a compact local view.
        zkw = {
            'size': 8 * self.psf.r68,
            'pixelsize': self.psf.r68 / 50,
            'figsize': (6, 5),
            'title': '',
        }
        zkw.update(kwargs)

        zfig = ZEAfigure(center, fig=fig, **zkw)
        zfig.imshow(np.log10(pixmap) if log else pixmap, )# norm=(colors.LogNorm() if log else None) )
        zfig.colorbar(label='log10(counts)' if log else 'counts', shrink=0.9, extend='max')
   
        zfig.axes_text(0.98, 0.98, f'{self.energy / 1e3:.2f} GeV',
                color='white', ha='right', va='top', fontsize=12)
        

class BandListLocalizationView:
    """Localization view for a BandList centered on the selected source.

    Wraps a ``LocalizedSourceView`` and provides a pre-bound ``delta_ts``
    method that uses the aggregated ``BandList.loglike`` for efficiency.
    """

    def __init__(self, bandlist, source_model_view):
        """Bind to a BandList and its underlying SourceModel localization view."""
        self.bandlist = bandlist
        self.source_model_view = source_model_view
        self.source = source_model_view.source

    @property
    def skydir(self):
        """Current sky position from the underlying source model view."""
        return self.source.skydir

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped ``LocalizedSourceView``."""
        return getattr(self.source_model_view, name)

    def delta_ts(self, position=None, baseline=None):
        """Evaluate delta TS using aggregated BandList likelihood.

        Parameters
        ----------
        position : SkyCoord or None, optional
            Trial sky position. If provided, return delta TS at that position.
            If omitted, return a callable `f(position)`.
        baseline : float, optional
            Reference log-likelihood. If omitted, uses current selected-source position.

        Returns
        -------
        float or callable
            `2 * (loglike(position) - baseline)` for a single position, or callable.
        """
        return self.source_model_view.delta_ts(self.bandlist.loglike, position=position, baseline=baseline)

    def make_grid(self, func, step=0.02, n=21,):
        """Make a local grid of function values for a given position-dependent function.
        
        Parameters
        ----------
        func : callable
            Function to evaluate on the grid. Should accept a `SkyCoord` and return a scalar.
        step : float, optional
            Grid spacing in degrees. Default is 0.02.
        n : int, optional
            Number of grid points along each axis. Default is 21.

        Returns
        -------
        ra_grid, dec_grid, func_grid : ndarray
            Grids of RA, Dec, and function values.
        """

        ra0  = self.source.skydir.icrs.ra.deg
        dec0 = self.source.skydir.icrs.dec.deg

        delta = np.linspace(-n // 2 * step, n // 2 * step, n)

        ra_grid, dec_grid = np.meshgrid(ra0 + delta, dec0 + delta)

        func_grid = np.array([
            func(SkyCoord(ra, dec, unit='deg', frame='icrs')) 
            for ra, dec in zip(ra_grid.ravel(), dec_grid.ravel())
        ]).reshape(n, n)
        return ra_grid, dec_grid, func_grid

class _BandListLocalizationContext:
    """Context-manager wrapper for BandList localization views."""

    def __init__(self, bandlist, source_model_context):
        """Bind to a BandList and its source-model context manager."""
        self.bandlist = bandlist
        self.source_model_context = source_model_context

    def __enter__(self):
        """Enter the source-model context and return a BandListLocalizationView."""
        source_model_view = self.source_model_context.__enter__()
        return BandListLocalizationView(self.bandlist, source_model_view)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the source-model context, restoring selected-source position."""
        return self.source_model_context.__exit__(exc_type, exc_val, exc_tb)


class BandList(list):
    """Collection of `Band` objects sharing a single source model.

    The class provides per-band count predictions, count gradients, and simple
    simulation/demo helpers.
    """
    bins = np.logspace(2,5,13) # energy bin edges: 12 bins from 100 MeV to 100 GeV
    # PSF3 nsides defined by MK
    nsides = np.array([  16,   32,   64,  128,  256,  512,  512,  512, 1024, 2048, 2048, 2048])
 
    def __init__(self, source_model, band_info=None): 
        """Initialize a list of bands for a shared source model.

        Parameters
        ----------
        source_model : SourceModel
            Source model to compute flux and gradient for each band.
        band_info : DataFrame or None
            Table containing `energy`, `nside`, and `psf` for each band. May
            also include optional `exposure_map` and `data` columns. If
            omitted, defaults derived from `bins` and `nsides` are used.
        """

        if band_info is None:
            # Default one-band-per-energy-bin table.
            band_info = pd.DataFrame(
                dict(
                    energy=np.sqrt(self.bins[1:] * self.bins[:-1]),
                    nside=self.nsides,
                    psf=[None] * len(self.nsides),
                )
            )
  
        for bi in band_info.to_dict(orient='records'):
            self.append(
                Band(
                    bi,
                    source_model=source_model,
                    exposure_map=bi.get('exposure_map'),
                    data=bi.get('data'),
                )
            )
        self.sources = source_model
        self.parameters = source_model.parameters
        self.parameter_names = source_model.parameter_names

        # Keep a simple default exposure scaling tied to energy.
        # energies = [band.energy for band in self]
        # self.exposure_factor = np.full_like(energies, 1e13) * energies / 100

        # Active band selection; None means all bands.
        self._selected: list | None = None

        # Fit outputs from the most recent ``fit`` call.
        self.fit_info = {'correlation': None, 'errors': None}

    @staticmethod
    def _pixel_table_exposure_map(pixel_band, *, use_scalar_fallback=True):
        """Adapt PixelTable exposure metadata to Band.pixel_counts semantics."""
        exposure_map = getattr(pixel_band, 'exposure_map', None)
        exposure_values = getattr(pixel_band, 'exposure_map_values', None)
        pixel_exposure = getattr(pixel_band, 'pixel_exposure', None)

        if pixel_exposure is not None:
            return _make_sparse_pixel_lookup(pixel_band, pixel_exposure)

        if exposure_values is not None:
            arr = np.asarray(exposure_values, dtype=float).ravel()
            map_nside = int(np.sqrt(arr.size / 12.0))
            if 12 * map_nside**2 == arr.size:
                frame = str(getattr(exposure_map, 'frame', 'galactic')).lower()
                nest = bool(getattr(exposure_map, 'nest', False))
                if frame == 'galactic' and not nest and map_nside == int(pixel_band.nside):
                    return lambda pix, arr=arr: arr[np.asarray(pix, dtype=int)]

        if exposure_map is not None:
            hpx = HEALPix(nside=int(pixel_band.nside), order='ring', frame='galactic')

            def map_from_pixels(pix, exposure_map=exposure_map, hpx=hpx):
                skycoord = hpx.healpix_to_skycoord(np.asarray(pix, dtype=int))
                return np.asarray(exposure_map(skycoord), dtype=float)

            return map_from_pixels

        if use_scalar_fallback:
            exposure = getattr(pixel_band, 'exposure', None)
            if exposure is not None and np.isfinite(float(exposure)):
                return lambda pix, value=float(exposure): np.full(len(np.asarray(pix, dtype=int)), value, dtype=float)

        return None

    @classmethod
    def from_pixel_table(
        cls,
        source_model,
        pixel_table,
        *,
        psf_table_path='files/loc/psf_psf_table.pkl', #or fb_...
        use_exposure=True,
        use_data=True,
    ):
        """Build a ``BandList`` from a ``SourceModel`` and ``PixelTable`` metadata.

        Parameters
        ----------
        source_model : SourceModel
            Source model to evaluate in each band.
        pixel_table : PixelTable
            Pixel-table object providing band metadata and optional exposure.
        psf_table_path : str, optional
            Path to the serialized PSF table used by ``pylib.psf_func.PSFlist``.
        use_exposure : bool, optional
            If True, adapt pixel-table exposure products onto each ``Band``.
        use_data : bool, optional
            If True, attach each pixel-table band's sparse photon counts as
            ``Band.data`` so ``BandList.fit`` can operate directly.
        """
        from pylib.psf_func import PSFlist

        if not hasattr(pixel_table, 'meta_df'):
            raise AttributeError('pixel_table must define meta_df')

        meta_df = pixel_table.meta_df.reset_index(drop=True).copy()
        pixel_bands = list(pixel_table.values())
        if len(meta_df) != len(pixel_bands):
            raise ValueError(
                f'PixelTable metadata has {len(meta_df)} rows but pixel table contains {len(pixel_bands)} bands'
            )

        if 'event_type_code' in meta_df.columns:
            event_codes = meta_df['event_type_code'].astype(int).to_numpy()
        else:
            event_codes = np.array([_event_type_code(value) for value in meta_df['event_type']], dtype=int)

        energies = np.sqrt(meta_df['emin'].to_numpy(dtype=float) * meta_df['emax'].to_numpy(dtype=float))
        assigned_psf = [None] * len(meta_df)

        for event_code in np.unique(event_codes):
            row_indices = np.flatnonzero(event_codes == int(event_code))
            available_psfs = []
            source_event_code = int(event_code)
            for candidate_event_code in [int(event_code), 0, 1]:
                available_psfs = list(PSFlist(event_type=candidate_event_code, table_path=psf_table_path))
                if available_psfs:
                    source_event_code = candidate_event_code
                    break
            if len(available_psfs) < len(row_indices):
                # Pad with clones of the last entry sorted by energy so the
                # highest-energy bands get a reasonable (if imprecise) PSF.
                n_missing = len(row_indices) - len(available_psfs)
                psf_energies = np.array(
                    [float(getattr(p, 'energy', np.nan)) for p in available_psfs], dtype=float
                )
                last_psf = available_psfs[int(np.argmax(psf_energies)) if np.any(np.isfinite(psf_energies)) else -1]
                available_psfs.extend([copy.copy(last_psf) for _ in range(n_missing)])

            for row_index in row_indices[np.argsort(energies[row_indices])]:
                if len(available_psfs) == 1:
                    psf_index = 0
                else:
                    candidate_energies = np.array(
                        [float(getattr(psf, 'energy', np.nan)) for psf in available_psfs],
                        dtype=float,
                    )
                    if np.all(np.isfinite(candidate_energies)):
                        psf_index = int(np.argmin(np.abs(np.log(candidate_energies) - np.log(energies[row_index]))))
                    else:
                        psf_index = 0
                psf = available_psfs.pop(psf_index)
                if source_event_code != int(event_code):
                    psf = _clone_psf_with_event_type(psf, event_code)
                assigned_psf[row_index] = psf

        band_rows = []
        for row_index, (row, pixel_band) in enumerate(zip(meta_df.itertuples(index=False), pixel_bands)):
            pixel_order = getattr(pixel_band, 'order', 'ring')
            band_info: dict[str, object] = dict(
                energy=float(np.sqrt(float(row.emin) * float(row.emax))),
                nside=int(row.nside),
                psf=assigned_psf[row_index],
                order=pixel_order,
            )
            if use_exposure:
                band_info['exposure_map'] = cls._pixel_table_exposure_map(pixel_band)
            if use_data:
                band_info['data'] = (
                    np.asarray(pixel_band.pix, dtype=int),
                    np.asarray(pixel_band.photons),
                )
            band_rows.append(band_info)

        return cls(source_model, pd.DataFrame(band_rows))

    def __iter__(self):
        """Iterate over selected bands, or all bands if no selection is active."""
        if self._selected is None:
            return super().__iter__()
        return (self[i] for i in self._selected)

    def select(self, indices=None):
        """Set active band selection for all iteration-based operations.

        Parameters
        ----------
        indices : array-like of int or None
            Band indices (0-based) to include in iteration. Pass ``None`` to
            reset to all bands.

        Returns
        -------
        self : BandList
            Returns ``self`` for method chaining.

        Examples
        --------
        Select only the first four bands (low energy)::

            bandlist.select(range(4)).simulate()

        Reset to all bands::

            bandlist.select()
        """
        self._selected = None if indices is None else list(indices)
        return self

    def counts(self):
        """Return predicted total counts per band."""
        return np.array([ band.counts() for band in self])

    def loglike(self, skydir=None):
        """Return total Poisson log-likelihood summed over all bands.

        Parameters
        ----------
        skydir : SkyCoord or None, optional
            Trial sky position forwarded to each ``Band.loglike`` call.

        Returns
        -------
        float
            Sum of per-band log-likelihood values.
        """
        return float(np.sum([band.loglike(skydir=skydir) for band in self]))
            
    def count_gradient(self):
        """Return count gradient array for all free model parameters by band."""
        g = np.array([band.pixel_gradient(band.data) for band in self])
        return g[:, :, 0].T
    
    def simulate(self, random_state=42): 
        """Simulate per-band counts, optionally with Poisson fluctuations.

        Parameters
        ----------
        random_state : int or None
            Random state for reproducibility. If None, no noise is added.
        """
        for band in self:
            band.data = band.simulate(random_state=random_state)

    # def source_position_loglike(self, source_name, data=None, frame='galactic', clip=1e-30):
    #     """Return a callable Poisson log-likelihood as a function of source position.

    #     The returned function evaluates the model log-likelihood while shifting a
    #     single source to each trial position and keeping all other model elements
    #     fixed.

    #     Parameters
    #     ----------
    #     source_name : str or Source
    #         Source identifier accepted by `SourceModel.find_source`.
    #     data : sequence[tuple[np.ndarray, np.ndarray]] or None
    #         Per-band observed data as `(pixels, counts)`. If omitted, uses
    #         `band.data` for each band and requires all bands to have data set.
    #     frame : str
    #         Coordinate frame used when trial positions are given as `(lon, lat)`.
    #     clip : float
    #         Lower bound applied to model counts to avoid `log(0)`.

    #     Returns
    #     -------
    #     callable
    #         Function `f(position) -> loglike`, where `position` can be a
    #         `SkyCoord` or a 2-tuple of degrees.
    #     """
    #     src = self.sources.find_source(source_name)
    #     if src.skydir is None:
    #         raise ValueError('source_position_loglike requires a localized source with skydir')

    #     if data is None:
    #         data = [band.data for band in self]
    #     if len(data) != len(self):
    #         raise ValueError('data length must match number of bands')
    #     if any(d is None for d in data):
    #         raise ValueError('missing band data; pass data explicitly or set band.data for all bands')

    #     def to_coord(position):
    #         if isinstance(position, SkyCoord):
    #             return position
    #         if hasattr(position, '__iter__') and len(position) == 2:
    #             return SkyCoord(position[0], position[1], unit='deg', frame=frame)
    #         raise ValueError(f'unrecognized position: {position}')

    #     original_skydir = src.skydir

    #     def loglike(position):
    #         src.skydir = to_coord(position)
    #         try:
    #             total = 0.0
    #             for band, band_data in zip(self, data):
    #                 keys, counts = band_data
    #                 model = np.zeros_like(counts, dtype=float)
    #                 for source in band.source_model:
    #                     flux = source.model(band.energy)
    #                     _, response_values = source.response(band).evaluate(keys)
    #                     model += response_values * flux
    #                 model *= band.exposure_map(keys)
    #                 model = np.clip(model, clip, None)
    #                 total += np.sum(counts * np.log(model) - model)
    #             return float(total)
    #         finally:
    #             src.skydir = original_skydir

    #     return loglike
    
    def localization_view(self, source_name=None):
        """Return a localization context-manager view for the selected source.

        The returned context manager yields a ``BandListLocalizationView``
        bound to the selected source, providing a pre-bound ``delta_ts``
        method that uses the aggregated ``BandList.loglike`` for efficiency.

        Parameters
        ----------
        source_name : str, Source-like, or None
            Source identifier accepted by ``SourceModel.localization_view``.
            May be a source name string, a source object with ``name``, or
            ``None`` to use the currently selected source.

        Returns
        -------
        _BandListLocalizationContext
            Context manager that yields ``BandListLocalizationView`` on ``__enter__``.

        Usage
        -----
        .. code-block:: python

            with bandlist.localization_view('Blazar') as loc:
                delta_ts = loc.delta_ts()
                ts_value = delta_ts(trial_position)
        """
        sm_context = self.sources.localization_view(source_name)
        return _BandListLocalizationContext(self, sm_context)

    def localize(self, source_name=None, sigma=0.1, verbose=True):
        """Run localization for a source and return a ``quadform.Localize`` result.

        Parameters
        ----------
        source_name : str, Source-like, or None
            Source identifier accepted by ``SourceModel.localization_view``.
            May be a source name string, a source object with ``name``, or
            ``None`` to use the currently selected source.
        sigma : float, optional
            Initial localization uncertainty in degrees passed to
            ``quadform.Localize``.
        verbose : bool, optional
            If True, print localization diagnostics.

        Returns
        -------
        like3.quadform.Localize
            Completed localization result object.

        Notes
        -----
        This is a convenience wrapper around::

            with bandlist.localization_view(source_name) as loc:
                result = Localize(loc, sigma=sigma, verbose=verbose)
        """
        from .quadform import Localize

        with self.localization_view(source_name) as loc:
            return Localize(loc, sigma=sigma, verbose=verbose)

    def fit(self, method='l-bfgs-b', quiet=True, use_gradient=True, **kwargs):
        """Optimize the free spectral parameters of the source model.

        Minimizes the negative log-likelihood summed over the active bands
        using :class:`~like3.fitter.Minimizer`.

        Parameters
        ----------
        method : str, optional
            Optimization method forwarded to ``Minimizer.__call__``.  One of
            ``'simplex'`` (default), ``'powell'``, or ``'l-bfgs-b'``.
        quiet : bool, optional
            Suppress optimizer diagnostic output.
        use_gradient : bool, optional
            If True, use the analytic gradient of the negative log-likelihood
            when the selected optimizer supports gradients.
        **kwargs
            Additional keyword arguments forwarded to ``Minimizer.__call__``.

        Returns
        -------
        fitvalue : float
            Negative log-likelihood at the optimum.
        parameters : np.ndarray
            Best-fit free-parameter vector (in fitter space).
        errors : np.ndarray
            1-sigma uncertainties on free parameters (NaN if estimation failed).

        Side Effects
        ------------
        Stores fit diagnostics on the instance in ``self.fit_info`` with keys
        ``'correlation'`` and ``'errors'`` from the most recent fit.

        Notes
        -----
        The fit updates the source model in place via ``parameters.set_parameters``.
        The active band selection (``self._selected``) is respected — only the
        selected bands contribute to the likelihood.
        """
        from .fitter import Minimizer, Fitted

        pset = self.sources.parameters
        bandlist = self  # capture for closure
        initial_loglike = self.loglike()
        use_gradient = kwargs.pop('use_gradient', use_gradient)

        class _Objective(Fitted):
            def __init__(self):
                self._cache_pars = None
                self._cache_value = None
                self._cache_grad = None

            @property
            def bounds(self):
                return bandlist.sources.bounds

            @property
            def parameter_names(self):
                return bandlist.sources.parameter_names

            def get_parameters(self):
                return pset.get_parameters()

            def set_parameters(self, par):
                pset.set_parameters(par)

            def _evaluate(self, pars, need_grad=False):
                pars = np.asarray(pars, dtype=float)
                if (
                    self._cache_pars is not None
                    and np.array_equal(pars, self._cache_pars)
                    and (not need_grad or self._cache_grad is not None)
                ):
                    return self._cache_value, self._cache_grad

                self.set_parameters(pars)
                loglike = 0.0
                grad = np.zeros_like(pars, dtype=float) if need_grad else None

                for band in bandlist:
                    if band.data is None:
                        continue
                    data_pix, counts = band.data
                    _, model = band.pixel_counts(data_pix)
                    model = np.clip(model, 1e-30, None)
                    loglike += np.sum(counts * np.log(model) - model)

                    if need_grad:
                        dm_dtheta = band.pixel_gradient(band.data)
                        grad -= ((counts / model - 1.0)[:, None] * dm_dtheta).sum(axis=0)

                value = -float(loglike) + initial_loglike
                self._cache_pars = np.array(pars, copy=True)
                self._cache_value = value
                self._cache_grad = None if grad is None else np.array(grad, copy=True)
                return value, grad

            def __call__(self, pars, *args):
                value, _ = self._evaluate(pars, need_grad=use_gradient)
                return value

            def gradient(self, pars):
                """Return gradient of the objective at ``pars``."""
                _, grad = self._evaluate(pars, need_grad=True)
                return grad

        objective = _Objective()
        minimizer = Minimizer(objective, quiet=quiet)
        fit_out = minimizer(method=method, use_gradient=use_gradient, **kwargs)
        best_pars = np.array(fit_out[1], copy=True)
        self.fit_info = {
            'correlation': np.array(minimizer.correlations(), copy=True),
            'errors': np.array(minimizer.sigmas(), copy=True),
            'gradient': np.array(objective.gradient(best_pars), copy=True),
        }
        return fit_out

    @classmethod
    def demo(cls, model=None):
        """Build a demo `BandList` and print per-band flux/count summaries."""
        if model is None:
            model = SourceModel.demo()
        print(f'Creating BandList for model: {model}')
        band_list = cls(model)
        for band in band_list:
            print(f'{band}: counts={band.counts():.2e}')
        print('Counts per band:', band_list.counts().astype(int))
        return band_list
    
    @classmethod
    def psf_demo(cls,):
        """Build a demo `BandList` populated with PSF metadata."""
        #from pylib import psf_func as pf; reload(pf)
        from pylib.psf_func import PSFlist

        df = PSFlist.demo_df()  # get PSF functions for each band in a DataFrame
        df['nside'] = BandList.nsides
      
        model = SourceModel.demo()
        print(f'Creating BandList with PSF for model: {model}')
        band_list = cls(model, df)
        print('Counts per band:', band_list.counts().astype(int))
        return band_list