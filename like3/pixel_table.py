

"""Utilities for loading, inspecting, plotting, and exporting pixel tables.

This module provides:
- `PixelTable` and `PixelTable.Band` for reading Kerr-style FITS pixel table
    files and working with per-band HEALPix data.
- Residual visualization helpers (`ResidualPlotter`, scatter/histogram helpers).
- Simple spatial clustering for significant residual points.
"""

from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy_healpix import HEALPix
from pathlib import Path
from typing import Callable, cast

from .sky_display import _catalog_source_summary, _install_text_hover, ait_plot as sky_ait_plot, zea_plot as sky_zea_plot


def _event_type_to_int(value):
    """Normalize event-type labels/codes to the FITS integer convention."""
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
        ivalue = int(value)
        if 0 <= ivalue <= 5:
            return ivalue
    raise ValueError(f'Unsupported event type: {value!r}')


def _event_type_to_label(value):
    """Convert FITS event-type integers to band labels."""
    ivalue = _event_type_to_int(value)
    if ivalue == 0:
        return 'FRONT'
    if ivalue == 1:
        return 'BACK'
    return f'PSF{ivalue - 2}'



_energy_index = lambda energy: (np.log10(energy) * 4 - 8).astype(int)
e_mev = lambda energy_index: (10** (energy_index/4 + 0.125)*1e2).astype(int)



class PixelTableLocalizationView:
    """Localization view centered on the selected source for a pixel table."""

    def __init__(self, pixel_table, source_model_view):
        self.pixel_table = pixel_table
        self.bandlist = pixel_table  # legacy alias used by existing tests/callers
        self.source_model_view = source_model_view
        self.source = source_model_view.source

    @property
    def skydir(self):
        return self.source.skydir

    def __getattr__(self, name):
        return getattr(self.source_model_view, name)

    def delta_ts(self, position=None, baseline=None):
        return self.source_model_view.delta_ts(self.pixel_table.loglike, position=position, baseline=baseline)


class _PixelTableLocalizationContext:
    """Context-manager wrapper for pixel-table localization views."""

    def __init__(self, pixel_table, source_model_context):
        self.pixel_table = pixel_table
        self.source_model_context = source_model_context

    def __enter__(self):
        source_model_view = self.source_model_context.__enter__()
        return PixelTableLocalizationView(self.pixel_table, source_model_view)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.source_model_context.__exit__(exc_type, exc_val, exc_tb)


class PixelTable(dict):
    """Container for pixel table bands and their sparse per-pixel arrays.

    The class loads a Kerr-style FITS file and exposes each
    ``(psf_index, energy_index)`` band through dictionary access.
    """

    # e_mev = lambda energy_index: (10** (energy_index/4 + 0.125)*1e2).astype(int)

    class Band(HEALPix):
        """Single event-type/energy slice of a pixel table.

        Each band is a HEALPix view plus aligned sparse arrays for photons and
        model components (`diffuse` and `sources`).
        """

        def __init__(self, meta, order, source_model=None):
            self.psf_name, self.e0, self.e1, nside, self.nocc = meta
            self.event_type = _event_type_to_int(self.psf_name)

            # key is (psf index, energy index) tuple
            psf_index = self.event_type if self.event_type < 2 else self.event_type - 2
            self.key = (int(psf_index), int(_energy_index(self.e0)))
            self.energy = int(np.sqrt(self.e0 * self.e1))  

            self.psf_cache = {}  # populated on demand by get_psf_cache()
            super().__init__(nside, frame='galactic', order=order)
            self.psf: Callable[[float], float] | None = None
            self.r68: float | None = None
            self.source_model = source_model
            self._ensure_psf()
            # self.roi = ROI(self, source_model) if source_model is not None else None
            # Dense array caches built once from sparse pixel arrays; see exposure_map.
            self._exposure_dense: np.ndarray | None = None  # shape (12*nside^2,)
            # Backward-compatible sparse lookup cache used by existing tests/callers.
            self._exposure_lookup: dict[int, float] | None = None
            self._pix_inv: np.ndarray | None = None          # pixel_index → pos in self.pix
            # Coverage DataFrame: columns pix, photons, diffuse_counts, source_counts,
            # restricted to pixels near sources.  When set, loglike/gradient computations
            # use this subset instead of the full self.pix arrays.
            self.coverage: pd.DataFrame | None = None
            # Sparse per-pixel arrays; populated by PixelTable._setup_from_arrays.
            # Mandatory arrays are initialised empty so the type checker knows
            # they are always ndarray (never None) once set up.
            self.pix: np.ndarray = np.empty(0, dtype=np.int64)
            self.photons: np.ndarray = np.empty(0, dtype=np.int32)
            self.diffuse_counts: np.ndarray = np.empty(0, dtype=np.float32)
            self.source_counts: np.ndarray = np.empty(0, dtype=np.float32)
            # Optional arrays: only present when the FITS file contains the column.
            self.pixel_exposure: np.ndarray | None = None
            self.count_exposure: np.ndarray | None = None
            self.slice: slice | None = None
            self.totals: dict | None = None

        def _ensure_psf(self):
            """Attach a PSF object to this band when one is available from the lookup table."""
            if getattr(self, 'psf', None) is not None:
                self.r68 = getattr(self.psf, 'r68', self.r68)
                return self.psf

            try:
                from like3.psf import PSFlookup
            except Exception:
                return None

            lookup = PSFlookup()
            try:
                self.psf = lookup(self)
            except ValueError:
                self.psf = None
            self.r68 = getattr(self.psf, 'r68', self.r68)
            return self.psf

        def __repr__(self) -> str:
            return f"Band{self.key}: {self.psf_name}@{self.energy * 1e-3:.2f} GeV nside {self.nside} occ {self.nocc/(12*self.nside**2):.3f}"

        def response(self, source, pixels):
            """Return PSF response for a source evaluated on given pixel indices."""
            if source is None:
                cpix = np.asarray([], dtype=np.int64)
                return cpix, np.asarray([], dtype=float)


            source_name = source.name if hasattr(source, 'name') else str(source)
            cache = self.psf_cache
            if source_name in cache:
                return cache[source_name]
            # Compute and cache the PSF list for this source
            sdir = source.skydir
            sdir = sdir.coord if hasattr(sdir, 'coord') else sdir

            cpix = np.asarray(pixels, dtype=np.int64)
            aa = sdir.separation(self.healpix_to_skycoord(cpix)).deg
            psf = self.psf
            vpix = np.array(list(map(psf, aa)), dtype=float) * self.pixel_area.value
            cache[source_name] = vpix
            return vpix

        @property
        def exposure_map(self):
            """Return a callable mapping pixel indices to their normalized exposure.

            The returned values are the pixel exposure already scaled by
            ``_exposure_normalization()`` (i.e. weighted exposure × ΔE for the
            band's reference power law).

            The callable accepts an integer array of pixel indices and returns a
            float array via numpy fancy indexing into a precomputed dense array —
            no Python loops.
            """
            pe = self.pixel_exposure
            if pe is not None:
                if self._exposure_dense is None:
                    dense = np.zeros(12 * self.nside ** 2, dtype=float)
                    dense[self.pix] = pe
                    self._exposure_dense = dense
                    if self._exposure_lookup is None:
                        self._exposure_lookup = {
                            int(p): float(v) for p, v in zip(self.pix.tolist(), pe.tolist())
                        }
                dense = self._exposure_dense
                return lambda q: dense[np.asarray(q, dtype=np.intp)]
            return lambda q: np.zeros(len(np.asarray(q)), dtype=float)

        
        def _exposure_normalization(self):
            """factor to convert input counts per pixel for this power law to the weighted exposure times delta E."""
            return 1/(1e-14 * (np.sqrt(self.e0 * self.e1) * 1e-3) ** (-2.1))

        def _component_values(self, component):
            """Resolve a coverafw component to a full HEALPix array of per-pixel values."""
            model = self._model_counts()
            match component:
                case 'exposure':
                    if self.pixel_exposure is None:
                        raise ValueError('No pixel exposure has been attached to this band')
                    return self.pixel_exposure
                case 'resid':
                    return self.photons - model
                case 'sigma':
                    return (self.photons - model) / np.sqrt(model.clip(1e-2, None))
                case 'model':
                    return model
                case 'data':
                    return self.photons
                case 'diffuse':
                    return self.diffuse_counts
                case 'sources':
                    # When a source_model is set, derive source counts dynamically;
                    # otherwise fall back to the pre-computed FITS array.
                    if self.source_model is not None:
                        return self.pixel_counts() - self.diffuse_counts
                    return self.source_counts
                case _:
                    raise ValueError(f"Unknown component: {component!r}")
        
        def _display_values(self, param):
            """Return a full HEALPix array of per-pixel values.
            The input param can be:
            - a string naming a coverage component (e.g. 'model_counts')
            - a numeric array of length equal to the coverage length (per-pixel values)
            - a numeric array of length equal to the active pixel list length (per-active-pixel values)
            - a numeric array of length equal to the full HEALPix length (per-pixel values)
            """
            if isinstance(param, str): # name of a coverage component
                component = param
                cov = self.coverage
                if component not in cov.columns:
                    if component+'_counts' in cov.columns:
                        component = component+'_counts'
                    else:
                        raise ValueError(f"Unknown coverage component: {component!r}")    
                arr = cov[component].values

            elif isinstance(param, (list, tuple, np.ndarray)):
                arr = np.asarray(param)
                if not np.issubdtype(arr.dtype, np.number):
                    raise ValueError(f"Array contains non-numeric values: {arr}")

            else:
                raise ValueError(f"Parameter must be a component name or numeric array, got {type(param)}")

            # Expand a per-pixel array to a full HEALPix array.

            if len(arr) == 12*self.nside**2:
                return arr
            
            if len(arr) == len(self.coverage):
                hpa = np.full(12*self.nside**2, np.nan, dtype=float)
                hpa[self.coverage.pix] = arr
                return hpa
            
            if len(arr) == len(self.pix):
                hpa = np.full(12*self.nside**2, np.nan, dtype=float)
                hpa[self.pix] = arr
                return hpa
            
            raise ValueError(f"""Array length {len(arr)} does not match coverage length {len(self.coverage)}
                            or active pixel list length {len(self.pix)} or full HEALPix length {12*self.nside**2}""")


        def _pixels_in_frame(self, frame):
            """Return NESTED pixel indices transformed to the requested frame."""
            if frame == 'galactic':
                pix = self.pix
            else:
                tsky = self.skycoords.transform_to(frame)
                lon = tsky.ra if frame == 'icrs' else tsky.lon
                lat = tsky.dec if frame == 'icrs' else tsky.lat
                pix = self.lonlat_to_healpix(lon, lat)

            if self.order == 'ring':
                # converted to ring: must convert back to nested first
                return self.ring_to_nested(pix)
            return pix

        def _pixels_for_map_order(self, *, nest=False):
            """Return current sparse pixels in the requested map ordering."""
            if nest:
                return self.pix if self.order == 'nested' else self.ring_to_nested(self.pix)
            return self.pix if self.order == 'ring' else self.nested_to_ring(self.pix)

        def pix_to_ring(self, *, inplace=False):
            """Convert and optionally store pixel indices using RING ordering."""
            if inplace:
                self.pix = self.nested_to_ring(self.pix)
                self.order = 'ring'

            return self.nested_to_ring(self.pix)

        @property
        def skycoords(self):
            """Return photon pixel centers as `SkyCoord` in the band frame."""
            return self.healpix_to_skycoord(self.pix)

        def cone_search(self, center, radius=5.0):
            """Return a boolean mask for pixels within `radius` degrees of `center`."""

            sc = self.healpix_to_skycoord(self.pix)
            return sc.separation(center) < Angle(radius, 'deg')

            # slower, more inclusive list
            # cone_pix = hp.cone_search_skycoord(center, radius=Angle(radius, 'deg'))
            # return  np.in1d(self.pix, cone_pix)

        def ring_map(self, nside=None, component='data', frame='galactic'):
            """Create a HEALPix RING map of a component at specified resolution.

            Parameters
            ----------
            nside : int, optional
                Target HEALPix resolution. If None or greater than band nside,
                uses band's native nside. Must be a valid HEALPix value.
            component : str, optional
                Component to map:
                - 'data': observed photon counts
                - 'diffuse': diffuse model component
                - 'sources': point-source model component
                - 'model': combined model (diffuse + sources)
                - 'resid': residuals (data - model)
                - 'sigma': normalized residuals (resid / sqrt(model))
                Default is 'data'.
            frame : str, optional
                Astropy coordinate frame for output map. Common values:
                'galactic' (default), 'icrs', 'geocentricmeanecliptic'.

            Returns
            -------
            np.ndarray
                1D array of length 12*nside^2 in RING ordering.
                Zero values replaced with NaN for display purposes.
            """
            from astropy_healpix import HEALPix
            if component not in ['data', 'diffuse', 'sources', 'model', 'resid', 'sigma', 'exposure']:
                raise ValueError(f"Invalid component: {component!r}")
            match component:
                case 'resid':
                    values = self.photons - (self.diffuse_counts + self.source_counts)
                case 'sigma':
                    model = self.diffuse_counts + self.source_counts
                    values = (self.photons - model) / np.sqrt(model.clip(1e-2, None))
                case 'model':
                    values = self.diffuse_counts + self.source_counts
                case 'data':
                    values = self.photons
                case 'diffuse':
                    values = self.diffuse_counts
                case 'sources':
                    values = self.source_counts 
                case 'exposure':
                    if self.pixel_exposure is None:
                        raise ValueError('No pixel exposure has been attached to this band')
                    values = self.pixel_exposure
                case _:
                    values = self._display_values(component)
            nside = self.nside if nside is None or nside > self.nside else nside
            ratio = (self.nside // nside) ** 2

            pix = self._pixels_in_frame(frame)
            # Aggregate to the requested nside in NESTED space, then convert to
            # RING so map consumers can assume standard HEALPix map ordering.
            pix = HEALPix(nside=nside).nested_to_ring(pix // ratio)

            mp = np.zeros(12 * nside**2)
            np.add.at(mp, pix, values)
            return mp
        
        def _display_values(self, component):
            """Return the display values for a given component."""

            if not isinstance(component, str): 
                return component
            
            if component not in ['data', 'diffuse', 'sources', 'model', 'resid', 'sigma', 'exposure']:
                raise ValueError(f"Invalid component: {component!r}")
            match component:
                case 'resid':
                    return self.photons - (self.diffuse_counts + self.source_counts)
                case 'sigma':
                    model = self.diffuse_counts + self.source_counts
                    return (self.photons - model) / np.sqrt(model.clip(1e-2, None))
                case 'model':
                    return self.diffuse_counts + self.source_counts
                case 'data':
                    return self.photons.astype(float)
                case 'diffuse':
                    return self.diffuse_counts
                case 'sources':
                    return self.source_counts 
                case 'exposure':
                    if self.pixel_exposure is None:
                        raise ValueError('No pixel exposure has been attached to this band')
                    return self.pixel_exposure
                case _:
                    raise ValueError(f"Unhandled component {component!r} expected one of {'data', 'diffuse', 'sources', 'model', 'resid', 'sigma', 'exposure'}")

        def _expand_to_ring(self, mp):
            """  make a full ring array
            """
            assert len(mp) == len(self.pix) 

            ringpix = self.pix if self.order=='ring' else self.nested_to_ring(self.pix)
            dense = np.full(12 * self.nside**2, np.nan, dtype=float)
            dense[ringpix] = mp
            return dense
        
        def ait_plot(self, component='data', *, figsize=(12,6), fig=None, colorbar=True,
                     label='{}: counts / pixel', title=None, log=True,
                     shrink=0.7, cmap='viridis', frame='galactic',
                     value_format='auto', **kwargs):
            """Render an all-sky AIT projection for one band component.

            This delegates to the shared sky-display helper so hover behavior,
            colorbar visibility, and value formatting stay consistent with the
            rest of the project.
            """
            mp = self._expand_to_ring(self._display_values(component))
            afig = sky_ait_plot(
                mp,
                figsize=figsize,
                fig=fig,
                colorbar=colorbar,
                label=label.format(component),
                title=title,
                shrink=shrink,
                cmap=cmap,
                frame=frame,
                log=log,
                value_format=value_format,
                **kwargs,
            )
            afig.axes_text(0.02, 0.98, component, color='white', ha='left', va='top', fontsize=10)
            return afig

        def zea_plot(self, center, component='data', *, figsize=(5, 4),
                pixelsize=None, size=None, fig=None, axes_visible=True,
                cmap='viridis', colorbar=True, title=None, label='{}: counts / pixel', log=True,
                vmin=None, vmax=None, frame='galactic', value_format='auto', **kwargs):
            """Render a local ZEA projection for a single band around a center coordinate.

            Delegates to the shared sky_display helper so the plotting and hover
            behavior is consistent with the rest of the codebase.
            """
            if type(center) is str:
                center = SkyCoord.from_name(center)

            mp = self._expand_to_ring(self._display_values(component))
            zfig = sky_zea_plot(
                center,
                mp,
                psf=getattr(self, 'psf', None),
                figsize=figsize,
                r68=getattr(self, 'r68', None),
                pixelsize=pixelsize,
                size=size,
                fig=fig,
                axes_visible=axes_visible,
                cmap=cmap,
                colorbar=colorbar,
                title=title,
                label=label.format(component),
                log=log,
                vmin=vmin,
                vmax=vmax,
                frame=frame,
                source_model=getattr(self, 'source_model', None),
                value_format=value_format,
                **kwargs,
            )

            zfig.axes_text(0.98, 0.98,
                           f'{self.energy / 1e3:.2f} GeV\n{self.psf_name}',
                           color='white', ha='right', va='top', fontsize=10)
            zfig.axes_text(0.02, 0.98, component, color='white', ha='left', va='top', fontsize=10)
            return zfig
        
        def get_outliers(self, sigma_min=4):
            """Extract pixels with normalized residual exceeding a threshold.

            Parameters
            ----------
            sigma_min : float, optional
                Significance threshold in sigma units. Default is 4.

            Returns
            -------
            pd.DataFrame
                Outlier data with columns:
                    - pixel (int): NESTED pixel index
                    - data (float): observed counts
                    - model (float): model counts
                    - sigma (float): normalized residual (data-model)/sqrt(model)
            """
            
            d, m = self.ring_map(None, 'data',), self.ring_map(None, 'model')
            r = (d-m)/np.sqrt(m.clip(1e-2, None))
            out = r > sigma_min
            pix = np.arange(12*self.nside**2)
            return pd.DataFrame( dict(pixel=self.ring_to_nested(pix[out]), data=d[out], model=m[out], sigma=r[out] )) 

  
    def __init__(self, root, source_model=[], *, emin=100, psf_path='files/loc'):
        """Load a pixel table from a Kerr-style FITS file.

        Parameters
        ----------
        root : str or Path
            Path to a FITS file containing ``SKYMAP`` and ``BANDS`` HDUs.
        source_model : SourceModel 
            Source model to attach to each band. When set, ``Band._model_counts()``
            evaluates the model dynamically rather than using the pre-computed FITS
            source arrays. Mirrors the ``source_model`` parameter of ``PixelTable``.
        emin : float or None, optional
            Minimum band energy in MeV. Bands with emin below this value are
            dropped at load time. Default is 100. Pass ``None`` to load all bands.
        psf_path : str or None, optional
            Path to a PSF file for the bands. 
        """
        root = Path(root).expanduser()
        self.version=root.name.split('.')[0].split('_')[-1]
        super().__init__()
        # self.source_model = source_model
        self._selected: list | None = None
        self.fit_info: dict = {}
        self._load_from_fits(root, emin_filter=emin)
  
        # print(f"Attached source model with {len(self.source_model)} sources to pixel table")
        # self.set_psf(psf_path)
        # self.build_coverage()
        # a list of mev values for the loaded bands, used for interpolation in the source model; derived from the unique emin values in the loaded bands.
        ebins = set()
        for a,b in self: ebins.add(b)
        self.energies = e_mev(np.asarray(list(ebins)))



    def __call__(self, *pars):
        return self[pars]

    def _load_from_fits(self, filename, *, emin_filter=None):
        """Load sparse arrays and metadata from a Kerr-style FITS file."""
        filename = Path(filename).expanduser()
        self.name = filename.stem

        with fits.open(filename, memmap=True) as hdul:
            skymap = cast(fits.BinTableHDU, hdul['SKYMAP'])
            bands = cast(fits.BinTableHDU, hdul['BANDS'])
            skymap_data = skymap.data
            bands_data = bands.data
            if skymap_data is None or bands_data is None:
                raise ValueError(f'Invalid FITS table payload in {filename}')

            ordering = str(skymap.header.get('ORDERING', 'NESTED')).upper()
            self.order = 'nested' if ordering == 'NESTED' else 'ring'

            nside = np.asarray(bands_data['NSIDE'], dtype=int)
            emin = np.asarray(bands_data['E_MIN'], dtype=float) * 1e-3
            emax = np.asarray(bands_data['E_MAX'], dtype=float) * 1e-3
            event_type = np.asarray(bands_data['EVENT_TYPE'], dtype=int)
            nbands = len(nside)
            band_exposure = None
            if 'EXPOSURE' in (bands_data.names or ()): 
                band_exposure = np.asarray(bands_data['EXPOSURE'], dtype=float)

            # Read channel column first to build a combined sort+filter index.
            # This avoids holding all columns in memory while sorting.
            chn_raw = np.asarray(skymap_data['CHANNEL'], dtype=np.int64)
            if chn_raw.size == 0:
                raise ValueError(f'No rows found in SKYMAP HDU: {filename}')

            order_idx = np.argsort(chn_raw, kind='stable')
            chn_sorted = chn_raw[order_idx]
            del chn_raw

            if emin_filter is not None:
                keep = np.where(emin >= emin_filter)[0]
                chn_mask = np.isin(chn_sorted, keep)
                order_idx = order_idx[chn_mask]          # combined sort+filter index
                old_to_new = np.full(nbands, -1, dtype=np.int64)
                old_to_new[keep] = np.arange(len(keep), dtype=np.int64)
                chn = old_to_new[chn_sorted[chn_mask]]
                del chn_sorted, chn_mask, old_to_new
            else:
                keep = np.arange(nbands)
                chn = chn_sorted
                del chn_sorted

            nbands_out = len(keep)
            nocc = np.bincount(chn, minlength=nbands_out)

            # Apply the combined index to each column individually so that only
            # one raw column is live at a time, bounding peak memory usage.
            self.columns = skymap_names = skymap_data.names or ()

            _raw = np.asarray(skymap_data['PIX'], dtype=np.int64)
            pix = _raw[order_idx]; del _raw

            _raw = np.asarray(skymap_data['COUNTS'], dtype=np.int32)
            photons = _raw[order_idx]; del _raw

            pixel_exposure = None
            if 'EXPOSURE' in skymap_names:
                _raw = np.asarray(skymap_data['EXPOSURE'], dtype=float)
                pixel_exposure = _raw[order_idx]; del _raw

            fits_diffuse = None
            if 'DIFFUSE' in skymap_names:
                _raw = np.asarray(skymap_data['DIFFUSE'], dtype=np.float32)
                fits_diffuse = _raw[order_idx]; del _raw
                if 'SUNMOON' in skymap_names:
                    _raw = np.asarray(skymap_data['SUNMOON'], dtype=np.float32)
                    fits_diffuse += _raw[order_idx]; del _raw

            _raw = np.asarray(skymap_data['EXTENDEDSOURCES'], dtype=np.float32)
            fits_sources = _raw[order_idx]; del _raw

            if True: #self.version<'v5':
                _raw = np.asarray(skymap_data['POINTSOURCES'], dtype=np.float32)
                fits_sources += _raw[order_idx]; del _raw
            else:
                print('*** Need new code for point source info in v5+ FITS files')



            del order_idx

        nside = nside[keep]
        emin = emin[keep]
        emax = emax[keep]
        event_type = event_type[keep]
        if band_exposure is not None:
            band_exposure = band_exposure[keep]

        event_labels = [_event_type_to_label(et) for et in event_type]
        meta = [
            (event_labels[i], float(emin[i]), float(emax[i]), int(nside[i]), int(nocc[i]))
            for i in range(nbands_out)
        ]

        self.photons = photons
        self.pix = pix
        if pixel_exposure is not None:
            self.pixel_exposure = pixel_exposure
        # FITS SKYMAP stores counts; initialize model arrays for API compatibility.
        self.diffuse_counts = fits_diffuse.astype(np.float32) if fits_diffuse is not None else np.zeros(len(pix), dtype=np.float32)
        self.source_counts = fits_sources.astype(np.float32)


        self._setup_from_arrays(meta, source=filename)

        # After _setup_from_arrays normalizes each band's pixel_exposure, rebuild
        # the table-level pixel_exposure array so it mirrors the normalized values.
        if hasattr(self, 'pixel_exposure'):
            self.pixel_exposure = np.concatenate([
                self[k].pixel_exposure for k in sorted(self.keys())
            ])

        # Optional per-band mean exposure from BANDS HDU.
        if band_exposure is not None:
            self.meta_df['exposure'] = np.asarray(band_exposure, dtype=float)
            for i, key in enumerate(self.keys()):
                self[key].exposure = float(band_exposure[i])
        elif hasattr(self, 'pixel_exposure'):
            means = []
            for key in self.keys():
                b = self[key]
                expo = float(np.mean(b.pixel_exposure)) if b.pixel_exposure is not None else np.nan
                b.exposure = expo
                means.append(expo)
            self.meta_df['exposure'] = np.asarray(means, dtype=float)

    @property
    def e_egom_mean(self):
        """Return the geometric mean energy of each band in MeV."""
        return np.sqrt(self.meta_df.emin * self.meta_df.emax) * 1e3

    def _setup_from_arrays(self, meta, *, source):
        """Build per-band objects from flattened sparse arrays and metadata."""
        self.meta_df = pd.DataFrame(meta, columns='event_type emin emax nside nocc'.split())
        self.meta_df['occupancy'] = (self.meta_df.nocc / (12 * self.meta_df.nside**2)).round(3)

        nbands = len(meta)
        offset = 0
        for i, m in enumerate(meta):
            b = self.Band(m, self.order, source_model=getattr(self, 'source_model', None))
            self[b.key] = b
            nocc = int(m[-1])
            sl = slice(offset, offset + nocc)
            b.slice = sl
            for attr in ('diffuse_counts', 'source_counts', 'photons',  'pix', 'pixel_exposure'):
                v = getattr(self, attr)[sl]
                if attr == 'pixel_exposure': # and getattr(b, attr) is not None:
                    setattr(b, attr, v * b._exposure_normalization())
                    setattr(b, 'count_exposure', v)
                elif attr == 'diffuse_counts':
                    setattr(b, 'diffuse_counts', v)
                elif attr == 'source_counts':
                    setattr(b, 'source_counts', v)
                else:
                    setattr(b, attr, v)

            offset += nocc
            b.totals = dict(diffuse=self.diffuse_counts[-nbands + i], sources=self.source_counts[-nbands + i])

        self.meta_df['event_type_code'] = [self[key].event_type for key in self.keys()]

        if len(self.diffuse_counts) >= offset + nbands and len(self.source_counts) >= offset + nbands:
            self.totals = dict(diffuse=self.diffuse_counts[offset:], sources=self.source_counts[offset:])
        else:
            self.totals = dict(
                diffuse=np.array([self.diffuse_counts.sum()], dtype=float),
                sources=np.array([self.source_counts.sum()], dtype=float),
            )

        keys = sorted(self.keys())
        if keys:
            self.band_summary = f"{self[keys[0]]} ... {self[keys[-1]]}"
        else:
            self.band_summary = "no bands"

        # Map energy_index -> sorted list of (psf_index, energy_index) keys.
        bands_by_energy: dict[int, list] = {}
        for key in keys:
            e_idx = int(key[1])
            bands_by_energy.setdefault(e_idx, []).append(key)
        self.bands_by_energy = {int(e): sorted(ks) for e, ks in sorted(bands_by_energy.items())}

        print(f"""Loaded pixel table from "{source}":
            {len(self)} bands {self.band_summary}
            {self.photons.sum().astype(int):,d} photons
            {len(self.pix):,d} pixels, order {self.order}
            """)

    @staticmethod
    def _frame_name(frame):
        name = getattr(frame, 'name', frame)
        return str(name).lower()

    def set_psf(self, table_path='files/loc'):
        """Assign PSF functor objects to each band from a PSF table.

        Loads :class:`~pylib.psf_func.PSFlist` entries from *table_path* and
        matches each band to the nearest-energy PSF for its event type.
        When *table_path* is a directory, ``fb_psf_table.pkl`` is used for
        FRONT/BACK bands (event types 0-1) and ``psf_psf_table.pkl`` for
        PSF-partition bands (event types 2-5).  If an event type is absent
        from the tables, the FRONT (event type 0) shapes are used as a
        fallback.

        Parameters
        ----------
        table_path : str or Path, optional
            Directory containing ``fb_psf_table.pkl`` and
            ``psf_psf_table.pkl``, or a direct path to a single pickle file.
            Default is ``'files/loc'``.

        Returns
        -------
        PixelTable
            Returns *self* for method chaining.
        """
        from pylib.psf_func import PSFlist
        import copy

        all_psfs = PSFlist(event_type=None, table_path=table_path)
        if not all_psfs:
            print(f'set_psf: no PSF entries loaded from {table_path!r}')
            return self

        et_names = PSFlist.PSF.et_name
        ets = sorted({p.event_type for p in all_psfs})
        et_labels = [et_names[e] if e < len(et_names) else str(e) for e in ets]
        print(f'set_psf: {len(all_psfs)} PSF entries '
              f'({", ".join(et_labels)}) from {table_path!r}')

        # Build event-type → sorted list-of-PSF lookup.
        psf_by_et: dict[int, list] = {}
        for psf in all_psfs:
            psf_by_et.setdefault(psf.event_type, []).append(psf)

        for band in self.values():
            et = band.event_type
            fallback = et not in psf_by_et
            lookup_et = et if not fallback else 0
            plist = psf_by_et[lookup_et]
            emean = np.sqrt(band.e0 * band.e1)  # MeV
            energies = np.array([p.energy for p in plist])
            psf = plist[int(np.argmin(np.abs(energies - emean)))]
            if fallback:
                psf = copy.copy(psf)
                psf['event_type'] = et
                psf.__dict__['event_type'] = et
            band.psf = psf

        return self

    @classmethod
    def from_fits(cls, filename, *, emin=None, source_model=None, set_psf=False):
        """Load a :class:`PixelTable` from a FITS file."""
        return cls(root=filename, emin=emin, source_model=source_model, set_psf=set_psf)

    def to_fits(self, filename):
        """Write this pixel table to a FITS file compatible with ``from_fits``."""
        from astropy.io import fits

        nocc = np.asarray(self.meta_df.nocc, dtype=np.int64)
        channels = np.repeat(np.arange(len(nocc), dtype=np.int64), nocc)
        npix = len(self.pix)
        diffuse_pix = np.asarray(self.diffuse_counts[:npix], dtype=np.float32)
        sources_pix = np.asarray(self.source_counts[:npix], dtype=np.float32)

        skymap_cols = [
            fits.Column(name='PIX', format='J', array=np.asarray(self.pix, dtype=np.int64)),
            fits.Column(name='CHANNEL', format='J', array=channels),
            fits.Column(name='COUNTS', format='J', array=np.asarray(self.photons, dtype=np.int32)),
        ]

        if hasattr(self, 'pixel_exposure'):
            # De-normalize before writing: store pixel_exposure / _exposure_normalization
            # so that _setup_from_arrays correctly restores physical values on reload.
            raw_exposure = np.concatenate([
                np.asarray(self[k].pixel_exposure, dtype=np.float64) / self[k]._exposure_normalization()
                for k in sorted(self.keys())
            ]).astype(np.float32)
            skymap_cols.append(
                fits.Column(name='EXPOSURE', format='E', array=raw_exposure)
            )

        # Keep diffuse/source component columns expected by from_fits loader.
        skymap_cols.extend([
            fits.Column(name='DIFFUSE', format='E', array=diffuse_pix),
            fits.Column(name='SUNMOON', format='E', array=np.zeros(npix, dtype=np.float32)),
            fits.Column(name='POINTSOURCES', format='E', array=sources_pix),
            fits.Column(name='EXTENDEDSOURCES', format='E', array=np.zeros(npix, dtype=np.float32)),
        ])
        skymap_hdu = fits.BinTableHDU.from_columns(skymap_cols, name='SKYMAP')
        skymap_hdu.header['ORDERING'] = 'NESTED' if str(self.order).lower() == 'nested' else 'RING'

        event_codes = (
            np.asarray(self.meta_df.event_type_code, dtype=np.int64)
            if 'event_type_code' in self.meta_df.columns
            else np.asarray([_event_type_to_int(v) for v in self.meta_df.event_type], dtype=np.int64)
        )
        bands_cols = [
            fits.Column(name='NSIDE', format='J', array=np.asarray(self.meta_df.nside, dtype=np.int64)),
            fits.Column(name='E_MIN', format='D', array=np.asarray(self.meta_df.emin, dtype=float) * 1e3),
            fits.Column(name='E_MAX', format='D', array=np.asarray(self.meta_df.emax, dtype=float) * 1e3),
            fits.Column(name='EVENT_TYPE', format='J', array=event_codes),
        ]
        if 'exposure' in self.meta_df.columns:
            bands_cols.append(
                fits.Column(name='EXPOSURE', format='D', array=np.asarray(self.meta_df.exposure, dtype=float))
            )
        bands_hdu = fits.BinTableHDU.from_columns(bands_cols, name='BANDS')

        fits.HDUList([fits.PrimaryHDU(), skymap_hdu, bands_hdu]).writeto(filename, overwrite=True)

    def attach_exposure(self, exposure_by_band, *, frame='galactic', nest=False):
        """Attach per-band exposure values/maps and update derived exposure metadata."""
        full = np.zeros(len(self.pix), dtype=float)
        means = []
        rows = []

        for i, key in enumerate(self.keys()):
            band = self[key]
            spec = exposure_by_band[key]

            if np.isscalar(spec):
                px = np.full(len(band.pix), float(spec), dtype=float)
            elif callable(spec):
                # ExposureMap-like callable over pixel indices.
                if hasattr(spec, 'map_values') or hasattr(spec, 'values'):
                    vals = np.asarray(getattr(spec, 'map_values', getattr(spec, 'values')), dtype=float)
                    band.exposure_map_values = vals
                    px = vals[np.asarray(band.pix, dtype=np.intp)]
                else:
                    px = np.asarray(spec(band.pix), dtype=float)
            else:
                arr = np.asarray(spec, dtype=float)
                if arr.ndim != 1:
                    raise ValueError(f'Exposure entry for band {key!r} must be 1D')
                if len(arr) == len(band.pix):
                    px = arr
                elif len(arr) == 12 * int(band.nside) ** 2:
                    band.exposure_map_values = arr
                    px = arr[np.asarray(band.pix, dtype=np.intp)]
                else:
                    raise ValueError(
                        f'Exposure entry for band {key!r} has length {len(arr)}, '
                        f'expected {len(band.pix)} or {12 * int(band.nside) ** 2}'
                    )

            band.pixel_exposure = np.asarray(px, dtype=float)
            band.count_exposure = np.asarray(px, dtype=float)
            band._exposure_dense = None
            band._exposure_lookup = None
            band.exposure = float(np.mean(px)) if len(px) else np.nan

            sl = band.slice
            if sl is not None:
                full[sl] = px
            means.append(band.exposure)
            rows.extend((key, int(p), float(e)) for p, e in zip(band.pix, px))

        self.pixel_exposure = full
        self.meta_df['exposure'] = np.asarray(means, dtype=float)
        self.pixel_exposure_df = pd.DataFrame(rows, columns=['band_key', 'pix', 'pixel_exposure'])
        return self

    def ring_map(self, nside=128, component='data', frame='galactic'):
        """Combine all bands with nside greater than or equal to the target nside into one HEALPix RING map.

        Parameters
        ----------
        nside : int, optional
            Target HEALPix resolution.
        component : str, optional
            Component name accepted by `PixelTable.Band.ring_map`.
        frame : str, optional
            Output sky coordinate frame.
        """
        hmap = np.zeros(12*nside**2)
        for band in self.values():
            if band.nside>=nside:
                hmap += band.ring_map(nside, component, frame=frame)
        return hmap
    
    def ait_plot(self, component='data', *, nside=128, figsize=(12,6), fig=None, colorbar=True, title='',
                log=True, shrink=0.7, vmin=None, vmax=None, cmap='viridis', frame='galactic', **kwargs):
        """Render an all-sky AIT projection aggregated across bands."""
        from utilities.skymaps import AITfigure

        mp = self.ring_map(nside, component=component, frame=frame)
        if log: mp[mp==0] = np.nan
        
        afig = AITfigure(fig=fig, figsize=figsize, title=title )

        afig.imshow(mp, log=log,  cmap=cmap, **kwargs)
        if colorbar:
            afig.colorbar(label=f'{component}: counts / nside {nside} pixel', shrink=shrink)
        return afig

    
    def zea_plot(self, center=None, *, component='data', nside=256, 
                figsize=(8,8), size=5, pixelsize=0.1, fig=None, log=True,
                frame='icrs', proj='ZEA', cmap='viridis', 
                colorbar=True, title=None,**kwargs):
        """Render a local ZEA projection aggregated across bands."""
        from utilities.skymaps import ZEAfigure

        if center is None:
            sm = getattr(self, 'source_model', None)
            selected = None if sm is None else getattr(sm, 'selected_source', None)
            if selected is None:
                raise ValueError(
                    "zea_plot center is None and no selected source is available; "
                    "pass center explicitly or select a source first"
                )
            center = selected.skydir

        mp = self.ring_map(nside, component=component, frame=frame)
        mp[mp==0] = np.nan

        zfig = ZEAfigure(center, size=size, fig=fig, proj=proj,figsize=figsize, title=title, frame=frame)
        zfig.imshow(mp, log=log, cmap=cmap, **kwargs)
        if colorbar:
            ## NOTE: this is not compatible with a following call to colorbar
            zfig.colorbar(label='log10(counts)', shrink=0.7)
        return zfig


    def _iter_bands(self):
        """Iterate over selected bands, or all bands when no selection is active."""
        if self._selected is None:
            return self.values()
        return (self[k] for k in self._selected)

    def select(self, keys=None, *, psf=None, emin=None, emax=None, energy=None):
        """Set active band selection for ``loglike``, ``simulate``, and ``fit``.

        Bands can be specified directly by key, or filtered by PSF type and/or
        energy range.  All active keyword filters are ANDed together.  Call
        with no arguments to reset to all bands.

        Parameters
        ----------
        keys : iterable of band keys or None
            Explicit band keys (2-tuples ``(psf_index, energy_index)``) to
            include.  When provided, keyword filters are ignored.
        psf : str, int, or list of str/int, optional
            PSF/event type(s) to keep.  Accepts label strings (``'FRONT'``,
            ``'BACK'``, ``'PSF0'`` … ``'PSF3'``) or integer event-type codes.
            Ignored when *keys* is given.
        emin : float or None, optional
            Lower energy bound in MeV (inclusive on the band's low edge ``e0``).
            Ignored when *keys* is given.
        emax : float or None, optional
            Upper energy bound in MeV (exclusive on the band's low edge ``e0``).
            Bands with ``e0 >= emax`` are excluded.
            Ignored when *keys* is given.
        energy : float or None, optional
            Select all bands whose energy interval ``[e0, e1)`` contains the
            given energy in MeV, i.e. ``b.e0 <= energy < b.e1``.
            Can be combined with *psf*.  Ignored when *keys* is given.

        Returns
        -------
        self : PixelTable
            Returns *self* for method chaining.

        Examples
        --------
        Reset to all bands::

            pt.select()

        Select PSF2 and PSF3 bands above 1 GeV::

            pt.select(psf=['PSF2', 'PSF3'], emin=1000)

        Select by explicit keys::

            pt.select(keys=[(2, 4), (2, 5)])

        Select all bands containing 1742 MeV::

            pt.select(energy=1742)
        """
        if keys is not None:
            # Accept a single band key (2-tuple) or an iterable of keys.
            if isinstance(keys, tuple):
                keys = [keys]
            self._selected = list(keys)
            return self
        if psf is None and emin is None and emax is None and energy is None:
            self._selected = None
            return self

        if psf is not None:
            if isinstance(psf, (str, int, np.integer)):
                psf = [psf]
            allowed_et = {_event_type_to_int(p) for p in psf}
        else:
            allowed_et = None

        selected = []
        for k, b in self.items():
            if allowed_et is not None and b.event_type not in allowed_et:
                continue
            if emin is not None and b.e0 < emin:
                continue
            if emax is not None and b.e0 >= emax:
                continue
            if energy is not None and not (b.e0 <= energy < b.e1):
                continue
            selected.append(k)
        if not selected:
            print('select: no bands match the given filters')
        self._selected = selected
        return self

    def show_selected(self, *, max_rows=20):
        """Print a compact table describing the active band selection.

        Parameters
        ----------
        max_rows : int, optional
            Maximum number of rows to print from the selected set.  When more
            rows are present, only the first ``max_rows`` entries are shown and
            a truncation message is printed.  Default is 20.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per active band in display order.
        """
        if max_rows < 1:
            raise ValueError('max_rows must be >= 1')

        if self._selected is None:
            keys = sorted(self.keys())
            scope = 'all bands (no active selection)'
        else:
            keys = list(self._selected)
            scope = 'active selection'

        if not keys:
            print('show_selected: no bands are currently selected')
            return pd.DataFrame(
                columns=['key', #'event_type', 
                         'label', 'emin_mev', 'emax_mev', 'energy_mev', 'nside', 'nocc']
            )

        rows = []
        for key in keys:
            band = self[key]
            rows.append(
                dict(
                    key=key,
                    # event_type=int(band.event_type),
                    label=_event_type_to_label(band.event_type),
                    emin_mev=int(band.e0),
                    emax_mev=int(band.e1),
                    energy_mev=int(band.energy),
                    nside=int(band.nside),
                    nocc=int(band.nocc),
                )
            )

        df = pd.DataFrame(rows)
        shown = df.head(max_rows)
        print(f'show_selected: {len(df)} band(s), scope={scope}')
        print(shown.to_string(index=False))
        if len(df) > len(shown):
            print(f'... truncated to first {len(shown)} rows (set max_rows to show more)')
        return df

    def make_bands_dataframe(self) -> pd.DataFrame:
        """ 
        Create a pandas DataFrame from the pixel table.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per band, containing columns for nside, mean exposure,
            energy, and deltaE.
        """
        dd = dict((key, dict(nside=pt.nside, mean=float(pt.pixel_exposure.mean()), 
                            energy=pt.energy,
                            deltaE=round(pt.e1-pt.e0,1))) 
                    for key, pt in self.items()); 
        
        return pd.DataFrame.from_dict(dd, orient='index')
          

    def declination_exposure_plots(self, band_keys: list, ncols=4) -> None: 
        """
        Generate declination exposure plots for the specified bands.

        Parameters
        ----------
        band_keys : list
            List of band keys to plot.
        ncols : int, optional
            Number of columns in the subplot grid. Default is 4.

        """
        import pandas as pd
        nrows = (len(band_keys) + ncols - 1) // ncols
        fig, axx = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,3), 
                                sharex=True, sharey=True, constrained_layout=True)
        for ax, key in zip(axx.flat, band_keys):

            band = self[key]
            dirs = band.healpix_to_skycoord(band.pix).icrs
            edf = pd.DataFrame.from_dict(dict(ra=dirs.ra.deg, dec=dirs.dec.deg, 
                                rel_exposure=band.pixel_exposure/(band.pixel_exposure.mean())
                                ))
            ax.scatter(np.sin(np.radians(edf.dec)), edf.rel_exposure, s=1)
            ax.set(#xlabel="sin(Declination) (deg)",
                ylabel="Relative Exposure",
                title= f"Band {band.key}")
            ax.axhline(1, color='r', linestyle='--')
            ax.set(ylim=(0.5,1.8), xlim=(-1,1))
            ax.grid('0.3', alpha=0.5)
        fig.text(0.5, -0.04, "sin(Declination)", ha='center')
        fig.suptitle("Declination Exposure dependence", fontsize=16)
        plt.show()
    
def multi_ait(pixel_table, et, component='diffuse'):
    """Generate a 3x4 panel of band-level AIT plots for one event-type prefix.

    Parameters
    ----------
    pixel_table : PixelTable
        Pixel table object holding band entries.
    et : str or int
        Event type selector. Accepts integer PSF index (0-3), strings like
        '0', 'PSF0', 'psf3', or legacy labels ending with a digit.
    component : str, optional
        Component name to visualize ('diffuse', 'sources', 'data', etc.).
        Defaults to 'diffuse'.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing 3x4 AIT projections (one per band with energy labels).

    Notes
    -----
    Creates a grid with 3 rows (event types) and 4 columns (energy bins),
    displaying AIT projections for each band with automatic nside=128 resolution.
    """
    if isinstance(et, str):
        s = et.strip().upper()
        if s.startswith('PSF'):
            psf_idx = int(s[3:])
        elif s.isdigit():
            psf_idx = int(s)
        else:
            psf_idx = int(s[-1])
    else:
        psf_idx = int(et)

    band_keys = sorted([k for k in pixel_table.keys() if int(k[0]) == psf_idx], key=lambda k: k[1])

    fig = plt.figure(layout='constrained', figsize=(13,5))
    subfigs = fig.subfigures(3,4, wspace=0.07)

    for sfig, key in zip(subfigs.flat, band_keys):
        if key not in pixel_table:
            continue
        b = pixel_table[key]
        ait = b.ait_plot( component, nside=128, fig=sfig, colorbar=False)
        ait.title(str(b), fontsize=10)

    return fig

def residual_scatter(model, norm, ax=None, ylim=np.array([-5,5])):
    """Plot normalized residuals against model counts per pixel.

    The x-axis is shown in log10(model-count) space, with tick labels rendered
    as powers of ten for readability. A binned mean and standard deviation are
    overlaid on top of the raw point cloud.

    Parameters
    ----------
    model : np.ndarray
        Model counts per pixel (linear scale).
    norm : np.ndarray
        Normalized residual values (sigma units).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, creates a new figure/axis.
    ylim : np.ndarray, optional
        Y-axis range (in sigma units). Default is [-5, 5].

    Returns
    -------
    matplotlib.axes.Axes
        The plot axis.

    Notes
    -----
    - X-axis displays log10(model) with scientific notation tick labels
    - Binned statistics (mean ± std) overlaid in yellow
    - Raw points clipped and plotted with light transparency
    """
    x = np.log10(model)
    y = norm
    xmax = x.max()
    bins = np.arange(x.min(),xmax,0.5)
    if np.histogram(x, bins=bins)[0][-1]<10:
        bins = bins[:-1]
        xmax -= 0.5

    _, ax = plt.subplots(figsize=(8,4)) if ax is None else (ax.figure, ax)

    bstat = BinnedStat(x, y, bins, )
    ax.axhline(0, color='0.5', ls='--', lw=2)
    ax.errorbar(x=bstat.x, y= bstat.mean, 
                xerr= bstat.xerr, yerr=bstat.std,#/np.sqrt(bstat.count), 
                fmt='o', ms=10, label='binned mean', color='yellow');
    
    ax.scatter(x, y.clip(*ylim),  s=5, alpha=0.3 ,color='0.5')

    ticks = np.arange(int(x.min() + 1), int(xmax) + 1)
    ax.set(xlabel='model counts/pixel', ylabel=r'residual ($\sigma$ units)', xscale='linear',
           ylim=ylim, yscale='linear',
           xticks=ticks, xticklabels=[f'$10^{{{int(t)}}}$' for t in ticks], xlim=(x.min(), xmax))


class ResidualPlotter:
    """Compute and visualize per-band residual diagnostics."""

    def __init__(self, band, nside=64, clipto=(-5,5)):
        """Precompute residual, model, and normalized residual maps.

        Parameters
        ----------
        band : PixelTable.Band
            Band to analyze.
        nside : int, optional
            Target resolution for map degradation. Default is 64.
            Uses minimum of requested nside and band's native nside.
        clipto : tuple of float, optional
            Min/max values to clip normalized residuals to. Default is (-5, 5).


        Attributes Set
        ---------------
        band : Band
            Reference to input band.
        nside : int
            Effective resolution (min of input and band nside).
        photons : np.ndarray
            Observed photon counts map (RING, length 12*nside^2).
        model : np.ndarray
            Model counts map (RING).
        resid : np.ndarray
            Residual map: photons - model (RING).
        rnorm : np.ndarray
            Normalized residuals: resid / sqrt(model) (RING).
        """
        self.nside = min(nside, band.nside) if nside is not None else band.nside
        self.resid = band.ring_map(component='resid', nside=self.nside) 
    
        self.model = band.ring_map(component='model', nside=self.nside)
        # clean up zeros in model to avoid div by zero
        self.model[self.model==0] = np.min(self.model[self.model>0])
        self.rnorm = (self.resid/np.sqrt(self.model))
        if clipto is not None:
            self.rnorm = np.clip(self.rnorm, *clipto)
        self.photons = band.ring_map(component='data', nside=self.nside)
        self.band = band

    def residual_adjustment(self, ylim=np.array([-10,10]), ax=None):
        """Fit a quadratic trend to percent residuals versus model level.

        Computes a polynomial correction to the model based on residual bias
        as a function of model intensity. Stores coefficients and adjusted model
        for later use.

        Parameters
        ----------
        ylim : np.ndarray, optional
            Y-range for diagnostic scatter plot, in percent. Default is [-10, 10].
        ax : matplotlib.axes.Axes, optional
            Axis to draw diagnostic plot on. If None, computes fit but does not
            plot. Default is None.

        Attributes Set
        ---------------
        coefficients : np.ndarray
            3 polynomial coefficients [a, b, c] for fit: y = a*x^2 + b*x + c
            where x = log10(model) and y = percent residual.
        adjusted_model : np.ndarray
            Bias-corrected model counts using the fitted polynomial.
        """
        rpct = 100*(self.photons/self.model -1)
        # Fit in log-count space to capture broad normalization drift.
        self.coefficients = np.polyfit(np.log10(self.model), rpct, 2)
        poly_fit = np.poly1d(self.coefficients)
        self.adjusted_model = self.model*(1+poly_fit(np.log10(self.model))/100)

        if ax is not None:
            ax.scatter( self.model, rpct.clip(*ylim),  s=15, alpha=0.5 ,color='0.5')
            ax.axhline(0, color='0.5', ls='--', lw=2)
            ax.set(xlabel='model counts/pixel', ylabel=r'residual (%)', xscale='log', 
                ylim = ylim, yscale='linear') 
            ax.plot((x:=(self.model.min(), self.model.max())),poly_fit(np.log10(x)),
                     color='red', lw=2, label='linear fit')
            ax.set_title('Percent residuals with polynomial fit')
            # ax.legend()

    def residual_hist(self, ax=None, rnorm=None, ylim=np.array([-5,5]), legend_fontsize=14):
        """Plot residual histogram with Gaussian fit overlay.

        Renders a normalized histogram of residual values with kernel density
        estimation and overlaid Gaussian probability density function showing
        fitted mean and standard deviation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. Creates new figure/axis if None. Default is None.
        rnorm : np.ndarray, optional
            Residual values (sigma units) to histogram. Uses self.rnorm if None.
        ylim : np.ndarray, optional
            Histogram x-range (sigma units). Default is [-5, 5].
        legend_fontsize : int, optional
            Font size for fitted parameters legend. Default is 14.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axis.
        """
        from scipy.stats import norm
        from matplotlib.colors import LogNorm

        fig, ax = plt.subplots(figsize=(4,3)) if ax is None else (ax.figure, ax)
    
        if rnorm is None:
            rnorm = self.rnorm

        nfit = norm.fit(rnorm[~np.isnan(rnorm)])
        ax.hist(rnorm.clip(*ylim), bins=25, range=(float(ylim[0]), float(ylim[1])), density=True,
                histtype='stepfilled', alpha=0.5,)
        ax.plot((x:=np.linspace(*ylim,num=25)), norm.pdf(x, *nfit), 'r-', lw=4,
            label =rf'$\mu$={nfit[0]:.2f}'+'\n'+ rf'$\sigma$={nfit[1]:.2f}')
        ax.legend(fontsize=legend_fontsize, loc='lower center')
        ax.set(xlabel=r'residual ($\sigma$ units)', ylabel='density', 
               yscale='log',xlim=ylim, ylim=(1e-4, 0.5))

    def plots(self):
        """Render a standard 4-panel diagnostic dashboard.

        Displays:
        1. All-sky AIT projection of observed photons
        2. All-sky AIT projection of normalized residuals (cooled with coolwarm cmap)
        3. Scatter plot of normalized residuals vs log10(model) with binned statistics
        4. Normalized residual histogram with Gaussian fit overlay

        Notes
        -----
        This is a convenience method for quick per-band diagnostics.
        For publication-quality plots, consider using individual plot methods.
        """

        from utilities.skymaps import AITfigure
        band = self.band

        fig = plt.figure(layout='constrained', figsize=(15,5))
        fig.suptitle(str(self.band), fontsize=18)
        fig1,fig2 = fig.subfigures(ncols=2, wspace=0.07)

        (AITfigure(fig=fig1, )
            .imshow(self.photons, log=True, cmap='viridis') #nside=self.nside, 
            .colorbar(label='photon counts', shrink=0.5)
            .title( 'photons\n'+f'nside {self.nside}', x=0, y = 0.9,ha='left', fontsize=16)
        )
         
        # ap = self.band.ait_plot(component='data', fig=fig1,) #nside=self.nside, 
        # ap.title( 'photons\n'+f'nside {self.nside}', x=0, y = 0.9,ha='left', fontsize=16)
        # )
        resid = self.resid 
        model = self.model
    
        afig = AITfigure(fig=fig2, )
        afig.imshow( resid/np.sqrt(model), 
                    cmap='coolwarm',  vmin=-2, vmax=2)#**kwargs)
        afig.colorbar(label='normalized residual', shrink=0.5)
        afig.title( f'residuals',x=0, ha='left', fontsize=16)
        plt.show()
    
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,4), gridspec_kw={'width_ratios': [2.5, 1]})
        ylim=np.array([-5,5])
        residual_scatter(self.model, self.rnorm, ax=ax1, ylim=ylim)

        self.residual_hist(ax=ax2, ylim=ylim)
        plt.show()


def multi_residual_plotter(pixel_table, nside=64):
    """Plot residual histograms in a PSF x energy grid.

    Parameters
    ----------
    pixel_table : PixelTable
        Pixel table containing bands to plot.
    nside : int, optional
        HEALPix resolution for residual map degradation. Default is 64.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with 4 rows (PSF0-3) × 9 columns (energy bins),
        displaying residual histograms with Gaussian fit overlays.

    Notes
    -----
    - Top row contains energy bin labels (e.g., '1.33 GeV')
    - Left column contains PSF labels (PSF0, PSF1, PSF2, PSF3)
    - Each cell shows normalized residual distribution with μ and σ from fit
    - Missing or invalid bands are hidden
    """
    fig, axx = plt.subplots(
        5, 9, figsize=(15, 6), sharex=True, sharey=True,
        gridspec_kw={'hspace': 0.1, 'wspace': 0,
                     'height_ratios': [0.1, 1, 1, 1, 1],
                     'width_ratios': [0.5, 1, 1, 1, 1, 1, 1, 1, 1]})

    axx[0, 0].axis('off')

    # Energy labels across the top row
    for energy_idx, ax in enumerate(axx[0, 1:]):
        ax.axis('off')
        ax.text(0.5, 0.5, pixel_table(3, energy_idx).energy,
                transform=ax.transAxes, fontsize=18, ha='center', va='center')

    # PSF label column and histogram grid
    for psf_idx, row in enumerate(axx[1:]):
        row[0].text(0.5, 0.5, pixel_table(psf_idx, 7).psf_name.upper(),
                    transform=row[0].transAxes, fontsize=18, ha='center', va='center')
        row[0].axis('off')
        for energy_idx, ax in enumerate(row[1:]):
            try:
                band = pixel_table(psf_idx, energy_idx)
            except KeyError:
                ax.set_visible(False)
                continue
            if band.key[1] < 0:
                ax.set_visible(False)
                continue
            ResidualPlotter(band, nside=nside).residual_hist(ax=ax, legend_fontsize=10)
            ax.set(ylabel='', xlabel='', yticks=[])

    axx[-1, -1].set(ylim=(1e-4, 0.5))
    plt.show()

class BinnedStat:
    """Compute per-bin summary statistics for profile-style plots.

    For ROOT-like profile plot.
    Example:
    bstat = BinnedStat(x,y,bins)

    plt.errorbar(x=bstat.x, y= bstat.mean, 
             xerr= bstat.xerr,yerr=bstat.std/np.sqrt(bstat.count), 
             fmt='o', label='binned mean', color='yellow')
    """
    def __init__(self, x, y, bins):
        """Precompute mean/std/count summaries for each user-supplied bin."""
        from scipy.stats import binned_statistic
        results = {s: binned_statistic(x, y, statistic=s, bins=bins)
                   for s in ('mean', 'std', 'count')}
        self.mean  = results['mean'].statistic
        self.std   = results['std'].statistic
        self.count = results['count'].statistic
        edges = results['mean'].bin_edges
        self.x    = 0.5 * (edges[:-1] + edges[1:])
        self.xerr = 0.5 * (edges[1:] - edges[:-1])
        self.bins = bins


def grouper(points, radius):
    """Group SkyCoord points into connected clusters via separation threshold.

    Uses depth-first traversal to find connected components where two points
    are neighbors if separated by <= radius degrees.

    Parameters
    ----------
    points : astropy.coordinates.SkyCoord
        Array of sky coordinates (length N).
    radius : float
        Maximum pairwise separation in degrees for graph connectivity.
        Must be > 0.

    Returns
    -------
    list[np.ndarray]
        List of clusters, each containing indices of grouped points.
        Cluster membership is returned as integer arrays into the original
        points array.

    Raises
    ------
    ValueError
        If radius <= 0.

    Notes
    -----
    Algorithm:
    1. Maintain unvisited set of all point indices
    2. Starting from unvisited seed, DFS to find all connected neighbors
    3. Repeat until all points visited
    4. Return list of clusters

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> coords = SkyCoord([0, 1, 5, 6], [0, 1, 0, 1], unit='deg')
    >>> clusters = grouper(coords, radius=1.5)
    >>> clusters
    [array([0, 1]), array([2, 3])]
    """

    if radius <= 0:
        raise ValueError("radius must be > 0")

    n_points = len(points)
    if n_points == 0:
        return []

    unvisited = np.ones(n_points, dtype=bool)
    clusters = []

    # Build connected components where edges join points within `radius`.
    for seed in range(n_points):
        if not unvisited[seed]:
            continue

        stack = [seed]
        unvisited[seed] = False
        cluster = []

        while stack:
            i = stack.pop()
            cluster.append(i)

            remaining = np.flatnonzero(unvisited)
            if len(remaining) == 0:
                continue

            dists = points[i].separation(points[remaining]).deg 
            
            neighbors = remaining[dists <= radius]
            if len(neighbors):
                unvisited[neighbors] = False
                stack.extend(neighbors.tolist())

        clusters.append(np.array(cluster, dtype=int))

    return clusters

def plot_residuals_for_given_energy(pixel_table, energy_index):
    """Scatter residuals vs model counts for one energy bin across all PSFs.

    Parameters
    ----------
    pixel_table : PixelTable
        Pixel table to analyze.
    energy_index : int
        Energy bin index (0–11 typical).

    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2×2 grid (one panel per PSF type) showing photons vs model
        scatter plots with normalized residual on y-axis and model counts (log scale)
        on x-axis.
    """
    def mdplot(band,ax=None):
        """Render one PSF panel for the selected energy slice."""
        d = band.photons; m = band.diffuse_counts+band.source_counts
        fig, ax = plt.subplots(figsize=(5,5)) if ax is None else (ax.figure, ax)
        ax.scatter(m.clip(1,1e4), ((d-m)/np.sqrt(m)).clip(-5,10), s=2);
        ax.set(xscale='log',yscale='linear',xlabel='model counts/pixel', ylabel=r'residual ($\sigma$ units)', )
        ax.text(1,8, f'{band.psf_name}\nnside {band.nside}', fontsize=14)
        
    fig, axx = plt.subplots(2,2, figsize=(12,8), sharey=True, sharex=True)
    for i, ax in enumerate(axx.flat):
        mdplot(pixel_table(i, energy_index), ax)
        if i < 2:
            ax.set(xlabel='')
        if i % 2 == 1:
            ax.set(ylabel='')
        ax.axhline(0, color='0.5', ls='--', lw=2)
    fig.suptitle(pixel_table(0,energy_index).energy, fontsize=16  )
    return fig

def histograms_of_residuals_for_given_energy(pixel_table, energy_index):
    """Plot residual histograms for one energy bin across all PSFs.

    Parameters
    ----------
    pixel_table : PixelTable
        Pixel table to analyze.
    energy_index : int
        Energy bin index (0–11 typical).

    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2×2 grid (one panel per PSF type) showing normalized
        residual distributions with Gaussian fit overlays.
    """
    fig, axx = plt.subplots(2,2, figsize=(8,6),sharey=True, sharex=True)
    for i, ax in enumerate(axx.flat):
        pt = pixel_table(i, energy_index)
        ResidualPlotter( pt, ).residual_hist(ax=ax,)
        ax.text(0.05, 0.9, f'PSF{i}', transform=ax.transAxes, fontsize=12, ha='left')
        if i%2>0: ax.set_ylabel('')
        if i<2: ax.set_xlabel('')
    fig.suptitle(f'Residual histograms for {pixel_table(0,energy_index).energy} ', fontsize=18)
    return fig

class ResidualPoints:
    """Collect and cluster significant residual pixels across PSF bands."""
    
    def __init__(self, pixel_table, energy_index, sigma_min=5):
        """Collect outliers across PSF bands at fixed energy and prepare for clustering.

        Parameters
        ----------
        pixel_table : PixelTable
            Pixel table instance.
        energy_index : int
            Energy bin index (selects the same energy across all 4 PSF bands).
        sigma_min : float, optional
            Significance threshold in sigma units. Outliers with |sigma| >= sigma_min
            are included. Default is 5.

        Attributes Set
        ---------------
        bands : list[Band]
            List of 4 Band objects (PSF0-3) at specified energy_index.
        sigma_min : float
            Threshold used for outlier selection.
        df : pd.DataFrame
            Merged outlier table from all bands with columns:
                - pixel, data, model, sigma (from Band.get_outliers)
                - glon, glat (in degrees, rescaled to [-180, 180])
                - psf (PSF index 0-3)
                - nside (band HEALPix nside)
        skycoord : SkyCoord
            Galactic sky coordinates of all outliers.
        cluster_idx : list[np.ndarray] or None
            Cluster membership (set by clusterer()). None until clusterer() called.
        cldf : pd.DataFrame or None
            Cluster summary (set by clusterer()). None until clusterer() called.
        clpoints : SkyCoord or None
            Representative points per cluster (set by ait_cluster_plot()).
        """
        
        self.bands = [pixel_table(i, energy_index) for i in range(4)]
        self.sigma_min = sigma_min

        dff = []
        for ie, band in enumerate(self.bands):
            df = band.get_outliers(self.sigma_min)
            skyc = band.healpix_to_skycoord(df.pixel)
            # Shift longitudes to [-180, 180] for plotting and visual grouping.
            glon = skyc.galactic.l.deg
            glon[glon > 180] -= 360
            df['glon'] = glon
            df['glat'] = skyc.galactic.b.deg
            df['psf'] = ie
            df['nside'] = band.nside
            dff.append(df)
        self.df = pd.concat(dff)
        self.skycoord = SkyCoord(self.df.glon, self.df.glat, unit='deg', frame='galactic')


    def ait_plot(self):
        """Plot all selected residual points on an AIT projection.

        Returns
        -------
        utilities.skymaps.AITfigure
            Chainable AIT figure showing all outlier points. Marker sizes
            scale with (data - model) to emphasize larger residuals.
            Title shows count and significance threshold.
        """
        from utilities.skymaps import AITfigure
        energy = self.bands[0].energy
        afig = AITfigure(
            figsize=(12,6),
            title=rf"""{len(self.skycoord)} residuals > {self.sigma_min} $\sigma$ at {energy}""",
        )
        (afig.scatter(self.skycoord, marker='o', s=5*np.sqrt(self.df.data-self.df.model), color='yellow')
        .show()
        )

    def clusterer(self, radius=1.5, ptmin=2):
        r"""Group high-sigma residual points into angularly connected clusters.

        Parameters
        ----------
        radius : float
            Separation threshold in degrees for graph connectivity.
        ptmin : int
            Minimum number of points required to keep a cluster.

        Notes
        -----
        The representative row for each cluster is chosen as the point with the
        largest modeled count level.
        """
        self.cluster_idx = grouper(self.skycoord, radius)
        # Keep only clusters large enough to be worth reporting.
        clgt1 = [cluster for cluster in self.cluster_idx if len(cluster) >= ptmin]

        # Represent each cluster by its highest-model point for concise reports.
        cld = {}
        for idx, cluster in enumerate(clgt1):
            t = self.df.iloc[cluster].sort_values('model', ascending=False).iloc[0]
            cld[idx] = dict(
                glon=round(t.glon, 3),
                glat=round(t.glat, 3),
                n=len(cluster),
                sigma=round(t.sigma, 1),
                data=t.data.astype(int),
                model=round(t.model, 1),
                ids=cluster,
            )

        self.cldf = pd.DataFrame.from_dict(cld, orient='index')['glon glat data model sigma n ids'.split()]

    def zea_plot(self, center, size=5, axes_visible=False, **kwargs):
        """Plot residual points in a local ZEA projection.

        Parameters
        ----------
        center : SkyCoord or tuple
            Center of projection.
        size : float, optional
            Field of view side length in degrees. Default is 5.
        **kwargs
            Additional arguments to ZEAfigure.

        Returns
        -------
        utilities.skymaps.ZEAfigure
            Local projection with points sized and colored by significance.
        """
        from utilities.skymaps import ZEAfigure

        zfig = ZEAfigure(center, size=size, fig=None, figsize=(8,8), title='Residual clusters', frame='galactic', axes_visible=axes_visible)
        zfig.scatter(self.skycoord, s=self.df.sigma*10, c=self.df.sigma, cmap='jet', vmin=5, **kwargs)
        zfig.colorbar(label=r'$\sigma$', shrink=0.7)
        return zfig
    
    def ait_cluster_plot(self, *, figsize=(10,10), title=None, **kwargs):
        """Plot one representative point per cluster in AIT projection.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (10, 10).
        title : str, optional
            Plot title. Auto-generated if None.
        **kwargs
            Additional arguments to AITfigure.

        Notes
        -----
        Representative point for each cluster is the pixel with largest model count.
        Marker sizes scale with cluster size (number of points).
        Colors scale with maximum significance per cluster.
        """
        from utilities.skymaps import AITfigure
        self.clpoints = SkyCoord(self.cldf.glon, self.cldf.glat, unit='deg', frame='galactic')
        if title is None:
            title = f"Residual clusters with >{self.sigma_min} sigma and >1 point"
        (AITfigure(figsize=figsize, title=title, **kwargs)
            .scatter(self.clpoints, s=self.cldf.n*20, c=self.cldf.sigma, cmap='jet', vmin=5)
            .colorbar(label=r'$\sigma$', shrink=0.4)
            .show())