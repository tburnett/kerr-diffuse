"""
FermiFit: fitting interface for a PixelTable + SourceModel pair.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import importlib
import io
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt

plt.style.use('dark_background')
import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy_healpix import HEALPix
import pandas as pd
from like3 import views, loglikelihood
from like3.pixel_table import PixelTable
from like3.sourcelist import SourceModel
import importlib

class PSFlookup:
    
    def __init__(self, table_path='files/loc'):
        """ A functor that returns the PSF for a given band, using the same PSF for all pixels in the band.

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
            print(f'PSFlookup: no PSF entries loaded from {table_path!r}')
            return self

        et_names = PSFlist.PSF.et_name
        ets = sorted({p.event_type for p in all_psfs})
        et_labels = [et_names[e] if e < len(et_names) else str(e) for e in ets]
        # print(f'PSFlookup: {len(all_psfs)} PSF entries '
        #       f'({", ".join(et_labels)}) from {table_path!r}')
        self.psf_list = all_psfs

    def __call__(self, band):
        """Return the PSF for *band*."""
        
        for candidate_psf in self.psf_list:
            if band.event_type != candidate_psf.event_type:
                continue
            if abs(candidate_psf.energy/band.energy-1)<0.1:
                # print(f'PSFlookup: found PSF {candidate_psf} for band {band}')
                return candidate_psf
        raise ValueError(f'PSFlookup: no PSF found for band {band} ')


from like3.fitter import Fitted   
class MultiBandLikelihood(dict, Fitted):
    def poisson_at_energy(self, energy, tol=0.2, **kwargs):
        """Return a Poisson object for the selected source at the given energy.

        Uses the eflux_view context manager and PoissonFitter to fit the likelihood curve
        as a function of differential energy flux at the specified energy.

        Parameters
        ----------
        energy : float
            Energy in MeV for which to compute the Poisson fit.
        tol : float, optional
            Fit-quality tolerance for PoissonFitter (default 0.2).
        **kwargs
            Additional arguments forwarded to PoissonFitter.

        Returns
        -------
        Poisson
            Fitted Poisson object representing the likelihood at this energy.
        """
       
        with self.eflux_view(energy) as eflux_ll:
            pf = loglikelihood.PoissonFitter(eflux_ll, tol=tol, **kwargs)
            return pf._poiss if hasattr(pf, '_poiss') else pf.poiss


    """A collection of BandLikelihood objects, one per band, that share a common SourceModel.
    It implements the Fitted interface by delegating parameter access to the shared SourceModel and summing log-likelihoods across bands.
    """
    
    def __init__(self, pixel_table, source_model):
        self.pixel_table = pixel_table
        self.source_model = source_model
        self.selected = None
        self.energies = pixel_table.energies
        self.fit_info: dict = {}
        super().__init__({key: BandLikelihood(band, source_model) for key, band in self.pixel_table.items()})

        self.llz = self.log_like()  # baseline log-likelihood for delta-TS calculations
        
    @property
    def sources(self):
        # For compatibility with FermiFit-like interface
        return self.source_model

    @property
    def parameter_names(self):
        return self.source_model.parameter_names

    @property
    def parameterset(self):
        """ParameterSet interface used by views.FitterView."""
        return self.source_model.parameters

    # ============== needed by Fitter interface ======================        
    @property
    def bounds(self):
        return self.source_model.bounds 
    def get_parameters(self):
        return self.parameters
    def set_parameters(self, pars):
        self.parameters = pars
    # =============================================================

    #======= Parameter accessors with coverage update ++++++++++++++++++++++
    @property
    def parameters(self):
        return self.source_model.parameters.get_parameters()
    
    @parameters.setter
    def parameters(self, pars):
        if pars is None:
            return
        if len(pars) != len(self.parameters):
            raise ValueError(f'Expected {len(self.source_model.parameters)} parameters, got {len(pars)}')
        self.source_model.parameters.set_parameters(pars)

        # update pixels in selected bands
        for bl in self._iter_bands():
            bl.evaluate_source_model()
    #============++++++++++++++++++++++


    def select(self, *pars, **kwargs):
        """Select bands, using pixel table selection."""
        self.pixel_table.select(*pars, **kwargs)
        self.selected =  self.pixel_table._selected
        return self.selected

    def _iter_bands(self):
        """Iterate over selected bands, or all bands when no selection is active."""
        if self.selected is None:
            return self.values()
        return (self[k] for k in self.selected)

    def update(self):
        """Refresh per-band model counts after parameter changes."""
        for bl in self._iter_bands():
            bl.evaluate_source_model()

    def log_like(self, pars=None, *, skydir=None, summed=True):
        """Evaluate the total log-likelihood across all selected bands.
        If `pars` is provided, update the source model parameters before computing.
        If `skydir` is provided, update the source position before computing."""

        if skydir is not None:
            src = self.source_model.selected_source
            if src is None:
                raise ValueError('No source selected in source model for position update')
 
            if isinstance(skydir, tuple):
                skydir = SkyCoord(*skydir, unit='deg', frame='icrs')  
            if not isinstance(skydir, SkyCoord):
                raise ValueError(f'Expected skydir as SkyCoord or (ra, dec) tuple, got {type(skydir)}') 

            src.skydir = skydir
            total_loglike = 0.0
            for bl in self._iter_bands():
                bl.update_position(src) # ensure pixels are updated for new position
                total_loglike += bl.log_like(skydir=skydir, )
            return total_loglike
        
        if pars is not None:
            self.parameters = pars
        return sum(bl.log_like() for bl in self._iter_bands())

    def loglike_grad(self, pars=None):
        """Compute the log-likelihood and its gradient for the current model parameters.
        If `pars` is provided, update the source model parameters before computing."""
        if pars is not None:
            self.parameters = pars
        total_loglike = 0.0
        total_grad = np.zeros_like(self.parameters)
        for bl in self._iter_bands():
            loglike, grad = bl.loglike_grad()
            total_loglike += loglike
            total_grad += grad
        return total_loglike, total_grad

    def __call__(self, pars=None, use_gradient=False, **kwargs):
        """Callable interface to negative log-likelihood, for compatibility with optimizers."""
        # print(f'MultiBandLikelihood called with pars={pars}' f' and kwargs={kwargs}')
        if use_gradient:
            ll, grad = self.loglike_grad(pars)
            return -ll, -grad
        return -self.log_like(pars)
    
    def gradient(self, pars=None):
        """Callable interface to negative log-likelihood gradient, for compatibility with optimizers."""
        _, grad = self.loglike_grad(pars)
        return -grad
    
    def delta_ts(self, skydir):
        """Return TS difference at ``skydir`` relative to nominal maximum."""
        # print(f'Computing delta-TS at {skydir.to_string()}')
        return 2 * (self.log_like(skydir=skydir) - self.llz)

    @contextmanager
    def tsmap_view(self, source_name=None):
        """Yield a TS-map callable for localization scans of one source.

        Parameters
        ----------
        source_name : str, source-like, or None, optional
            Source selector. When ``None``, the currently selected source is
            used.

        Yields
        ------
        object
            Callable TS-map helper with ``source``, ``skydir``,
            ``saved_skydir``, and ``reset()`` attributes.

        Raises
        ------
        ValueError
            If no source is selected and ``source_name`` is not provided.
        """

        source_model = self.source_model
        selected_source = getattr(source_model, 'selected_source', None)
        selected_source_index = getattr(source_model, 'selected_source_index', None)

        if source_name is not None and not isinstance(source_name, str) and hasattr(source_name, 'name'):
            if source_name in source_model:
                source_model.selected_source = source_name
                source_model.selected_source_index = source_model.index(source_name)
                source = source_name
            else:
                source = source_model.find_source(source_name.name)
        elif source_name is None:
            source = source_model.selected_source
        else:
            source = source_model.find_source(source_name)

        if source is None:
            raise ValueError('No source is selected in the source model for a tsmap')

        log_like = self.log_like

        class TSmap_function:

            def __init__(self, ts_source):
                self.source = ts_source
                self.skydir = ts_source.skydir
                self.saved_skydir = ts_source.skydir
                self._llz = log_like(skydir=ts_source.skydir)

            def __call__(self, skydir):
                """Return ``2 * (log_like(skydir) - log_like(nominal))``."""
                return 2 * (log_like(skydir=skydir) - self._llz)

            def reset(self):
                self.source.skydir = self.saved_skydir

        tsm = TSmap_function(source)
        try:
            yield tsm
        finally:
            source.skydir = tsm.saved_skydir
            source_model.selected_source = selected_source
            source_model.selected_source_index = selected_source_index
    
    
    def localize(self, update=False, sigma=0.1, **kwargs):
        """Localize the currently selected source with a TS-map fit.

        Parameters
        ----------
        update : bool, optional
            If ``True``, keep the localized sky position applied by
            ``Localization.localize``. If ``False``, restore the original
            source position after the fit and return only the ellipse result.
        sigma : float, optional
            Initial localization scale in degrees. If the source already has
            an ``ellipse`` entry with a ``sigma`` value, that value is used
            instead.
        **kwargs
            Additional keyword arguments forwarded to ``Localization``.

        Returns
        -------
        dict
            Ellipse parameters returned by ``Localization.localize``. The
            result is also attached to ``source.ellipse``.

        Raises
        ------
        ValueError
            If no source is currently selected in the source model.
        """

        from .localization import Localization

        source = self.source_model.selected_source
        if source is None:
            raise ValueError("No source selected in the MultiBandLikelihood's source model")

        # use existing ellipse sigma if available, otherwise default to 0.1 deg 
        if hasattr(source, 'ellipse'):
            sigma = source.ellipse.get('sigma', sigma)

        with self.tsmap_view(source) as tsm:
            ellipse = Localization(tsm, **kwargs).localize(sigma=sigma)
            if update and ellipse is not None:
                tsm.saved_skydir = SkyCoord(ellipse['ra'], ellipse['dec'], unit='deg', frame='icrs')

        # attach result in any case
        source.ellipse = ellipse
        return ellipse

    def plot_tsmap(
        self,
        source_name=None,
        *,
        size=None,
        npix=None,
        ax=None,
        figsize=(6, 6),
        cmap='viridis',
        colorbar=True,
        contour_levels=None,
        show_source=True,
        show_peak=True,
    ):
        """Plot a square TS map in ICRS RA/Dec coordinates.

        Parameters
        ----------
        source_name : str, source-like, or None, optional
            Source selector. When ``None``, the currently selected source is
            used.
        size : float or None, optional
            Full map width in degrees. When omitted, it is inferred from the
            current localization ellipse if available, otherwise a default of
            ``0.25`` deg is used.
        npix : int or None, optional
            Number of pixels per side. When omitted, it is chosen from ``size``
            at roughly ``0.02`` deg sampling and rounded up to an odd value so
            the central source falls on a pixel center.
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw into. A new square figure is created when ``None``.
        figsize : tuple[float, float], optional
            Figure size used only when creating a new figure.
        cmap : str, optional
            Matplotlib colormap name.
        colorbar : bool, optional
            Whether to draw a colorbar.
        contour_levels : sequence[float] or None, optional
            Optional contour levels to overlay.
        show_source : bool, optional
            Mark the source center with a white ``+``.
        show_peak : bool, optional
            Mark the maximum sampled TS pixel with a black ``x``.

        Returns
        -------
        dict
            Plot payload with ``fig``, ``ax``, ``contour``, ``tsmap``, ``ra``,
            ``dec``, ``size``, and ``npix``.
        """
        import astropy.units as u

        source_model = self.source_model
        if source_name is None:
            source = source_model.selected_source
        else:
            source = source_model.find_source(source_name)
        if source is None:
            raise ValueError('No source is selected in the source model for a tsmap plot')

        if size is None:
            size = 0.25
            ellipse = getattr(source, 'ellipse', None)
            if isinstance(ellipse, dict):
                scale = ellipse.get('a', ellipse.get('sigma', None))
                if scale is not None and np.isfinite(scale):
                    size = float(np.clip(max(0.25, 15.0 * scale), 0.25, 2.0))

        if npix is None:
            npix = max(25, int(np.ceil(size / 0.02)) + 1)
        npix = int(npix)
        if npix < 3:
            raise ValueError('npix must be at least 3')
        if npix % 2 == 0:
            npix += 1

        center = source.skydir.icrs
        half_size = 0.5 * float(size)
        offsets = np.linspace(-half_size, half_size, npix)
        lon_offsets, lat_offsets = np.meshgrid(offsets, offsets)

        with self.tsmap_view(source) as tsm:
            coords = center.spherical_offsets_by(lon_offsets * u.deg, lat_offsets * u.deg)
            flat_coords = coords.reshape((coords.size,))
            tsmap = np.fromiter((tsm(coord) for coord in flat_coords), dtype=float).reshape((npix, npix))

        # scale TS values to 5 sigmas
        scaled_tsmap = tsmap #5- np.sqrt(-np.clip(tsmap, -25, 0)) 

        ra = coords.ra.deg
        dec = coords.dec.deg

        # fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)

        # if contour_levels is None:
        #     contour_levels = np.linspace(float(np.nanmin(scaled_tsmap)), float(np.nanmax(scaled_tsmap)), 6)

        # contour = ax.contourf(ra, dec, tsmap, cmap=cmap, 
        #         levels = np.linspace(float(np.nanmin(scaled_tsmap)), float(np.nanmax(scaled_tsmap)), 26))
        
        # ax.contour(ra, dec, tsmap, levels=contour_levels, colors='white', linewidths=0.8)
        # ax.clabel(contour)#, inline=True, fontsize=8, fmt='%.1f')

        # if show_source:
        #     ax.plot(center.ra.deg, center.dec.deg, marker='+', color='white', markersize=10, mew=1.5)

        # if show_peak:
        #     peak_index = np.unravel_index(np.nanargmax(tsmap), tsmap.shape)
        #     ax.plot(ra[peak_index], dec[peak_index], marker='x', color='black', markersize=7, mew=1.5)

        # ax.set(
        #     xlabel='RA (deg)',
        #     ylabel='Dec (deg)',
        #     title=f'{source.name} TS map',
        # )
        # ax.invert_xaxis()
        # ax.set_box_aspect(1)

        # if colorbar:
        #     fig.colorbar(contour, ax=ax,)# label=r'$\sigma (\sqrt{TS_{\max}-TS})$')

        return dict(
            # fig=fig,
            # ax=ax,
            # contour=contour,
            # image=contour,
            tsmap=tsmap,
            ra=ra,
            dec=dec,
            size=float(size),
            npix=npix,
        )
    
       
    
    def fitter_view(self, select=None, setpars=None, **kwargs):
        """Return a fitter view over all or a subset of free parameters.

        Parameters
        ----------
        select : str, list, or None
            If None, returns a :class:`views.FitterView` over all free
            parameters. Otherwise, constructs a
            :class:`views.SubsetFitterView` for the selection.
        setpars : dict or None
            If provided, set these parameter values before constructing the
            view.
        **kwargs
            Forwarded to the view constructor.

        Returns
        -------
        views.FitterView or views.SubsetFitterView
        """
        if setpars is not None:
            self.sources.parameters.setitems(setpars)

        if select is None:
            return views.FitterView(self, **kwargs)
        return views.SubsetFitterView(self, select, **kwargs)

    def energy_flux_view(self, source_name, energy=None, **kw):
        """Return a functor expressing log-likelihood as energy flux.

        Parameters
        ----------
        source_name : str
            Source whose normalization is profiled.
        energy : float or None
            Energy in MeV. If None, uses the model reference energy e0.
        **kw
            Forwarded to :class:`views.EnergyFluxView`.

        Returns
        -------
        views.EnergyFluxView
        """
        try:
            source = self.sources.find_source(source_name)
        except Exception as msg:
            raise Exception(
                'could not create energy flux function for source %s;%s'
                % (source_name, msg)
            )
        return views.EnergyFluxView(self, source.name, energy, **kw)

    def selected_source_energy_flux_view(self, energy=None, **kw):
        """Return an energy-flux view for the currently selected source."""
        src = self.source_model.selected_source
        if src is None:
            raise ValueError('No source is selected')
        return self.energy_flux_view(src.name, energy=energy, **kw)

    @contextmanager
    def eflux_view(self, energy):
        """Yield an energy-flux likelihood view for the selected source.

        The active band selection is temporarily reduced to the band(s)
        containing ``energy`` and the currently selected source is temporarily
        assigned a power-law spectral model.  On exit (including exceptions),
        both the original spectral model and the original band selection are
        restored.

        Parameters
        ----------
        energy : float
            Energy in MeV used to select active band(s).

        Yields
        ------
        views.EnergyFluxView
            Callable view of negative log-likelihood as a function of
            differential energy flux.
        """
        if energy is None:
            raise ValueError('eflux_view requires an energy value in MeV')

        source = self.source_model.selected_source
        if source is None:
            raise ValueError('No source is selected')

        energy = float(energy)
        saved_selection = None if self.selected is None else list(self.selected)
        saved_model = source.spectral_model
        saved_changed = bool(getattr(source, 'changed', False))

        def _powerlaw_at(model, e):
            """Create a power-law proxy matching model normalization/slope at e."""
            from like3 import spectral_models
            from like3.sources import set_default_bounds

            if getattr(model, 'name', None) == 'PowerLaw':
                pl = model.copy()
            else:
                pl = None
                if hasattr(model, 'create_powerlaw'):
                    candidate = model.create_powerlaw()
                    if getattr(candidate, 'name', None) == 'PowerLaw':
                        pl = candidate.copy()
                if pl is None:
                    e1 = max(1e-3, e * (1 - 1e-3))
                    e2 = e * (1 + 1e-3)
                    f1 = max(float(model(e1)), 1e-300)
                    f2 = max(float(model(e2)), 1e-300)
                    gamma = -np.log(f2 / f1) / np.log(e2 / e1)
                    norm = max(float(model(e)), 1e-300)
                    pl = spectral_models.PowerLaw(p=[norm, gamma], e0=e)

            if hasattr(pl, 'e0'):
                pl.e0 = e
            if hasattr(pl, 'free') and len(pl.free) >= 2:
                pl.free[0] = True
                pl.free[1] = False
            set_default_bounds(pl, force=True)
            return pl

        try:
            self.select(energy=energy)
            source.spectral_model = _powerlaw_at(saved_model, energy)
            self.update()
            yield views.EnergyFluxView(self, source.name, energy)
        finally:
            source.spectral_model = saved_model
            source.changed = saved_changed
            # Ensure all bands are refreshed before restoring the prior selection.
            self.select()
            self.update()
            if saved_selection is None:
                self.select()
            else:
                self.select(saved_selection)

        # ------------------------------------------------------------------
    # Freeze / thaw
    # ------------------------------------------------------------------

    def freeze(self, param, source_name=None, ):
        """Freeze one or all parameters of a source's spectral model.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector forwarded to ``SourceModel.find_source``.
        param : str, int, or None, optional
            Parameter name or index to freeze.  When ``None`` all parameters
            of the source's model are frozen.
        """
        src = self.source_model.find_source(source_name)
        if param is None:
            src.model.free[:] = False
        else:
            src.model.freeze(param)
        self.source_model.reinitialize()  # Ensure parameter set is updated after thawing.

    def thaw(self,  param,source_name=None,):
        """Thaw (unfreeze) one or all parameters of a source's spectral model.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector forwarded to ``SourceModel.find_source``.
        param : str, int, or None, optional
            Parameter name or index to thaw.  When ``None`` all parameters
            of the source's model are thawed.
        """
        src = self.source_model.find_source(source_name)
        if param is None:
            src.model.free[:] = True
        else:
            src.model.thaw(param)
        self.source_model.reinitialize()  # Ensure parameter set is updated after thawing.

    
    def fit(self, select=None, *, exclude=None, summarize=True, setpars=None, **kwargs):
        """Fit free parameters using :class:`~like3.likelihood.Likelihood`.

        Parameters
        ----------
        select : None, item, or list of items
            Optional parameter selector forwarded to ``Likelihood.select``.
        exclude : None, item, or list of items
            Currently unused; reserved for future subset exclusion.
        summarize : bool, default=True
            If True, print the fit summary after a successful fit.
        setpars : dict or None
            Optional parameter values to set before fitting.

        Returns
        -------
        Likelihood
            The fitted :class:`~like3.likelihood.Likelihood` instance, with
            ``fit_info`` populated.
        """
        from like3.likelihood import Likelihood

        if len(self.source_model.parameters) == 0:
            print('No parameters to fit')
            return

        if setpars is not None:
            self.source_model.parameters.setitems(setpars, quiet=True)

        mbl = self

        class _MultiBandModel:
            """Adapter exposing the Likelihood interface over all selected bands."""
            parameters = mbl.source_model.parameters
            parameter_names = mbl.source_model.parameter_names
            source_model = mbl.source_model

            @property
            def data(self):
                return np.concatenate([
                    bl.coverage['photons'].to_numpy() for bl in mbl._iter_bands()
                ])

            def counts(self):
                """Fresh model counts across all selected bands."""
                parts = []
                for bl in mbl._iter_bands():
                    pix = bl.coverage['pix'].to_numpy()
                    exp = bl.coverage['exposure'].to_numpy()
                    c = np.zeros(len(pix))
                    for src in bl.source_model:
                        c += bl.response(src, pix) * src.model(bl.band.energy)
                    c *= exp
                    parts.append(c)
                return np.concatenate(parts)

            def count_gradient(self):
                """Gradient (n_params, total_pixels) across all selected bands."""
                return np.vstack([bl.pixel_gradient() for bl in mbl._iter_bands()]).T

            def parsubset(self, *args):
                return mbl.source_model.parsubset(*args)

        model = _MultiBandModel()
        lik = Likelihood(model)

        if select is not None:
            if isinstance(select, (list, tuple)):
                lik.select(*select)
            else:
                lik.select(select)

        lik.maximize()
        self.fit_info = lik.fit_info

        if summarize:
            self.summary()

    def summary(self, out=None, title=None, gradient=True, ts=True):
        """Print a summary table of fitted parameter values and diagnostics.

        Parameters
        ----------
        out : file-like or None
            Output stream; defaults to stdout.
        title : str or None
            Optional title line.
        gradient : bool
            Include likelihood-gradient column when available.
        ts : bool
            Include TS column when available.
        """
        if title is not None:
            print(title, file=out)

        fmt_hdr = '%-21s%6s%10s%10s'
        tup_hdr = ('Name', 'index', 'value', 'error(%)')

        pset = self.source_model.parameters
        all_names = pset.parameter_names
        all_model_params = np.asarray(pset.model_parameters)
        n_all = len(all_names)

        param_mask = np.asarray(getattr(pset, 'mask', np.ones(n_all, dtype=bool)), dtype=bool)
        grad = None
        ts_values = None

        if self.fit_info:
            fit_mask = self.fit_info.get('param_mask')
            if fit_mask is not None:
                param_mask = np.asarray(fit_mask, dtype=bool)
            if gradient:
                grad = self.fit_info.get('grad')
                if grad is not None:
                    grad = np.asarray(grad, dtype=float)
                if grad is not None and len(grad) == int(param_mask.sum()):
                    fmt_hdr += '%10s'
                    tup_hdr += ('ll_grad',)
            if ts:
                ts_values = self.fit_info.get('ts_values')
                if ts_values is not None:
                    fmt_hdr += '%10s'
                    tup_hdr += ('TS',)

        if gradient and grad is None:
            try:
                _, grad = self.loglike_grad()
                grad = np.asarray(grad, dtype=float)[param_mask]
                fmt_hdr += '%10s'
                tup_hdr += ('ll_grad',)
            except Exception:
                grad = None

        print(fmt_hdr % tup_hdr, file=out)

        active_names = all_names[param_mask]
        active_model_params = all_model_params[param_mask]
        index_array = np.arange(n_all)[param_mask]

        n_active = int(param_mask.sum())
        uncertainties = pset.uncertainties[param_mask] if self.fit_info else np.zeros(n_active)

        prev = ''
        for i, (name, value) in enumerate(zip(active_names, active_model_params)):
            t = name.split('_')
            pname = t[-1]
            sname = '_'.join(t[:-1])
            display_name = name if sname != prev else len(sname) * ' ' + '_' + pname
            prev = sname

            rsig = float(uncertainties[i]) if i < len(uncertainties) else 0.0
            psig = '%.1f' % (rsig * 100) if rsig > 0 and not np.isnan(rsig) else '***'

            truncname = display_name[:20] + '*' if len(display_name) > 20 else display_name
            fmt = '%-21s%6d%10.4g%10s'
            tup = (truncname, index_array[i], value, psig)

            if gradient and grad is not None:
                fmt += '%10.1f'
                tup += (float(grad[i]),)
            if ts and ts_values is not None:
                fmt += '%10s'
                ts_val = ts_values[i]
                tup += (f'{ts_val:.0f}' if np.isfinite(ts_val) else '',)

            print(fmt % tup, file=out)


    def get_sed_poisson_table(self, source_name=None, event_type=None, tol=0.2):
        """Return an SED table with one Poisson object per energy bin.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector forwarded to ``SourceModel.find_source``.
        event_type : None, int, or str
            Event-type selection forwarded to ``sedfuns.sed_poisson_table``.
        tol : float
            Fit-quality tolerance forwarded to ``PoissonFitter``.

        Returns
        -------
        pandas.DataFrame
            Per-band SED table containing a ``poiss`` column with
            ``like3.loglikelihood.Poisson`` entries.
        """
        source = self.source_model.find_source(source_name)
        pkg = __package__ if __package__ else 'like3'
        sedfuns = importlib.import_module(f'{pkg}.sedfuns')
        return sedfuns.sed_poisson_table(
            self,
            source_name=source.name,
            event_type=event_type,
            tol=tol,
        )

    def plot_sed_with_band_points(
        self,
        source=None,
        *,
        sed_table=None,
        set_kwargs=None,
        ax=None,
        update=False,
        event_type=None,
        tol=0.2,
        emin=100,
        emax=1e5,
        xlim=None,
        ylim=(0.1, None),
        npts=100,
        model_label=None,
        points_label='Per-band SED',
        title=None,
        show_upper_limits=True,
    ):
        """Plot source SED model with per-band errorbar points.

        Parameters
        ----------
        source : Source, str, or None, optional
            Source selector forwarded to ``SourceModel.find_source``.
            When ``None`` (default), the currently selected source is used.
        sed_table : pandas.DataFrame or None, optional
            Precomputed SED Poisson table containing ``elow``, ``ehigh``,
            ``flux``, ``lflux``, and ``uflux`` columns. When provided,
            this table is used directly and no call to
            :meth:`get_sed_poisson_table` is made.
        set_kwargs : dict or None, optional
            Keyword arguments forwarded to ``Axes.set``. These values override
            the default axis settings used by this method.
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw into. A new figure is created when ``None``.
        update : bool, optional
            Force regeneration of the source SED Poisson table when
            ``sed_table`` is not provided. Default ``False``.
        event_type : None, int, or str, optional
            Event-type selection forwarded to :meth:`get_sed`.
        tol : float, optional
            Poisson-fit tolerance forwarded to :meth:`get_sed`.
        emin, emax : float, optional
            Model-curve plotting range in MeV.
        xlim : tuple[float, float] or None, optional
            X-axis limits in MeV. Defaults to ``(emin, emax)``.
        npts : int, optional
            Number of model-curve points.
        model_label : str or None, optional
            Legend label for the model curve. Defaults to source name.
        points_label : str, optional
            Legend label for the binned points.
        title : str or None, optional
            Plot title. Defaults to source name.
        show_upper_limits : bool, optional
            If True, plot upper-limit markers for bins with ``flux <= 0``.

        Returns
        -------
        matplotlib.axes.Axes
            Axes with model SED and per-band points.
        """
        import matplotlib.pyplot as plt

        src = self.source_model.find_source(source)
        if sed_table is None:
            if not update and hasattr(src, 'sedrec'):
                sed_table = src.sedrec
            else:
                sed_table = self.get_sed_poisson_table(
                    source_name=src.name,
                    event_type=event_type,
                    tol=tol,
                )
                src.sedrec = sed_table

        if sed_table is None:
            raise ValueError(f'No SED table available for source {src.name}')

        fields = set(getattr(sed_table, 'columns', ()))
        needed = {'elow', 'ehigh', 'flux', 'lflux', 'uflux'}
        missing = needed - fields
        if missing:
            raise ValueError(f'sed table missing required fields: {sorted(missing)}')

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        if xlim is None:
            xlim = (emin, emax)

        # Plot model SED directly in eV cm^-2 s^-1 units.
        model = src.model
        energies = np.logspace(np.log10(emin), np.log10(emax), npts)
        dnde = model(energies)  # ph cm^-2 s^-1 MeV^-1
        e2dnde_ev = energies**2 * dnde * 1e6
        ax.loglog(
            energies,
            e2dnde_ev,
            label=src.name.strip() if model_label is None else model_label,
        )

        if model.has_errors():
            g = model.external_gradient(energies)
            cov = model.get_cov_matrix()
            var_dnde = np.sum((cov @ g) * g, axis=0)
            var_dnde = np.clip(var_dnde, 0, None)
            sigma_e2dnde_ev = energies**2 * np.sqrt(var_dnde) * 1e6
            ax.fill_between(
                energies,
                e2dnde_ev - sigma_e2dnde_ev,
                e2dnde_ev + sigma_e2dnde_ev,
                alpha=0.3,
            )

        elow = np.asarray(sed_table['elow'], dtype=float)
        ehigh = np.asarray(sed_table['ehigh'], dtype=float)
        flux = np.asarray(sed_table['flux'], dtype=float)
        lflux = np.asarray(sed_table['lflux'], dtype=float)
        uflux = np.asarray(sed_table['uflux'], dtype=float)

        ecent = np.sqrt(elow * ehigh)
        xerr = np.vstack([
            np.clip(ecent - elow, 0, np.inf),
            np.clip(ehigh - ecent, 0, np.inf),
        ])
        # sed_table fluxes are energy-flux values from EnergyFluxView (eV units).
        y = flux
        ylo = lflux
        yhi = uflux

        det_mask = np.isfinite(y) & np.isfinite(ylo) & np.isfinite(yhi) & (flux > 0)
        if np.any(det_mask):
            yerr = np.vstack([
                np.clip(y[det_mask] - ylo[det_mask], 0, np.inf),
                np.clip(yhi[det_mask] - y[det_mask], 0, np.inf),
            ])
            ax.errorbar(
                ecent[det_mask],
                y[det_mask],
                xerr=xerr[:, det_mask],
                yerr=yerr,
                fmt='o',
                ms=5,
                capsize=2,
                lw=1,
                color='tab:orange',
                label=points_label,
            )

        if show_upper_limits:
            ul_mask = np.isfinite(uflux) & (flux <= 0)
            if np.any(ul_mask):
                y_ul = uflux[ul_mask]
                yerr_ul = 0.35 * np.clip(y_ul, 0, np.inf)
                ax.errorbar(
                    ecent[ul_mask],
                    y_ul,
                    xerr=xerr[:, ul_mask],
                    yerr=yerr_ul,
                    uplims=True,
                    fmt='v',
                    ms=4,
                    lw=1,
                    color='tab:red',
                    label='95% UL',
                )

        defaults = dict(
            xlabel='Energy (MeV)',
            ylabel=r'$E^2\,dN/dE\ [\mathrm{eV\,cm^{-2}\,s^{-1}}]$',
            title=src.name.strip() if title is None else title,
            xlim=xlim,
            ylim=ylim,
            xscale='log',
            yscale='log',
        )
        if set_kwargs is not None:
            defaults.update(set_kwargs)
        ax.set(**defaults)

        ax.grid(True, which='both', alpha=0.25)
        ax.legend()
        return ax

    def norm_profile(self, source_name=None, tol=0.5):
        """Return a Norm profile fitted in log-space.

        All other free parameters are held fixed at their current values.
        The fit is performed against ``x = log(norm / norm_floor)`` (with an
        additional linear rescaling for numerical stability), then exposed via a
        callable object in physical Norm units.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector forwarded to ``SourceModel.find_source``.  Defaults
            to the currently selected source.
        tol : float, optional
            Fit-quality tolerance forwarded to ``PoissonFitter``.

        Returns
        -------
        object
            Profile object with ``__call__``, ``flux``, ``errors``, ``limit``,
            and ``ts`` attributes in physical Norm units.

        Raises
        ------
        ValueError
            If no ``Norm`` parameter exists in the source's spectral model.
        """
        pkg = __package__ if __package__ else 'like3'
        PoissonFitter = importlib.import_module(f'{pkg}.loglikelihood').PoissonFitter

        source = self.source_model.find_source(source_name)
        model = source.model
        best_norm = float(model.getp('Norm'))
        norm_floor = 1e-30
        safe_best = max(best_norm, norm_floor)

        def _set_norm(norm):
            model.setp('Norm', max(float(norm), norm_floor))
            source.changed = True

        def _loglike_x(x):
            # x is log(norm / norm_floor), constrained to x >= 0 by PoissonFitter.
            x = max(float(np.asarray(x, dtype=float).reshape(-1)[0]), 0.0)
            norm = norm_floor * np.exp(np.clip(x, 0.0, 700.0))
            _set_norm(norm)
            return self.log_like()

        # Use local curvature around the current best value to set a linear
        # scale so the PoissonFitter variable has O(1) width.
        x0 = float(np.log(safe_best / norm_floor))
        h = 1e-3
        try:
            f0 = _loglike_x(x0)
            fp = _loglike_x(x0 + h)
            fm = _loglike_x(max(x0 - h, 0.0))
            curvature = max(2.0 * f0 - fp - fm, 0.0) / (h * h)
            sigma_x = 1.0 / np.sqrt(curvature) if curvature > 0 else 1.0
            sigma_x = float(np.clip(sigma_x, 1e-4, 1e4))
            y0 = x0 / sigma_x

            def _loglike_y(y):
                return _loglike_x(y * sigma_x)

            try:
                pf = PoissonFitter(_loglike_y, scale=max(y0, 1.0), tol=tol)
            except Exception:
                # Some low-information or strongly non-Poisson-like profiles can
                # exceed the strict maxdev test; retry with a looser tolerance.
                pf = PoissonFitter(_loglike_y, scale=max(y0, 1.0), tol=max(1.0, 2.0 * tol))
            poiss_y = pf.poiss
        finally:
            _set_norm(best_norm)

        class _LogNormProfile:
            def __init__(self, poiss, sigma, floor):
                self._poiss = poiss
                self._sigma = float(sigma)
                self._floor = float(floor)

            def __str__(self):
                flux = self.flux
                lo, hi = self.errors
                limit = self.limit
                if flux > 0:
                    return (
                        f'LogNormProfile(flux={flux:.4g}, '
                        f'errors=({lo:.4g}, {hi:.4g}), '
                        f'ts={self.ts:.2f})'
                    )
                return f'LogNormProfile(flux=0, limit95={limit:.4g}, ts={self.ts:.2f})'

            __repr__ = __str__

            def _norm_to_y(self, norm):
                n = np.clip(np.asarray(norm, dtype=float), self._floor, np.inf)
                x = np.log(n / self._floor)
                return x / self._sigma

            def __call__(self, norm):
                y = self._norm_to_y(norm)
                if np.ndim(y) == 0:
                    return float(self._poiss(float(y)))
                return np.asarray(self._poiss(y), dtype=float)

            @property
            def flux(self):
                y_peak = max(self._poiss.flux, 0.0)
                return float(self._floor * np.exp(np.clip(y_peak * self._sigma, 0.0, 700.0)))

            @property
            def errors(self):
                y_lo, y_hi = self._poiss.errors
                lo = self._floor * np.exp(np.clip(y_lo * self._sigma, 0.0, 700.0))
                hi = self._floor * np.exp(np.clip(y_hi * self._sigma, 0.0, 700.0))
                return (float(lo), float(hi))

            @property
            def limit(self):
                y_lim = self._poiss.limit if hasattr(self._poiss, 'limit') else self._poiss.percentile(0.95)
                return float(self._floor * np.exp(np.clip(y_lim * self._sigma, 0.0, 700.0)))

            @property
            def ts(self):
                return float(self._poiss.ts)

        return _LogNormProfile(poiss_y, sigma_x, norm_floor)

    def zea_plot(self, what):
        """
        """
        raise NotImplementedError('ZEA plotting not yet implemented for MultiBandLikelihood')


class BandLikelihood(HEALPix):
    """ For a given band and source model, select active pixels from the SourceModel, 
        evaluate PSF responses for those pixels,
        and provide the likelihood function."""

    def __init__(self, band, source_model):
        self.band = band
        self.source_model = source_model    
        self.psf = PSFlookup()(band)
        super().__init__(nside=band.nside, order=band.order, frame=band.frame)
        self.center = source_model[0].skydir if len(source_model) > 0 else SkyCoord(0, 0, unit='deg')
        self.coverage = None  # populated on demand by build_coverage()
        self.empty_coverage = False
        self.psf_cache = {}  # populated on demand by response()
        self.build_coverage() if source_model else None


    def __repr__(self):
        return f'BandLikelihood(band={self.band}, source_model={self.source_model})'
    
    @property
    def parameter_names(self):
        return self.source_model.parameter_names
        
    @property
    def bounds(self):
        return self.source_model.bounds 

    #======= Parameter accessors with coverage update ++++++++++++++++++++++
    @property
    def parameters(self):
        return self.source_model.parameters.get_parameters()
    
    @parameters.setter
    def parameters(self, pars):
        if pars is None:
            return
        if len(pars) != len(self.parameters):
            raise ValueError(f'Expected {len(self.source_model.parameters)} parameters, got {len(pars)}')
        self.source_model.parameters.set_parameters(pars)
        self.evaluate_source_model()
    #============++++++++++++++++++++++

    def build_coverage(self, r68_radius: float = 4.0) -> None:
        """Build and cache a coverage DataFrame for pixels to the source footprint."""
        import pandas as pd
        radius_deg = r68_radius * self.band.psf.r68 if self.band.psf is not None else 2.0
        mask = np.zeros(len(self.band.pix), dtype=bool)
        for src in self.source_model:
            mask |= self._coverage_mask(src.skydir, radius_deg)
        pix = self.band.pix[mask]
        photons = self.band.photons[mask]
        diffuse_counts = self.band.diffuse_counts[mask]
        source_counts = self.band.source_counts[mask]
        
        self.coverage = pd.DataFrame(dict(
            pix=pix,
            photons=photons,
            diffuse_counts=diffuse_counts,
            source_counts=source_counts,
            background_counts=diffuse_counts + source_counts,
            exposure=self.band.exposure_map(pix).astype(np.float32),
        ))
        self.empty_coverage = len(self.coverage) == 0
        if len(self.coverage) == 0:
            self.coverage['model_counts'] = np.array([], dtype=float)
            warnings.warn(self._coverage_error_message('build coverage'), stacklevel=2)
            return
        self.evaluate_source_model()

    def _normalize_skydir(self, center):
        """Return a SkyCoord-compatible center from either SkyCoord or legacy SkyDir."""
        return center.coord if hasattr(center, 'coord') else center

    def _coverage_padding_deg(self):
        """Approximate half-width used to include coarse pixels that overlap a search cone."""
        resolution = getattr(self.band, 'pixel_resolution', None)
        if resolution is None:
            return 0.0
        if hasattr(resolution, 'to_value'):
            return float(resolution.to_value('deg'))
        return float(resolution)

    def _coverage_mask(self, center, radius_deg):
        """Return a sparse-pixel mask for source coverage using an overlap-aware cone search."""
        center = self._normalize_skydir(center)
        padded_radius = radius_deg + self._coverage_padding_deg()
        if hasattr(self.band, 'cone_search_skycoord'):
            cone_pix = np.asarray(
                self.band.cone_search_skycoord(center, Angle(padded_radius, 'deg')),
                dtype=np.int64,
            )
            if len(cone_pix) > 0:
                return np.isin(self.band.pix, cone_pix)
        return self.band.cone_search(center, padded_radius)

    def _coverage_error_message(self, action):
        """Return a diagnostic string describing an empty coverage selection."""
        names = [getattr(src, 'name', str(src)) for src in self.source_model]
        src_text = ', '.join(names) if names else '<no sources>'
        return (
            f'Band {self.band.key} ({self.band.energy:.0f} MeV) has empty coverage; '
            f'cannot {action}. This usually means the selected source footprint '
            f'did not intersect any sparse pixels in the band. Sources: {src_text}'
        )

    def _require_coverage(self, action):
        """Return whether coverage is usable, tracking empty coverage on the instance."""
        if self.coverage is None:
            raise RuntimeError(f'Band {self.band.key} has no coverage table; cannot {action}.')
        if len(self.coverage) == 0:
            self.empty_coverage = True
            return False
        self.empty_coverage = False
        return True

    def _free_parameter_count(self):
        """Return the number of free source-model parameters for empty-coverage fallbacks."""
        return int(sum(np.count_nonzero(src.model.free) for src in self.source_model))

    def response(self, source, pixels=None,*, ignore_cache_for=None):
        """Return PSF response for a source evaluated on given pixel indices.
        If *pixels* is None, the coverage pixels are used.  
        The PSF response is cached per source for efficiency.
         Parameters
        ----------
        source : object
            The source for which to compute the PSF response.
        pixels : array-like, optional
            The pixel indices on which to evaluate the PSF response. If None, the coverage pixels are used.
        ignore_cache_for : str, optional
            If the cache contains an entry for this source, it will be ignored and recomputed. 
            This is useful when the source position has changed and the PSF response needs to be updated.

        Returns
        -------
        np.ndarray
            The PSF response values for the specified pixels.
        """
        if source is None:
            cpix = np.asarray([], dtype=np.int64)
            return cpix, np.asarray([], dtype=float)

        source_name = source.name if hasattr(source, 'name') else str(source)
        cache = self.psf_cache
        if source_name in cache and ignore_cache_for != source_name:
            return cache[source_name]
        # Compute and cache the PSF list for this source
        sdir = source.skydir
        sdir = sdir.coord if hasattr(sdir, 'coord') else sdir

        if pixels is None:
            cpix = self.coverage['pix'].to_numpy() if self.coverage is not None else np.asarray([], dtype=np.int64)
        else:
            cpix = np.asarray(pixels, dtype=np.int64)
        aa = sdir.separation(self.healpix_to_skycoord(cpix)).deg
        psf = self.psf
        vpix = np.array(list(map(psf, aa)), dtype=float) * self.pixel_area.value
        cache[source_name] = vpix
        return vpix

    def evaluate_source_model(self, pix=None):
        """Evaluate the source model counts for coverage pixels."""
        if pix is None:
            if self.coverage is not None:
                if not self._require_coverage('evaluate source model'):
                    self.coverage['model_counts'] = np.array([], dtype=float)
                    return
                pix = self.coverage['pix'].to_numpy()
            else:
                pix = self.band.pix
        counts = np.zeros(len(pix), dtype=float)
        exp = self.coverage['exposure'].to_numpy() #if self.coverage is not None else self.band.exposure_map(pix)
        for src in self.source_model:
            flux = src.model(self.band.energy)
            v = self.response(src, pix)
            counts += v * flux
        counts *= exp
        self.coverage['model_counts'] = counts
    
    def update_position(self, source, new_skydir=None):
        """Update the position of a source and invalidate the PSF cache for that source.
        Parameters
        ----------
        source : object
            The source for which to update the position.
        new_skydir : SkyCoord or tuple, optional
            The new sky coordinates for the source. If None, the position is not updated but the pixels are.
        """
  
        if new_skydir is not None:
            source.skydir = new_skydir
        self.psf_cache.pop(source.name, None)  # Invalidate cache for this source
        self.evaluate_source_model()  # Update model counts based on new position

        
    def log_like(self, pars=None, skydir=None):
        """Compute the log-likelihood for the current model parameters.
        If `pars` is provided, update the source model parameters before computing.
        If `skydir` is provided, evaluate the model with the selected source at that position .
            `pars` must be None in this case
         """
        
        if skydir is not None:
            if pars is not None:
                raise ValueError('skydir-based loglike evaluation ignores pars')
            if isinstance(skydir, SkyCoord):
                pass
            elif isinstance(skydir, tuple):
                skydir = SkyCoord(*skydir, unit='deg', frame='icrs')
            src = self.source_model.selected_source
            if src is None:
                raise ValueError('No source is selected for skydir-based loglike evaluation')
            # src.skydir = skydir # need check for skydir attribute and type conversion here

            self.update_position(src, skydir)
        
        elif pars is not None:
            self.parameters = pars

        else:
            # case where neither pars nor skydir is provided: just evaluate with current parameters and position,
            # which may be needed to populate model counts in coverage if parameters were updated externally
            self.evaluate_source_model()

        # Data and model (diffuse + other sources + active source counts for coverage pixels)
        if not self._require_coverage('evaluate log-likelihood'):
            return 0.0
        cov = self.coverage
        data = cov['photons'].to_numpy()
        # model = cov['model_counts'].to_numpy() # temporary for bright source testing!
        model = (cov.model_counts.array + cov.background_counts.array) 
        

        # Poisson log-likelihood (ignoring constant term)
        ll = np.sum(data * np.log(model + 1e-12) - model)
        return float(ll)

    def pixel_gradient(self):
        """Evaluate per-pixel count gradients for free source-model parameters.

        Returns
        -------
        np.ndarray
            Gradient matrix of shape (n_pixels, n_free_parameters)."""
        if not self._require_coverage('evaluate pixel gradients'):
            return np.zeros((0, self._free_parameter_count()), dtype=float)
        g = []
        keys = self.coverage['pix'].to_numpy() 
        for src in self.source_model:
            grad = src.model.gradient(self.band.energy)[src.model.free]
            v = self.response(src)
            g.append(v[:, None] * grad[None, :])
        g_arr = np.hstack(g) if g else np.zeros((len(keys), 0))
        # g_arr *= self.band.exposure_map(keys)[:, None]
        g_arr *= self.coverage['exposure'].to_numpy()[:, None] #if self.coverage is not None else self.band.exposure_map(keys)[:, None]
        return g_arr

    def loglike_grad(self, pars=None):
        """Compute the log-likelihood and its gradient for the current model parameters.
        If `pars` is provided, update the source model parameters before computing."""
        if pars is not None:
            self.parameters = pars
        if not self._require_coverage('evaluate log-likelihood gradient'):
            return 0.0, np.zeros(self._free_parameter_count(), dtype=float)
        cov = self.coverage
        data = cov['photons'].to_numpy()
        model = cov['model_counts'].to_numpy()
        grad_matrix = np.zeros((len(model), 0))
        if hasattr(self, 'pixel_gradient') and callable(self.pixel_gradient):
            grad_matrix = self.pixel_gradient()
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(model > 0, data / model, 0.0)
        grad = np.sum((ratio[:, None] - 1) * grad_matrix, axis=0) if grad_matrix.shape[1] > 0 else np.zeros(0)
        ll = np.sum(data * np.log(model + 1e-12) - model)
        return float(ll), grad

    def expand_healpix_array(self,arr):
            """Expand a per-pixel array to a full HEALPix array."""

            nside = getattr(self, 'nside', self.band.nside)
            npix = 12 * nside**2

            if len(arr) == npix:
                return arr
            
            if len(arr) == len(self.coverage):
                hpa = np.full(npix, np.nan, dtype=float)
                hpa[self.coverage.pix] = arr
                return hpa
            raise ValueError(f'Cannot expand array of length {len(arr)} to HEALPix array of length {npix}')

    def _plot_component_values(self, component):
        """Return a full HEALPix array for a coverage component or numeric array."""
        if self.coverage is None:
            raise RuntimeError('BandLikelihood has no coverage table for plotting')

        if isinstance(component, str):
            component = {'residual': 'resid'}.get(component, component)
            if component == 'data':
                arr = self.coverage['photons'].to_numpy(dtype=float)
            elif component == 'diffuse':
                arr = self.coverage['diffuse_counts'].to_numpy(dtype=float)
            elif component == 'sources':
                arr = self.coverage['source_counts'].to_numpy(dtype=float)
            elif component == 'model':
                arr = self.coverage['model_counts'].to_numpy(dtype=float)
            elif component == 'resid':
                arr = self.residual
            elif component == 'sigma':
                arr = self.sigma
            elif component == 'exposure':
                arr = self.coverage['exposure'].to_numpy(dtype=float)
            elif component in self.coverage:
                arr = self.coverage[component].to_numpy(dtype=float)
            else:
                raise ValueError(f'Unknown coverage component: {component!r}')
        else:
            arr = np.asarray(component, dtype=float)

        return self.expand_healpix_array(arr)

    def _default_plot_log(self, component, log):
        """Choose a plotting log-scale default appropriate for the component."""
        if log is not None:
            return log
        if isinstance(component, str) and component in {'resid', 'residual', 'sigma'}:
            return False
        return True

    def _plot_center(self, center=None):
        """Resolve a plotting center from explicit input or the first/selected source."""
        if center is not None:
            return center.coord if hasattr(center, 'coord') else center

        source_model = self.source_model
        selected = getattr(source_model, 'selected_source', None)
        if selected is not None:
            return selected.skydir.coord if hasattr(selected.skydir, 'coord') else selected.skydir

        if len(source_model) > 0:
            skydir = source_model[0].skydir
            return skydir.coord if hasattr(skydir, 'coord') else skydir

        raise ValueError('No source is available for plotting; pass center explicitly')

    @property
    def residual(self):
        """Compute the residual counts (data - model) for coverage pixels."""
        model_key = 'background_counts' if 'background_counts' in self.coverage else 'model_counts'
        return self.coverage['photons'].to_numpy() - self.coverage[model_key].to_numpy()
    
    @property
    def sigma(self):
        """Compute residual in (approximate) sigma units for coverage pixels."""
        model_key = 'background_counts' if 'background_counts' in self.coverage else 'model_counts'
        model = self.coverage[model_key].to_numpy()
        data = self.coverage['photons'].to_numpy()
        return np.where(model > 0, (data - model) / np.sqrt(model), 0.0)

    def ait_plot(self, component='data', *, figsize=(12, 6), fig=None, colorbar=True,
                 label='counts/pixel', title=None, shrink=0.7, cmap='viridis',
                 log=None, **kwargs):
        """Render an all-sky AIT projection for a BandLikelihood coverage component."""
        from matplotlib.colors import LogNorm, Normalize
        from utilities.skymaps import AITfigure

        log = self._default_plot_log(component, log)
        mp = self._plot_component_values(component)
        if log:
            mp = mp.copy()
            mp[mp == 0] = np.nan

        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        norm_fn = LogNorm if log else Normalize

        afig = AITfigure(fig=fig, figsize=figsize, title=title)
        afig.imshow(mp, norm=norm_fn(vmin=vmin, vmax=vmax), cmap=cmap, **kwargs)
        if colorbar:
            afig.colorbar(label=label, shrink=shrink)
        return afig

    def zea_plot(self, component='data', center=None, *, figsize=(6, 5), pixelsize=None,
                 size=None, fig=None, axes_visible=True, cmap='viridis', colorbar=True,
                 title=None, label='counts/pixel', log=None, vmin=None, vmax=None,
                 frame='galactic', **kwargs):
        """Render a local ZEA projection for a BandLikelihood coverage component."""
        from matplotlib.patches import Circle
        from utilities.skymaps import ZEAfigure

        center = self._plot_center(center)
        log = self._default_plot_log(component, log)
        psf = self.psf
        if psf is not None:
            size = size if size is not None else 16 * psf.r68
            pixelsize = pixelsize if pixelsize is not None else psf.r68 / 50
        else:
            size = size if size is not None else 5
            pixelsize = pixelsize if pixelsize is not None else 0.05

        zfig = ZEAfigure(
            center,
            size=size,
            fig=fig,
            figsize=figsize,
            frame=frame,
            pixelsize=pixelsize,
            axes_visible=axes_visible,
            title='' if title is None else title,
        )

        if component is not None:
            mp = self._plot_component_values(component)
            mp = mp.copy()
            mp[mp == 0] = np.nan
            zfig.imshow(mp, log=log, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
            if colorbar:
                zfig.colorbar(label=label, shrink=0.9, extend='max')

        band_label = getattr(self.band, 'psf_name', None)
        if band_label is None:
            event_type = getattr(self.psf, 'event_type', None)
            band_label = f'PSF{event_type - 2}' if event_type is not None and event_type >= 2 else 'Band'
        zfig.axes_text(
            0.98,
            0.98,
            f'{self.band.energy / 1e3:.2f} GeV\n{band_label}',
            color='white',
            ha='right',
            va='top',
            fontsize=12,
        )

        if psf is not None:
            ax = zfig.ax
            r68_px = psf.r68 / pixelsize
            cx, cy = (ax.transAxes + ax.transData.inverted()).transform((0.12, 0.12))
            ax.add_patch(Circle((cx, cy), r68_px, fill=False, edgecolor='white', linewidth=1.5))

        return zfig

def gradient_check(bl,  eps=1e-3):
    """Compare analytic gradient from grad_fn to numerical gradient of loglike_fn at pars."""

    def numerical_gradient(loglike_fn, pars):
        """Compute numerical gradient of loglike_fn at pars using central differences."""
        grad = np.zeros_like(pars, dtype=float)
        for i in range(len(pars)):
            p_hi = np.array(pars, dtype=float)
            p_lo = np.array(pars, dtype=float)
            p_hi[i] += eps
            p_lo[i] -= eps
            f_hi = loglike_fn(p_hi)
            f_lo = loglike_fn(p_lo)
            grad[i] = (f_hi - f_lo) / (2 * eps)
        return grad
    
    pars0 = bl.parameters# if hasattr(bl.source_model, 'parameters') else np.concatenate([src.model.parameters[src.model.free] for src in bl.source_model])
    ll, analytic_grad = bl.loglike_grad(pars0)
    num_grad = numerical_gradient(bl.loglike, pars0)
    bl.parameters = pars0

    print(f'Analytic gradient: {analytic_grad.round().astype(int)}')
    print(f'Numerical gradient: {num_grad.round().astype(int)}')
    print(f'Difference: {(analytic_grad - num_grad).round(1)}')

class FermiFit(views.LikelihoodViews):
    """Fitting engine that wraps a PixelTable and its attached SourceModel.

    Parameters
    ----------
    pixel_table : like3.pixel_table.PixelTable
        A loaded pixel table with a ``source_model`` attribute set.
    """

    def __init__(self, pixel_table, verbose=False):
        if pixel_table.source_model is None:
            raise ValueError('FermiFit requires a PixelTable with a source_model attached')
        super().__init__(pixel_table, verbose=verbose)
        self.pixel_table = pixel_table
        self.fermi_catalog = getattr(pixel_table.source_model, 'fermi_catalog', None)
        self.fit_info: dict = {}

    # ------------------------------------------------------------------
    # Parameter accessors
    # ------------------------------------------------------------------

    @property
    def source_model(self):
        return self.pixel_table.source_model

    @property
    def parameters(self):
        """Free-parameter set of the attached source model."""
        return self.source_model.parameters
    @parameters.setter
    def parameters(self, pars):
        """Set free parameters of the attached source model."""
        if pars is None:
            return
        if len(pars) != len(self.parameters):
            raise ValueError(f'Expected {len(self.source_model.parameters)} parameters, got {len(pars)}')
        self.source_model.parameters.set_parameters(pars)


    @property
    def parameter_names(self):
        """Names of the free parameters."""
        return self.source_model.parameter_names

    @property
    def bounds(self):
        """Fitter-space parameter bounds."""
        return self.source_model.bounds

    # def get_sed(self, source_name=None, event_type=None, update=False, tol=0.2):
    #     """ return the SED recarray for the source, including npred info
    #     source_name : string
    #         Name of a source in the ROI, with possible wildcards
    #     event_type : None, or integer, 0/1 for front/back, 2-5 for psf0-3
    #     update : bool
    #         set True to force recalculation of sed recarray
    #     """
        
    #     source = self.source_model.find_source(source_name)
    #     if not hasattr(source, 'sedrec') or source.sedrec is None\
    #              or (update and np.any(source.model.free)):
    #         pkg = __package__ if __package__ else 'like3'
    #         sedfuns = importlib.import_module(f'{pkg}.sedfuns')
    #         with sedfuns.SED(self, source.name) as sf:
    #             source.sedrec = sf.sed_rec(event_type=event_type, tol=tol)
        
    #     return source.sedrec

    def get_sed_poisson_table(self, source_name=None, event_type=None, tol=0.2):
        """Return an SED table with one Poisson object per energy bin.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector forwarded to ``SourceModel.find_source``.
        event_type : None, int, or str
            Event-type selection forwarded to ``sedfuns.sed_poisson_table``.
        tol : float
            Fit-quality tolerance forwarded to ``PoissonFitter``.

        Returns
        -------
        pandas.DataFrame
            Per-band SED table containing a ``poiss`` column with
            ``like3.loglikelihood.Poisson`` entries.
        """
        source = self.source_model.find_source(source_name)
        pkg = __package__ if __package__ else 'like3'
        sedfuns = importlib.import_module(f'{pkg}.sedfuns')
        return sedfuns.sed_poisson_table(
            self,
            source_name=source.name,
            event_type=event_type,
            tol=tol,
        )

    def selected_source_energy_flux_view(self, energy=None, **kw):
        """Return an energy-flux likelihood view for the selected source.

        Parameters
        ----------
        energy : float or None, optional
            Evaluation energy in MeV. If ``None``, the source model reference
            energy is used.
        **kw
            Forwarded to :meth:`energy_flux_view`.

        Returns
        -------
        like3.views.EnergyFluxView
            Callable object that maps energy flux to negative log-likelihood.

        Raises
        ------
        ValueError
            If no source is currently selected.
        """
        src = self.source_model.selected_source
        if src is None:
            raise ValueError('No source is selected')
        return self.energy_flux_view(src.name, energy=energy, **kw)

    def plot_sed_with_band_points(
        self,
        source=None,
        *,
        sed_table=None,
        set_kwargs=None,
        ax=None,
        update=False,
        event_type=None,
        tol=0.2,
        emin=100,
        emax=1e5,
        xlim=None,
        ylim=(0.1, None),
        npts=100,
        model_label=None,
        points_label='Per-band SED',
        title=None,
        show_upper_limits=True,
    ):
        """Plot source SED model with per-band errorbar points.

        Parameters
        ----------
        source : Source, str, or None, optional
            Source selector forwarded to ``SourceModel.find_source``.
            When ``None`` (default), the currently selected source is used.
        sed_table : pandas.DataFrame or None, optional
            Precomputed SED Poisson table containing ``elow``, ``ehigh``,
            ``flux``, ``lflux``, and ``uflux`` columns. When provided,
            this table is used directly and no call to
            :meth:`get_sed_poisson_table` is made.
        set_kwargs : dict or None, optional
            Keyword arguments forwarded to ``Axes.set``. These values override
            the default axis settings used by this method.
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw into. A new figure is created when ``None``.
        update : bool, optional
            Force regeneration of the source SED Poisson table when
            ``sed_table`` is not provided. Default ``False``.
        event_type : None, int, or str, optional
            Event-type selection forwarded to :meth:`get_sed`.
        tol : float, optional
            Poisson-fit tolerance forwarded to :meth:`get_sed`.
        emin, emax : float, optional
            Model-curve plotting range in MeV.
        xlim : tuple[float, float] or None, optional
            X-axis limits in MeV. Defaults to ``(emin, emax)``.
        npts : int, optional
            Number of model-curve points.
        model_label : str or None, optional
            Legend label for the model curve. Defaults to source name.
        points_label : str, optional
            Legend label for the binned points.
        title : str or None, optional
            Plot title. Defaults to source name.
        show_upper_limits : bool, optional
            If True, plot upper-limit markers for bins with ``flux <= 0``.

        Returns
        -------
        matplotlib.axes.Axes
            Axes with model SED and per-band points.
        """
        import matplotlib.pyplot as plt

        src = self.source_model.find_source(source)
        if sed_table is None:
            if not update and hasattr(src, 'sedrec'):
                sed_table = src.sedrec
            else:
                sed_table = self.get_sed_poisson_table(
                    source_name=src.name,
                    event_type=event_type,
                    tol=tol,
                )
                src.sedrec = sed_table

        if sed_table is None:
            raise ValueError(f'No SED table available for source {src.name}')

        fields = set(getattr(sed_table, 'columns', ()))
        needed = {'elow', 'ehigh', 'flux', 'lflux', 'uflux'}
        missing = needed - fields
        if missing:
            raise ValueError(f'sed table missing required fields: {sorted(missing)}')

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        if xlim is None:
            xlim = (emin, emax)

        # Plot model SED directly in eV cm^-2 s^-1 units.
        model = src.model
        energies = np.logspace(np.log10(emin), np.log10(emax), npts)
        dnde = model(energies)  # ph cm^-2 s^-1 MeV^-1
        e2dnde_ev = energies**2 * dnde * 1e6
        ax.loglog(
            energies,
            e2dnde_ev,
            label=src.name.strip() if model_label is None else model_label,
        )

        if model.has_errors():
            g = model.external_gradient(energies)
            cov = model.get_cov_matrix()
            var_dnde = np.sum((cov @ g) * g, axis=0)
            var_dnde = np.clip(var_dnde, 0, None)
            sigma_e2dnde_ev = energies**2 * np.sqrt(var_dnde) * 1e6
            ax.fill_between(
                energies,
                e2dnde_ev - sigma_e2dnde_ev,
                e2dnde_ev + sigma_e2dnde_ev,
                alpha=0.3,
            )

        elow = np.asarray(sed_table['elow'], dtype=float)
        ehigh = np.asarray(sed_table['ehigh'], dtype=float)
        flux = np.asarray(sed_table['flux'], dtype=float)
        lflux = np.asarray(sed_table['lflux'], dtype=float)
        uflux = np.asarray(sed_table['uflux'], dtype=float)

        ecent = np.sqrt(elow * ehigh)
        xerr = np.vstack([
            np.clip(ecent - elow, 0, np.inf),
            np.clip(ehigh - ecent, 0, np.inf),
        ])
        # sed_table fluxes are energy-flux values from EnergyFluxView (eV units).
        y = flux
        ylo = lflux
        yhi = uflux

        det_mask = np.isfinite(y) & np.isfinite(ylo) & np.isfinite(yhi) & (flux > 0)
        if np.any(det_mask):
            yerr = np.vstack([
                np.clip(y[det_mask] - ylo[det_mask], 0, np.inf),
                np.clip(yhi[det_mask] - y[det_mask], 0, np.inf),
            ])
            ax.errorbar(
                ecent[det_mask],
                y[det_mask],
                xerr=xerr[:, det_mask],
                yerr=yerr,
                fmt='o',
                ms=5,
                capsize=2,
                lw=1,
                color='tab:orange',
                label=points_label,
            )

        if show_upper_limits:
            ul_mask = np.isfinite(uflux) & (flux <= 0)
            if np.any(ul_mask):
                y_ul = uflux[ul_mask]
                yerr_ul = 0.35 * np.clip(y_ul, 0, np.inf)
                ax.errorbar(
                    ecent[ul_mask],
                    y_ul,
                    xerr=xerr[:, ul_mask],
                    yerr=yerr_ul,
                    uplims=True,
                    fmt='v',
                    ms=4,
                    lw=1,
                    color='tab:red',
                    label='95% UL',
                )

        defaults = dict(
            xlabel='Energy (MeV)',
            ylabel=r'$E^2\,dN/dE\ [\mathrm{eV\,cm^{-2}\,s^{-1}}]$',
            title=src.name.strip() if title is None else title,
            xlim=xlim,
            ylim=ylim,
            xscale='log',
            yscale='log',
        )
        if set_kwargs is not None:
            defaults.update(set_kwargs)
        ax.set(**defaults)

        ax.grid(True, which='both', alpha=0.25)
        ax.legend()
        return ax

    def sed_poisson_delta_decomposition(
        self,
        source,
        *,
        alpha=1.05,
        sed_table=None,
        sed_table_attr='sed_poisson_table',
        update=False,
        event_type=None,
        tol=0.2,
    ):
        """Compare per-band Poisson delta sum to direct all-band delta.

        This evaluates each SED-bin Poisson object at two flux points:
        the model-predicted reference flux and a perturbed flux scaled by
        ``alpha``. It then compares the sum of per-bin Poisson deltas to the
        direct all-band log-likelihood delta under the same normalization shift.

        Parameters
        ----------
        source : Source
            Source object from the attached ``SourceModel``.
            The source is expected to carry a SED Poisson table on
            ``getattr(source, sed_table_attr)``. If not present (or when
            ``update=True``), one is generated and attached.
        alpha : float, optional
            Multiplicative factor applied to the source ``Norm`` parameter for
            the perturbed state. Default is ``1.05``.
        sed_table : pandas.DataFrame or None, optional
            Precomputed SED Poisson table. When provided, this table is used
            directly and no call to ``get_sed_poisson_table`` is made.
        sed_table_attr : str, optional
            Attribute name on ``source`` used to store/read the SED Poisson
            table. Default is ``'sed_poisson_table'``.
        update : bool, optional
            Force regeneration of the source SED Poisson table when
            ``sed_table`` is not provided. Default ``False``.
        event_type : int, str, or None, optional
            Event-type selection passed to ``get_sed_poisson_table`` when
            generating the table.
        tol : float, optional
            Poisson-fit tolerance passed when generating the table.

        Returns
        -------
        dict
            Dictionary containing:
            - ``per_band``: DataFrame with per-bin fluxes and Poisson deltas
            - ``sum_poisson_delta``: sum of per-bin Poisson deltas
            - ``all_band_delta``: direct all-band delta
            - ``difference``: ``sum_poisson_delta - all_band_delta``
            - ``alpha``: the input perturbation scale
            - ``source_name``: source name used
        """
        import pandas as pd

        if not hasattr(source, 'name') or not hasattr(source, 'model'):
            raise TypeError('source must be a Source-like object with name and model attributes')

        source_name = source.name
        if sed_table is None:
            if update or not hasattr(source, sed_table_attr) or getattr(source, sed_table_attr) is None:
                sed_table = self.get_sed_poisson_table(
                    source_name=source_name,
                    event_type=event_type,
                    tol=tol,
                )
                setattr(source, sed_table_attr, sed_table)
            else:
                sed_table = getattr(source, sed_table_attr)

        if not isinstance(sed_table, pd.DataFrame):
            raise TypeError(f'source.{sed_table_attr} must be a pandas.DataFrame')
        needed = {'elow', 'ehigh', 'poiss'}
        missing = needed - set(sed_table.columns)
        if missing:
            raise ValueError(
                f'source.{sed_table_attr} missing required columns: {sorted(missing)}'
            )

        valid = sed_table[sed_table.poiss.notna()].copy()
        if len(valid) == 0:
            raise ValueError('No valid Poisson entries found in source SED table')

        pt = self.pixel_table
        src = self.source_model.find_source(source_name)
        norm0 = float(src.model.getp('Norm'))
        saved_keys = pt._selected
        efv = self.energy_flux_view(source_name, bound=-20)
        rows = []

        try:
            for energy_idx, row in valid.sort_index().iterrows():
                elow, ehigh = float(row.elow), float(row.ehigh)
                chosen = [
                    b for b in pt.values()
                    if float(b.e0) >= elow and float(b.e1) <= ehigh
                ]
                if len(chosen) == 0:
                    continue

                pt.select(keys=[b.key for b in chosen])
                efv.set_energy(np.sqrt(elow * ehigh))

                flux_ref = float(efv.eflux)
                flux_test = alpha * flux_ref
                dll_poiss = float(row.poiss(flux_test) - row.poiss(flux_ref))

                rows.append(dict(
                    energy=energy_idx,
                    elow=elow,
                    ehigh=ehigh,
                    flux_ref=flux_ref,
                    flux_test=flux_test,
                    dll_poiss=dll_poiss,
                ))

            per_band = pd.DataFrame(rows)
            sum_poisson_delta = float(per_band.dll_poiss.sum()) if len(per_band) else 0.0

            # Direct all-band delta for the same normalization perturbation.
            pt.select(keys=saved_keys)
            src.model.setp('Norm', norm0)
            src.changed = True
            ll_ref_all = float(self.log_like())

            src.model.setp('Norm', alpha * norm0)
            src.changed = True
            ll_test_all = float(self.log_like())

            all_band_delta = ll_test_all - ll_ref_all
        finally:
            src.model.setp('Norm', norm0)
            src.changed = True
            pt.select(keys=saved_keys)
            if hasattr(efv, 'restore'):
                efv.restore()

        return dict(
            per_band=per_band,
            sum_poisson_delta=sum_poisson_delta,
            all_band_delta=float(all_band_delta),
            difference=float(sum_poisson_delta - all_band_delta),
            alpha=float(alpha),
            source_name=source_name,
        )


    # ------------------------------------------------------------------
    # Freeze / thaw
    # ------------------------------------------------------------------

    def freeze(self, param, source_name=None, ):
        """Freeze one or all parameters of a source's spectral model.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector forwarded to ``SourceModel.find_source``.
        param : str, int, or None, optional
            Parameter name or index to freeze.  When ``None`` all parameters
            of the source's model are frozen.
        """
        src = self.source_model.find_source(source_name)
        if param is None:
            src.model.free[:] = False
        else:
            src.model.freeze(param)
        self.source_model.reinitialize()  # Ensure parameter set is updated after thawing.

    def thaw(self,  param,source_name=None,):
        """Thaw (unfreeze) one or all parameters of a source's spectral model.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector forwarded to ``SourceModel.find_source``.
        param : str, int, or None, optional
            Parameter name or index to thaw.  When ``None`` all parameters
            of the source's model are thawed.
        """
        src = self.source_model.find_source(source_name)
        if param is None:
            src.model.free[:] = True
        else:
            src.model.thaw(param)
        self.source_model.reinitialize()  # Ensure parameter set is updated after thawing.

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    def preserve_parameters(self, restore=True):
        """Context manager that restores free-parameter values on exit.

        Parameters
        ----------
        restore : bool, optional
            Initial value of the restore flag.  When ``True`` (default) the
            saved parameter values are restored on exit.  Can be changed at
            any point inside the ``with`` block via the yielded flag object::

                with ff.preserve_parameters() as ctx:
                    ff.fit()
                    ctx.restore = False   # keep fitted values on exit
        """
        class _Ctx:
            def __init__(self, restore):
                self.restore = restore

        @contextmanager
        def _ctx():
            pset = self.parameters
            saved = np.array(pset.get_parameters(), copy=True)
            ctx = _Ctx(restore)
            try:
                yield ctx
            finally:
                if ctx.restore:
                    pset.set_parameters(saved)
        return _ctx()

    def preserve_position(self):
        """Context manager that restores the selected source's sky position on exit."""
        @contextmanager
        def _ctx():
            src = self.source_model.selected_source
            if src is None:
                raise ValueError('preserve_position requires a selected source')
            saved = src.skydir
            try:
                yield
            finally:
                src.skydir = saved
        return _ctx()

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def log_like(self, skydir=None, summed=True):
        """Total Poisson log-likelihood summed over all selected bands.

        Parameters
        ----------
        skydir : SkyCoord or None, optional
            Trial sky position forwarded to each band's ``loglike`` call.
        summed : bool, optional
            If True (default), return the sum of log-likelihoods over all bands.
            If False, return an array of log-likelihoods for each band.

        Returns
        -------
        float or np.ndarray
        """
        pt = self.pixel_table
        if summed:
            return float(sum(band.log_like(skydir=skydir) for band in pt._iter_bands()))
        else:
            return np.array([band.log_like(skydir=skydir) for band in pt._iter_bands()])

    def gradient(self, pars=None):
        """Analytic gradient of the total log-likelihood for free parameters."""
        if pars is not None:
            self.parameters.set_parameters(pars)
            self.update()

        n_all = len(self.parameters.parameter_names)
        full_grad = np.zeros(n_all, dtype=float)

        for band in self.pixel_table._iter_bands():
            counts = band.coverage['photons'].to_numpy() if band.coverage is not None else band.photons
            model, dm_dtheta = band.pixel_counts_and_gradient()
            model = np.clip(model, 1e-30, None)
            full_grad += ((counts / model - 1.0)[:, None] * dm_dtheta).sum(axis=0)

        return full_grad

    def hessian(self, pars=None):
        """Fisher-approximation Hessian of the total log-likelihood."""
        if pars is not None:
            self.parameters.set_parameters(pars)
            self.update()

        n_all = len(self.parameters.parameter_names)
        fisher = np.zeros((n_all, n_all), dtype=float)

        for band in self.pixel_table._iter_bands():
            model, dm_dtheta = band.pixel_counts_and_gradient()
            model = np.clip(model, 1e-30, None)
            g = dm_dtheta.T
            fisher += (g / model) @ g.T

        return -fisher

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, random_state=42):
        """Replace per-band photon counts with Poisson samples from the model.

        Parameters
        ----------
        random_state : int or np.random.Generator, optional
            Seed or RNG for reproducible sampling.
        """
        pt = self.pixel_table
        rng = np.random.default_rng(random_state)
        for band in pt._iter_bands():
            _, model = band.pixel_counts()
            band.photons[:] = rng.poisson(model)

    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------

    def fit(self, select=None, exclude=None, summarize=True, setpars=None, **kwargs):
        """Perform a legacy-style fit using the shared fitter-view workflow.

        Parameters
        ----------
        select : None, item, or list of items
            Optional parameter selector forwarded to ``fitter_view``.
        exclude : None, item, or list of items
            Optional parameter selector to remove from ``select``.
        summarize : bool, default=True
            If True, print the fit summary after a successful fit.
        setpars : dict or None
            Optional parameter values to set before fitting.
        **kwargs
            Legacy fitter options. Supported convenience keywords handled here
            are ``ignore_exception``, ``update_by``, ``tolerance``, and
            ``plot``; remaining keywords are forwarded to ``fv.maximize``.
        """
        if len(self.sources.parameters) == 0:
            print('No parameters to fit')
            return

        ignore_exception = kwargs.pop('ignore_exception', False)
        update_by = kwargs.pop('update_by', 1.0)
        tolerance = kwargs.pop('tolerance', 0.0)
        plot = kwargs.pop('plot', False)

        if setpars is not None:
            self.sources.parameters.setitems(setpars, quiet=True)

        fit_kw = dict(use_gradient=True, estimate_errors=True)
        fit_kw.update(kwargs)

        with self.fitter_view(select, exclude=exclude) as fv:
            if tolerance > 0:
                qual = fv.delta_log_like()
                if qual < tolerance and qual > 0:
                    if summarize:
                        print(
                            'Not fitting, estimated improvement, %.2f, is less than tolerance= %.1f'
                            % (qual, tolerance)
                        )
                    return
            try:
                qual = 99.0
                fv.maximize(**fit_kw)
                w = fv.log_like()
                self.fmin_ret = fv.fmin_ret
                if summarize:
                    print(
                        '%d calls: improvement, quality: %.2f, %.2f'
                        % (fv.calls, w - fv.initial_likelihood, fv.delta_log_like())
                    )
                with redirect_stdout(io.StringIO()):
                    fv.modify(update_by)
                if fit_kw['estimate_errors']:
                    fv.save_covariance()

                if plot:
                    fv.plot_all()

            except Exception as msg:
                print('Fit Failure %s: quality: %.2f' % (msg, qual))
                fv.summary()
                if not ignore_exception:
                    raise

            covariance = getattr(fv, 'covariance', None)
            sigmas = None
            cor = None
            param_mask = np.array(getattr(fv, 'mask', []), dtype=bool)
            grad = None
            ts_values = None

            # Preserve active-parameter diagnostics expected by summary().
            try:
                grad = np.asarray(fv.gradient(), dtype=float)
            except Exception:
                grad = None

            # TS-like values: 2 * delta-loglike forcing each active Norm
            # parameter to a very small internal value (-20), matching the
            # previous summary-column behavior.
            try:
                active_names = np.asarray(fv.parameter_names)
                x_fit = np.asarray(fv.get_parameters(), dtype=float)
                logl_opt = float(fv.log_like())
                ts_values = np.full(len(active_names), np.nan)
                if len(active_names) == len(x_fit):
                    for k, name in enumerate(active_names):
                        if str(name).endswith('_Norm'):
                            trial = x_fit.copy()
                            trial[k] = -20.0
                            fv.set_parameters(trial)
                            ts_values[k] = round(2.0 * (logl_opt - float(fv.log_like())), 1)
                    fv.set_parameters(x_fit)
            except Exception:
                ts_values = None

            if covariance is not None:
                cov = np.asarray(covariance, dtype=float)
                if cov.ndim == 2 and cov.size > 0:
                    diag = np.diag(cov).copy()
                    diag[diag < 0] = np.nan
                    sigmas = np.sqrt(diag)
                    outer = np.outer(sigmas, sigmas)
                    cor = np.full_like(cov, np.nan, dtype=float)
                    valid = outer > 0
                    cor[valid] = cov[valid] / outer[valid]


            self.fit_info = dict(
                # loglike=fv.log_like(),
                calls=fv.calls,
                # pars=fv.parameters[:],
                # covariance=covariance,
                # param_mask=param_mask if len(param_mask) > 0 else None,
                grad=grad.round(),
                ts_values=ts_values,
                sigmas=sigmas.round(3) if sigmas is not None else None,
                cor=100*cor.round(2) if cor is not None else None,
                deltaTS=round(2.0 * (fv.log_like() - fv.initial_likelihood), 1),
                # mask_indeces=np.arange(len(fv.mask))[fv.mask],
                qual=float(fv.delta_log_like().round(2)),
            )
        if summarize:
            self.summary()
        return

    # ------------------------------------------------------------------
    # Repivot
    # ------------------------------------------------------------------

    def repivot(self, summarize=True, **fit_kwargs):
        """Set each source's reference energy to its pivot energy, then refit.

        The pivot energy is the energy at which the flux uncertainty is
        minimised (the knot of the bow-tie). Repivoting removes the
        correlation between the normalisation and spectral-index parameters
        and is good practice after an initial fit has been performed.

        Workflow
        --------
        1. If no fit has been performed yet (no covariance available on any
           free source model), run :meth:`fit` first.
        2. For each non-global source with at least one free parameter, call
           ``model.pivot_energy()`` to find the pivot, then ``model.set_e0()``
           to move the reference energy (adjusting the norm so the model
           prediction is unchanged).
        3. Run :meth:`fit` again.
        4. Return a :class:`pandas.DataFrame` comparing old and new values of
           ``e0``, ``Norm``, and ``Index`` (where present) for each source.

        Parameters
        ----------
        summarize : bool, optional
            Forwarded to each :meth:`fit` call.  Default ``True``.
        **fit_kwargs
            Additional keyword arguments forwarded to :meth:`fit`.

        Returns
        -------
        pandas.DataFrame
            One row per repivoted source with columns:
            ``source``, ``e0_before``, ``e0_after``, ``pivot``,
            ``norm_before``, ``norm_after``,
            and ``index_before`` / ``index_after`` when the model has an
            ``Index`` parameter.
        """
        import pandas as pd

        # --- Step 1: initial fit if covariance is not yet available ----------
        any_errors = any(
            src.model.has_errors()
            for src in self.source_model
            if not getattr(src, 'isglobal', False) and np.any(src.model.free)
        )
        if not any_errors:
            print('repivot: no covariance found – running initial fit first')
            self.fit(summarize=summarize, **fit_kwargs)

        # --- Step 2: collect before-state and set new e0 --------------------
        rows = []
        for src in self.source_model:
            if getattr(src, 'isglobal', False):
                continue
            m = src.model
            if not np.any(m.free):
                continue

            pivot = m.pivot_energy(exception=False)
            if not np.isfinite(pivot) or pivot <= 0:
                continue

            e0_before = float(m.e0)
            norm_before = float(m.getp('Norm'))
            param_names = set(getattr(m, 'param_names', []))
            index_before = float(m.getp('Index')) if 'Index' in param_names else None

            m.set_e0(float(pivot))

            rows.append(dict(
                source=src.name,
                e0_before=round(e0_before, 1),
                pivot=round(pivot, 1),
                e0_after=round(float(m.e0), 1),
                norm_before=norm_before,
                norm_after=float(m.getp('Norm')),
                index_before=index_before,
                index_after=float(m.getp('Index')) if 'Index' in param_names else None,
            ))

        if not rows:
            print('repivot: no sources could be repivoted (fit with errors first?)')
            return pd.DataFrame()

        # --- Step 3: refit with new pivot energies ---------------------------
        self.fit(summarize=summarize, **fit_kwargs)

        # Update norm_after from the refitted values.
        for row in rows:
            src = self.source_model.find_source(row['source'])
            row['norm_after'] = float(src.model.getp('Norm'))
            if 'Index' in set(getattr(src.model, 'param_names', [])):
                row['index_after'] = float(src.model.getp('Index'))

        # --- Step 4: build and return the comparison table ------------------
        df = pd.DataFrame(rows)
        # Drop index columns if no source has an Index parameter.
        if df['index_before'].isna().all():
            df = df.drop(columns=['index_before', 'index_after'])
        return df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, out=None, title=None, gradient=True, ts=True):
        """Print a summary table of fitted parameter values and diagnostics.

        Parameters
        ----------
        out : file-like or None
            Output stream; defaults to stdout.
        title : str or None
            Optional title line.
        gradient : bool
            Include gradient column when available.
        ts : bool
            Include TS column when available.
        """
        if title is not None:
            print(title, file=out)

        fmt_hdr = '%-21s%6s%10s%10s'
        tup_hdr = ('Name', 'index', 'value', 'error(%)')

        grad = None
        ts_values = None
        param_mask = None

        if self.fit_info:
            param_mask = self.fit_info.get('param_mask')
            if gradient:
                grad = self.fit_info.get('grad')
                if grad is not None:
                    fmt_hdr += '%10s'
                    tup_hdr += ('gradient',)
            if ts:
                ts_values = self.fit_info.get('ts_values')
                if ts_values is not None:
                    fmt_hdr += '%10s'
                    tup_hdr += ('TS',)

        print(fmt_hdr % tup_hdr, file=out)

        pset = self.source_model.parameters
        all_names = pset.parameter_names
        all_model_params = np.asarray(pset.model_parameters)
        n_all = len(all_names)

        if param_mask is None:
            param_mask = np.ones(n_all, dtype=bool)

        active_names = all_names[param_mask]
        active_model_params = all_model_params[param_mask]
        index_array = np.arange(n_all)[param_mask]

        # Relative uncertainties in external (physical) parameter space.
        # pset.uncertainties applies the internal->external Jacobian and divides
        # by the external value, giving the correct fractional errors.
        n_active = int(param_mask.sum())
        uncertainties = pset.uncertainties[param_mask] if self.fit_info else np.zeros(n_active)

        prev = ''
        for i, (name, value) in enumerate(zip(active_names, active_model_params)):
            t = name.split('_')
            pname = t[-1]
            sname = '_'.join(t[:-1])
            display_name = name if sname != prev else len(sname) * ' ' + '_' + pname
            prev = sname

            rsig = float(uncertainties[i]) if i < len(uncertainties) else 0.0
            psig = '%.1f' % (rsig * 100) if rsig > 0 and not np.isnan(rsig) else '***'

            truncname = display_name[:20] + '*' if len(display_name) > 20 else display_name
            fmt = '%-21s%6d%10.4g%10s'
            tup = (truncname, index_array[i], value, psig)

            if gradient and grad is not None:
                fmt += '%10.1f'
                tup += (float(grad[i]),)
            if ts and ts_values is not None:
                fmt += '%10s'
                ts_val = ts_values[i]
                tup += (f'{ts_val:.0f}' if np.isfinite(ts_val) else '',)

            print(fmt % tup, file=out)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def fit_source(self, source=None, energy_range=None, **kwargs):
        """Fit over an optional energy range, then restore the prior selection.

        Parameters
        ----------
        source : str, Source-like, or None, optional
            Source to fit (currently informational; free parameters come from
            ``source_model.parameters``).
        energy_range : tuple[float, float] or None, optional
            ``(emin, emax)`` in GeV.  Bands outside this range are excluded.
        **kwargs
            Forwarded to :meth:`fit`.

        Returns
        -------
        tuple
            Return value of :meth:`fit`.
        """
        pt = self.pixel_table
        prior_selected = pt._selected
        try:
            if energy_range is not None:
                emin_mev = energy_range[0] * 1e3
                emax_mev = energy_range[1] * 1e3
                candidates = list(prior_selected) if prior_selected is not None else list(pt.keys())
                pt._selected = [
                    k for k in candidates
                    if pt[k].e0 >= emin_mev and pt[k].e1 <= emax_mev
                ]
            return self.fit(**kwargs)
        finally:
            pt._selected = prior_selected

    def norm_profile(self, source_name=None, tol=0.5):
        """Return a Norm profile fitted in log-space.

        All other free parameters are held fixed at their current values.
        The fit is performed against ``x = log(norm / norm_floor)`` (with an
        additional linear rescaling for numerical stability), then exposed via a
        callable object in physical Norm units.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector forwarded to ``SourceModel.find_source``.  Defaults
            to the currently selected source.
        tol : float, optional
            Fit-quality tolerance forwarded to ``PoissonFitter``.

        Returns
        -------
        object
            Profile object with ``__call__``, ``flux``, ``errors``, ``limit``,
            and ``ts`` attributes in physical Norm units.

        Raises
        ------
        ValueError
            If no ``Norm`` parameter exists in the source's spectral model.
        """
        pkg = __package__ if __package__ else 'like3'
        PoissonFitter = importlib.import_module(f'{pkg}.loglikelihood').PoissonFitter

        source = self.source_model.find_source(source_name)
        model = source.model
        best_norm = float(model.getp('Norm'))
        norm_floor = 1e-30
        safe_best = max(best_norm, norm_floor)

        def _set_norm(norm):
            model.setp('Norm', max(float(norm), norm_floor))
            source.changed = True

        def _loglike_x(x):
            # x is log(norm / norm_floor), constrained to x >= 0 by PoissonFitter.
            x = max(float(np.asarray(x, dtype=float).reshape(-1)[0]), 0.0)
            norm = norm_floor * np.exp(np.clip(x, 0.0, 700.0))
            _set_norm(norm)
            return self.log_like()

        # Use local curvature around the current best value to set a linear
        # scale so the PoissonFitter variable has O(1) width.
        x0 = float(np.log(safe_best / norm_floor))
        h = 1e-3
        try:
            f0 = _loglike_x(x0)
            fp = _loglike_x(x0 + h)
            fm = _loglike_x(max(x0 - h, 0.0))
            curvature = max(2.0 * f0 - fp - fm, 0.0) / (h * h)
            sigma_x = 1.0 / np.sqrt(curvature) if curvature > 0 else 1.0
            sigma_x = float(np.clip(sigma_x, 1e-4, 1e4))
            y0 = x0 / sigma_x

            def _loglike_y(y):
                return _loglike_x(y * sigma_x)

            try:
                pf = PoissonFitter(_loglike_y, scale=max(y0, 1.0), tol=tol)
            except Exception:
                # Some low-information or strongly non-Poisson-like profiles can
                # exceed the strict maxdev test; retry with a looser tolerance.
                pf = PoissonFitter(_loglike_y, scale=max(y0, 1.0), tol=max(1.0, 2.0 * tol))
            poiss_y = pf.poiss
        finally:
            _set_norm(best_norm)

        class _LogNormProfile:
            def __init__(self, poiss, sigma, floor):
                self._poiss = poiss
                self._sigma = float(sigma)
                self._floor = float(floor)

            def __str__(self):
                flux = self.flux
                lo, hi = self.errors
                limit = self.limit
                if flux > 0:
                    return (
                        f'LogNormProfile(flux={flux:.4g}, '
                        f'errors=({lo:.4g}, {hi:.4g}), '
                        f'ts={self.ts:.2f})'
                    )
                return f'LogNormProfile(flux=0, limit95={limit:.4g}, ts={self.ts:.2f})'

            __repr__ = __str__

            def _norm_to_y(self, norm):
                n = np.clip(np.asarray(norm, dtype=float), self._floor, np.inf)
                x = np.log(n / self._floor)
                return x / self._sigma

            def __call__(self, norm):
                y = self._norm_to_y(norm)
                if np.ndim(y) == 0:
                    return float(self._poiss(float(y)))
                return np.asarray(self._poiss(y), dtype=float)

            @property
            def flux(self):
                y_peak = max(self._poiss.flux, 0.0)
                return float(self._floor * np.exp(np.clip(y_peak * self._sigma, 0.0, 700.0)))

            @property
            def errors(self):
                y_lo, y_hi = self._poiss.errors
                lo = self._floor * np.exp(np.clip(y_lo * self._sigma, 0.0, 700.0))
                hi = self._floor * np.exp(np.clip(y_hi * self._sigma, 0.0, 700.0))
                return (float(lo), float(hi))

            @property
            def limit(self):
                y_lim = self._poiss.limit if hasattr(self._poiss, 'limit') else self._poiss.percentile(0.95)
                return float(self._floor * np.exp(np.clip(y_lim * self._sigma, 0.0, 700.0)))

            @property
            def ts(self):
                return float(self._poiss.ts)

        return _LogNormProfile(poiss_y, sigma_x, norm_floor)

    def localization_view(self, source_name=None):
        """Return a localization context manager for the selected source.

        Returns
        -------
        _PixelTableLocalizationContext
        """
        pkg = __package__ if __package__ else 'like3'
        pixel_table_mod = importlib.import_module(f'{pkg}.pixel_table')
        context_cls = pixel_table_mod._PixelTableLocalizationContext
        sm_context = self.source_model.localization_view(source_name)
        return context_cls(self.pixel_table, sm_context)

    def localize(self, source_name=None, sigma=0.1, verbose=False):
        """Run localization for a source.

        Parameters
        ----------
        source_name : str, Source-like, or None
        sigma : float, optional
            Initial uncertainty in degrees.
        verbose : bool, optional

        Returns
        -------
        like3.quadform.Localize
        """
        from like3.quadform import Localize
        with self.localization_view(source_name) as loc:
            return Localize(loc, sigma=sigma, verbose=verbose)


def main(
    source_name, cone_size=1.0,
    *,
    pixel_table_path='files/kerr/toby_v4.fits',
    catalog='v40',
    query=None,
    frame='galactic',
    verbose=False,
):
    """Build and return a ``MultiBandLikelihood`` for one catalog source.

    Parameters
    ----------
    source_name : str
        Catalog source name, forwarded to
        ``SourceModel.from_fermi_catalog`` as the cone-center selector.
    pixel_table_path : str, default='files/kerr/toby_v4.fits'
        Pixel-table FITS path passed to ``PixelTable``.
    catalog : str, default='v40'
        Fermi catalog version passed as ``version=``.
    query : str or None, optional
        Optional catalog query filter.
    cone_size : float, default=1.0
        Cone radius in degrees used when building the source model.
    verbose : bool, default=False
        Verbosity flag forwarded to ``FermiFit``.

    Returns
    -------
    MultiBandLikelihood
        Multi-band likelihood wrapper with attached ``pixel_table`` and
        ``source_model``.
    """
    package_root = Path(__file__).resolve().parent.parent
    path_candidates = [
        package_root,
        package_root / 'wtlike',
        package_root.parent / 'wtlike',
    ]
    for candidate in path_candidates:
        if not (candidate / 'utilities').is_dir():
            continue
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    pkg = __package__ if __package__ else 'like3'
    pixel_table_mod = importlib.import_module(f'{pkg}.pixel_table')
    sourcelist_mod = importlib.import_module(f'{pkg}.sourcelist')

    source_model = sourcelist_mod.SourceModel.from_fermi_catalog(
        source_name, frame=frame,
        version=catalog,
        query=query,
        cone_size=cone_size,
    )
    pixel_table = pixel_table_mod.PixelTable(
        pixel_table_path,
        source_model=source_model,
    )
    plt.style.use('dark_background')
    likelihood = MultiBandLikelihood(pixel_table, source_model)
    return likelihood


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Build a MultiBandLikelihood from a catalog source name.')
    parser.add_argument('source_name', help='Catalog source name used to build the SourceModel.')
    parser.add_argument('cone_size',  help='Cone radius in degrees for catalog selection.')

    parser.add_argument(
        '--pixel-table-path',
        default='files/kerr/toby_v4.fits',
        help='Pixel-table FITS path.',
    )
    parser.add_argument(
        '--catalog',
        default='v40',
        help='Fermi catalog version passed to SourceModel.from_fermi_catalog.',
    )
    parser.add_argument(
        '--query',
        default=None,
        help='Optional catalog query filter.',
    )
    parser.add_argument(
        '--frame',
        default='galactic',
        help='Coordinate frame for the center (e.g., "galactic" or "icrs").',
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Forward verbose=True to FermiFit.',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    from utilities.ipynb_docgen import show, show_date, capture_hide
    args = _parse_args()
    show_date()
    with capture_hide(f'Setup output for {args.source_name}') as setup_output:
        mbl = main(
            args.source_name, 
            float(args.cone_size),
            pixel_table_path=args.pixel_table_path,
            catalog=args.catalog,
            query=args.query,
            verbose=args.verbose,
            frame=args.frame,
        )

        pt = mbl.pixel_table
        sm = mbl.source_model
        selected = getattr(sm, 'selected_source', None)
        selected_name = None if selected is None else selected.name
 
    
        print(f'Built MultiBandLikelihood for {selected_name or args.source_name}')
        print(f'Pixel table: {pt.name}')
        print(f'Sources, selected near {args.source_name}: {", ".join(s.name for s in sm)}')

    show(setup_output)
    plt.style.use('dark_background')
