"""
FermiFit: fitting interface for a PixelTable + SourceModel pair.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import importlib
import io
from pathlib import Path
import sys

import matplotlib.pyplot as plt
plt.style.use('dark_background')
import numpy as np
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
import pandas as pd
from like3 import views

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
    
class MultiBandLikelihood(dict):
    """Class to manage likelihood evaluation from multiple bands."""
    def __init__(self, pixel_table, source_model):
        self.pixel_table = pixel_table
        self.source_model = source_model
        self.selected = None
        super().__init__({key: BandLikelihood(band, source_model) for key, band in self.pixel_table.items()})

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

    def loglike(self, pars=None, **kwargs):
        """Evaluate the total log-likelihood across all selected bands.
        If `pars` is provided, update the source model parameters before computing."""
        if pars is not None:
            self.parameters = pars
        return sum(bl.loglike() for bl in self._iter_bands())

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
            mask |= self.band.cone_search(src.skydir, radius_deg)
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
        self.evaluate_source_model()   

    def response(self, source, pixels=None):
        """Return PSF response for a source evaluated on given pixel indices.
        If *pixels* is None, the coverage pixels are used.  The PSF response is cached per source for efficiency."""
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
    
    def loglike(self, pars=None):
        """Compute the log-likelihood for the current model parameters.
        If `pars` is provided, update the source model parameters before computing."""
        if pars is not None:
            self.parameters = pars

        # Model counts for coverage pixels
        cov = self.coverage
        data = cov['photons'].to_numpy()
        model = cov['model_counts'].to_numpy()

        # Poisson log-likelihood (ignoring constant term)
        ll = np.sum(data * np.log(model + 1e-12) - model)
        return float(ll)

    def pixel_gradient(self):
        """Evaluate per-pixel count gradients for free source-model parameters.

        Returns
        -------
        np.ndarray
            Gradient matrix of shape (n_pixels, n_free_parameters)."""
        
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

            if len(arr) == 12*self.nside**2:
                return arr
            
            if len(arr) == len(self.coverage):
                hpa = np.full(12*self.nside**2, np.nan, dtype=float)
                hpa[self.coverage.pix] = arr
                return hpa
            raise ValueError(f'Cannot expand array of length {len(arr)} to HEALPix array of length {12*self.nside**2}')

    @property
    def residual(self):
        """Compute the residual counts (data - model) for coverage pixels."""
        return self.coverage['photons'].to_numpy() - self.coverage['source_counts'].to_numpy()
    
    @property
    def sigma(self):
        """Compute residual in (approximate) sigma units for coverage pixels."""
        model = self.coverage['model_counts'].to_numpy()
        data = self.coverage['photons'].to_numpy()
        return np.where(model > 0, (data - model) / np.sqrt(model), 0.0)

    def zea_plot(self, what, **kwargs):
        """
         Plot a HEALPix map of the given per-pixel quantity *what* in ZEA projection 
         centered on the source model."""
        from like3 import sky_display
        if isinstance(what, str) and what in self.coverage:
            arr = self.expand_healpix_array(self.coverage[what])
        elif isinstance(what, np.ndarray):
            arr = self.expand_healpix_array(what)
        else:
            raise ValueError(f"Unknown coverage key or array: {what}")
        
        zea = sky_display.zea_plot(self.center, arr, r68 = self.psf.r68, 
                                   source_model=self.source_model, **kwargs)
        zea.axes_text(0.98, 0.98,
                        f'{self.band.energy / 1e3:.2f} GeV\nPSF{self.psf.event_type-2}',
                        color='white', ha='right', va='top', fontsize=12)
        return zea



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
            ll_ref_all = float(self.loglike())

            src.model.setp('Norm', alpha * norm0)
            src.changed = True
            ll_test_all = float(self.loglike())

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

    def loglike(self, skydir=None):
        """Total Poisson log-likelihood summed over all selected bands.

        Parameters
        ----------
        skydir : SkyCoord or None, optional
            Trial sky position forwarded to each band's ``loglike`` call.

        Returns
        -------
        float
        """
        pt = self.pixel_table
        return float(sum(band.loglike(skydir=skydir) for band in pt._iter_bands()))

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
                qual = fv.delta_loglike()
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
                        % (fv.calls, w - fv.initial_likelihood, fv.delta_loglike())
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
                qual=float(fv.delta_loglike().round(2)),
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
            x = max(float(x), 0.0)
            norm = norm_floor * np.exp(np.clip(x, 0.0, 700.0))
            _set_norm(norm)
            return self.loglike()

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
    """Build and return a ``FermiFit`` for one catalog source.

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
    FermiFit
        Fit wrapper with attached ``pixel_table`` and ``source_model``.
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
    return FermiFit(pixel_table, verbose=verbose)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Build a FermiFit from a catalog source name.')
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
        ff = main(
            args.source_name, 
            float(args.cone_size),
            pixel_table_path=args.pixel_table_path,
            catalog=args.catalog,
            query=args.query,
            verbose=args.verbose,
            frame=args.frame,
        )

        pt = ff.pixel_table
        sm = ff.source_model
        selected = getattr(sm, 'selected_source', None)
        selected_name = None if selected is None else selected.name
 
    
        print(f'Built FermiFit for {selected_name or args.source_name}')
        print(f'Pixel table: {pt.name}')
        print(f'Sources, selected near {args.source_name}: {", ".join(s.name for s in sm)}')

    show(setup_output)
    plt.style.use('dark_background')
