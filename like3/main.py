"""
FermiFit: fitting interface for a PixelTable + SourceModel pair.
"""
from contextlib import contextmanager

import numpy as np


class FermiFit:
    """Fitting engine that wraps a PixelTable and its attached SourceModel.

    Parameters
    ----------
    pixel_table : pylib.pixel_table.PixelTable
        A loaded pixel table with a ``source_model`` attribute set.
    """

    def __init__(self, pixel_table):
        if pixel_table.source_model is None:
            raise ValueError('FermiFit requires a PixelTable with a source_model attached')
        self.pixel_table = pixel_table
        self.fit_info: dict = {}

    # def __getattr__(self, name):
    #     # Delegate unknown attributes to the wrapped PixelTable.
    #     # __getattr__ is only called when normal lookup fails, so this
    #     # does not shadow any attribute defined directly on FermiFit.
    #     return getattr(self.pixel_table, name)

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

    @property
    def parameter_names(self):
        """Names of the free parameters."""
        return self.source_model.parameter_names

    @property
    def bounds(self):
        """Fitter-space parameter bounds."""
        return self.source_model.bounds

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

    def preserve_parameters(self):
        """Context manager that restores free-parameter values on exit."""
        @contextmanager
        def _ctx():
            pset = self.parameters
            saved = np.array(pset.get_parameters(), copy=True)
            try:
                yield
            finally:
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

    def fit(self, select=None, *, method='l-bfgs-b', quiet=True, use_gradient=True, **kwargs):
        """Optimize the free spectral parameters of the source model.

        Parameters
        ----------
        select : str | int | list[str | int] or None, optional
            Subset of parameters to optimize, forwarded to ``ParSubSet``.
            Supports source names (with ``*`` wildcards), parameter names
            prefixed with ``_``, and integer indices.
        method : str, optional
            Optimization algorithm: ``'l-bfgs-b'`` (default), ``'simplex'``,
            or ``'powell'``.
        quiet : bool, optional
            Suppress optimizer diagnostic output.
        use_gradient : bool, optional
            Pass the analytic gradient to the optimizer.
        **kwargs
            Forwarded to ``Minimizer.__call__``.

        Returns
        -------
        tuple
            ``(fitvalue, parameters, errors)`` as returned by ``Minimizer``.

        Side Effects
        ------------
        Updates ``source_model`` parameters in place and populates
        ``self.fit_info``.
        """
        from like3.fitter import Minimizer, Fitted
        from like3.parameterset import ParSubSet

        pt = self.pixel_table
        pset = self.source_model.parameters
        initial_loglike = self.loglike()
        use_gradient = kwargs.pop('use_gradient', use_gradient)

        all_names = np.asarray(pset.parameter_names)
        n_all = len(all_names)

        if select is not None:
            select_args = select if isinstance(select, (list, tuple)) else [select]
            subset = ParSubSet(self.source_model, *select_args)
            param_mask = subset._mask
        else:
            param_mask = np.ones(n_all, dtype=bool)

        x_init = np.asarray(pset.get_parameters(), dtype=float)[param_mask].copy()

        fit_obj = self

        class _Objective(Fitted):
            def __init__(self):
                self._cache_pars = None
                self._cache_value = None
                self._cache_grad = None

            @property
            def bounds(self):
                b = fit_obj.bounds
                return b[param_mask] if b is not None else None

            @property
            def parameter_names(self):
                return all_names[param_mask]

            def get_parameters(self):
                return np.asarray(pset.get_parameters())[param_mask]

            def set_parameters(self, par):
                full = np.asarray(pset.get_parameters(), dtype=float)
                full[param_mask] = par
                pset.set_parameters(full)

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
                full_grad = np.zeros(n_all, dtype=float) if need_grad else None

                for band in pt._iter_bands():
                    m = band._fit_mask
                    counts = band.photons if m is None else band.photons[m]
                    if need_grad:
                        model, dm_dtheta = band.pixel_counts_and_gradient()
                    else:
                        _, model = band.pixel_counts()
                    model = np.clip(model, 1e-30, None)
                    loglike += float(np.sum(counts * np.log(model) - model))
                    if need_grad:
                        full_grad -= ((counts / model - 1.0)[:, None] * dm_dtheta).sum(axis=0)

                grad = full_grad[param_mask] if need_grad else None
                value = -float(loglike) + initial_loglike
                self._cache_pars = np.array(pars, copy=True)
                self._cache_value = value
                self._cache_grad = None if grad is None else np.array(grad, copy=True)
                return value, grad

            def __call__(self, pars, *args):
                value, _ = self._evaluate(pars, need_grad=use_gradient)
                return value

            def gradient(self, pars):
                _, grad = self._evaluate(pars, need_grad=True)
                return grad

        objective = _Objective()
        minimizer = Minimizer(objective, quiet=quiet)
        fit_out = minimizer(method=method, use_gradient=use_gradient, **kwargs)
        x_fit = np.array(fit_out[1], copy=True)
        logl_opt = initial_loglike - float(fit_out[0])

        # Analytical Hessian (Fisher information matrix).
        n_active = int(param_mask.sum())
        hess = np.zeros((n_active, n_active), dtype=float)
        for band in pt._iter_bands():
            band_model, dm_dtheta = band.pixel_counts_and_gradient()
            band_model = np.clip(band_model, 1e-30, None)
            G = dm_dtheta[:, param_mask].T  # (n_active, n_pix)
            hess += (G / band_model) @ G.T

        try:
            cov = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            cov = np.full_like(hess, np.nan)
        sigs = np.sqrt(np.clip(cov.diagonal(), 0.0, None))
        outer = np.outer(sigs, sigs)
        corr = np.where(outer > 0, cov / np.where(outer > 0, outer, 1.0), np.nan).round(2)
        grad = objective.gradient(x_fit)

        # TS values: 2 × Δloglike forcing each Norm parameter to -20.
        active_names = all_names[param_mask]
        ts_values = np.full(n_active, np.nan)
        for k, name in enumerate(active_names):
            if name.endswith('_Norm'):
                trial = x_fit.copy()
                trial[k] = -20.0
                objective.set_parameters(trial)
                ts_values[k] = round(2.0 * (logl_opt - self.loglike()), 1)
                objective.set_parameters(x_fit)

        # Write covariance back into source models so that pset.uncertainties
        # (which applies the internal->external Jacobian) gives correct values.
        old_mask = pset.mask.copy()
        pset.mask = param_mask
        pset.set_covariance(cov)
        pset.mask = old_mask

        self.fit_info = dict(
            hess=hess,
            cov=cov,
            sigs=sigs.round(4),
            corr=corr,
            grad=grad,
            x_fit=x_fit,
            x_init=x_init,
            delta_loglike=round(logl_opt - initial_loglike, 2),
            ts_values=ts_values,
            param_mask=param_mask,
        )
        return fit_out

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

    def localization_view(self, source_name=None):
        """Return a localization context manager for the selected source.

        Parameters
        ----------
        source_name : str, Source-like, or None

        Returns
        -------
        _BandListLocalizationContext
        """
        from like3.bands import _BandListLocalizationContext
        sm_context = self.source_model.localization_view(source_name)
        return _BandListLocalizationContext(self.pixel_table, sm_context)

    def localize(self, source_name=None, sigma=0.1, verbose=True):
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
