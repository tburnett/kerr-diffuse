"""Likelihood and optimization helpers for the like3 model layer.

The `Likelihood` class wraps Poisson log-likelihood evaluation and numerical
optimization of model parameters, then stores post-fit diagnostics such as
covariance, correlation matrix, gradients, and TS-like values for normalization
parameters.
"""

import contextlib
import numpy as np
from scipy import optimize


class Likelihood:
    """Poisson likelihood wrapper for fitting model counts to observed data."""
    
    def __init__(self, model, data=None):
        """Initialize a likelihood object.

        Parameters:
        ----------
        model : object
            Model object exposing parameter and count-evaluation interfaces used
            by this class (`parameters`, `counts`, `count_gradient`, etc.).
        data : array-like
            Observed counts to fit.
        """
        self.model = model
        self.data = model.data if data is None else data
        self.mp = model.parameters

    @contextlib.contextmanager
    def saved_state(self):
        """Context manager that restores SourceModel parameter state on exit.

        Saves the active ``ParameterSet`` (which ``select()`` can replace with a
        ``ParSubSet``) and the full internal parameter vector, then restores both
        when the ``with`` block exits — whether normally or via exception.

        Example
        -------
        with likelihood.saved_state():
            likelihood.select('Norm')
            likelihood.maximize()
        # model.parameters and all source-model values are back to what they were
        """
        saved_parameters = self.model.parameters
        saved_values = saved_parameters.values.copy()
        try:
            yield
        finally:
            self.model.parameters = saved_parameters
            self.mp = saved_parameters
            saved_parameters.values = saved_values

    def select(self, *select):
        """Select a subset of fit parameters and update active parameter view.

        Parameters:
        ----------
        *select : list[int|str]
            Selection arguments forwarded to `model.parsubset`.
        """
        self.model.parameters = self.model.parsubset(*select)
        self.mp = self.model.parameters

    def _evaluate(self, x):
        """Return log-likelihood and gradient at parameter vector `x`."""
        self.mp.values = x
        d = self.data
        m = self.model.counts()
        safe_m = np.maximum(m, 1e-12)
        # Constant factorial terms are omitted because they do not affect argmax.
        logl = np.sum(d * np.log(safe_m) - m)
        # Gradient of log L for Poisson model: sum((d/m - 1) * dm/dtheta).
        # count_gradient() returns shape (n_total_free, n_pixels); restrict to the
        # active subset defined by the current parameter mask (set via select()).
        ratio = np.zeros_like(d, dtype=float)
        with np.errstate(invalid='ignore'):
            np.divide(d, safe_m, out=ratio, where=m > 0)
        full_grad = ((ratio - 1) * self.model.count_gradient()).sum(axis=1)
        grad = full_grad[self.mp.mask]
        return logl, grad
 
    @property
    def value(self):
        """Return the log-likelihood at the best-fit parameters."""
        if not hasattr(self, 'fit_info'):
            raise RuntimeError('fit has not been run yet; call maximize() first')
        return -self(self.fit_info['x_fit'])[0]
    

    @property
    def grad(self):
        """Return the gradient at the best-fit parameters."""
        if not hasattr(self, 'fit_info'):
            raise RuntimeError('fit has not been run yet; call maximize() first')
        return -self(self.fit_info['x_fit'])[1]
    
    def log_like(self, x):
        """Evaluate Poisson log-likelihood at parameter vector `x`."""
        return self._evaluate(x)[0]
    
    def __call__(self, x):
        """Return objective and gradient for L-BFGS-B minimization.

        Returns
        -------
        tuple[float, np.ndarray]
            Negative log-likelihood and its negative gradient at `x`.
        """
        logl, grad = self._evaluate(x)
        return -logl, -grad
    
    def maximize(self, x0=None):
        """Maximize log-likelihood, store fit diagnostics in `self.fit_info`.

        Parameters
        ----------
        x0 : array-like or None
            Optional initial parameter vector. Uses current model parameter
            values when omitted.

        Notes
        -----
        Stores covariance, 1-sigma estimates, correlation matrix, gradient,
        fitted values, and TS-like values for normalization parameters.
        """
        def evaluate_ts():
            """Compute TS-like drops by forcing `_Norm` parameters to -20."""
            model = self.model
            logl = self.log_like
            x_fit = self.fit_info['x_fit'].copy()
            val_fit = -self(x_fit)[0]
            values = x_fit.copy()
            ts_array = np.full_like(values, np.nan)

            for k, name in enumerate(model.parameter_names[model.parameters.mask]):
                if name.endswith('_Norm'):
                    values[k] = -20
                    ts_array[k] = round(2*(val_fit - logl(values)),1)
                    values[k] = x_fit[k]
                    model.parameters.values = x_fit
            return ts_array

        if x0 is None:
            x0 = self.model.parameters.values.copy()
        initial_val, _ = self(x0)
        x_fit, val, d = optimize.fmin_l_bfgs_b(
            self, x0, bounds=self.model.parameters.bounds, factr=1e3, pgtol=1e-2)
        if d['warnflag'] == 2:
            # Line-search abnormal: accept if objective improved, otherwise raise.
            if val > initial_val:
                raise RuntimeError('fit_plot: optimization failed: %s' % d['task'])
        elif d['warnflag'] != 0:
            raise RuntimeError('fit_plot: optimization failed: %s' % d['task'])
        # Re-evaluate at reported optimum to synchronize objective + gradient.
        val, gradient = self(x_fit)
        self.model.parameters.values = x_fit
  
        # Analytic Fisher information matrix: H_ij = sum_n (1/m_n)(dm_n/dtheta_i)(dm_n/dtheta_j).
        # Positive-definite by construction; avoids numdifftools finite differences which
        # become unreliable when many parameters are correlated (overlapping sources).
        G = self.model.count_gradient()  # (n_total_free, n_pix)
        G_active = G[self.mp.mask]       # (n_active, n_pix)
        m = self.model.counts()          # (n_pix,)
        safe_m = np.maximum(m, 1e-12)
        hess = (G_active / safe_m) @ G_active.T  # (n_active, n_active)
        cov = np.linalg.inv(hess)
        sigs = np.sqrt(cov.diagonal())

        self.model.parameters.set_covariance(cov)

        self.fit_info = dict(
            hess = hess,
            cov = cov,
            sigs = sigs.round(4),
            corr = (cov / np.outer(sigs,sigs)).round(2),
            grad = -gradient,
            x_fit = x_fit,
            x_init = x0,
            delta_loglike = round(initial_val-val,2), 

            funcalls = d['funcalls'],)
        self.fit_info['ts_values'] = evaluate_ts()
        
    @property
    def model_parameters(self):
        """Return model/external parameter values for active fit parameters."""
        return self.model.parameters.model_parameters

    def norm_profile(self, source_name=None):
        """Return the log-likelihood as a function of the Norm parameter for a named source.

        All other free parameters are held fixed at their current values.

        Parameters
        ----------
        source_name : str
            Source name 

        Returns
        -------
        callable
            ``f(norm_values)`` accepts a scalar or array of internal (log-space)
            Norm values and returns the corresponding Poisson log-likelihood
            value(s).  Model state is restored to its original Norm value after
            each call.

        Raises
        ------
        ValueError
            If no ``<source_name>_Norm`` parameter exists among the free parameters.
        """
        all_names = self.model.parameter_names   # all free-param names (unmasked)
        
        try:
            source_name =  self.model.source_model.find_source(source_name).name
        except Exception as e:
            raise ValueError(f'No source found matching {source_name!r}') from e
        norm_name = source_name + '_Norm'
        matches = np.where(all_names == norm_name)[0]
        if len(matches) == 0:
            raise ValueError(
                f'No Norm parameter found for source {source_name!r}; '
                f'available names: {list(all_names)}'
            )
        norm_global_idx = int(matches[0])

        # Build a one-element ParSubSet selecting only this Norm parameter.
        norm_subset = self.model.parsubset()
        m = np.zeros(len(all_names), bool)
        m[norm_global_idx] = True
        norm_subset.set_mask(m)

        d = self.data

        def f(norm_values):
            scalar = np.ndim(norm_values) == 0
            norm_values = np.atleast_1d(np.asarray(norm_values, dtype=float))
            saved = float(norm_subset.values[0])
            result = np.empty(len(norm_values))
            for i, v in enumerate(norm_values):
                norm_subset.values = np.array([v])
                counts = self.model.counts()
                result[i] = np.sum(d * np.log(counts) - counts)
            norm_subset.values = np.array([saved])   # restore
            return float(result[0]) if scalar else result

        return f

    def summary(self,  out=None, title=None, gradient=True, ts=True):
        """Print a summary table of fitted parameter values and diagnostics.

        Parameters:
        ----------

        out : open file or None
            Output stream; defaults to stdout.
        title : str or None
            Optional title line.
        gradient : bool
            If true, include gradient column.
        ts : bool
            If true, include TS-like column when available.
        """
        if title is not None:
            print(title, file=out)

        fmt, tup = '%-21s%6s%10s%10s', tuple('Name index value error(%)'.split())

        mask = self.model.parameters.mask
        grad = None
        ts_values = None
        if hasattr(self, 'fit_info'):
            if gradient:
                # fit_info['grad'] is already the active-subset gradient (length
                # mask.sum()), so do NOT re-index with mask.
                grad = self.fit_info['grad']
                fmt += '%10s'; tup += ('gradient',)
            if ts:
                ts_all = self.fit_info.get('ts_values', None)
                if ts_all is not None:
                    # Same: ts_values was computed for the active subset only.
                    ts_values = ts_all
                    fmt += '%10s'; tup += ('TS',)
        print(fmt %tup, file=out)
        prev=''
        

        index_array = np.arange(len(mask))[mask]
        for index, (name, value, rsig) in enumerate(zip(self.model.parameter_names[mask], 
                                                        self.model_parameters,
                                                        self.model.parameters.uncertainties)):
            t = name.split('_')
            pname = t[-1]
            sname = '_'.join(t[:-1])
            if sname==prev: name = len(sname)*' '+'_'+pname
            prev = sname
            fmt = '%-21s%6d%10.4g%10s'
            psig = '%.1f'%(rsig*100) if rsig>0 and not np.isnan(rsig) else '***'
            truncname = name[:20]+'*' if len(name)>20 else name
            tup = (truncname, index_array[index], value,psig)
            if gradient and grad is not None:
                fmt +='%10.1f'; tup += (grad[index],)
            if ts and ts_values is not None:
                fmt += '%10s'
                ts_val = ts_values[index]
                tup += (f'{ts_val:.0f}' if np.isfinite(ts_val) else '',)
            print(fmt % tup, file=out)
    
    @classmethod
    def test_plots(cls, model):
        """Generate per-parameter 1D fit scans for quick visual checks."""
        import matplotlib.pyplot as plt
        from like3.views import fit_plot
        data = model.data
        x0 = model.parameters.values.copy()
        fig, axx = plt.subplots(ncols=len(x0), figsize=(4*len(x0),4), sharey=True)
        for k,ax in enumerate(axx):
            model.parameters.values=x0
            func = cls(model, data)
            func.select(k)
            fit_plot(func, x0[k], ax=ax, title=model.parameter_names[k])
        plt.show()

    @classmethod
    def test_fit(cls, src_key=0, random_state=42):
        """Run a self-consistency fit on simulated data from a demo model."""
        from like3.demo_model import DemoModel
        model=DemoModel.test(plot=False, random_state=random_state, src_key=src_key)
        data = model.data
        x0 = model.parameters.values.copy()
        lk = cls(model, data)
        lk.maximize(x0)
        print(model)


class BandModel:
    """Adapt a :class:`~like3.pixel_table.PixelTable.Band` to the Likelihood interface.

    Parameters
    ----------
    band : PixelTable.Band
        A band with ``pix``, ``photons``, ``diffuse``, ``source_model``,
        ``exposure_map``, and ``pixel_gradient`` populated.

    Notes
    -----
    Active pixels are the intersection of the band's data pixels and the pixels
    illuminated by the source model (returned by ``band.pixel_counts()``).  Only
    these pixels participate in ``counts()``, ``count_gradient()``, and
    ``self.data``.

    The PSF response × exposure for each source is cached at construction time
    so that ``counts()`` and ``count_gradient()`` avoid recomputing the spatial
    response on every likelihood evaluation (including every Hessian
    finite-difference step).
    """

    def __init__(self, band):
        self.band = band
        self.source_model = band.source_model
        self.parameters = self.source_model.parameters

        # Pixels illuminated by the source model
        illum_pix, _ = band.pixel_counts()

        # Restrict to the intersection with the band's data pixels
        mask = np.isin(band.pix, illum_pix)
        self._pix = band.pix[mask]
        self.data = band.photons[mask].astype(float)

        # Diffuse contribution aligned to the restricted pixel set (fixed)
        if band.diffuse_counts is not None:
            self._diffuse = band.diffuse_counts[mask].astype(float)
        else:
            self._diffuse = np.zeros(len(self._pix), dtype=float)

        # Cache spatial response × exposure per source (shape: n_active_pix).
        # These are constant w.r.t. model parameters and expensive to recompute.
        exp = band.exposure_map(self._pix)
        self._src_responses = []
        for src in self.source_model:
            _, v = band.response(src, self._pix)
            self._src_responses.append(v * exp)

    @property
    def parameter_names(self):
        return self.source_model.parameter_names

    def parsubset(self, *select):
        return self.source_model.parsubset(*select)

    def counts(self):
        """Total model counts (diffuse + source) per active pixel."""
        model = self._diffuse.copy()
        for src, resp in zip(self.source_model, self._src_responses):
            model += resp * src.model(self.band.energy)
        return model

    def count_gradient(self):
        """Gradient of source counts w.r.t. free params, shape (n_params, n_pixels).

        Diffuse counts are treated as fixed (no gradient contribution).
        Returns ``(n_params, n_pixels)`` as expected by :class:`Likelihood`.
        """
        g = []
        for src, resp in zip(self.source_model, self._src_responses):
            grad = src.model.gradient(self.band.energy)[src.model.free]
            g.append(resp[:, None] * grad[None, :])   # (n_pix, n_free_for_src)
        return np.hstack(g).T                          # (n_params, n_pixels)
        return lk