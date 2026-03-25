"""Likelihood and optimization helpers for the like3 model layer.

The `Likelihood` class wraps Poisson log-likelihood evaluation and numerical
optimization of model parameters, then stores post-fit diagnostics such as
covariance, correlation matrix, gradients, and TS-like values for normalization
parameters.
"""

import numpy as np
from scipy import optimize


class Likelihood:
    """Poisson likelihood wrapper for fitting model counts to observed data."""
    
    def __init__(self, model, data,):
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
        self.data = data
        self.mp = model.parameters

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
        # Constant factorial terms are omitted because they do not affect argmax.
        logl = np.sum(d * np.log(m) - m)
        # Gradient of log L for Poisson model: sum((d/m - 1) * dm/dtheta).
        grad = ((d / m - 1) * self.model.count_gradient()).sum(axis=1)
        return logl, grad
 
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
        import numdifftools

        def evaluate_ts():
            """Compute TS-like drops by forcing `_Norm` parameters to -20."""
            model = self.model
            logl = self.log_like
            x_fit = self.fit_info['x_fit'].copy()
            val_fit = -self(x_fit)[0]
            values = x_fit.copy()
            ts_array = np.full_like(values, np.nan)

            for k, name in enumerate(model.parameter_names):
                if name.endswith('_Norm'):
                    values[k] = -20
                    ts_array[k] = round(2*(val_fit - logl(values)),1)
                    values[k] = x_fit[k]
                    model.parameters.values = x_fit
            return ts_array

        if x0 is None:
            x0 = self.model.parameters.values.copy()
        initial_val, _ = self(x0)
        x_fit, val, d = optimize.fmin_l_bfgs_b(self, x0,  bounds=self.model.parameters.bounds); 
        if d['warnflag'] != 0:
            raise RuntimeError('fit_plot: optimization failed: %s' % d['task'])
        # Re-evaluate at reported optimum to synchronize objective + gradient.
        val, gradient = self(x_fit)
        self.model.parameters.values = x_fit
  
        hess = numdifftools.Hessian(self.log_like)(x_fit) 
        cov = np.linalg.inv(-hess)
        sigs = np.sqrt(cov.diagonal())
        self.model.parameters.set_covariance(cov)

        self.fit_info = dict(
            cov = cov,
            sigs = sigs,
            corr = (cov / np.outer(sigs,sigs)).round(2),
            grad = -gradient,
            x_fit = x_fit,
            delta_loglike = round(initial_val-val,2), 

            funcalls = d['funcalls'],)
        self.fit_info['ts_values'] = evaluate_ts()
        
    @property
    def model_parameters(self):
        """Return model/external parameter values for active fit parameters."""
        return self.model.parameters.model_parameters
    
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
                grad = self.fit_info['grad'][mask]
                fmt += '%10s'; tup += ('gradient',)
            if ts:
                ts_all = self.fit_info.get('ts_values', None)
                if ts_all is not None:
                    ts_values = ts_all[mask]
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
        return lk