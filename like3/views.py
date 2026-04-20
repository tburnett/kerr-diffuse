"""Likelihood engine view classes and fitting helpers.

This module defines view and mixin classes used to fit and inspect ROI likelihood
models. The core classes expose:

- optimization wrappers around ``scipy.optimize.fmin_l_bfgs_b``
- convenience summaries and diagnostic plotting
- context-manager state restoration via ``WithMixin``

Historical source:
    $Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/views.py,v 1.22 2017/11/17 22:50:36 burnett Exp $
    Author: T.Burnett <tburnett@uw.edu> (based on pioneering work by M. Kerr)
"""

import sys, types, importlib
from typing import Any
import numpy as np
from scipy import misc, optimize
from . skydir import SkyDir
# from . import roimodel, bandlike
from . import parameterset
_pkg = __package__ if __package__ else 'like3'
tools = importlib.import_module(f'{_pkg}.tools')


def _is_verbose(obj):
    """Return the effective verbose flag for a view-like object."""
    if hasattr(obj, 'verbose'):
        return bool(obj.verbose)
    blike = getattr(obj, 'blike', None)
    if blike is not None and hasattr(blike, 'verbose'):
        return bool(blike.verbose)
    return True


def _verbose_print(obj, *args, **kwargs):
    """Print only when the owning view has verbose output enabled."""
    if _is_verbose(obj):
        print(*args, **kwargs)


def bounded_root(func, x0, bounds, method='hybr', tol=1e-8, maxiter=1000):
    """Find a bounded root using an unbounded-variable transform.

    The solver runs in transformed coordinates and maps each iterate back into
    ``[lb, ub]`` using a logistic transform, so iterates always remain in bounds.

    Parameters
    ----------
    func : callable
        Function f(x) returning array-like residuals.
    x0 : array-like
        Initial guess (within bounds).
    bounds : tuple
        (lower_bounds, upper_bounds) as lists/arrays.
    method : str
        Method for scipy.optimize.root (default 'hybr').
    tol : float
        Tolerance for convergence.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    result : OptimizeResult
        SciPy optimization result with .x as the bounded root.
    """
    lb, ub = np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)

    # Safety check
    if np.any(lb >= ub):
        raise ValueError("Lower bounds must be strictly less than upper bounds.")

    # Transform bounded x -> unbounded u (inverse sigmoid)
    def to_unbounded(x):
        return np.log((x - lb) / (ub - x))

    # Transform unbounded u -> bounded x (sigmoid mapping)
    def to_bounded(u):
        return lb + (ub - lb) / (1 + np.exp(-u))

    # Wrapped function for root solver
    def wrapped(u):
        x = to_bounded(u)
        return np.atleast_1d(func(x))

    # Ensure initial guess is inside bounds
    x0 = np.clip(x0, lb + 1e-12, ub - 1e-12)
    u0 = to_unbounded(x0)

    # Solve in unbounded space
    res = optimize.root(wrapped, u0, method=method, tol=tol, options={'maxfev': maxiter})

    # Map back to bounded space
    res.x = to_bounded(res.x)
    return res




def fit_plot( tupfun, x0, 
             *,ax=None,  nolabels=False , y2lim=(-5,5), **kwargs):
    """Plot likelihood and derivative around the local optimum.

    The x-axis is shown in units of the estimated local sigma, derived from the
    second derivative at the fitted root of the gradient.

    Parameters
    ----------
    tupfun : callable
        Function taking a single float argument and returning a tuple of
        (-log likelihood, -derivative)
    x0 : float
        Initial position used by the root finder.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    nolabels : bool, optional
        If True, do not add axis labels.
    y2lim : tuple, optional
        Y-axis limits for derivative panel.
    **kwargs
        Additional keyword arguments forwarded to ``ax.set(...)``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the diagnostic plot.

    Raises
    ------
    RuntimeError
        If the root finder does not converge.
    """
    import matplotlib.pyplot as plt
    from scipy import optimize
    func = lambda x: -tupfun(x)[0]
    fprime = lambda x : -tupfun(x)[1] 

    # find root and hessian (second derivative)
    sol = optimize.root(fprime, x0)
    if not sol.success:
        raise RuntimeError('fit_plot: root finding failed: %s' % sol.message)
    mu, sigma = sol.x[0], 1/np.sqrt(sol.r[0]) # Normal approximation: mu and sigma
    ref = func(mu)

    fig, ax = plt.subplots( figsize=(5,4)) if ax is None else (ax.figure, ax)
   
    xsig = np.linspace(-3, 3, 27)
    x =  mu + xsig * sigma 
    ax.plot(xsig, np.array(list(map(func,x)))-ref, '-b') # plot of log likelihood
    ax.plot(xsig, -0.5*((xsig)/sigma)**2, '--b') # plot of expected likelihood shape

    ax.set( ylim=(-5,0.5), xlim=(-4,4))
    ax.axvline(0, color='grey', ls = ':')
    ax.axhline(0, color='grey', ls = ':')
    ax.plot([-1,1], [-0.5,-0.5], '|-w')
    ax.set_xticks([-2,0,2])
    ax.grid(False)
    ax.text( 0,-4, rf'{mu:9.3f} $\pm$ {sigma:5.3f}', size=12, ha='center', backgroundcolor='k' )   
    ax.set(**kwargs)
    if not np.isnan(sigma):
        ax2 = ax.twinx()
        gradvals = -sigma*np.array(list(map(fprime, x)))
        ax2.plot(xsig, gradvals, '-r')
        ax2.axhline(0, color='r', ls=':')
        ax2.set_ylim( y2lim)
        if not nolabels: 
            ax.set_ylabel('log likelihood', fontsize=14)
            ax2.set_ylabel(r'derivative ($\sigma$ units)', fontsize=14)
            ax.set_xlabel(r'value ($\sigma$ units)', fontsize=14)
        else: ax2.set_yticklabels([])
        ax2.grid(False)
    return fig



class FitterSummaryMixin(object):
    """Mixin that formats tabular summaries for fitter parameters."""

    # Interface expected from concrete fitter views.
    def gradient(self):
        raise NotImplementedError

    def hessian(self):
        raise NotImplementedError

    @property
    def mask(self):
        raise NotImplementedError

    @property
    def parameter_names(self):
        raise NotImplementedError

    @property
    def model_parameters(self):
        raise NotImplementedError

    @property
    def uncertainties(self):
        raise NotImplementedError
    
    def summary(self: Any, select=None, exclude=None, out=None, title=None, gradient=True):
        """Print summary table for current free parameters.

        Parameters
        ----------
        select : list[int] or str or None
            Reserved selector argument (currently not applied in body).
        exclude : list[int] or str or None
            Reserved exclusion argument (currently not applied in body).
        out : file-like or None
            Output stream passed to ``print``.
        title : str or None
            Optional heading line.
        gradient : bool
            If True, include gradient values.
        """
        if out is None and not _is_verbose(self):
            return

        if title is not None:
            print(title, file=out)

        fmt, tup = '%-21s%6s%10s%10s', tuple('Name index value error(%)'.split())
        if gradient:
            grad = self.gradient()
            fmt +='%10s'; tup += ('gradient',)
        print(fmt %tup, file=out)
        prev=''
        selected = (select, exclude)
        index_array = np.arange(len(self.mask))[self.mask]
        for index, (name, value, rsig) in enumerate(zip(self.parameter_names, 
                                                        self.model_parameters, 
                                                        self.uncertainties)):
            t = name.split('_')
            pname = t[-1]
            sname = '_'.join(t[:-1])
            if sname==prev: name = len(sname)*' '+'_'+pname
            prev = sname
            fmt = '%-21s%6d%10.4g%10s'
            psig = '%.1f'%(rsig*100) if rsig>0 and not np.isnan(rsig) else '***'
            truncname = name[:20]+'*' if len(name)>20 else name
            tup = (truncname, index_array[index], value,psig)
            if gradient:
                fmt +='%10.1f'; tup += (grad[index],)
            print(fmt % tup, file=out)
    
    def delta_loglike(self: Any, quiet=True):
        """Estimate expected log-likelihood improvement from local quadratic form.

        Parameters
        ----------
        quiet : bool
            Reserved compatibility argument.

        Returns
        -------
        float
            Approximate expected improvement in log-likelihood. Returns 99.0 if
            the estimate fails.
        """
        try:
            # Old code assuming now-deprecated numpy.matrix
            # gm = np.matrix(self.gradient())
            # H = self.hessian()
            # return (gm * H.I * gm.T)[0,0]/4
            gv = self.gradient()
            H = np.array(self.hessian()) # in case matrix type
            return np.dot(np.dot(gv, np.linalg.inv(H)), gv)/4
        except Exception as msg:
            _verbose_print(self, 'Failed log likelihood estimate, returning 99.: %s' % msg)
            return 99.


class FitPlotMixin(object):
    """Mixin with likelihood-profile plotting helpers."""

    # Interface expected from concrete fitter views.
    def get_parameters(self):
        raise NotImplementedError

    def set_parameters(self, pars):
        raise NotImplementedError

    def gradient(self, pars=None):
        raise NotImplementedError

    def hessian(self, pars=None):
        raise NotImplementedError

    def __call__(self, pars=None):
        raise NotImplementedError

    parameters: Any  # concrete subclass must provide this attribute

    @property
    def parameter_names(self):
        raise NotImplementedError

    @property
    def mask(self):
        raise NotImplementedError
    
    def estimate_solution(self: Any):
        """Estimate local optimum and parameter sigmas from Hessian.

        Returns
        -------
        tuple
            ``(parz, parmax, sigs)`` where ``parz`` are current parameters,
            ``parmax`` is a one-step quadratic estimate of the optimum, and
            ``sigs`` are 1-sigma estimates from the covariance diagonal.
        """
        parz = self.get_parameters()
        hess = np.array(self.hessian(parz), dtype=float)

        # Some callers provide Hessian(logL) (negative-definite near optimum)
        # while others provide Hessian(-logL). Pick the sign that yields a
        # physically meaningful covariance (non-negative diagonal).
        if hess.ndim == 0:
            candidates = [np.array([[hess]], dtype=float), np.array([[-hess]], dtype=float)]
        else:
            candidates = [hess, -hess]

        cov = None
        best_diag = None
        for h in candidates:
            try:
                c = np.linalg.inv(h)
            except np.linalg.LinAlgError:
                continue
            diag = np.diag(c).copy()
            good = np.isfinite(diag) & (diag >= 0)
            score = int(np.count_nonzero(good))
            if cov is None or score > int(np.count_nonzero(np.isfinite(best_diag) & (best_diag >= 0))):
                cov = c
                best_diag = diag

        if cov is None:
            # Final fallback for near-singular cases used only for plotting.
            h = candidates[0]
            cov = np.linalg.pinv(h)
            best_diag = np.diag(cov).copy()

        best_diag[best_diag < 0] = np.nan
        sigs = np.sqrt(best_diag)
        parmax = parz - self.gradient(parz) * sigs**2 / 2.
        return parz, parmax, sigs

    def plot(self: Any, index, ax=None, nolabels=False , y2lim=(-5,5), estimate=None):
        """Plot 1D likelihood profile for a selected parameter.

        Parameters
        ----------
        index : int
            Index of parameter to profile.
        ax : matplotlib.axes.Axes or None
            Target axis; if None, create a new one.
        nolabels : bool
            If True, suppress axis labels.
        y2lim : tuple
            Limits for derivative axis.
        estimate : tuple or None
            Optional precomputed ``(parz, parmax, sigs)`` tuple.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the profile plot.
        """
        import matplotlib.pyplot as plt
        # get current parameters, gradient, and the Hessian for estimate of max liklihood position
        parz, parmax, sigs = self.estimate_solution() if estimate is None else estimate
        pz = parz[index]
        part = parz.copy()
        def func(x):
            part[index]=x
            return -self(part) #(restore sign for minimization)
        def gradf(x):
            part[index]=x
            return self.gradient(part)[index]
        x0, sig = (parmax)[index], sigs[index]
        ref = func(x0)
        if ax is None:
            fig, ax = plt.subplots( figsize=(3,3))
        else: fig = ax.figure
        if not np.isnan(sig):
            xsig = np.linspace(-3, 3, 27)
            x =  x0 + xsig * sig 
            profile = np.array(list(map(func, x)), dtype=float)
            ax.plot(xsig, profile - ref, '-', color='orange')
            ax.plot(xsig, -0.5*((x-x0)/sig)**2, '--', color='orange')
            ax.plot((pz-x0)/sig, func(pz)-ref, 'dk')
            ax.plot([-1,1], [-0.5,-0.5], '|-')
            ax.grid(False)
        plt.setp(ax, ylim=(-5,0.5), xlim=(-4,4))
        if not nolabels: ax.set_ylabel('log likelihood')
        j = np.arange(len(self.mask))[self.mask][index]if hasattr(self,'mask') else index
        ax.set_title('#{}: {}'.format(j,self.parameter_names[index]), size=10)
        ax.axvline(0, color='grey', ls = ':')
        ax.set_xticks([-2,0,2])
        if not np.isnan(sig):
            ax2 = ax.twinx()
            gradvals = -sig*np.array(list(map(gradf, x)))
            ax2.plot(xsig, gradvals, '-r')
            ax2.axhline(0, color='r', ls=':')
            ax2.set_ylim( y2lim)
            ax2.grid(False)
            if not nolabels: 
                ax2.set_ylabel('derivative (sig units)')
                ax.set_xlabel('value (sig units)')
            else: ax2.set_yticklabels([])
        ax.text( 0,-4, r'{:9.3f}$\pm$ {:5.3f}'.format(pz,sig), size=10, family='monospace', 
            backgroundcolor= fig.get_facecolor(),  ha='center' )
        self.set_parameters(parz) # restore when done
        return fig
    
    def plot_all(self: Any, perrow=5, figsize=None): 
        """Plot profile diagnostics for all free parameters.

        Parameters
        ----------
        perrow : int
            Number of panels per row.
        figsize : tuple or None
            Figure size passed to ``matplotlib``; defaults to a size derived
            from the number of rows.
        """
        import matplotlib.pyplot as plt
        n = len(self.parameters)
        if n==1:
            return self.plot(0)
        estimate = self.estimate_solution()
        rows = (n+perrow-1)//perrow
        if figsize is None:
            figsize = (12, 2.5*rows)
        fig, axx = plt.subplots(rows,perrow, 
            figsize=figsize, sharex=True, sharey=True)
        for i, ax in enumerate(axx.flatten()):
            if i>=n: 
                ax.set_visible(False)
            else:
                self.plot(i, ax = ax, nolabels=True, estimate=estimate)
        plt.show()
        return


class FitterMixin(object):
    """Mixin providing likelihood maximization and derivative diagnostics."""

    # Interface expected from concrete fitter views.
    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def parameter_names(self):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def set_parameters(self, pars):
        raise NotImplementedError

    def gradient(self, pars=None):
        raise NotImplementedError

    def log_like(self, summed=True):
        raise NotImplementedError

    def __call__(self, pars=None):
        raise NotImplementedError

    def maximize(self: Any,  **kwargs):
        """Maximize likelihood and optionally estimate parameter uncertainties.

        Uses ``scipy.optimize.fmin_l_bfgs_b``.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the optimizer and behavior switches.

        Other Parameters
        ----------------
        m : int
            The maximum number of variable metric corrections
            used to define the limited memory matrix. (The limited memory BFGS
            method does not store the full hessian but uses this many terms in an
            approximation to it.)
        factr : float
            The iteration stops when
            ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
            where ``eps`` is the machine precision, which is automatically
            generated by the code. Typical values for `factr` are: 1e12 for
            low accuracy; 1e7 for moderate accuracy; 10.0 for extremely
            high accuracy.
        pgtol : float
            The iteration will stop when
            ``max{|proj g_i | i = 1, ..., n} <= pgtol``
            where ``pg_i`` is the i-th component of the projected gradient.
        epsilon : float
            Step size used when `approx_grad` is True, for numerically
            calculating the gradient
        iprint : int
            Controls the frequency of output. ``iprint < 0`` means no output;
            ``iprint == 0`` means write messages to stdout; ``iprint > 1`` in
            addition means write logging information to a file named
            ``iterate.dat`` in the current working directory.
        disp : int, optional
            If zero, then no output.  If a positive number, then this over-rides
            `iprint` (i.e., `iprint` gets the value of `disp`).
        maxfun : int
            Maximum number of function evaluations.
        maxiter : int
            Maximum number of iterations.
        quiet : bool
            If False, print minimizer diagnostics.
        use_gradient : bool
            Kept for compatibility; gradients are used when available.
        estimate_errors : bool
            If True, invert Hessian at optimum to estimate 1-sigma errors.
        approx_grad : bool
            If True, use finite-difference gradients in L-BFGS-B.

        Returns
        -------
        tuple
            ``(fmin, pars, sigmas)`` where ``fmin`` is minimized objective,
            ``pars`` are best-fit parameters, and ``sigmas`` are estimated
            standard deviations or ``np.nan`` if ``estimate_errors=False``.

        Raises
        ------
        Exception
            If minimization fails according to the optimizer warn flag.

        """
        from scipy import optimize
        quiet = kwargs.pop('quiet', True)
        if not kwargs.pop('use_gradient', True):
            _verbose_print(self, 'Warning: ignoring use_gradient=False')
        estimate_errors = kwargs.pop('estimate_errors', True)
        if not quiet:
            _verbose_print(self, 'using optimize.fmin_l_bfgs_b with parameter bounds %s\n, kw= %s'% (
                self.bounds, kwargs
            ))
        parz = self.get_parameters()
        winit = self.log_like()
        # assert len(parz)==len(self.gradient()), 'tracking a bug'

        # Defaults from scipy.optimize.fmin_l_bfgs_b, with project-specific tuning.
        m = int(kwargs.pop('m', 10))
        factr = float(kwargs.pop('factr', 1e9))
        pgtol = float(kwargs.pop('pgtol', 1e-3))
        epsilon = float(kwargs.pop('epsilon', 1e-08))
        iprint = int(kwargs.pop('iprint', -1))
        maxfun = int(kwargs.pop('maxfun', 15000))
        maxiter = int(kwargs.pop('maxiter', 15000))
        approx_grad = bool(kwargs.pop('approx_grad', False))

        # run the fit
        ret = optimize.fmin_l_bfgs_b(self, parz, 
                bounds=self.bounds,  
                fprime=None if approx_grad else self.gradient, 
                approx_grad=approx_grad, 
            m=m,
            factr=factr,
            pgtol=pgtol,
            epsilon=epsilon,
            iprint=iprint,
            maxfun=maxfun,
            maxiter=maxiter,
            **kwargs)
        self.fmin_ret=ret
        if ret[2]['warnflag']>0: 
            _verbose_print(self, 'Fit failure: check parameters')
            self.set_parameters(parz) #restore if error 
            self.covariance = None 
            raise Exception( 'Fit failure:\n%s' % ret[2])
        if not quiet:
            _verbose_print(self, ret[2])
        f = ret 
        if estimate_errors:
            # maximize(logL) by minimizing(-logL): cov = inv(-d2logL/dp2)
            self.covariance = cov = np.linalg.inv(-self.hessian(f[0]))
            diag = np.array(cov.diagonal()).flatten()
            bad = diag<0
            if np.any(bad):
                _verbose_print(self, 'Minimizer warning: bad errors for values %s'\
                    %np.asarray(self.parameter_names)[bad]) 
                diag[bad]=0
            return f[1], f[0], np.sqrt(diag)
        else:
            self.covariance = None
            return f[1], f[0], np.nan

    def hessian(self: Any, pars=None, **kwargs):
        """Compute numerical Hessian matrix using ``numdifftools``.

        Parameters
        ----------
        pars : array-like or None
            Parameter vector. If None, uses current parameters.
        **kwargs
            Forwarded to ``numdifftools.Hessian``.

        Returns
        -------
        numpy.matrix
            Hessian matrix at ``pars``.
        """
        import numdifftools
        if pars is None: pars = self.get_parameters()
        return np.matrix(numdifftools.Hessian(self,  **kwargs)(pars))

        
        
    def modify(self: Any, fraction):
        """Move saved initial parameters toward current parameters.

        Parameters
        ----------
        fraction : float
            Fraction of ``(current - initial)`` to apply to ``initial``.
        """
        if fraction==0 : return
        delta = self.get_parameters()-self.initial_parameters
        self.initial_parameters += fraction * delta
        
    def restore(self: Any):
        """Restore parameters to the saved initial state."""
        _verbose_print(self, f'set parameters to {self.initial_parameters}')
        self.set_parameters(self.initial_parameters)

    def check_gradient(self: Any, delta=1e-5):
        """Compare analytic gradient with central-difference estimate.

        Parameters
        ----------
        delta : float
            Step size used for central finite differences.

        Returns
        -------
        tuple
            ``(analytic, numeric)`` gradient arrays.
        """
        
        parz = self.get_parameters()
        fz = self(parz)
        grad = self.gradient(parz)
        fprime=[]
        for i in range(len(parz)):
            parz[i]+=delta
            fplus = self(parz)
            assert abs(fplus-fz)>1e-7, 'Fail consistency: variable %d not changing' % i
            parz[i]-=2*delta
            fminus = self(parz)
            parz[i]+= delta
            fzero = self(parz)
            assert abs(fzero-fz)<1e-2, 'Fail consistency: %e, %e ' % (fzero, fz)
            fprime.append((fplus-fminus)/(2*delta))
        return grad, np.array(fprime) 

class WithMixin(object):
    """Mixin enabling context-manager state restoration.

    Classes using this mixin must define ``restore()``. On context exit,
    ``restore()`` is always invoked.

    Examples
    --------
    with ClassName(...) as obj:
        ...
    """
    def __enter__(self: Any):
        return self

    def restore(self):
        raise NotImplementedError
        
    def __exit__(self: Any, type, value, traceback):
        self.restore()


class FitterView(FitPlotMixin, FitterMixin, FitterSummaryMixin, WithMixin): 
    """Full-parameter fitter view over all currently free source parameters."""

    def __init__(self, blike,  **kwargs):
        """Initialize fitter for all free parameters in *blike*.

        Parameters
        ----------
        blike : LikelihoodViews
            Band-likelihood container providing ``parameterset``,
            ``log_like``, ``gradient``, ``hessian``, and ``update``.
        **kwargs
            Reserved; currently unused.
        """
        self.blike = blike
        self.verbose = getattr(blike, 'verbose', True)
        self.parameters = blike.parameterset
        # self.parameters = blike.sources.parameters
        self.sources = self.parameters.free_sources
        self.initial_parameters = self.parameters[:]
        self.initial_likelihood = self.log_like()
        self.calls=0
        
    def get_parameters(self):
        """Return current free-parameter vector."""
        return self.parameters.get_parameters()
    def set_parameters(self, pars):
        """Set free parameters and refresh bandlike state."""
        self.parameters.set_parameters(pars)
        self.blike.update()
        
    def save_covariance(self):
        """Store per-source covariance submatrices into source model objects.

        Notes
        -----
        Cross-source covariance terms are not preserved by this projection.
        """
        assert hasattr(self, 'covariance'), 'maximize was not run: no covariance to save'
        self.parameters.set_covariance(self.covariance)
    
    def modify(self, fraction):
        """Update saved initial parameters toward current values.

        Parameters
        ----------
        fraction : float
            Interpolation factor between saved and current parameters.
        """
        if fraction==0 : return
        delta = self.get_parameters()-self.initial_parameters
        if fraction==1:
            _verbose_print(self, 'set parameters to current values')
            self.initial_parameters = self.get_parameters()
            return
        # fractional change
        self.initial_parameters += fraction * delta
        _verbose_print(self, f'parameters to save {self.initial_parameters}')
        
    def restore(self):
        """Restore parameters and print post-restore gradient."""
        _verbose_print(self, f'set parameters to {self.initial_parameters}')
        self.set_parameters(self.initial_parameters)
        _verbose_print(self, f'gradient after restore: {self.gradient()}')

    @property 
    def bounds(self):
        """Parameter bounds for L-BFGS-B, concatenated from all free source models."""
        return np.concatenate([s.model.bounds[s.model.free] for s in self.sources]) 
    def __call__(self, pars=None):
        """Return negative log-likelihood, optionally setting free parameters first."""
        if pars is not None: self.set_parameters(pars)
        self.calls+=1
        return -self.blike.log_like()
    def log_like(self, summed=True):
        """Return log-likelihood for current state.

        Parameters
        ----------
        summed : bool
            If True, return scalar total; otherwise return per-component values.
        """
        return self.blike.log_like(summed=summed)

    def gradient(self,pars=None):
        """Return gradient of objective (-log-likelihood) for free parameters."""
        if pars is not None: self.set_parameters(pars)
        return -self.blike.gradient()
    def hessian(self, pars=None):
        """Return Hessian matrix, computing at pars if provided."""
        if pars is not None: self.set_parameters(pars)
        return self.blike.hessian()
    @property
    def parameter_names(self):
        """Names of all currently free parameters, in order."""
        return self.parameters.parameter_names
    @property
    def model_parameters(self):
        """Current values of all free model parameters."""
        return self.parameters.model_parameters
    @property
    def uncertainties(self):
        """Relative uncertainties for all free parameters."""
        return self.parameters.uncertainties
    @property
    def mask(self):
        """Boolean mask marking free parameters in the full parameter vector."""
        return self.parameters.mask
             

class SubsetFitterView(parameterset.ParSubSet, FitPlotMixin, FitterMixin, FitterSummaryMixin, WithMixin):
    """Fitter view restricted to a selected subset of free parameters."""

    def __init__(self, blike, select=None, exclude=None):
        """Initialize a subset fitter for a selection of free parameters.

        Parameters
        ----------
        blike : LikelihoodViews
            Band-likelihood container.
        select : str, list, or None
            Parameter names or indices to include; see :class:`parameterset.ParSubSet`.
        exclude : str, list, or None
            Parameter names or indices to exclude.
        """
        self.blike = blike
        self.verbose = getattr(blike, 'verbose', True)
        selected = []
        if select is not None:
            selected.append(select)
        super().__init__(blike.sources, *selected)
        if exclude is not None:
            excluded = self.select_parameters(exclude)
            mask = self.mask.copy()
            mask[list(excluded)] = False
            self.set_mask(mask)
        self.initial_parameters = self.parameters[:]
        self.initial_likelihood = self.log_like()
        self.calls=0

    def __repr__(self):
        """Short string including the module, class, and selection description."""
        return '%s.%s: %s '% (self.__module__, self.__class__.__name__, self.selection_description)
    def restore(self):
        """Restore subset parameters to their saved initial values."""
        self.set_parameters(self.initial_parameters)

    def save_covariance(self):
        """Store subset covariance matrix back into model parameter container.

        Notes
        -----
        Cross-source covariance terms are not preserved by this projection.
        """
        assert hasattr(self, 'covariance'), 'maximize was not run: no covariance to save'
        self.set_covariance(self.covariance)

    @property
    def parameters(self):
        """Current subset parameter values (re-read on each access)."""
        return self.get_parameters()
    def set_parameters(self, pars):
        """Set subset parameters and refresh band-likelihood state."""
        super().set_parameters(pars)
        self.blike.update()
    def __call__(self, pars=None):
        """Return negative log-likelihood, optionally setting subset parameters first."""
        if pars is not None: self.set_parameters(pars)
        self.calls +=1
        return -self.blike.log_like()
        
    def log_like(self, *,summed=True):
        """Return log-likelihood for current state.

        Parameters
        ----------
        summed : bool
            If True, return scalar total; otherwise return per-component values.
        """
        return self.blike.log_like(summed=summed)
    def gradient(self, pars=None):
        """Return subset gradient by masking the full band-likelihood gradient."""
        if pars is not None: self.set_parameters(pars)
        return self.blike.gradient()[self.mask]
    #### This seems wrong! #####
    # def hessian(self,pars=None):
    #     if pars is not None: self.set_parameters(pars)
    #     return self.blike.hessian(self.mask) 
        
    def ts(self):
        """Compute simple TS by forcing first subset parameter to a low value."""
        lnow = self()
        pars = self.parameters
        pars[0] = -20 ##### override to be really small self.bounds[0][0]
        return 2 * (self(pars)-lnow)
        

class TSmapView(WithMixin):
    """Context-managed view for source-position scanning (TS maps).

    Moves a source's ``skydir`` attribute and evaluates the likelihood
    difference ``2*(L - L0)`` relative to the saved reference likelihood.

    Parameters
    ----------
    blike : LikelihoodViews
        Band-likelihood container.
    func : FitterView
        Fitter view with the source normalization parameter selected.
    quiet : bool
        If True, suppress diagnostic output.
    """

    def __init__(self, blike, func, quiet=True):
        self.quiet = quiet
        self.func = func
        self.blike = blike
        self.source = self.func.source
        self.saved_skydir = self.get_dir()
        self.wzero = func.log_like()
    
    def __repr__(self):
        """Short representation showing the source name."""
        return '%s.%s: source %s' % (self.__module__, self.__class__.__name__, self.source.name)
        
    def set_dir(self, skydir):
        """Move the source to skydir and reinitialize band-likelihood."""
        self.source.skydir = skydir
        self.blike.initialize(None, self.source.name ) #sourcenane=self.source.name)
    
    def get_dir(self):
        """Return the current source sky direction."""
        return self.source.skydir
    skydir = property(get_dir, set_dir)
    
    def restore(self):
        """Restore source to its saved initial position."""
        self.set_dir(self.saved_skydir)

    def __call__(self, skydir=None):
        """Return TS = 2*(L(skydir) - L0) at the given position.

        Parameters
        ----------
        skydir : SkyDir, (ra, dec) tuple, or None
            Position to evaluate. If None, evaluates at the current position.

        Returns
        -------
        float
            Test statistic value.
        """
        if skydir is not None:
            if not isinstance(skydir, SkyDir):
                skydir = SkyDir(*skydir)
            self.set_dir(skydir)
        return 2*(self.func.log_like()-self.wzero)


class EnergyFluxView(WithMixin):
    """Context-managed view expressing likelihood as a function of differential energy flux.

    The underlying normalization parameter is mapped to energy flux in eV
    units at the specified energy.  ``__call__`` is decorated as a ufunc so
    it can be evaluated over arrays.

    Parameters
    ----------
    blike : LikelihoodViews
        Band-likelihood container.
    source_name : str
        Name of the source whose Norm parameter is selected for fitting.
    energy : float or None
        Evaluation energy in MeV. If None, uses the model reference energy.
    **kw
        ``bound`` (float): lower bound for the internal log-norm parameter.
    """

    def __init__(self, blike, source_name, energy, **kw):

        self.blike = blike
        self.source = source = blike.sources.find_source(source_name)
        self.model = model = source.spectral_model
        norm_param = source_name + '_' + model.param_names[0]
        self.func = blike.fitter_view(norm_param)
        #assert model[0]==model['norm']
        self.norm = model[0]
        self.tointernal = model.mappers[0].tointernal
        self.bound = kw.get('bound', -20)# !!! model.bounds[0][0])
        self.set_energy(energy)

    def set_energy(self, energy=None):
        """Reset evaluation energy (or restore to reference energy if None).

        Parameters
        ----------
        energy : float or None
            New energy in MeV; defaults to the model reference energy.
        """
        if energy is None:
            energy=self.model.e0
        self.model[0]=self.norm # get original norm 
        self.source.changed=True
        self.blike.update()
        self.energy = energy
        self.eflux = self.model(energy) * energy**2 * 1e6
        self.ratio = self.model[0]/self.eflux
    
    def __repr__(self):
        """Short representation showing func name and evaluation energy."""
        return '%s.%s: func=%s, at %.0f MeV' % (self.__module__, self.__class__.__name__,self.func,self.energy)
    def restore(self):
        """Restore normalization to the model reference energy."""
        self.set_energy()

    @tools.ufunc_decorator # make this behave like a ufunc
    def __call__(self, eflux):
        """Return negative log-likelihood as a function of differential energy flux.

        Parameters
        ----------
        eflux : float
            Differential energy flux at ``self.energy`` in eV units.

        Returns
        -------
        float
            Negative log-likelihood value.
        """
        if eflux<=0:
            par = self.bound
        else:
            par = max(self.bound, self.tointernal(eflux*self.ratio))
        return -self.func([par])
        
class NormalizationView(WithMixin):
    """Context-managed view expressing likelihood as a function of source normalization.

    If the normalization parameter is frozen, it is temporarily thawed and
    restored on context exit.  ``__call__`` is decorated as a ufunc.

    Parameters
    ----------
    blike : LikelihoodViews
        Band-likelihood container.
    source_name : str
        Name of the source whose normalization parameter is varied.
    """
    def __init__(self, blike, source_name):
        """Initialize, thawing normalization if needed before constructing the view."""
        self.blike = blike
        source = blike.sources.find_source(source_name)
        self.model = model= source.model
        self.par = model[0]
        parname = model.param_names[0]
        if not model.free[0]:
            self.freed = (parname, source_name)
            blike.thaw(parname, source_name)
        else: self.freed=None
        self.func = blike.fitter_view(source_name + '_' + parname)

        self.tointernal = model.mappers[0].tointernal
        self.bound = model.bounds[0][0]
    
    @tools.ufunc_decorator
    def __call__(self, norm):
        """Return negative log-likelihood as a function of normalization factor.

        Parameters
        ----------
        norm : float
            Normalization factor to evaluate.

        Returns
        -------
        float
            Negative log-likelihood value.
        """
        if norm <=0:
            par = self.bound
        else:
            par = max(self.bound, self.tointernal(norm*self.par))
        return -self.func([par])
    
    def restore(self):
        """If had to thaw, restore"""
        self.func.restore()
        if self.freed is not None and self.model.free[0]:
            self.blike.freeze(*self.freed)
        

class LikelihoodViews(object):
    """Container of per-band likelihoods with factory methods for analysis views.

    Provides convenience constructors for:

    * ``fitter_view`` — :class:`FitterView` or :class:`SubsetFitterView`
    * ``energy_flux_view`` — :class:`EnergyFluxView`
    * ``tsmap_view`` — :class:`TSmapView`
    * ``normalization_view`` — :class:`NormalizationView`
    """

    sources: Any  # provided by base class or initializer

    @property
    def parameterset(self):
        """Current parameter set for the attached source model.

        When the underlying source model is reinitialized after thaw/freeze or
        model replacement, this property follows the current
        ``sources.parameters`` object instead of holding a stale cached view.
        """
        if getattr(self, 'sources', None) is not None:
            current = getattr(self.sources, 'parameters', None)
            if current is not None:
                return current
        return getattr(self, '_parameterset', None)

    @parameterset.setter
    def parameterset(self, value):
        self._parameterset = value

    def __init__(self, bands_or_pixel_table, sources=None, verbose=True):
        """Initialize likelihood views from either PixelTable or legacy parts.

        Parameters
        ----------
        bands_or_pixel_table : object
            Preferred: ``like3.pixel_table.PixelTable`` instance.
            Legacy: band container object.
        sources : object or None
            Legacy source-model object when using the 2-argument constructor.
        verbose : bool, optional
            If False, suppress view-managed diagnostic printing.

        Notes
        -----
        New code should pass a PixelTable directly. Legacy callers that still
        pass ``(bands, sources)`` remain supported.
        """
        self.verbose = bool(verbose)

        # New preferred path: PixelTable-like object.
        if sources is None and hasattr(bands_or_pixel_table, '_iter_bands') \
                and hasattr(bands_or_pixel_table, 'source_model'):
            self.pixel_table = bands_or_pixel_table
            self.bands = bands_or_pixel_table
            self.sources = bands_or_pixel_table.source_model
            self.parameterset = self.sources.parameters if self.sources is not None else None
            return

        # Legacy path: explicit bands + sources.
        if sources is not None:
            self.bands = bands_or_pixel_table
            self.sources = sources
            self.parameterset = getattr(self.sources, 'parameters', None)
            return

        raise TypeError(
            'LikelihoodViews expects a PixelTable instance or (bands, sources)'
        )
    
    def fitter_view(self, select=None, setpars=None, **kwargs):
        """Return a fitter view over all or a subset of free parameters.

        Parameters
        ----------
        select : str, list, or None
            If None, returns a :class:`FitterView` over all free parameters.
            Otherwise, constructs a :class:`SubsetFitterView` for the selection.
        setpars : dict or None
            If provided, set these parameter values before constructing the view.
        **kwargs
            Forwarded to the view constructor.

        Returns
        -------
        FitterView or SubsetFitterView
        """
        if setpars is not None: 
            self.sources.parameters.setitems(setpars)

        if select is None:
            return FitterView(self, **kwargs)
        return SubsetFitterView(self, select, **kwargs)

    def log_like(self, summed=True):
        """Return total log-likelihood for current parameter state.

        For PixelTable-backed instances this delegates to ``pixel_table.loglike``.
        Legacy containers are supported if they expose ``log_like`` or ``loglike``.
        """
        if hasattr(self, 'pixel_table') and hasattr(self.pixel_table, 'loglike'):
            return self.pixel_table.loglike()
        if hasattr(self.bands, 'log_like'):
            return self.bands.log_like(summed=summed)
        if hasattr(self.bands, 'loglike'):
            return self.bands.loglike()
        raise AttributeError('No log-likelihood provider found for LikelihoodViews')

    def update(self):
        """Refresh cached likelihood state after parameter changes."""
        if hasattr(self.bands, 'update'):
            self.bands.update()

    def initialize(self, *args, **kwargs):
        """Reinitialize band-likelihood internals when available."""
        if hasattr(self.bands, 'initialize'):
            return self.bands.initialize(*args, **kwargs)

    def gradient(self, pars=None):
        """Numerical gradient of total log-likelihood for current free parameters."""
        import numdifftools
        if self.parameterset is None:
            raise AttributeError('No parameterset available to compute gradient')
        pset = self.parameterset
        saved = pset.get_parameters().copy()
        try:
            if pars is not None:
                pset.set_parameters(pars)
            x0 = pset.get_parameters().copy()

            def f(x):
                pset.set_parameters(x)
                self.update()
                return self.log_like()

            return np.array(numdifftools.Gradient(f)(x0), float)
        finally:
            pset.set_parameters(saved)
            self.update()

    def hessian(self, pars=None):
        """Numerical Hessian of total log-likelihood for current free parameters."""
        import numdifftools
        if self.parameterset is None:
            raise AttributeError('No parameterset available to compute hessian')
        pset = self.parameterset
        saved = pset.get_parameters().copy()
        try:
            if pars is not None:
                pset.set_parameters(pars)
            x0 = pset.get_parameters().copy()

            def f(x):
                pset.set_parameters(x)
                self.update()
                return self.log_like()

            return np.array(numdifftools.Hessian(f)(x0), float)
        finally:
            pset.set_parameters(saved)
            self.update()

    def energy_flux_view(self, source_name, energy=None, **kw):
        """Return a functor expressing log-likelihood as a function of energy flux.

        Parameters
        ----------
        source_name : str
            Source whose normalization is profiled.
        energy : float or None
            Energy in MeV. If None, uses the model reference energy e0.
        **kw
            Forwarded to :class:`EnergyFluxView`.

        Returns
        -------
        EnergyFluxView

        Raises
        ------
        Exception
            If the source cannot be found or the view cannot be constructed.
        """
        try:
            source = self.sources.find_source(source_name)
        except Exception as msg:
            raise Exception('could not create energy flux function for source %s;%s' %(source_name, msg))
        return EnergyFluxView(self, source.name, energy, **kw)
        
    def tsmap_view(self, source_name, **kw):
        """Return a TS-map view for position scanning of the named source.

        Parameters
        ----------
        source_name : str or None
            Source name. If None, uses the currently selected source.
        **kw
            Forwarded to :class:`TSmapView`.

        Returns
        -------
        TSmapView

        Raises
        ------
        Exception
            If no source is identified or the view cannot be constructed.
        """
        if source_name is None and self.sources.selected_source is not None:
            source_name = self.sources.selected_source.name 
        if source_name is None: 
            raise Exception('No source is selected for a tsmap')
        try:
            func = self.fitter_view(source_name+'_Norm')
        except Exception as msg:
            raise Exception('could not create tsmap function for source %s;%s' %(source_name, msg))
        return TSmapView(self, func, **kw)
        
    def normalization_view(self, source_name):
        """Return a normalization view for the named source.

        Parameters
        ----------
        source_name : str
            Source name.

        Returns
        -------
        NormalizationView
        """
        return NormalizationView(self, source_name)

def make_views(roi_index, rings=2):
    """Convenience factory to build a :class:`LikelihoodViews` for testing.

    .. note::
        Requires legacy modules ``configuration``, ``bands.BandSet``, and
        ``from_healpix.ROImodelFromHealpix`` that are not part of like3.
        Calling this function will raise ``ImportError`` unless those modules
        are available on ``sys.path``.

    Parameters
    ----------
    roi_index : int
        HEALPix ROI index.
    rings : int
        Number of neighbour rings to include around the central pixel.

    Returns
    -------
    LikelihoodViews
    """
    from . import (configuration, bands, from_healpix)  # type: ignore[attr-defined]
    config = configuration.Configuration(quiet=True)
    roi_bands = bands.BandSet(config, roi_index, load=True)  # type: ignore[attr-defined]
    roi_sources = from_healpix.ROImodelFromHealpix(config, roi_index, load_kw=dict(rings=rings))  # type: ignore[attr-defined]
    return LikelihoodViews(roi_bands, roi_sources)  # type: ignore[call-arg]
