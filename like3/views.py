"""
classes presenting views of the likelihood engine in the module bandlike

Each has a mixin to allow the with ... as ... construction, which should restore the BandLikeList


$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/views.py,v 1.22 2017/11/17 22:50:36 burnett Exp $
Author: T.Burnett <tburnett@uw.edu> (based on pioneering work by M. Kerr)
"""

import sys, types
import numpy as np
from scipy import misc, optimize
from . skydir import SkyDir
# from . import (roimodel, bandlike, tools,)
from . import parameterset
"""
Got it ✅
Here’s a ready-made SciPy helper that lets you do bounded root finding for any dimension by wrapping scipy.optimize.root with a variable transformation so the solver never leaves your bounds.

Bounded Root Finder Wrapper
Pythonimport numpy as np

from scipy.optimize import root
"""

def bounded_root(func, x0, bounds, method='hybr', tol=1e-8, maxiter=1000):
    """
    Find a root of a function with bounds using variable transformation.

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
    res = root(wrapped, u0, method=method, tol=tol, options={'maxfev': maxiter})

    # Map back to bounded space
    res.x = to_bounded(res.x)
    return res
"""
# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Example: Solve cos(x) - x = 0 with bounds [0, 2]
    def f1(x):
        return np.cos(x[0]) - x[0]

    result = bounded_root(f1, x0=[1.0], bounds=([0], [2]))
    if result.success:
        print("Bounded root:", result.x)
    else:
        print("No solution:", result.message)

    # Example: 2D system with bounds
    def f2(vars):
        x, y = vars
        return [
            x**2 + y - 4,
            x - y**2 + 1
        ]

    result2 = bounded_root(f2, x0=[1, 1], bounds=([0, 0], [3, 3]))
    if result2.success:
        print("Bounded root (2D):", result2.x)
    else:
        print("No solution:", result2.message)
"""

# How It Works

# Sigmoid mapping keeps variables inside [lb, ub] automatically.
# root works in unbounded space (u variables).
# After solving, we map back to the bounded space.
# Works for 1D and multi-D problems.


# Advantages
# ✅ Works with any scipy.optimize.root method (hybr, lm, broyden1, etc.)
# ✅ No need for sign-change requirement like brentq
# ✅ Handles multi-dimensional problems with bounds
# ✅ Keeps solver stable by avoiding hard clipping during iterations

# If you want, I can extend this so it also supports equality and inequality constraints like minimize does, while still solving for roots.
# That would make it a full bounded + constrained root solver.
# Do you want me to build that extended version?




def fit_plot( tupfun, x0, 
             *,ax=None,  nolabels=False , y2lim=(-5,5), **kwargs):
    """make a plot showing the log likelihood and its derivative as a function of
    expected sigma, evaluated from the second derivative at the current point
   
    Parameters
    ----------
    tupfun : function
        function that takes a single float argument and returns a tuple of
        (-log likelihood, -derivative)
    x0 : float
        initial position to evaluate log likelihood and derivative
    ax : matplotlib.axes.Axes, optional
        axis to plot on, by default None
    nolabels : bool, optional
        if True, do not add axis labels, by default False
    y2lim : tuple, optional
        y-axis limits for derivative plot, by default (-5,5)
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
    """mixin to summarize variables"""
    
    def summary(self, select=None, exclude=None, out=None, title=None, gradient=True):
        """ summary table of free parameters, values uncertainties gradient
        
        Parameters:
        ----------
        select : list of integers or string
            integers are indices of parameters
            string is the wildcarded name of a source
        out : open file or None
        title: None or string
        gradient: bool
            set False to not print gradient
            
        """
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
    
    def delta_loglike(self, quiet=True):
        """ estimate change in log likelihood from current gradient 
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
            print('Failed log likelihood estimate, returning 99.: %s' % msg)
            return 99.


class FitPlotMixin(object):
    """mixin  for likelihood function to generate a plot, or set of all plots"""
    
    def estimate_solution(self):
        """ return a tuple with:
            current parameters, 
            estimated parmeters at maximum
            sigmas
        """
        parz = self.get_parameters()
        hess = self.hessian(parz)
        cov = np.linalg.inv(hess) if len(parz)>1 else 1./hess
        sigs = np.sqrt(np.asarray(cov.diagonal()).flatten())
        parmax = parz-self.gradient(parz)*sigs**2/2.
        return parz, parmax, sigs

    def plot(self, index, ax=None, nolabels=False , y2lim=(-5,5), estimate=None):
        """make a plot showing the log likelihood and its derivative as a function of
        expected sigma, evaluated from the second derivative at the current point
        
        index : int
            index of the parameter
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
            ax.plot(xsig, list(map(func,x))-ref, '-b')
            ax.plot(xsig, -0.5*((x-x0)/sig)**2, '--b')
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
    
    def plot_all(self, perrow=5, figsize=None): 
        """
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

    def maximize(self,  **kwargs):
        """Maximize likelihood and estimate errors.
        Uses scipy.optimize.fmin_l_bfgs_b
        
        keyword args
        ------------
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

        """
        from scipy import optimize
        quiet = kwargs.pop('quiet', True)
        if not kwargs.pop('use_gradient', True):
            print('Warning: ignoring use_gradient=False')
        estimate_errors = kwargs.pop('estimate_errors', True)
        if not quiet: print('using optimize.fmin_l_bfgs_b with parameter bounds %s\n, kw= %s'% (
                            self.bounds, kwargs))
        parz = self.get_parameters()
        winit = self.log_like()
        # assert len(parz)==len(self.gradient()), 'tracking a bug'

        # list of default from the function statement, mods shown
        fit_args=dict(m=10, 
            factr=1e9,  #1e8
            pgtol=1e-3, #1e-05
            epsilon=1e-08, 
            iprint=-1, maxfun=15000, maxiter=15000)
        fit_args.update(kwargs)
        approx_grad = fit_args.pop('approx_grad', False)

        # run the fit
        ret = optimize.fmin_l_bfgs_b(self, parz, 
                bounds=self.bounds,  
                fprime=None if approx_grad else self.gradient, 
                approx_grad=approx_grad, 
                **fit_args)
        self.fmin_ret=ret
        if ret[2]['warnflag']>0: 
            print('Fit failure: check parameters')
            self.set_parameters(parz) #restore if error 
            self.covariance = None 
            raise Exception( 'Fit failure:\n%s' % ret[2])
        if not quiet:
            print(ret[2])
        f = ret 
        if estimate_errors:
            self.covariance = cov = np.linalg.inv(self.hessian(f[0])) # was .I
            diag = np.array(cov.diagonal()).flatten()
            bad = diag<0
            if np.any(bad):
                print('Minimizer warning: bad errors for values %s'\
                    %np.asarray(self.parameter_names)[bad]) 
                diag[bad]=0
            return f[1], f[0], np.sqrt(diag)
        else:
            self.covariance = None
            return f[1], f[0], np.nan

    def hessian(self, pars=None, **kwargs):
        """    
        Return the Hessian matrix  
        For sigmas and correlation coefficients, invert to covariance
                cov =  self.hessian().I
                sigs = np.sqrt(cov.diagonal())
                corr = cov / np.outer(sigs,sigs)
        """
        import numdifftools
        if pars is None: pars = self.get_parameters()
        return np.matrix(numdifftools.Hessian(self,  **kwargs)(pars))

        
        
    def modify(self, fraction):
        """change iniital set to fraction of current change; restore will make it permanent
        """
        if fraction==0 : return
        delta = self.get_parameters()-self.initial_parameters
        self.initial_parameters += fraction * delta
        
    def restore(self):
        print(f'set parameters to {self.initial_parameters}')
        self.set_parameters(self.initial_parameters)

    def check_gradient(self, delta=1e-5):
        """compare the analytic gradient with a numerical derivative"""
        
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
    """Mixin to allow simple restore of an object's state
        supports the 'with' construction, guarantees that restore is called to restore the state of the model
        example:
        -------
        with ClassName(...) as something:
            # use something ...
    """
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        self.restore()


class FitterView(FitPlotMixin, FitterMixin, FitterSummaryMixin, WithMixin): 

    def __init__(self, blike,  **kwargs):
        self.blike = blike
        self.parameters = blike.parameterset
        # self.parameters = blike.sources.parameters
        self.sources = self.parameters.free_sources
        self.initial_parameters = self.parameters[:]
        self.initial_likelihood = self.log_like()
        self.calls=0
        
    def get_parameters(self):
        return self.parameters.get_parameters()
    def set_parameters(self, pars):
        self.parameters.set_parameters(pars)
        self.blike.update()
        
    def save_covariance(self):
        """ store source submatrices of the fit covariance matrix into the models.
        this loses the correlations between sources
        """
        assert hasattr(self, 'covariance'), 'maximize was not run: no covariance to save'
        self.parameters.set_covariance(self.covariance)
    
    def modify(self, fraction):
        """change iniital set to fraction of current change; restore will make it permanent
        """
        if fraction==0 : return
        delta = self.get_parameters()-self.initial_parameters
        if fraction==1:
            print(f'set parameters to current values')
            self.initial_parameters = self.get_parameters()
            return
        # fractional change
        self.initial_parameters += fraction * delta
        print(f'parameters to save {self.initial_parameters}')
        
    def restore(self):
        print(f'set parameters to {self.initial_parameters}')
        self.set_parameters(self.initial_parameters)
        print(f'gradient after restore: {self.gradient()}')

    @property 
    def bounds(self):
        return np.concatenate([s.model.bounds[s.model.free] for s in self.sources]) 
    def __call__(self, pars=None):
        if pars is not None: self.set_parameters(pars)
        self.calls+=1
        return -self.blike.log_like()
    def log_like(self, summed=True):
        """assume that parameters are set, possibility of individual likelihoods"""
        return self.blike.log_like(summed=summed)

    def gradient(self,pars=None):
        if pars is not None: self.set_parameters(pars)
        return self.blike.gradient()
    def hessian(self, pars=None):
        if pars is not None: self.set_parameters(pars)
        return self.blike.hessian()
    @property
    def parameter_names(self):
        return self.parameters.parameter_names
    @property
    def model_parameters(self):
        return self.parameters.model_parameters
    @property
    def uncertainties(self):
        return self.parameters.uncertainties
    @property
    def mask(self):
        return self.parameters.mask
             

class SubsetFitterView(parameterset.ParSubSet, FitPlotMixin, FitterMixin, FitterSummaryMixin, WithMixin):

    def __init__(self, blike, select=None, exclude=None):
        self.blike = blike
        super().__init__(blike.sources, select, exclude)
        self.initial_parameters = self.parameters[:]
        self.initial_likelihood = self.log_like()
        self.calls=0

    def __repr__(self):
        return '%s.%s: %s '% (self.__module__, self.__class__.__name__, self.selection_description)
    def restore(self):
        self.set_parameters(self.initial_parameters)

    def save_covariance(self):
        """ store source submatrices of the fit covariance matrix into the models.
        this loses the correlations between sources
        """
        assert hasattr(self, 'covariance'), 'maximize was not run: no covariance to save'
        self.set_covariance(self.covariance)

    @property
    def parameters(self):
        return self.get_parameters()
    def set_parameters(self, pars):
        super().set_parameters(pars)
        self.blike.update()
    def __call__(self, pars=None):
        if pars is not None: self.set_parameters(pars)
        self.calls +=1
        return -self.blike.log_like()
        
    def log_like(self, *,summed=True):
        """assume that parameters are set, possibility of individual likelihoods"""
        return self.blike.log_like(summed=summed)
    def gradient(self, pars=None):
        if pars is not None: self.set_parameters(pars)
        return self.blike.gradient()[self.mask]
    #### This seems wrong! #####
    # def hessian(self,pars=None):
    #     if pars is not None: self.set_parameters(pars)
    #     return self.blike.hessian(self.mask) 
        
    def ts(self):
        """ simple test statistic """
        lnow = self()
        pars = self.parameters
        pars[0] = -20 ##### override to be really small self.bounds[0][0]
        return 2 * (self(pars)-lnow)
        

# class TSmapView(tools.WithMixin):

#     def __init__(self, blike, func, quiet=True):
#         self.quiet = quiet
#         self.func = func
#         self.blike = blike
#         self.source = self.func.source
#         self.saved_skydir = self.get_dir()
#         self.wzero = func.log_like()
    
#     def __repr__(self):
#         return '%s.%s: source %s' % (self.__module__, self.__class__.__name__, self.source.name)
        
#     def set_dir(self, skydir):
#         self.source.skydir = skydir
#         self.blike.initialize(None, self.source.name ) #sourcenane=self.source.name)
    
#     def get_dir(self):
#         return self.source.skydir
#     skydir = property(get_dir, set_dir)
    
#     def restore(self):
#         self.set_dir(self.saved_skydir)

#     def __call__(self, skydir=None):
#         if skydir is not None:
#             if not isinstance(skydir, SkyDir):
#                 skydir = SkyDir(*skydir)
#             self.set_dir(skydir)
#         return 2*(self.func.log_like()-self.wzero)


# class EnergyFluxView(tools.WithMixin):

#     def __init__(self, blike, func, energy, **kw):
        
#         self.func = func
#         self.blike=blike
#         self.source = source = self.func.source
#         self.model=model = source.spectral_model
#         #assert model[0]==model['norm']
#         self.norm = model[0]
#         self.tointernal = model.mappers[0].tointernal
#         self.bound = kw.get('bound', -20)# !!! model.bounds[0][0])
#         self.set_energy(energy)

#     def set_energy(self, energy=None):
#         if energy is None:
#             energy=self.model.e0
#         self.model[0]=self.norm # get original norm 
#         self.source.changed=True
#         self.blike.update()
#         self.energy = energy
#         self.eflux = self.model(energy) * energy**2 * 1e6
#         self.ratio = self.model[0]/self.eflux
    
#     def __repr__(self):
#         return '%s.%s: func=%s, at %.0f MeV' % (self.__module__, self.__class__.__name__,self.func,self.energy)
#     def restore(self):
#         self.set_energy()

#     @tools.ufunc_decorator # make this behave like a ufunc
#     def __call__(self, eflux):
#         if eflux<=0:
#             par = self.bound
#         else:
#             par = max(self.bound, self.tointernal(eflux*self.ratio))
#         return -self.func([par])
        
# class NormalizationView(tools.WithMixin):
#     """Manage a view defining a function of the normalization factor for a source
    
#     """
#     def __init__(self, blike, source_name):
#         self.blike = blike
#         source = blike.sources.find_source(source_name)
#         self.model = model= source.model
#         self.par = model[0]
#         parname = model.param_names[0]
#         if not model.free[0]:
#             self.freed = (parname, source_name)
#             blike.thaw(parname, source_name)
#         else: self.freed=None
#         self.func = blike.fitter_view(source_name + '_' + parname)

#         self.tointernal = model.mappers[0].tointernal
#         self.bound = model.bounds[0][0]
    
#     @tools.ufunc_decorator
#     def __call__(self, norm):
#         if norm <=0:
#             par = self.bound
#         else:
#             par = max(self.bound, self.tointernal(norm*self.par))
#         return -self.func([par])
    
#     def restore(self):
#         """If had to thaw, restore"""
#         self.func.restore()
#         if self.freed is not None and self.model.free[0]:
#             self.blike.freeze(*self.freed)
        

# class LikelihoodViews(bandlike.BandLikeList):

#     """Subclass of BandLikeList with  methods to return views for specific analyses.
    
#     * fits: fitter_view, return a FitterView or SubsetFitterView
#     * SED : energy_flux_view, a fitterView with a source selected
#     * TSmap : tsmap_view : a FitterView with the source flux selected which can have the position changed.
#     """
    
#     def fitter_view(self, select=None, setpars=None, **kwargs):
#         """ return a object to use with a fitter.
#             Two versions, one with full set of parameters, other if a subset is specified
#         """
#         if setpars is not None: 
#             self.sources.parameters.setitems(setpars)

#         if select is None:
#             return FitterView(self, **kwargs)
#         return SubsetFitterView(self, select, **kwargs)

#     def energy_flux_view(self, source_name, energy=None, **kw):
#         """ a functor for a source, which returns log likelihood as a 
#                 function of the differential energy flux, in eV units, at the given energy
                
#         parameters
#         ----------
#         source_name : string
#         energy : [None | float]
#             if None, use the reference energy e0
#         """
#         try:
#             source = self.sources.find_source(source_name)
#             model = source.model
#             func = self.fitter_view(source_name + '_' + model.param_names[0])
#         except Exception as msg:
#             raise Exception('could not create energy flux function for source %s;%s' %(source_name, msg))
#         return EnergyFluxView(self, func, energy, **kw)
        
#     def tsmap_view(self, source_name, **kw):
#         """Return TSmap function for the source
#         """
#         if source_name is None and self.sources.selected_source is not None:
#             source_name = self.sources.selected_source.name 
#         if source_name is None: 
#             raise Exception('No source is selected for a tsmap')
#         try:
#             func = self.fitter_view(source_name+'_Norm')
#         except Exception as msg:
#             raise Exception('could not create tsmap function for source %s;%s' %(source_name, msg))
#         return TSmapView(self, func, **kw)
        
#     def normalization_view(self, source_name):
#         return NormalizationView(self, source_name)

# def make_views(roi_index, rings=2):
#     """convenience function to return a LikelihoodViews object for testing
#     """
#     from . import (configuration, bands, from_healpix)
#     config = configuration.Configuration(quiet=True)
#     roi_bands = bands.BandSet(config, roi_index, load=True)
#     roi_sources = from_healpix.ROImodelFromHealpix(config, roi_index, load_kw=dict(rings=rings))
#     return LikelihoodViews(roi_bands, roi_sources)
