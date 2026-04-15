"""Numerical fitting utilities used by like3 models.

This module provides:
- `Fitted`: lightweight interface for fit-capable function objects.
- `Minimizer`: wrappers around SciPy (and optional Minuit) optimizers.
- `Projector` / `Profile`: adapters for subset and 1D profile fits.
- `TestFunc` and `test(...)`: simple sanity checks against parabola-like inputs.

Authors
-------
Matthew Kerr, Toby Burnett
"""
import numpy as np
from scipy import optimize #for fmin,fmin_powell,fmin_bfgs
from numpy import linalg  #for inv
import numdifftools

class FitterException(Exception):
    """Raised for fitter-interface and parameter-selection errors."""

class Fitted(object):
    """Base protocol for function objects consumed by the fitter utilities."""
    @property
    def bounds(self):
        return None
    @property
    def parameter_names(self):
        return None
    def get_parameters(self):
        """Return current internal parameter vector."""
        raise FitterException('get_parameters is not implemented')
    def set_parameters(self, par):
        """Update current internal parameter vector."""
        raise FitterException('set_parameters is not implemented')
        
    def minimize(self, **kwargs):
        """Minimize with L-BFGS-B starting from current parameters."""
        use_gradient = kwargs.pop('use_gradient',True)#, self.gradient(self.get_parameters()) is None)
        ret =optimize.fmin_l_bfgs_b(self, self.get_parameters(), 
            bounds=self.bounds, 
            fprime= None, # expect gradient calculated by function
            approx_grad = not use_gradient,
            args = (use_gradient,),  # pass to the function
            **kwargs)
        if ret[2]['warnflag']==0: 
            self.set_parameters(ret[0])
        else:
            print('Fit failure:\n%s' % ret[2])
        return ret
        
    def hessian(self, pars=None, **kwargs):
        """Return numerical Hessian matrix at `pars` (or current parameters).

        Notes
        -----
        For 1-sigma errors and correlation estimates, invert the Hessian:
            `cov = self.hessian().I`
            `sigs = np.sqrt(cov.diagonal())`
            `corr = cov / np.outer(sigs, sigs)`
        """
        if pars is None: pars = self.get_parameters()
        return np.matrix(numdifftools.Hessian(self,  **kwargs)(pars))

def _make_testfunc(fn=None, p0=None, pars=None):
    if fn is None:
        pars=[1.0, 2.]
        fn = lambda p: 1.+ 0.5*((p[0]-pars[0])/pars[1])**2
    return TestFunc(fn, [1.1])   
 
       
class Minimizer(object):
    """General-purpose minimizer wrapper around several optimization methods."""

    def __init__(self, fn, parameters=None, args=(), quiet=True):
        """Initialize minimizer.

        Parameters
        ----------
        fn : callable
            Objective function to minimize (typically negative log-likelihood).
        parameters : array-like or None
            Optional explicit parameter vector. If None, `fn.get_parameters()` is used.
        args : tuple
            Extra arguments forwarded to objective evaluations.
        quiet : bool
            Controls optimizer logging.
        """
        self.quiet = quiet
        self.par = parameters
        self.args = args
        self.fn = fn 
        npar = len(self.get_parameters())
        self.cov_matrix=np.zeros([npar,npar])
    
    def gradient(self,parameters,*args):
        """Return analytic gradient from the wrapped objective."""
        assert hasattr(self.fn, 'gradient'), 'Minimize: use_gradient set, but function did not define a gradient'
        return self.fn.gradient(parameters)

    def get_parameters(self):
        return self.fn.get_parameters() if self.par is None else self.par

    def set_parameters(self, par):
        if self.par is None:
            self.fn.set_parameters(par)
        else:
            self.par = par

    def get_free_errors(self):
        """Return the diagonal elements of the covariance matrix -- useful for step sizes in minimization, if known.
        """
        assert False, 'get_free_errors not implemented yet'
    
    def optimize(self, optimizer,  **kwargs):
        return optimizer( self.fn, self.get_parameters(),  **kwargs)
        
    def __call__(self, method='simplex', tolerance = 0.01, save_values = True, 
                      estimate_errors=True, error_for_steps=False,
                     use_gradient = True, gtol = 1e-1, **kwargs):
        """Run minimization and optional covariance/error estimation.

        Parameters
        ----------
        method : str
            One of `simplex`, `powell`, `minuit`, or `l-bfgs-b`.
        tolerance : float
            Approximate absolute convergence tolerance.
        estimate_errors : bool
            If true, estimate covariance at optimum and return errors.
        use_gradient : bool
            If true, prefer analytic-gradient paths when available.
        """
        if method.lower() not in ['simplex','powell','minuit', 'l-bfgs-b']:
            raise Exception('Unknown fitting method for F.fit(): "%s"' % method)

        use_gradient = use_gradient and hasattr(self.fn, 'gradient')
        use_bounds = kwargs.pop('use_bounds', self.fn.bounds is not None)
        if method == 'minuit':
            return self.minuit(
                tolerance=tolerance,
                save_values=save_values,
                estimate_errors=estimate_errors,
                error_for_steps=error_for_steps,
                use_gradient=use_gradient,
            )
        # Evaluate objective scale once for relative tolerance settings.
        ll_0 = self.fn(self.get_parameters(), *self.args) 
        if ll_0==0: ll_0=1.0
        if use_gradient and not use_bounds:
            f0 = optimize.fmin_bfgs(self.fn,self.get_parameters(),self.gradient,full_output=1,maxiter=500,gtol=gtol,disp=0)
            # Retry BFGS if first attempt does not stabilize objective.
            for i in range(10):
                f = self._save_bfgs = optimize.fmin_bfgs(self.fn,self.get_parameters(),self.gradient,
                        full_output=1,maxiter=500,gtol=gtol,disp=0)
                if abs(f0[1] - f[1]) < tolerance: break # note absolute tolerance
                if not self.quiet:
                    print('Did not converge on first gradient iteration.  Trying again.')
                    print(f0[1],f[1],abs(f0[1]-f[1]))
                f0 = f
        elif use_gradient:
            if not self.quiet: print('using optimize.fmin_l_bfgs_b with parameter bounds %s\n, kw= %s'% (self.fn.bounds, kwargs))
            ret = optimize.fmin_l_bfgs_b(self.fn, self.get_parameters(), 
                bounds=self.fn.bounds, 
                fprime=self.gradient ,  
                **kwargs)
            if ret[2]['warnflag']>0: 
                print('Fit failure:\n%s' % ret[2])
            if not self.quiet:
                print(ret[2])
            f = ret 
        else:
            minimizer  = optimize.fmin_powell if method == 'powell' else optimize.fmin
            f = minimizer(self.fn, self.get_parameters(),full_output=1,
                              maxiter=10000, maxfun=20000, ftol=0.01/abs(ll_0), disp=0 if self.quiet else 1)
        
        if not self.quiet: print('Function value at minimum: %.8g'%f[1])
        self.set_parameters(f[0])
        self.fitvalue=f[1]
        if estimate_errors: 
            self.__set_error__(use_gradient)
        if estimate_errors:
            diag = self.cov_matrix.diagonal().copy()
            bad = diag<0
            if np.any(bad):
                if not self.quiet: print('Minimizer warning: bad errors for values %s'\
                     %np.asarray(self.fn.parameter_names)[bad]) #    %np.arange(len(bad))[bad]
                diag[bad]=np.nan
            return f[1], f[0], np.sqrt(diag)
        return f[1], f[0]
    
    def minuit(self, tolerance=0.01, save_values=True, estimate_errors=True,
               error_for_steps=False, use_gradient=True):
        from uw.utilities.minuit import Minuit
        temp_params = self.get_parameters()
        npars = temp_params.shape[0]
        param_names = ['p%i'%i for i in range(npars)]
        
        if use_gradient :
            gradient         = self.gradient
            force_gradient = 1
        else:
            gradient         = None
            force_gradient = 0

        if error_for_steps:
            steps = self.get_free_errors()
            steps[steps<1e-6] = 0.04 # for models without error estimates, put in the defaults
            steps[steps > 1]  = 1     # probably don't want to step more than 100%...
            m = Minuit(self.fn,temp_params,up=.5,maxcalls=20000,tolerance=tolerance,printMode=-self.quiet,param_names=param_names,steps=steps)
        else:
            m = Minuit(self.fn,temp_params,up=.5,maxcalls=20000,tolerance=tolerance,printMode=-self.quiet,param_names=param_names)

        params,fval = m.minimize()

        if save_values:
            if estimate_errors == True:
                self.__set_error_minuit(m,'HESSE')
            self.fn(params) # reset values to the ones found by minimization step
        self.fitvalue= fval
        return fval
 
    def __set_error_minuit(self,m,method='HESSE'):
        """Compute errors for minuit fit."""
        #Not sure yet if there will be problems with including the backgrounds.
        self.cov_matrix = m.errors(method=method)
        print('Minuit error not done?')
        #self.bgm.set_covariance_matrix(self.cov_matrix,current_position = 0)
        #self.psm.set_covariance_matrix(self.cov_matrix,current_position = len(self.bgm.parameters()))

    def sigmas(self):
        """Return 1-sigma estimates from covariance diagonal (NaN if invalid)."""
        diag = self.cov_matrix.diagonal()
        bad = diag<0
        if np.any(bad): diag[bad]=np.nan
        return np.sqrt(diag)

    def correlations(self, percent=False):
        """Return linear correlation coefficients derived from covariance matrix.

        Any rows/columns with zero uncertainty are returned as NaN.
        """
        s = self.sigmas()
        s[s==0] = np.nan
        t =self.cov_matrix / np.outer(s,s)
        return t*100. if percent else t

    def __set_error__(self,use_gradient=False):
        """Estimate covariance matrix near current optimum."""

        npar = len(self.get_parameters())
        if use_gradient:
            save_pars = self.get_parameters().copy()
            cov_matrix,hessian = Minimizer.mycov(self.gradient,self.get_parameters(),full_output=True)[:2]
            self.set_parameters(save_pars)
            mask = hessian.diagonal()>0
        else:
            hessian, bad_mask = Minimizer.hessian(self.fn, self.get_parameters(), quiet=self.quiet)
            cov_matrix = None
            mask = bad_mask==0
        if np.all(~mask):
            self.cov_matrix = np.zeros([npar,npar])
            success = False
            return
        full = np.all(mask)
        if not full:
            h = hessian[mask].T[mask]
            hessian = h
        success = False
        npar = len(self.get_parameters())
        try:
            if linalg.det(hessian)<=0:
                full=False
                
            if not self.quiet: print('Attempting to invert full hessian...')
            self.cov_matrix =t = cov_matrix if cov_matrix is not None else linalg.inv(hessian)
            if np.any(np.isnan(self.cov_matrix)):
                if not self.quiet: print('Found NaN in covariance matrix!')
                raise Exception('Found NaN in covariance matrix!')
            # Expand reduced covariance back to full parameter-space shape.
            if not full:
                # must be better way to expand a matrix
                self.cov_matrix =np.zeros([npar,npar])
                k = np.arange(npar)[mask]
                for i in range(len(k)):
                    ki = k[i]
                    self.cov_matrix[k[i],k[i]] = t[i,i] 
                    for j in range(i+1, len(k)):
                        self.cov_matrix[ki,k[j]] =self.cov_matrix[k[j],ki] = t[i,j]
            success = True
        except linalg.LinAlgError as e:
            if not self.quiet:
                print('Error generating cov matrix, %s' % e)
            self.cov_matrix = np.zeros([npar,npar])
            success = False
        return success

    @staticmethod
    def hessian(mf, pars, quiet=True, *args):
        """Calculate Hessian by finite differences around `pars`.

        Parameters
        ----------
        mf : callable
            Objective function.
        pars : array-like
            Parameter vector (typically at optimum).
        args : tuple
            Extra args passed to `mf`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Hessian matrix and per-parameter return/error code array.
        """
        p  = pars.copy()
        npar = len(pars)
        deltas    = np.abs(0.01 * p) #initial guess
        hessian  = np.zeros([npar,npar])
        bad_mask = np.asarray([False] * npar)
        return_code = np.zeros(npar)

        l0 = mf(p, *args)

        # Tune per-parameter step sizes so diagonal finite differences are
        # informative but numerically stable.
        for i in range(npar):
            if not quiet: print('Working on parameter %d'%(i))
            h,l = p.copy(),p.copy()
            delta_f_1 = 0.0
            delta_f_2 = 0.0
            for j in range(10):
                h[:] = p[:]; l[:] = p[:];
                h[i] += deltas[i]
                l[i] -= deltas[i]

                delta_f_1 = mf(h, *args) - l0
                delta_f_2 = mf(l, *args) - l0
                delta_f = max(delta_f_1 + delta_f_2,0) #twice difference, really
                deltas[i] /= max(delta_f**0.5,0.33) # can change by half decade
                if delta_f < 5 and delta_f > 0.5: break

            if delta_f < 5e-3:
                # no constraint on parameter -- ignore it in further fittingor :
                bad_mask[i] = True
                return_code[i] = 1
            ratio = np.inf if delta_f_2 == 0 else (delta_f_1 / delta_f_2)
            if (ratio > 10 or ratio < 1./10):
                # significant asymmetry in likelihood             
                bad_mask[i] = True
                return_code[i] = 2
            if (delta_f_2 < 5e-3 and delta_f_1 > 0.5):
                # not actually at maximum of likelihood -- upper limit condition
                bad_mask[i] = True
                return_code[i] = 3
                if not quiet: print('fail, need upper limit')
                import pdb; pdb.set_trace()

        for i in range(npar):
            if bad_mask[i]:
                hessian[i,:] = 0 #no correlation?
                hessian[:,i] = 0
                continue
            for j in range(i,npar): #Second partials by finite difference
                
                xhyh,xhyl,xlyh,xlyl=p.copy(),p.copy(),p.copy(),p.copy()
                xdelt = deltas[i]
                ydelt = deltas[j]
                xhyh[i] += xdelt;  xhyh[j] += ydelt
                xhyl[i] += xdelt;  xhyl[j] -= ydelt
                xlyh[i] -= xdelt;  xlyh[j] += ydelt
                xlyl[i] -= xdelt;  xlyl[j] -= ydelt
                hessian[i][j]=hessian[j][i]=(mf(xhyh, *args)-mf(xhyl, *args)
                                            -mf(xlyh, *args)+mf(xlyl, *args))/\
                                                        (4*xdelt*ydelt)

        mf(p, *args) #call likelihood with original values; this resets model and any other values that might be used later
        return hessian,return_code

    @staticmethod
    def mycov(grad,par,full_output=False,init_step=0.04,min_step=1e-6,max_step=1,max_iters=5,target=0.5,min_func=1e-4,max_func=4):
        """Estimate covariance from finite differences of analytic gradients.

        Positional args:
            grad                : a function to return a gradient
            par                 : vector of parameters (should be function minimum for covariance matrix calculation)

        Keyword args:

            full_output [False] : if True, return information about convergence, else just the covariance matrix
            init_step   [1e-3]  : initial step size (0.04 ~ 10% in log10 space); can be a scalar or vector
            min_step    [1e-6]  : the minimum step size to take in parameter space
            max_step    [1]     : the maximum step size to take in parameter sapce
            max_iters   [5]     : maximum number of iterations to attempt to converge on a good step size
            target      [0.5]   : the target change in the function value for step size
            min_func    [1e-4]  : the minimum allowable change in (abs) function value to accept for convergence
            max_func    [4]     : the maximum allowable change in (abs) function value to accept for convergence
        """

        nparams   = len(par)
        step_size = np.ones(nparams)*init_step
        step_size = np.maximum(step_size,min_step*1.1)
        step_size = np.minimum(step_size,max_step*0.9)
        hess      = np.zeros([nparams,nparams])
        min_flags = np.asarray([False]*nparams)
        max_flags = np.asarray([False]*nparams)

        def revised_step(delta_f,current_step,index):
            if (current_step == max_step):
                max_flags[index] = True
                return True,0
            elif (current_step == min_step):
                min_flags[index] = True
                return True,0
            else:
                adf = abs(delta_f)
                if adf < 1e-8:
                    # need to address a step size that results in a likelihood change that's too
                    # small compared to precision
                    pass
                    
                if (adf < min_func) or (adf > max_func):
                    new_step = current_step/(adf/target)
                    new_step = min(new_step,max_step)
                    new_step = max(new_step,min_step)
                    return False,new_step
                else:
                    return True,0
        
        iters = np.zeros(nparams)
        for i in range(nparams):
            converged = False
            # Initialize in case `max_iters` is set to 0 by caller.
            g_up = grad(par)
            g_dn = g_up.copy()
            for j in range(max_iters):
                iters[i] += 1
                di = step_size[i]
                par[i] += di
                g_up    = grad(par)
                par[i] -= 2*di
                g_dn    = grad(par)
                par[i] += di
                delta_f = (g_up - g_dn)[i]
                converged,new_step = revised_step(delta_f,di,i)
                #print 'Parameter %d -- Iteration %d -- Step size: %.2e -- delta: %.2e'%(i,j,di,delta_f)
                if converged: break
                else: step_size[i] = new_step
            hess[i,:] = (g_up - g_dn) / (2*di)  # central difference
            if not converged:
                print('Warning: step size for parameter %d (%.2g) did not result in convergence.'%(i,di))
        try:
            cov = np.linalg.inv(hess)
        except:
            raise Exception('Error inverting hessian')
        if full_output:
            return cov,hess,step_size,iters,min_flags,max_flags
        else:
            return cov

class Projector(Fitted):
    """Adapter exposing a subset of parameters as the active optimization vector.

    The wrapped function is still evaluated in full parameter space; non-selected
    parameters are held fixed at their current values.
    """
    def __init__(self, fn, select=[0], par=None, ):
        """ 
        
        parameters:
            fn: function of par: should be minimizable 
            par: array type or None
                default parameters to use: if None, get from fn.get_parameters)
            select: list of free parameter 
            TODO: use mask instead or optionally
        """
        self.fn=fn
        self.select = select
        self.mask = np.zeros(len(fn.get_parameters()),bool)
        self.mask[select]=True
        self.fpar= fn.get_parameters().copy()
        self.par = np.asarray(par[:]) if par is not None else self.fpar[self.mask]
        assert len(self.par)==sum(self.mask), 'wrong number of specified parameters'
    def get_parameters(self):
        return self.par
    def set_parameters(self,par=None):
        p = par if par is not None else self.par
        self.par = p
        self.fpar[self.mask] = p
        self.fn.set_parameters(self.fpar) # note this sets the original set
        
    def __call__(self, x):
        """Evaluate wrapped function for selected-parameter vector `x`."""
        self.fpar[self.mask]=x
        ret= self.fn(self.fpar)
        #print 'value(%.2f)=%.2f' % (x,ret)
        return ret
    def gradient(self, x):
        """ the function object may not support this
        """
        self.fpar[self.mask]=x
        t = self.fn.gradient(self.fpar)[self.mask]
        #print 'gradient(%.2f)=%.2f' % (x, t)
        return t
    @property
    def parameter_names(self):
        return None if not hasattr(self.fn,'parameter_names') else self.fn.parameter_names[self.mask]
        
    @property
    def bounds(self):
        return None if self.fn.bounds is None else np.array(self.fn.bounds)[self.mask]
        
    def fmin(self, x=None, **kwargs):
        """ run simple fmin """
        try:
            par = optimize.fmin(self, [x] if x is not None else self.par, **kwargs)
            self.set_parameters(par)
        except:
            raise
        
    def minimize(self, par0=None, **fit_kw):
        """ create  Minimizer of this, run it, update original parameters
        parameters:
            par0 : array type of float or None
                pass to Minimizer
                
        return value, parameter values, errors
        """
        self.fitter = Minimizer(self, par0)
        
        result = self.fitter(**fit_kw)
        if len(result) == 3:
            c2, par, dpar = result
        else:
            c2, par = result
            dpar = None
        self.par = par
        self.set_parameters(par)
        return c2, par, dpar


class Profile(Fitted):
    """One-parameter profile/projection view of a multi-parameter objective.

    Optionally profiles out the remaining parameters via internal minimization.
    """

    def __init__(self, fn, index, par=None, profile=True):
        """
        parameters
        ---------
        fn : function of a set of parameters
            Must implement Fitted interface
        index : integer or string
            the index to examine, or its parameter name
        par: arary type or None
           initial set of parameters for fn if not None
        profile: bool
            set False to not apply profile
        """
        # local reference to the basic function, copy of original parametes
        self.fn = fn
        if type(index)==bytes:
            try:
                self.index = list(fn.parameter_names).index(index)
            except ValueError:
                raise FitterException('parameter name "%s" not one of %s' % (index, fn.parameter_names))
            except Exception as msg:
                raise
        else:  self.index = index
        self.fpar =  par if par is not None else fn.get_parameters().copy()
        npar = len(self.fpar)
        self.mask = np.ones(npar,bool)
        self.mask[self.index]=False
       
        # set up function of the selected parameter (self) and a function of the rest
        select = list(range(npar))
        assert self.index in select, 'Expect index to select to be one of parameters'
        self.par = self.fpar[self.index:self.index+1]
        select.remove(self.index)
        self.pfun = Projector(fn, select)
        self.profile = profile
        
        # set up a fitter for the remaining parameters
        self.fitter = Minimizer(self.pfun) 
        
    def __call__(self, x):
        """Evaluate profile function at scalar parameter value `x[0]`."""
        self.fpar[self.index]=x[0]
        # if don't optimize the other parameters
        if self.profile: 
            fit_result = self.fitter()
            p = fit_result[1]
            v = fit_result[0]
            self.fpar[self.mask]=p
            r = self.fn(self.fpar)
            print(v,r)
        else:
            r = self.fn(self.fpar)
        return r
    
    @property
    def parameter_names(self):
        return self.fn.parameter_names[self.index:self.index+1]
        
    def get_parameters(self):
        return self.par
        
    def set_parameters(self, par=None):
        p = par if par is not None else self.par
        self.par = p
        self.fpar[self.index] = p
        
    
class TestFunc(Fitted):
    """Minimal `Fitted` implementation for simple functional tests."""
    def __init__(self, fn, pars):
        self.fn = fn
        self.pars = pars
    @property 
    def bounds(self):
        return  [(0.9,2), ]
    def __call__(self, pars):
        return self.fn(pars)
    def get_parameters(self): return self.pars
    def set_parameters(self,pars):
        self.pars = pars

 
def test(x0=1.1, pars=[1.0, 1.5], **kwargs):
    """Run a simple parabola minimization test and print fitted result.
    
    >>> pars=[1.0, 1.5]; x0=1.1
    >>> testf = lambda p: 1.+ 0.5*((p[0]-pars[0])/pars[1])**2
    >>> func = TestFunc(testf, [x0])
    >>> m = Minimizer(func) # create minimizer object
    >>> m() # run default fit
    (1.0000000000211928, array([ 0.99999023]), array([ 1.5]))
    """
    testf = lambda p: 1.+ 0.5*((p[0]-pars[0])/pars[1])**2
        
    print('input parameters:', pars)
    func = TestFunc(testf, [x0])
    m = Minimizer(func)
    #m =  Minimizer(testf, [x0], )
    f = m(use_gradient=False)
    print('solution at %.2f, +/- %.2f ' % (m.get_parameters(), np.sqrt(m.cov_matrix.diagonal())))
    return func, m, f

if __name__ == "__main__":
    print(__doc__)
    import doctest
    doctest.testmod()
  