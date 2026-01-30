import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt


class Likelihood:
    """
    Class to handle likelihood calculations and fitting for a given model and data.   
    
    """
    
    def __init__(self, model, data,):
        """
        Docstring for __init__
        
        Parameters:
        ----------
        model : SourceList object
            The model to be fitted
        data : array-like
            The observed data to fit the model to
        """
        self.model = model
        self.data = data
        self.mp = model.parameters
 
    def log_like(self, x):
        """ Evaluate the log likelihood with a parameter set x """

        self.mp.values = x
        model_predictions = self.model.counts()
        ret = self.data * np.log(model_predictions) - model_predictions  # Poisson likelihood
        return np.sum(ret) 
    
    def __call__(self, x):
        """ Return a 2-tuple for use with optimize.fmin_l_bfgs_b:
        * the negative of the log likelihood at x
        * negative gradient of the log likelihood wrt parameters (or parameter) at x 
        """  
        self.mp.values = x
        d = self.data
        # m = self.model.flux(self.model.energies)*self.model.exposure_factor
        # g = self.model.gradient(self.model.energies)*self.model.exposure_factor
        m = self.model.counts()
        g = self.model.count_gradient()
        gsum = ((d/m-1)*g).sum(axis=1)
        return -np.sum(d * np.log(m) - m),  -gsum
    
    def maximize(self, x0=None):
        """
        Maximize the log likelihood function starting from initial guess x0 if present
        
        Sets fit_info, a dictionary containing the covariance matrix, standard deviations,
        correlation matrix, gradient at the optimum, and the fitted parameters.
        """
        import numdifftools

        def evaluate_ts():
            """ Evaluate TS for Norm parameters
            """
            model = self.model
            logl = self.log_like
            x_fit =  self.fit_info['x_fit'].copy()
            val_fit = -self(x_fit)[0]
            values = x_fit.copy()
            ts_array = np.full_like(values, np.nan)

            for k,name in enumerate(model.parameter_names):
                if name.endswith('_Norm'):
                    values[k] = -20
                    ts_array[k] = round(2*(val_fit - logl(values)),1)
                    values[k] = x_fit[k]
                    model.parameters.values = x_fit
            return ts_array

        if x0 is None:
            x0 = self.model.parameters.values.copy()
        x_fit, val, d = optimize.fmin_l_bfgs_b(self, x0,  bounds=self.model.bounds); 
        if d['warnflag'] != 0:
            raise RuntimeError('fit_plot: optimization failed: %s' % d['task'])
        # do this since final values are projections 
        val, gradient = self(x_fit)
        self.model.parameters.values = x_fit
  
        hess = numdifftools.Hessian(self.log_like)(x_fit) 
        cov = np.linalg.inv(-hess)
        sigs = np.sqrt(cov.diagonal())
        self.model.parameters.set_covariance(cov)

        self.fit_info = dict(
            cov = cov,
            sigs = sigs,
            corr = cov / np.outer(sigs,sigs),
            grad = -gradient,
            x_fit = x_fit,
            value = -val, 

            funcalls = d['funcalls'],)
        self.fit_info['ts_values'] = evaluate_ts()
        
    @property
    def model_parameters(self):
        """ Return the external parameters of the model
        """
        return self.model.parameters.model_parameters
    
    def summary(self,  out=None, title=None, gradient=True, ts=True):
        """ summary table of free parameters, values, uncertainties, gradient
        
        Parameters:
        ----------

        out : open file or None
        title: None or string
        gradient: bool
            set False to not print gradient
        ts: bool
            set False to not print TS values
            
        """
        if title is not None:
            print(title, file=out)

        fmt, tup = '%-21s%6s%10s%10s', tuple('Name index value error(%)'.split())
        if gradient:
            grad = self.fit_info['grad']
            fmt +='%10s'; tup += ('gradient',)
        if ts:
            ts_values = self.fit_info.get('ts_values', None)
            if ts_values is not None:
                fmt +='%10s'; tup += ('TS',)
        print(fmt %tup, file=out)
        prev=''

        index_array = np.arange(len(self.model.parameters.mask))[self.model.parameters.mask]
        for index, (name, value, rsig) in enumerate(zip(self.model.parameter_names, 
                                                        self.model_parameters,#self.fit_info['x_fit'], 
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
            if gradient:
                fmt +='%10.1f'; tup += (grad[index],)
            if ts and ts_values is not None:
                fmt +='%10s'; tup += (f'{ts_values[index]:.0f}' if not pd.isna(ts_values[index]) else '',)
            print(fmt % tup, file=out)
    
    @classmethod
    def test_plots(cls, model):
        from like3.views import fit_plot
        data = model.data
        x0 = model.parameters.values.copy()
        fig, axx = plt.subplots(ncols=len(x0), figsize=(4*len(x0),4), sharey=True)
        for k,ax in enumerate(axx):
            model.parameters.values=x0
            func = cls(model, data, k)
            fit_plot(func, x0[k], ax=ax, title=model.parameter_names[k])
        plt.show()

    @classmethod
    def test_fit(cls, src_key=0, random_state=42):
        """ Test fitting the model to its own simulated data
        """
        model=DemoModel.test(plot=False, random_state=random_state, src_key=src_key)
        data = model.data
        x0 = model.parameters.values.copy()
        result = cls(model, data,).maximize(x0)    
        print(model)
        return result