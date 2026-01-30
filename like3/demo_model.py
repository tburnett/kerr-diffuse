
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from . sourcelist import SourceList
from . sources import PointSource 
from . parameterset import ParameterSet
from . spectral_models import (LogParabola, PLSuperExpCutoff4, PowerLaw)   

class DemoModel(SourceList):  #, FitterView):
    """
    Inherit from a SourceList, which encapsulates a list of sources, to create a model.
    The model has a set of energy bins and an exposure_factor_factor factor to predict counts per bin
    
    Also generates a simulation to produce a data set for testing.
    """
    
    bins = np.logspace(2,5,13) # energy bin edges: 12 bins from 100 MeV to 100 GeV


    def __init__(self, sources, *, random_state=42):
        super().__init__(sources)
        self.sources = sources
        self.values = self.parameters.values # property access to parameters
        self.energies = np.sqrt(self.bins[1:]*self.bins[:-1])
        self.exposure_factor = np.full_like(self.energies,1e13) * self.energies/100  # simple energy-dependent exposure

        self.data = self.simulate(random_state=random_state)
    
    def simulate(self, random_state=42): 
        """
        Simulate data from the model with Poisson noise

        Parameters
        ----------
        random_state : int or None
            Random state for reproducibility. If None, no noise is added.
        """
        from scipy import stats
        # compute predicted counts with current parameters
        predicted = self(self.energies)*self.exposure_factor
        if random_state is None:
            return predicted

        # add Poisson noise if random_state is not None
        return  stats.poisson(predicted).rvs(random_state=random_state)
    
    def dataframe(self):
        assert hasattr(self, 'data'), "No data available. Please run simulate() first."
        return pd.DataFrame.from_dict(dict(energy=self.energies.astype(int), 
                                           model=(self(self.energies)*self.exposure_factor).round(1), 
                                           data=self.data)
                                           )

    def spectral_plot(self, ax =None):
        """ plot the spectrum
        """
        df = self.dataframe()
        xlim = (self.bins[0], self.bins[-1])
        fig, ax = plt.subplots(1,1, figsize=(6,4)) if ax is None else ax
        efactor = 1e5*df.energy**2 /self.exposure_factor
        
        ax.errorbar(df.energy, df.data*efactor, yerr=np.sqrt(df.data)*efactor, fmt='+', color='C1', label='data')
        ax.stairs(df.data*efactor, self.bins,  color='C1')        
        ax.plot(df.energy, df.model*efactor, 'x', label='model', color='red')
        ax.set(xscale='log', xlim=xlim, yscale='log',
                xlabel='Energy (MeV)',        ylabel='Flux', ylim=(0.1, 10))
        ax.legend()
        plt.show()
        return

    @classmethod
    def test(cls, plot=True, src_key=0, random_state=42):
        """ Create a simple model with one (or two) point source and plot the spectrum
        src_key : int
            0 : PLSuperExpCutoff source
            1 : PowerLaw source
            2 : both sources

        """ 
        ps = PointSource(name='Pulsar', skydir=(0,0), 
                        model=PLSuperExpCutoff4((1e-11, 2., 0.7, 0.69),free=[True,True,True,False] ))  
        
        pl = PointSource(name='Blazar', skydir=(0,0), 
                        model=LogParabola((4e-12, 2, 0, 1e3), free=[True, True, True, False]))
        
        pp = []
        if src_key==0:
            pp = [ps]
        elif src_key==1:
            pp = [pl]
        else:
            pp = [ps, pl]
        model = cls(pp, random_state=random_state)

        print(f'Model: {str(model)}')
    
        if plot:
            df = model.dataframe()
            model.spectral_plot()
        return model


class Likelihood:
    
    def __init__(self, model, data,):
        self.model = model
        self.data = data
        self.mp = model.parameters
 
    def log_like(self, x):
        """ Evaluate the log likelihood with a parameter set x """

        self.mp.values = x
        model_predictions = self.model(self.model.energies)*self.model.exposure_factor
        ret = self.data * np.log(model_predictions) - model_predictions  # Poisson likelihood
        return np.sum(ret) 
    
    def __call__(self, x):
        """ Return a 2-tuple for use with optimize.fmin_l_bfgs_b:
        * the negative of the log likelihood at x
        * negative gradient of the log likelihood wrt parameters (or parameter) at x 
        """  
        self.mp.values = x
        d = self.data
        m = self.model(self.model.energies)*self.model.exposure_factor
        g = self.model.gradient(self.model.energies)*self.model.exposure_factor
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

