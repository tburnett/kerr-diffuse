import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . sources import PointSource 
from . parameterset import ParameterSet
from . spectral_models import (LogParabola, PLSuperExpCutoff4, PowerLaw)

class SimpleModel(ParameterSet):
    """
    Inherit from a ParamterSet, which encapsulates a list of sources, to create a model.
    The model has a set of energy bins and an exposure_factor_factor factor to predict counts per bin
    
    Also generates a simulation to produce a data set for testing.
    """
    
    bins = np.logspace(2,5,13) # energy bins edges: 12 bins from 100 MeV to 100 GeV
    exposure_factor = np.full_like(bins,1e12)

    def __init__(self, sources, *, random_state=42):
        super().__init__(sources)
        self.sources = sources
        self.energies = np.sqrt(self.bins[1:]*self.bins[:-1])
        self.exposure_factor = 1e13 #np.full_like(self.energies,1e13)#*100/self.energies  # simple energy-dependent exposure

        self.parameters = self # since we inherit from ParameterSet
        self.data = self.simulate(random_state=random_state)

    def __call__(self):
        """ Compute model counts for given parameter set """
        # if pars is None:
        #     pars = self.get_parameters()
                
        r = np.zeros_like(self.energies)
        for source in self.free_sources:
            r += source.model(self.energies)*self.exposure_factor
        return r     
    
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
        predicted = self()

        # add Poisson noise if random_state is not None
        self.data = stats.poisson(predicted).rvs(random_state=random_state)\
              if random_state is not None else predicted
        return self.data 
    
    def gradient(self):
        """Derive the gradient of the model with respect to free parameters
        """
        return np.vstack([source.model.gradient(self.energies)[source.model.free]*self.exposure_factor 
                          for source in self.free_sources])
    
    def dataframe(self):
        assert hasattr(self, 'data'), "No data available. Please run simulate() first."
        return pd.DataFrame.from_dict(dict(energy=self.energies.astype(int), 
                                           model=self().round(1), 
                                           data=self.data)
                                           )
    
    def count_plot(self, ax =None):
        """ plot the counts vs energy
        """
        df = self.dataframe()
        xlim = (self.bins[0], self.bins[-1])
        fig, ax = plt.subplots(1,1, figsize=(6,4)) if ax is None else ax
        ax.errorbar(df.energy, df.data, yerr=np.sqrt(df.data), fmt='+', color='C1', label='data')
        ax.stairs(df.data, self.bins,  color='C1')
        ax.plot(df.energy, df.model, '--', label='model')
        ax.set(xscale='log', xlim=xlim, yscale='log',
               xlabel='Energy (MeV)',        ylabel='Counts')
        ax.legend()
        plt.show()
        return
    
    def spectral_plot(self, ax =None):
        """ plot the spectrum
        """
        df = self.dataframe()
        xlim = (self.bins[0], self.bins[-1])
        fig, ax = plt.subplots(1,1, figsize=(6,4)) if ax is None else ax
        efactor = 1e-6*df.energy**2
        
        ax.errorbar(df.energy, df.data*efactor, yerr=np.sqrt(df.data)*efactor, fmt='+', color='C1', label='data')
        ax.stairs(df.data*efactor, self.bins,  color='C1')
        
        ax.plot(df.energy, df.model*efactor, 'x', label='model')

        ax.set(xscale='log', xlim=xlim, yscale='log',
                xlabel='Energy (MeV)',        ylabel='Flux', ylim=(1e1, 1e4))
        ax.legend()
        plt.show()
        return


    @classmethod
    def demo(cls, plot=True, src_key=0, random_state=42):
        """ Create a simple model with one (or two) point source and plot the spectrum
        src_key : int
            0 : PLSuperExpCutoff source
            1 : PowerLaw source
            2 : both sources

        """ 
        ps = PointSource(name='PSR', skydir=(0,0), 
                        model=PLSuperExpCutoff4((1e-10, 2., 0.7, 0.69),free=[True,True,True,False] ))  
        
        pl = PointSource(name='PL', skydir=(0,0), 
                        model=PowerLaw((1e-11, 1), free=[True, True]))
        
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
