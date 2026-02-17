
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix 

from . sourcelist import SourceList
from . sources import PointSource 
from . parameterset import ParameterSet
from . spectral_models import (LogParabola, PLSuperExpCutoff4, PowerLaw)   

class Pixels:
    """ class describing the pixels in the model
    Here, a set of energies with one pixel bin per energy
    Includes also exposure information to convert flux to counts
    """
    bins = np.logspace(2,5,13) # energy bin edges: 12 bins from 100 MeV to 100 GeV
    # PSF3 nsides defined by MK
    nsides = np.array([  16,   32,   64,  128,  256,  512,  512,  512, 1024, 2048, 2048, 2048])

    def __init__(self):
        self.energies = np.sqrt(self.bins[1:]*self.bins[:-1])
        self.exposure_factor = np.full_like(self.energies,1e13) * self.energies/100  # simple energy-dependent exposure

    def counts(self, source_model):
        """ Convert flux to counts using exposure_factor
        """
        return source_model.flux(self.energies) * self.exposure_factor
    
    def count_gradient(self, source_model):
        """ Convert flux gradient to counts gradient using exposure_factor
        """
        return source_model.gradient(self.energies) * self.exposure_factor
    

# class Band(HEALPix):
#     """ class describing an energy band: energy, nside, and source model
#     """
#     def __init__(self, nside, energy, source_model):
#         """ Initialize a band with given nside, energy, and source model
        
#         Parameters        ----------
#         nside : int
#             HEALPix nside for this band
#         energy : float
#             Geometric mean energy for this band
#         source_model : SourceList
#             Source model to compute flux and its gradient for this band"""
#         self.source_model = source_model
#         self.energy = energy  # geometric mean energy for the band
#         super().__init__(nside, order='ring', frame='galactic')  # initialize HEALPix part

#     def __repr__(self):
#         return f'Band(energy={self.energy:.1f} MeV, nside={self.nside})'

#     def flux(self):
#         return self.source_model.flux(self.energy)
    
#     def flux_gradient(self):
#         return self.source_model.gradient([self.energy])  


# class BandList(list):
#     """ Combine a list of bands with a source model to create a model for the counts in each band, and its gradient with respect to free parameters.
#     Also includes a method to simulate data from the model with Poisson noise, and a demo method to create a BandList for a given model and print the counts for each band.
#     """
#     bins = np.logspace(2,5,13) # energy bin edges: 12 bins from 100 MeV to 100 GeV
#     # PSF3 nsides defined by MK
#     nsides = np.array([  16,   32,   64,  128,  256,  512,  512,  512, 1024, 2048, 2048, 2048])
 
#     def __init__(self, source_model):
#         """ Initialize the list of bands with given source model"""
#         self.energies = np.sqrt(self.bins[1:]*self.bins[:-1])
#         self.exposure_factor = np.full_like(self.energies,1e13) * self.energies/100  # simple energy-dependent exposure

#         for energy, nside in zip(self.energies, self.nsides):
#             self.append(
#                 Band(nside, energy, source_model=source_model)
#                 )
#         self.sources = source_model
#         self.parameters = source_model.parameters
#         self.parameter_names = source_model.parameter_names

#     def counts(self):
#         """ Convert flux to counts using the exposure factor for each band
#         """
#         return np.array([band.source_model.flux(band.energy) * self.exposure_factor[i] 
#                          for i, band in enumerate(self)])
            
#     def count_gradient(self):
#         """ Convert flux gradient to counts gradient using the exposure factor for each band
#         """
#         g = np.array([band.source_model.gradient([band.energy]) * self.exposure_factor[i] 
#                          for i, band in enumerate(self)])
#         return g[:,:,0].T#.transpose(1,0,2)  # reshape to (n_parameters, n_bands, n_pixels_per_band)
    
#     def simulate(self, random_state=42): 
#         """
#         Simulate data from the model with Poisson noise

#         Parameters
#         ----------
#         random_state : int or None
#             Random state for reproducibility. If None, no noise is added.
#         """
#         from scipy import stats
#         # compute predicted counts with current parameters
#         predicted = self.counts()
#         if random_state is None:
#             return predicted

#         # add Poisson noise if random_state is not None
#         return  stats.poisson(predicted).rvs(random_state=random_state)
    
    # @classmethod
    # def demo(cls, model=None):
    #     """ Create a BandList for a given model and print the counts for each band
    #     """
    #     if model is None:
    #         model = SourceList.demo()
    #     print(f'Creating BandList for model: {model}')
    #     band_list = cls(model)
    #     for band in band_list:
    #         print(f'{band}: flux={band.flux():.2e}')
    #     print('Counts per band:', band_list.counts().astype(int))
    #     return band_list

class DemoModel(SourceList, Pixels):  
    """
    Inherit from a SourceList, which encapsulates a list of sources, to create a model.
    
    Inherot from a special Pixels class encapsulating the pixel descriptonn

    Generates a simulation to produce a data set for testing.
    """

    def __init__(self, sources, *, random_state=42):
        super().__init__(sources)
        Pixels.__init__(self)  # initialize Pixels part
        
        self.sources = sources
        assert isinstance(self.parameters, ParameterSet)
        assert hasattr(self, 'energies'), "Pixels not initialized properly"

        # this is a read/write property
        self.values = self.parameters.values # property access to parameters
        
        self.data = self.simulate(random_state=random_state)

    def __repr__(self):
        return super().__repr__()+ f' {len(self.data)} pixels'
    
    def counts(self):
        """ return counts predicted by the model, ovverriding Pixel method
        """
        return super().counts(self)
    
    def count_gradient(self):
        """ Return the gradient of the counts with respect to free parameters
        """
        return super().count_gradient(self)

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
        predicted = self.counts()
        if random_state is None:
            return predicted

        # add Poisson noise if random_state is not None
        return  stats.poisson(predicted).rvs(random_state=random_state)
    
    def dataframe(self):
        assert hasattr(self, 'data'), "No data available. Please run simulate() first."
        return pd.DataFrame.from_dict(dict(energy=self.energies.astype(int), 
                                           model=self.counts().round(1), 
                                           data=self.data)
                                           )

    def spectral_plot(self, ax =None):
        """ plot the spectrum
        """
        df = self.dataframe()
        xlim = (self.bins[0], self.bins[-1])
        _, ax = plt.subplots(1,1, figsize=(6,4)) if ax is None else ax
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
                        model=LogParabola((4e-12, 2, 0, 1e3), free=[True, True, False, False]))
        
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

