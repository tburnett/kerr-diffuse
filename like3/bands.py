import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix 
from . sourcelist import SourceList

class Band(HEALPix):
    """ class describing an energy band: energy, nside, and source model
    TODO: add PSF information for each band, and use it to convolve the source model when computing flux and gradient
    """
    def __init__(self, nside, energy, source_model):
        """ Initialize a band with given nside, energy, and source model
        
        Parameters        ----------
        nside : int
            HEALPix nside for this band
        energy : float
            Geometric mean energy for this band
        source_model : SourceList
            Source model to compute flux and its gradient for this band"""
        self.source_model = source_model
        self.energy = energy  # geometric mean energy for the band
        super().__init__(nside, order='ring', frame='galactic')  # initialize HEALPix part

    def __repr__(self):
        return f'Band(energy={self.energy:.1f} MeV, nside={self.nside})'

    def flux(self):
        return self.source_model.flux(self.energy)
    
    def flux_gradient(self):
        return self.source_model.gradient([self.energy])  


class BandList(list):
    """ Combine a list of bands with a source model to create a model for the counts in each band, and its gradient with respect to free parameters.
    Also includes a method to simulate data from the model with Poisson noise, and a demo method to create a BandList for a given model and print the counts for each band.
    """
    bins = np.logspace(2,5,13) # energy bin edges: 12 bins from 100 MeV to 100 GeV
    # PSF3 nsides defined by MK
    nsides = np.array([  16,   32,   64,  128,  256,  512,  512,  512, 1024, 2048, 2048, 2048])
 
    def __init__(self, source_model):
        """ Initialize the list of bands with given source model"""
        self.energies = np.sqrt(self.bins[1:]*self.bins[:-1])
        self.exposure_factor = np.full_like(self.energies,1e13) * self.energies/100  # simple energy-dependent exposure

        for energy, nside in zip(self.energies, self.nsides):
            self.append(
                Band(nside, energy, source_model=source_model)
                )
        self.sources = source_model
        self.parameters = source_model.parameters
        self.parameter_names = source_model.parameter_names

    def counts(self):
        """ Convert flux to counts using the exposure factor for each band
        """
        return np.array([band.source_model.flux(band.energy) * self.exposure_factor[i] 
                         for i, band in enumerate(self)])
            
    def count_gradient(self):
        """ Convert flux gradient to counts gradient using the exposure factor for each band
        """
        g = np.array([band.source_model.gradient([band.energy]) * self.exposure_factor[i] 
                         for i, band in enumerate(self)])
        return g[:,:,0].T#.transpose(1,0,2)  # reshape to (n_parameters, n_bands, n_pixels_per_band)
    
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
    
    @classmethod
    def demo(cls, model=None):
        """ Create a BandList for a given model and print the counts for each band
        """
        if model is None:
            model = SourceList.demo()
        print(f'Creating BandList for model: {model}')
        band_list = cls(model)
        for band in band_list:
            print(f'{band}: flux={band.flux():.2e}')
        print('Counts per band:', band_list.counts().astype(int))
        return band_list