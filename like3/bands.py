import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix 
from . sourcelist import SourceList

class Band(HEALPix):
    """ class describing an energy band: energy, nside, and source model
    TODO: add PSF information for each band, and use it to convolve the source model when computing flux and gradient
    """
    def __init__(self, band_info, source_model):
        """ Initialize a band with given nside, energy, and source model
        
        Parameters        
        ----------
        band_info: dict
            Dictionary containing 'energy' (geometric mean energy for the band) and 'nside
        source_model : SourceList
            Source model to compute flux and its gradient for this band"""
        self.source_model = source_model
        self.__dict__.update(band_info)  
        super().__init__(self.nside, order='ring', frame='galactic')  # initialize HEALPix part

    def __repr__(self):
        return f'Band(energy={self.energy:.1f} MeV, nside={self.nside})'

    def flux(self):
        """ Compute flux for this band using the source model at the band's energy"""
        return self.source_model.flux(self.energy)
    
    def flux_gradient(self):
        return self.source_model.gradient([self.energy])  
    
    def pixel_flux(self, ):
        """Evaluate the spatial response of the model 
        Returns the pixel coordinates and the corresponding fluxes for the model
        at the given energy.
        Returns
        -------
        k : array-like
            Pixel coordinates (HEALPix pixel indices) where the model flux is evaluated.
        v : array-like
            Corresponding flux values at the given energy for each pixel in `k`."""

        from collections import Counter
        r = Counter()
        for src in  self.source_model:
            flux = src.model(self.energy)
            k,v = src.response(self).evaluate()
            r += dict(zip(k,v*flux))  
        k = np.array([j for j in r.keys()])
        v = np.array([x for x in r.values()])
        return k,v

    def pixel_gradient(self, data):
        """Evaluate the gradient of the flux with respect to the model parameters 
        at the pixels with nonzero counts in the data.
        Parameters
        ----------
        self : Band
            The band for which to evaluate the gradient.
        data : tuple (pixels, counts)
            The pixel indices and corresponding counts for the data. The gradient will be evaluated at these pixels.

        Returns
        -------
        g : np.ndarray
            The gradient of the fluxwith respect to the model parameters, evaluated at the pixels with non
            
            """

        keys, _ = data 
        g = []
        
        for src in self.source_model:
            grad = src.model.gradient(self.energy)[src.model.free]
            _,v = src.response(self).evaluate(keys)
            g.append(v[:,None]*grad[None,:])
   
        return  np.hstack(g)    

    
    def predict(self, exposure):
        """
        Return the model prediction for  the counts for this band given an exposure.
        :param exposure: Exposure factor to scale counts.
        :return: Predicted counts in pixels with non-zero counts
        """
        k,v = self.pixel_flux()
        v *= exposure
        return k,v

    def simulate(self, *, total_counts=None, exposure=None, random_state=None):
        """
        Simulate data for the likelihood analysis.

        Parameters:
        :param total_counts: Total expected counts to simulate.
        :param exposure: Optional exposure factor to scale counts.
        :param random_state: Optional random state for reproducibility.
        :return: Simulated data (k, counts) where k are pixels with non-zero counts
        """
        from scipy import stats

        k, v = self.pixel_flux()

        if total_counts is not None:
            counts = total_counts * v/v.sum()
        elif exposure is not None:
            counts = v * exposure
        else:
            raise ValueError("Either total_counts or exposure_factor must be provided.")
        
        if random_state is None:
            return k, counts
        
        # add Poisson noise to the counts
        counts = stats.poisson(counts).rvs(random_state=random_state)
    
        select = counts > 0
        return k[select], counts[select]

    def plot_pixel_map(self, center, *, data=None, fig=None, label = None, log=True, **kwargs):
        """
        Plot a map of pixel values for this band.
        :param center: tuple or SkyCoord
            The center of the plot in (l, b) coordinates. 
        :param data: tuple (pixels, counts) or dict(pixel: count)
            The data to be plotted, the counts in each pixel. If None, the spatial response of the model will be plotted.
        :param label: str or None
            The label for the colorbar. If None, a default label will be used.
        :param kwargs: Additional keyword arguments to pass to the ZEAfigure
        """
        from utilities.skymaps import ZEAfigure
        
        pixmap = np.zeros(self.npix)
        if isinstance(data, dict):
            k = np.array(list(data.keys()))
            v = np.array(list(data.values()))
        else:
            k,v = data if data is not None else self.pixel_flux()
        pixmap[k] = v
        pixmap[pixmap == 0] = np.nan  # Set zero values to NaN for log evaluation

        kw = dict(size=8*self.psf.r68, pixelsize=self.psf.r68/50,  figsize=(6,5),
                title=f'')
        kw.update(kwargs)

        zfig = ZEAfigure(center, fig=fig, **kw)
        if log:
            zfig.imshow(np.log10(pixmap), )
            zfig.colorbar(label='log(flux)' if label is None else f'log({label})')
        else:
            zfig.imshow(pixmap, )
            zfig.colorbar(label='flux' if label is None else label)
        zfig.axes_text(0.98, 0.98, f'{(self.energy)/1e3:.2f} GeV',
                color='white', ha='right', va='top', fontsize=12)
        if fig is not None: zfig.show()

class BandList(list):
    """ Combine a list of bands with a source model to create a model for the counts in each band, and its gradient with respect to free parameters.
    Also includes a method to simulate data from the model with Poisson noise, and a demo method to create a BandList for a given model and print the counts for each band.
    """
    bins = np.logspace(2,5,13) # energy bin edges: 12 bins from 100 MeV to 100 GeV
    # PSF3 nsides defined by MK
    nsides = np.array([  16,   32,   64,  128,  256,  512,  512,  512, 1024, 2048, 2048, 2048])
 
    def __init__(self, source_model, band_info=None): 
        """ Initialize the list of bands with given source model
        Parameters
        ----------
        source_model : SourceList
            Source model to compute flux and gradient for each band
        band_info : DataFrame or None
            DataFrame containing 'energy','nside' and 'psf' for each band. If None, default values will be used based on self.bins and self.nsides.
        """

        if band_info is None:
            # create a DataFrame with energy, nside, and psf for each band
            band_info = pd.DataFrame(dict(energy = np.sqrt(self.bins[1:]*self.bins[:-1]),
                             nside = self.nsides,
                             psf=[None]*len(self.nsides)), # dummy PSF functions for now, to be filled in later
                             )
  
        for bi in band_info.to_dict(orient='records'):
            self.append(
                Band(bi, source_model=source_model)
                )
        self.sources = source_model
        self.parameters = source_model.parameters
        self.parameter_names = source_model.parameter_names
        # get energies and exposure factor for easy access
        energies = [band.energy for band in self]
        self.exposure_factor = np.full_like(energies,1e13) * energies/100  # simple energy-dependent exposure


    def counts(self):
        """ Convert flux to counts per pixel using the exposure factor for each band
        """
        return np.array([band.source_model.flux(band.energy) * self.exposure_factor[i] 
                         for i, band in enumerate(self)])
            
    def count_gradient(self):
        """ Convert flux gradient to counts gradient using the exposure factor for each band
        """
        g = np.array([band.source_model.gradient([band.energy]) * self.exposure_factor[i] 
                         for i, band in enumerate(self)])
        return g[:,:,0].T #.transpose(1,0,2)  # reshape to (n_parameters, n_bands, n_pixels_per_band)
    
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
    
    @classmethod
    def psf_demo(cls,):
        """ Create a BandList for a given model and print the counts for each band
        """
        #from pylib import psf_func as pf; reload(pf)
        from pylib.psf_func import PSFlist

        df = PSFlist.demo_df()  # get PSF functions for each band in a DataFrame
        df['nside'] = BandList.nsides
      
        model = SourceList.demo()
        print(f'Creating BandList with PSF for model: {model}')
        band_list = cls(model, df)
        print('Counts per band:', band_list.counts().astype(int))
        return band_list