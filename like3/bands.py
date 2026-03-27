"""Band-level model evaluation utilities for likelihood workflows.

This module defines:
- `Band`: a HEALPix-backed view of one energy bin with methods to evaluate
    model flux, gradients, predictions, simulation, and local map plotting.
- `BandList`: a container of `Band` objects with helpers for per-band counts,
    count gradients, and simple simulation/demo setups.
"""

import numpy as np
import pandas as pd
from astropy_healpix import HEALPix 
from astropy.coordinates import SkyCoord
from .sourcelist import SourceModel
from collections import namedtuple

# Define a namedtuple type for a key,value pair of lists
Pixel = namedtuple('Pixel', ['key', 'value'])

def create_pixel(keys, values):
    """Create a Pixel namedtuple from lists of keys and values.
    """
    if not isinstance(keys, (list, np.ndarray)) or not isinstance(values, (list, np.ndarray)):
        raise ValueError("Keys and values must be lists or numpy arrays.")
    return Pixel(keys, values)



class Band(HEALPix):
    """HEALPix representation of a single energy band for a source model.

    Notes
    -----
    TODO: include full PSF handling per band and use it to convolve source
    model terms during flux and gradient evaluation.
    """

    def __init__(self, band_info, source_model, exposure_map=None, data=None):
        """Initialize one analysis band from metadata and a source model.

        Parameters        
        ----------
        band_info : dict
            Band metadata. Expected keys include `energy`, `nside`, and `psf`
        source_model : SourceModel
            Source model, a list of sources, used to evaluate count fluxes and gradients for this band.
        exposure_map : optional
            Exposure map for the band, if any. 
        data : optional
            Data, a tuple (pixels, counts) for the band, if any.
            If provided, this is used for sparcifationn of model evaluation and gradient calculation.
            It can be set with the `simulate` method if not provided at initialization.
        """
        self.source_model = source_model
        self.exposure_map = exposure_map
        self.data = data
        for attr in 'energy nside psf'.split():
            setattr(self, attr, band_info.get(attr))

        # Initialize the HEALPix geometry used by response evaluators.
        super().__init__(self.nside, order='ring', frame='galactic')
        # set up exposure calculation function for this band based on energy, if not provided by a data-based exposure model
        if self.exposure_map is None:
            self.exposure_map = lambda pix: np.ones_like(pix) * 1e13 * self.energy / 100

    def __repr__(self):
        return f'Band(energy={self.energy:.1f} MeV, et={self.psf.event_type} nside={self.nside})'

    def response(self, source, pixels=None):
        """Return the response, or evaluation of the PSF, for a given source and pixel set.
        """
        return source.response(self).evaluate( pixels)

    def pixel_counts(self, pixels=None):
        """Evaluate model counts on a set of pixels on the sparse set of illuminated pixels

        Parameters
        ----------
        pixels : array-like, optional
            Pixel indices to evaluate. If None, all illuminated pixels are used.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pixel indices and their corresponding model counts.
        """

        from collections import defaultdict

        # Accumulate contributions from all sources into a sparse pixel map.
        accum = defaultdict(float)
        for src in self.source_model:

            flux = src.model(self.energy)
            k, v = self.response(src, pixels)
            for pix, value in zip(k, v):
                accum[pix] += value * flux

        k = np.fromiter(accum.keys(), dtype=int)
        v = np.fromiter(accum.values(), dtype=float)
        v *= self.exposure_map(k)  # apply exposure scaling to model flux
        return k, v

    def counts(self):
        """Return predicted total counts."""
        return np.sum(self.pixel_counts()[1])
    
    def pixel_gradient(self, data):
        """Evaluate per-pixel count gradients for the currently free model parameters.

        Parameters
        ----------
        data : tuple (pixels, counts)
            Pixel indices and corresponding counts; only the pixel index array is
            used to evaluate responses.

        Returns
        -------
        g : np.ndarray
            Gradient matrix with shape `(n_selected_pixels, n_free_parameters)`.
        """

        keys, _ = data 
        g = []
        
        for src in self.source_model:
            # Restrict to currently free parameters before projecting to pixels.
            grad = src.model.gradient(self.energy)[src.model.free]
            _, v = src.response(self).evaluate(keys)
            g.append(v[:, None] * grad[None, :])
        g *= self.exposure_map(keys)[:, None]  # apply exposure scaling to gradients
        return  np.hstack(g)    

    def predicted_counts(self):
        """Return predicted counts in occupied pixels """
        k, v = self.pixel_counts() 
        v *= self.exposure_map(k)
        return k, v

    def simulate(self, random_state=None, total_counts=None,):
        """Simulate pixel counts for this band.

        If `random_state` is provided, Poisson fluctuations are applied/
        Only non-zero pixels are returned.

        Parameters
        ----------
        random_state : int or np.random.Generator, optional
            Random seed/state for reproducible Poisson sampling.
        total_counts : float, optional
            Total expected counts to distribute proportionally to model weights.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pixel indices and counts, with Poisson noise if `random_state` is provided,
            and only non-zero pixels returned.
        """
        k, counts = self.pixel_counts()

        if total_counts is not None:
            # Normalize the model shape to the requested total counts.
            counts = total_counts * counts / counts.sum()
        
        if random_state is not None:
            # Apply Poisson noise when a seed or Generator is provided.
            rng = np.random.default_rng(random_state)
            counts = rng.poisson(counts)
        else:
            counts = counts.astype(int)
        
        # return only non-zero pixels to avoid unnecessary computation in likelihood evaluation
        select = counts > 0
        return k[select], counts[select]

    def loglike(self, skydir=None):
        """Compute the Poisson log-likelihood ."""

        if skydir is not None:
            self.source_model.setposition(skydir)
            
        data_pix, counts = self.data

        _, model = self.pixel_counts(data_pix)

        return np.sum(counts * np.log(model) - model)

    def plot_pixel_map(self, center, *, data=None, fig=None, label=None, log=True, **kwargs):
        """Plot per-pixel values for this band in a local ZEA projection.

        Parameters
        ----------
        center : tuple or SkyCoord
            Plot center in sky coordinates.
        data : tuple[np.ndarray, np.ndarray] or dict, optional
            Pixel/value data to display. If omitted, uses `self.pixel_counts()`.
        fig : matplotlib.figure.Figure, optional
            Existing figure target.
        label : str, optional
            Colorbar label.
        log : bool, optional
            If true, plot `log10` values.
        **kwargs
            Forwarded to `utilities.skymaps.ZEAfigure`.
        """
        from utilities.skymaps import ZEAfigure
        from matplotlib import colors
        
        pixmap = np.zeros(self.npix)
        if isinstance(data, dict):
            k = np.array(list(data.keys()))
            v = np.array(list(data.values()))
        else:
            k, v = data if data is not None else self.pixel_counts()
        pixmap[k] = v
        # Mask empty pixels so they do not dominate the color scale.
        pixmap[pixmap == 0] = np.nan

        # PSF width sets both field size and resolution for a compact local view.
        zkw = {
            'size': 8 * self.psf.r68,
            'pixelsize': self.psf.r68 / 50,
            'figsize': (6, 5),
            'title': '',
        }
        zkw.update(kwargs)

        zfig = ZEAfigure(center, fig=fig, **zkw)
        zfig.imshow(np.log10(pixmap) if log else pixmap, )# norm=(colors.LogNorm() if log else None) )
        zfig.colorbar(label='log10(counts)' if log else 'counts', shrink=0.9, extend='max')
   
        zfig.axes_text(0.98, 0.98, f'{self.energy / 1e3:.2f} GeV',
                color='white', ha='right', va='top', fontsize=12)
        

class BandList(list):
    """Collection of `Band` objects sharing a single source model.

    The class provides per-band count predictions, count gradients, and simple
    simulation/demo helpers.
    """
    bins = np.logspace(2,5,13) # energy bin edges: 12 bins from 100 MeV to 100 GeV
    # PSF3 nsides defined by MK
    nsides = np.array([  16,   32,   64,  128,  256,  512,  512,  512, 1024, 2048, 2048, 2048])
 
    def __init__(self, source_model, band_info=None): 
        """Initialize a list of bands for a shared source model.

        Parameters
        ----------
        source_model : SourceModel
            Source model to compute flux and gradient for each band.
        band_info : DataFrame or None
            Table containing `energy`, `nside`, and `psf` for each band. If
            omitted, defaults derived from `bins` and `nsides` are used.
        """

        if band_info is None:
            # Default one-band-per-energy-bin table.
            band_info = pd.DataFrame(
                dict(
                    energy=np.sqrt(self.bins[1:] * self.bins[:-1]),
                    nside=self.nsides,
                    psf=[None] * len(self.nsides),
                )
            )
  
        for bi in band_info.to_dict(orient='records'):
            self.append(Band(bi, source_model=source_model))
        self.sources = source_model
        self.parameters = source_model.parameters
        self.parameter_names = source_model.parameter_names

        # Keep a simple default exposure scaling tied to energy.
        # energies = [band.energy for band in self]
        # self.exposure_factor = np.full_like(energies, 1e13) * energies / 100

    def counts(self):
        """Return predicted total counts per band."""
        return np.array([ band.counts() for band in self])
            
    def count_gradient(self):
        """Return count gradient array for all free model parameters by band."""
        g = np.array([band.pixel_gradient()  for band in self])
        return g[:, :, 0].T
    
    def simulate(self, random_state=42): 
        """Simulate per-band counts, optionally with Poisson fluctuations.

        Parameters
        ----------
        random_state : int or None
            Random state for reproducibility. If None, no noise is added.
        """
        # compute predicted counts with current parameters
        # predicted = self.pixel_counts()
        # if random_state is None:
        #     return predicted

        # # Draw one Poisson realization per band.
        # rng = np.random.default_rng(random_state)
        # return rng.poisson(predicted)

    def source_position_loglike(self, source_name, data=None, frame='galactic', clip=1e-30):
        """Return a callable Poisson log-likelihood as a function of source position.

        The returned function evaluates the model log-likelihood while shifting a
        single source to each trial position and keeping all other model elements
        fixed.

        Parameters
        ----------
        source_name : str or Source
            Source identifier accepted by `SourceModel.find_source`.
        data : sequence[tuple[np.ndarray, np.ndarray]] or None
            Per-band observed data as `(pixels, counts)`. If omitted, uses
            `band.data` for each band and requires all bands to have data set.
        frame : str
            Coordinate frame used when trial positions are given as `(lon, lat)`.
        clip : float
            Lower bound applied to model counts to avoid `log(0)`.

        Returns
        -------
        callable
            Function `f(position) -> loglike`, where `position` can be a
            `SkyCoord` or a 2-tuple of degrees.
        """
        src = self.sources.find_source(source_name)
        if src.skydir is None:
            raise ValueError('source_position_loglike requires a localized source with skydir')

        if data is None:
            data = [band.data for band in self]
        if len(data) != len(self):
            raise ValueError('data length must match number of bands')
        if any(d is None for d in data):
            raise ValueError('missing band data; pass data explicitly or set band.data for all bands')

        def to_coord(position):
            if isinstance(position, SkyCoord):
                return position
            if hasattr(position, '__iter__') and len(position) == 2:
                return SkyCoord(position[0], position[1], unit='deg', frame=frame)
            raise ValueError(f'unrecognized position: {position}')

        original_skydir = src.skydir

        def loglike(position):
            src.skydir = to_coord(position)
            try:
                total = 0.0
                for band, band_data in zip(self, data):
                    keys, counts = band_data
                    model = np.zeros_like(counts, dtype=float)
                    for source in band.source_model:
                        flux = source.model(band.energy)
                        _, response_values = source.response(band).evaluate(keys)
                        model += response_values * flux
                    model *= band.exposure_map(keys)
                    model = np.clip(model, clip, None)
                    total += np.sum(counts * np.log(model) - model)
                return float(total)
            finally:
                src.skydir = original_skydir

        return loglike
    
    @classmethod
    def demo(cls, model=None):
        """Build a demo `BandList` and print per-band flux/count summaries."""
        if model is None:
            model = SourceModel.demo()
        print(f'Creating BandList for model: {model}')
        band_list = cls(model)
        for band in band_list:
            print(f'{band}: counts={band.pixel_counts():.2e}')
        print('Counts per band:', band_list.counts().astype(int))
        return band_list
    
    @classmethod
    def psf_demo(cls,):
        """Build a demo `BandList` populated with PSF metadata."""
        #from pylib import psf_func as pf; reload(pf)
        from pylib.psf_func import PSFlist

        df = PSFlist.demo_df()  # get PSF functions for each band in a DataFrame
        df['nside'] = BandList.nsides
      
        model = SourceModel.demo()
        print(f'Creating BandList with PSF for model: {model}')
        band_list = cls(model, df)
        print('Counts per band:', band_list.counts().astype(int))
        return band_list