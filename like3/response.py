"""
Manage the instrument rsponse for non-diffuse sources
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from astropy_healpix import HEALPix 



class Response:
    def __init__(self, source, band, roi=None, **kwargs):
        """
        Given a source and a band, set values for set of pixels
        """    
        self.source = source
        self.band = band   
        raise NotImplementedError(f'Called with source {source.name}')


class PointResponse(HEALPix):
    """
    Class to evaluate and visualize the PSF response of a given Fermi-LAT PSF model (e.g., PSF3)
     at any sky direction."""

    def __init__(self, source, band): 
        """Initialize with a PSF band (e.g., PSF3) which contains the R68 and PSF function."""
        self.source = source
        self.sdir = source.skydir
        self.band = band
        self.r68 = band.psf.r68
        super().__init__(nside=band.nside, order='ring', frame='galactic')
    
    def evaluate(self,  cpix=None, *,r68_radius=3):
        """Evaluate the PSF response at a given sky direction `sdir` (SkyCoord). 
        Returns a tuple of HEALPix pixel indices and corresponding PSF values times the pixel area."""
        
        if cpix is None:
            cpix = self.cone_search_skycoord(self.sdir, Angle(r68_radius*self.r68, 'deg'))
        aa = self.sdir.separation(self.healpix_to_skycoord(cpix)).deg
        vpix = np.array(list(map(self.band.psf, aa))) * self.pixel_area.value
        return cpix, vpix

    def plot_response(self, *, r68_radius=3):
        """Plot the PSF response as a function of angular separation from `sdir`, for pixels within `r68_radius`*R68."""
        cpix, vpix = self.evaluate(r68_radius=r68_radius)
        aa = self.sdir.separation(self.healpix_to_skycoord(cpix)).deg
        
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(aa, vpix, '.', c='cyan', label='PSF3 PSF')
        ax.axvline(self.r68, c='red', ls='--', label='R68')
        ax.set_xlabel('Angular separation (deg)')
        ax.set_ylabel('PSF value')
        ax.legend(fontsize=12)
        ax.set_title(f"{self.band.energy} MeV:  nside={self.nside}, r68={self.r68:.3f} deg", fontsize=14)
        ax.set(xlim=(0, None), yscale='log')
        plt.show()

    def add_response_to_map(self,  pixmap):
        """Add the PSF response centered on `sdir` to an existing HEALPix map `pixmap`."""
        cpix, vpix = self.evaluate()
        pixmap[cpix] += vpix
    
    def plot_psf_map(self, *, frame='galactic', **kwargs):
        """Plot the PSF response as a HEALPix map centered on `sdir`."""
        from utilities.skymaps import ZEAfigure
        k,v = self.evaluate()
        pixmap = np.zeros(self.npix)
        pixmap[k] = v
        pixmap[pixmap == 0] = np.nan  # Set zero values to NaN for log evaluation
        kw = dict(size=8*self.r68, pixelsize=self.r68/50,  fig=None, figsize=(6,5))
        kw.update(kwargs)
        
        zfig = ZEAfigure(self.sdir, frame=frame, **kw)
        zfig.imshow(np.log10(pixmap), )
        zfig.colorbar(label='log(PSF value)')
        zfig.scatter(self.sdir, c='red', marker='x', label='Center')
        zfig.legend()
        zfig.show()

    @classmethod
    def example_plots(cls, band, sdir=None):
        """Class method to demonstrate the PSF response and map plotting."""   
        self = cls(None, band)
        if sdir is None:
            sdir = SkyCoord(10,10, unit='deg', frame='galactic')
        elif type(sdir) is tuple:
            sdir = SkyCoord(*sdir, unit='deg', frame='galactic')
        self.sdir = sdir
        self.plot_response()
        self.plot_psf_map()


    
class ExtendedResponse(Response):
    pass
