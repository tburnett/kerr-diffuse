"""Instrument-response utilities for non-diffuse source components.

This module currently provides point-source PSF response evaluation and
visualization helpers, plus placeholders for extended-source response support.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from astropy_healpix import HEALPix 



class Response:
    """Base response interface for source/band combinations."""

    def __init__(self, source, band,):
        """Store shared source/band references for derived response classes."""
        self.source = source
        self.band = band   
        raise NotImplementedError(f'Called with source {source.name}')


class PointResponse(HEALPix):
    """Evaluate and visualize a point-source PSF response on a HEALPix grid."""

    def __init__(self, source, band): 
        """Initialize from a source and a band carrying PSF and nside metadata."""
        self.source = source
        self.sdir = source.skydir
        self.band = band
        self.r68 = band.psf.r68
        super().__init__(nside=band.nside, order='ring', frame='galactic')
    
    def evaluate(self,  cpix=None, *,r68_radius=3):
        """Evaluate PSF weights over pixels near the source direction.

        Parameters
        ----------
        cpix : array-like or None
            Optional explicit pixel index list. If omitted, uses a cone search.
        r68_radius : float
            Cone radius in units of `r68` when `cpix` is not supplied.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pixel indices and pixel-integrated PSF weights.
        """
        
        if cpix is None:
            cpix = self.cone_search_skycoord(self.sdir, Angle(r68_radius*self.r68, 'deg'))
        # Evaluate PSF at angular distance to each selected pixel center.
        aa = self.sdir.separation(self.healpix_to_skycoord(cpix)).deg
        vpix = np.array(list(map(self.band.psf, aa))) * self.pixel_area.value
        return cpix, vpix

    def plot_response(self, *, r68_radius=3):
        """Plot PSF value versus angular distance from source center."""
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
        """Accumulate this source response into an existing HEALPix map."""
        cpix, vpix = self.evaluate()
        pixmap[cpix] += vpix
    
    def plot_psf_map(self, *, frame='galactic', **kwargs):
        """Render local PSF-response map around the source direction."""
        from utilities.skymaps import ZEAfigure
        k,v = self.evaluate()
        pixmap = np.zeros(self.npix)
        pixmap[k] = v
        # Hide zero bins before log scaling to avoid `-inf` in the image.
        pixmap[pixmap == 0] = np.nan
        size = kwargs.pop('size', 8 * self.r68)
        pixelsize = kwargs.pop('pixelsize', self.r68 / 50)
        fig = kwargs.pop('fig', None)
        figsize = kwargs.pop('figsize', (6, 5))

        zfig = ZEAfigure(
            self.sdir,
            frame=frame,
            size=size,
            pixelsize=pixelsize,
            fig=fig,
            figsize=figsize,
            **kwargs,
        )
        zfig.imshow(np.log10(pixmap), )
        zfig.colorbar(label='log(PSF value)')
        zfig.scatter(self.sdir, c='red', marker='x', label='Center')
        zfig.legend()
        zfig.show()

    @classmethod
    def example_plots(cls, band, sdir=None):
        """Demonstrate response curve and local PSF map for one band."""
        # Construct a lightweight source-like object with `skydir` for demo use.
        if sdir is None:
            sdir = SkyCoord(10,10, unit='deg', frame='galactic')
        elif isinstance(sdir, tuple):
            sdir = SkyCoord(*sdir, unit='deg', frame='galactic')

        demo_source = type('DemoSource', (), {'skydir': sdir})()
        self = cls(demo_source, band)
        self.sdir = sdir
        self.plot_response()
        self.plot_psf_map()


    
class ExtendedResponse(Response):
    """Placeholder for future extended-source response implementation."""

    def __init__(self, source, band, roi=None, **kwargs):
        self.source = source
        self.band = band
        self.roi = roi
