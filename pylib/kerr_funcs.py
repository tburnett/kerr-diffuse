import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from astropy_healpix import HEALPix 
from pathlib import Path


class PixelTable(dict):
    """ Manage a pixel table for a Kerr diffuse model, with methods to access and visualize the data
      in various ways.
    
    """

    class Band(HEALPix):
        """ A band of the Kerr model, defined by an event type PSF and energy range, 
        with associated photon data and model components."""

        def __init__(self, meta):
            self.psf, self.e0, self.e1, nside, self.nocc = meta
            self.counts=0 # 
            ekey = lambda energy:  (np.log10(energy)*4-8).astype(int) #energy bin index

            self.key = (int(self.psf[-1]), ekey(self.e0) ) # key is (psf index, energy index) tuple
            self.energy = f'{np.sqrt(self.e0*self.e1)*1e-3:.2f} GeV'
            super().__init__(nside, frame='galactic', order='nested')

        def __repr__(self) -> str:
            return f"Band{self.key}: {self.psf}@{self.energy} nside {self.nside} occ {self.nocc/(12*self.nside**2):.3f}"

        def pix_to_ring(self, *, inplace=False):
            """ Convert and return the pixel array with RING indexing
            """
            if inplace:
                self.pix = self.nested_to_ring(self.pix)
                self.order='ring'
     
            return self.nested_to_ring(self.pix)

        @property
        def skycoords(self):
            """ Return SkyCoord array for photons in this band """
            return self.healpix_to_skycoord(self.pix)

        def cone_search(self, center, radius=5.0):
            """ Return mask for photons within radius (deg) of center (SkyCoord)"""            

            sc = self.healpix_to_skycoord(self.pix)
            return sc.separation(center) < Angle(radius, 'deg')

            # slower, more inclusive list
            # cone_pix = hp.cone_search_skycoord(center, radius=Angle(radius, 'deg'))
            # return  np.in1d(self.pix, cone_pix)
        
        def ring_map(self, nside=None, component='data', frame='galactic'):
            """Return, for display purposes, a HEALPix RING map of the selected component or combination.
                nside: if set, and less than the Band's, combine pixels to this nside
                component: 'data', 'diffuse', 'ptsrc', 'model' (diffuse+ptsrc), 'resid' (data-model)
                frame: 'galactic', 'geocentricmeanecliptic', 'equatorial', etc.
            """
            from astropy_healpix import HEALPix

            def model():
                return self.diffuse + self.ptsrc +\
                      (self.extsrc if hasattr(self, 'extsrc') else 0) + (self.sunmoon if hasattr(self, 'sunmoon') else 0)

            if component=='resid':
                values = self.photons - model()
            elif component=='sigma': #normalized residual
                d, m = self.photons, model()
                return (d-m)/np.sqrt(m.clip(1e-2,None)) # avoid div by zero
            elif component=='model':
                values = model()
            else:
                values = dict(data=self.photons, 
                              diffuse=self.diffuse, 
                              ptsrc = self.ptsrc,
                              extsrc = self.extsrc if hasattr(self, 'extsrc') else np.zeros_like(self.photons),
                              sunmoon = self.sunmoon if hasattr(self, 'sunmoon') else np.zeros_like(self.photons),
                              )[component]

            if nside is None or nside>self.nside:
                nside = self.nside
            ratio = (self.nside//nside)**2

            if frame=='galactic':
                pix = self.pix
            elif frame=='icrs':
                tsky = self.skycoords.transform_to(frame)
                pix = self.lonlat_to_healpix(tsky.ra, tsky.dec)
            else:
                tsky = self.skycoords.transform_to(frame)
                pix = self.lonlat_to_healpix(tsky.lon, tsky.lat) 

            if self.order=='ring':
                # converted to ring: must convert back to nested first
                pix = self.ring_to_nested( pix )
            # now convert to lower nside and then to RING
            pix = HEALPix(nside=nside).nested_to_ring(pix//ratio)
            
            mp = np.zeros(12*nside**2) # RING sequence of values
            np.add.at(mp, pix, values)
            return mp
        
        def ait_plot(self, component, *, nside=128, figsize=(12,6), fig=None, colorbar=True, 
                     shrink=0.7, cmap='viridis', frame='galactic', log=True, **kwargs):
            from utilities.skymaps import AITfigure

            mp = self.ring_map(nside, component=component, frame=frame)
            if log: mp[mp==0] = np.nan

            afig = AITfigure(fig=fig, figsize=figsize, title=f'{component} for {self}')
            afig.imshow(np.log10(mp) if log else mp, cmap=cmap, **kwargs)
            if colorbar:
                afig.colorbar(label='log10(counts)' if log else 'counts', shrink=shrink)
            return afig   

        def zea_plot(self, component, center, *, nside=256, figsize=(8,8), 
                    pixelsize=0.05, size=5, fig=None,
                     cmap='viridis', colorbar=True, title=None,**kwargs):
            from utilities.skymaps import ZEAfigure

            zfig = ZEAfigure(center, size=size, fig=fig, figsize=figsize, pixelsize=pixelsize,
                             title=f'{component} for {self}' if title is None else title) 
            
            if component is not None:
                mp = self.ring_map(nside, component=component)
                mp[mp==0] = np.nan
                zfig.imshow(np.log10(mp), cmap=cmap, **kwargs)

                if colorbar:
                    zfig.colorbar(label='log10(counts)', shrink=0.7)
            return zfig    
        
        def get_outliers(self, sigma_min=4):
            """ Return a dataframe with nested pixel indeces, data, model, and sigma for all pixels with residuals > sigma_min """
            
            d, m = self.ring_map(None, 'data',), self.ring_map(None, 'model')
            r = (d-m)/np.sqrt(m.clip(1e-2, None))
            out = r > sigma_min
            pix = np.arange(12*self.nside**2)
            return pd.DataFrame( dict(pixel=self.ring_to_nested(pix[out]), data=d[out], model=m[out], sigma=r[out] )) 
         
       
    def __init__(self, root, *, ring=False):
        """ 
        Load Kerr model from files root+'.npz' and root+'.pickle'
        root : path root for files
        ring : if True, convert pixel indices to RING ordering"""

        import pickle
        root = Path(root).expanduser()
  
        filename, meta = root.with_suffix('.npz'), root.with_suffix('.pickle')
        super().__init__()
        self.name = root.name
        self.ring = ring

        with np.load(filename) as f:
            # print('keyes', f.keys())
            self.diffuse = f['diffuse']
            self.ptsrc  = f['pointsources']
            self.photons = f['counts'].astype(np.int32)
            self.pix = f['indices']
            if 'extendedsources' in f:
                self.extsrc = f['extendedsources']
            if 'sunmoon' in f:
                self.sunmoon = f['sunmoon']

        with open(meta, 'rb') as inp:
            meta = pickle.load(inp)
            self.meta_df = pd.DataFrame(meta, columns='event_type emin emax nside nocc'.split())
        self.meta_df['occupancy']= (self.meta_df.nocc/(12*self.meta_df.nside**2)).round(3)

        nbands = len(meta)
        offset = 0
        for i,m in enumerate(meta):
            b = self.Band(m)
            # if b.e0<100: continue
            self[b.key] = b
            nocc = m[-1]
            b.diffuse = self.diffuse[offset:offset+nocc]
            b.ptsrc   = self.ptsrc[offset:offset+nocc]
            b.photons = self.photons[offset:offset+nocc]
            if hasattr(self, 'extsrc'):
                b.extsrc  = self.extsrc[offset:offset+nocc]
            if hasattr(self, 'sunmoon'):
                b.sunmoon = self.sunmoon[offset:offset+nocc]
            b.pix     = self.pix[offset:offset+nocc]            
            offset += nocc
            b.totals = dict(diffuse=self.diffuse[-nbands+i], ptsrc=self.ptsrc[-nbands+i],)
        # the total pixel sums
        self.totals = dict(diffuse=self.diffuse[offset:], ptsrc=self.ptsrc[offset:],)
            
        print(f"""Loaded Kerr model from "{filename}":
            {len(self)} bands {self[(0,4)]} ... {self[(3,11)]}
            {self.photons.sum().astype(int):,d} photons
            {len(self.pix):,d} pixels
            """)
        
        if ring:
            for b in self.values():
                b.pix_to_ring(inplace=True)

            # it seems that the Band arrays are copies. So copy back pix changes
            for b in self.values():
                # key = b.key
                # nocc = b.nocc
                start = sum(self[k].b.nocc for k in self if k<b.key)
                self.pix[start:start+b.nocc] = b.pix
        
        

    def __call__(self, *pars):
        assert len(pars)==2, "Provide psf and energy bin index"
        return self[pars]
    
    def ring_map(self, nside=128, component='data', frame='galactic'):
        """Return a HEALPix ring map at the given nside, combining all bands with that nside or larger
          for the given component."""
        hmap = np.zeros(12*nside**2)
        for band in self.values():
            if band.nside>=nside:
                hmap += band.ring_map(nside, component, frame=frame)
        return hmap
    
    def ait_plot(self, component='data', *, nside=128, figsize=(12,6), fig=None, colorbar=True, 
                 shrink=0.7, cmap='viridis', frame='galactic', **kwargs):
        from utilities.skymaps import AITfigure

        mp = self.ring_map(nside, component=component, frame=frame)
        mp[mp==0] = np.nan

        afig = AITfigure(fig=fig, figsize=figsize, title=f'{component} for PixelTable {self.name}')
        afig.imshow(np.log10(mp), cmap=cmap, **kwargs)
        if colorbar:
            afig.colorbar(label='log10(counts)', shrink=shrink)
        return afig
    
    def zea_plot(self, center, *, component='data', nside=256, 
                figsize=(8,8), size=5, pixelsize=0.1, fig=None,
                frame='icrs', proj='ZEA', cmap='viridis', 
                colorbar=True, title=None,**kwargs):
        from utilities.skymaps import ZEAfigure

        mp = self.ring_map(nside, component=component, frame=frame)
        mp[mp==0] = np.nan

        zfig = ZEAfigure(center, size=size, fig=fig, proj=proj,figsize=figsize, title=title, frame=frame)
        zfig.imshow(np.log10(mp), cmap=cmap, **kwargs)
        if colorbar:
            ## NOTE: this is not compatible with a following call to colorbar
            zfig.colorbar(label='log10(counts)', shrink=0.7)
        return zfig

        
def multi_ait(et, component='diffuse'):
    fig = plt.figure(layout='constrained', figsize=(13,5))
    subfigs = fig.subfigures(3,4, wspace=0.07)
    keys = [f'{et}'+ k for k in '0123456789ABCDEF']

    for sfig, key in zip(subfigs.flat, keys):
        if key not in self:
            continue
        b  = self[key]
        ait = b.ait_plot( component, nside=128, fig=sfig, colorbar=False)
        ait.title(str(b), fontsize=10)

def residual_scatter(model, norm, ax=None, ylim=np.array([-5,5])):
    """ 
    Scatter plot of residuals vs model counts/pixel
    Includes binned mean and stddev
    """
    x = np.log10(model)
    y = norm
    xmax = x.max()
    bins = np.arange(x.min(),xmax,0.5)
    if np.histogram(x, bins=bins)[0][-1]<10:
        bins = bins[:-1]
        xmax -= 0.5

    _, ax = plt.subplots(figsize=(8,4)) if ax is None else (ax.figure, ax)

    bstat = BinnedStat(x, y, bins, )
    ax.axhline(0, color='0.5', ls='--', lw=2)
    ax.errorbar(x=bstat.x, y= bstat.mean, 
                xerr= bstat.xerr, yerr=bstat.std,#/np.sqrt(bstat.count), 
                fmt='o', ms=10, label='binned mean', color='yellow');
    
    ax.scatter(x, y.clip(*ylim),  s=5, alpha=0.3 ,color='0.5')

    ax.set(xlabel='model counts/pixel', ylabel=r'residual ($\sigma$ units)', xscale='linear', 
        ylim = ylim, yscale='linear') 
    ax.set(xticks=(ticks:=np.arange(int(x.min()+1), int(xmax)+0.5,1)), 
        xticklabels=[f'$10^{{{int(tick)}}}$' for tick in ticks], xlim=(x.min(),xmax))


class ResidualPlotter:

    def __init__(self, band, nside=64):
        self.nside = min(nside, band.nside) if nside is not None else band.nside
        self.resid = band.ring_map(component='resid', nside=nside) 
        self.model = band.ring_map(component='model', nside=nside)
        # clean up zeros in model to avoid div by zero
        self.model[self.model==0] = np.min(self.model[self.model>0])
        self.rnorm = (self.resid/np.sqrt(self.model))
        self.photons = band.ring_map(component='data', nside=nside)
        self.band = band

    def residual_adjustment(self, ylim=np.array([-10,10]), ax=None):
        """Fit polynomial function to percent residuals and adjust model accordingly
        ax : optional axis to plot on
        """
        rpct = 100*(self.photons/self.model -1)
        # linear fit to relative residuals
        self.coefficients = np.polyfit(np.log10(self.model), rpct, 2)
        poly_fit = np.poly1d(self.coefficients)
        self.adjusted_model = self.model*(1+poly_fit(np.log10(self.model))/100)

        if ax is not None:
            ax.scatter( self.model, rpct.clip(*ylim),  s=15, alpha=0.5 ,color='0.5')
            ax.axhline(0, color='0.5', ls='--', lw=2)
            ax.set(xlabel='model counts/pixel', ylabel=r'residual (%)', xscale='log', 
                ylim = ylim, yscale='linear') 
            ax.plot((x:=(self.model.min(), self.model.max())),poly_fit(np.log10(x)),
                     color='red', lw=2, label='linear fit')
            ax.set_title('Percent residuals with polynomial fit')
            # ax.legend()

    def residual_hist(self, ax=None, rnorm=None, ylim=np.array([-5,5]), legend_fontsize=14):
        """ax : optional axis to plot on
        rnorm : optional residuals to plot: if None, use self.rnorm
        """
        from scipy.stats import norm

        fig, ax = plt.subplots(figsize=(4,3)) if ax is None else (ax.figure, ax)
    
        if rnorm is None:
            rnorm = self.rnorm

        nfit = norm.fit(rnorm[~np.isnan(rnorm)])
        ax.hist(rnorm.clip(*ylim), bins=25, range=ylim, density=True, 
                histtype='stepfilled', alpha=0.5,)
        ax.plot((x:=np.linspace(*ylim,num=25)), norm.pdf(x, *nfit), 'r-', lw=4,
            label =rf'$\mu$={nfit[0]:.2f}'+'\n'+ rf'$\sigma$={nfit[1]:.2f}')
        ax.legend(fontsize=legend_fontsize, loc='lower center')
        ax.set(xlabel=r'residual ($\sigma$ units)', ylabel='density', 
               yscale='log',xlim=ylim, ylim=(1e-4, 0.5))

    def plots(self):

        from utilities.skymaps import AITfigure

        fig = plt.figure(layout='constrained', figsize=(15,5))
        fig.suptitle(str(self.band), fontsize=18)
        fig1,fig2 = fig.subfigures(ncols=2, wspace=0.07)
        ap = self.band.ait_plot(component='data', nside=self.nside, fig=fig1,)
        ap.title( 'photons\n'+f'nside {self.nside}', x=0, y = 0.9,ha='left', fontsize=16)

        resid = self.resid 
        model = self.model
    
        afig = AITfigure(fig=fig2, )
        afig.imshow( resid/np.sqrt(model), 
                    cmap='coolwarm',  vmin=-2, vmax=2)#**kwargs)
        afig.colorbar(label='normalized residual', shrink=0.5)
        afig.title( f'residuals',x=0, ha='left', fontsize=16)
        plt.show()
    
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,4), gridspec_kw={'width_ratios': [2.5, 1]})
        ylim=np.array([-5,5])
        residual_scatter(self.model, self.rnorm, ax=ax1, ylim=ylim)

        self.residual_hist(ax=ax2, ylim=ylim)
        plt.show()


def multi_residual_plotter(self, nside=64):
    fig, axx = plt.subplots(5, 9, figsize=(15,6),# constrained_layout=True, 
                            sharex=True, sharey=True,gridspec_kw={'hspace':0.1, 'wspace':0},
                            height_ratios=[0.1,1,1,1,1] , width_ratios=[0.5,1,1,1,1,1,1,1,1] ) 
 
    axx[0,0].axis('off')
    for i, ax in enumerate(axx.flat[1:9]):
        ax.axis('off')
        ax.text(0.5, 0.5, self(3,i).energy, transform=ax.transAxes, fontsize=18, ha='center', va='center')
    for i, ax in enumerate(axx.flat[9:]):
        col = i%9
        row = i//9
        if col==0:
            ax.text(0.5, 0.5, self(row,7).psf.upper(), transform=ax.transAxes, fontsize=18, ha='center', va='center',)
            ax.axis('off')
            continue
        try:
            band = self(row, col-1)
        except KeyError:
            ax.set_visible(False)
            continue
        if band.key[1]<0:
            ax.set_visible(False)
            continue
        rp = ResidualPlotter(band, nside=nside)
        rp.residual_hist(ax=ax, legend_fontsize=10) 
        ax.set(ylabel='', xlabel='', yticks=[])
    ax.set(ylim=(1e-4, 0.5)) 
    plt.show()

class BinnedStat:
    """ For ROOT-like profile plot 
    Example:
    bstat = BinnedStat(x,y,bins)

    plt.errorbar(x=bstat.x, y= bstat.mean, 
             xerr= bstat.xerr,yerr=bstat.std/np.sqrt(bstat.count), 
             fmt='o', label='binned mean', color='yellow')
    """
    def __init__(self, x,y, bins):
        from scipy.stats import binned_statistic
        self.mean, edges, _ = binned_statistic(x, y, statistic='mean', bins=bins)
        self.std, _, _ = binned_statistic(x, y, statistic='std', bins=bins)
        self.count, _, _ = binned_statistic(x, y, statistic='count', bins=bins)
        self.x = 0.5 * (edges[:-1] + edges[1:])
        self.xerr = 0.5*(edges[1:]-edges[:-1])
        self.bins = bins


from astropy.io import fits

class KerrDataFile:
    """
    To render in FITS format

    Filename: files/16years_zmax100_4bpd.fits
    No.    Name      Ver    Type      Cards   Dimensions   Format
    0  PRIMARY       1 PrimaryHDU      31   ()      
    1  SKYMAP        1 BinTableHDU     21   26261758R x 3C   [J, I, J]   
    2  BANDS         1 BinTableHDU     20   32R x 4C   [J, D, D, J]   
    ColDefs(
        name = 'PIX'; format = 'J'
        name = 'CHANNEL'; format = 'I'
        name = 'VALUE'; format = 'J'
    )ColDefs(
        name = 'NSIDE'; format = 'J'
        name = 'E_MIN'; format = 'D'; unit = 'keV'
        name = 'E_MAX'; format = 'D'; unit = 'keV'
        name = 'EVENT_TYPE'; format = 'J'
    )
    """
    def __init__(self, kerrmodel, *,order='ring'):
        """ 
        kerrmodel : 
        order
        """
        self.pixeltable = kerrmodel
        self.order = order

    def __repr__(self):
        return f'KerrDataFile for {self.pixeltable}'
    

    def skymap_hdu(self):
        """ create a skymap HDU            
        """
        km = self.pixeltable

        # make the channel list
        et = km.meta_df.event_type.apply(lambda x: int(x[-1])+2)
        nocc = km.meta_df.nocc
        # channels: index of BANDS entry for each pixel
        chn = np.array([], dtype=np.uint32)
        for n,v in enumerate(nocc):
            chn = np.append(chn, np.full(v, n, dtype=np.uint32))

        cols = [
            fits.Column(name='PIX', format='J',    array=km.pix),
            fits.Column(name='CHANNEL', format='I',array=chn),
            fits.Column(name='VALUE', format='J',  array=km.photons),
        ]
        hdu=fits.BinTableHDU.from_columns(cols, name='SKYMAP')
        hdu.header.update(
            PIXTYPE='HEALPIX',
            INDXSCHM='SPARSE',
            ORDERING='RING' if self.pixeltable.ring else 'NESTED',
            COORDSYS='GAL',
            BANDSHDU='BANDS',
            AXCOLS='E_MIN,E_MAX',
            )
        return hdu  

    def band_hdu(self, version=5):
        """ create a bands HDU
        """
        df = self.pixeltable.meta_df
        band_cols = [
            fits.Column(name='NSIDE', format='J', array=df.nside),
            fits.Column(name='E_MIN', format='D', array=df.emin*1e+3, unit='keV'),
            fits.Column(name='E_MAX', format='D', array=df.emax*1e+3, unit='keV'),
            fits.Column(name='EVENT_TYPE', format='J', array=df.event_type.apply(lambda x: int(x[-1])+2)),
        ]
        hdu=fits.BinTableHDU.from_columns(band_cols, name='BANDS')
        hdu.header.update(VERSION=version)
        return hdu

    def writeto(self, filename, overwrite=True):
        """write to a FITS file        """

        hdus=[fits.PrimaryHDU(), 
              self.skymap_hdu(), 
              self.band_hdu()] 
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
        print(f'wrote file {filename}' + (f' (ring={self.pixeltable.ring})' if self.pixeltable.ring else ''))

    @classmethod
    def readfrom(cls, filename, kerrmodel):
        """ read from a file
        """
        hdus = fits.open(filename)
        print(f'Read KerrDataFile from {filename}:')
        hdus.info()
        return cls(kerrmodel)

    
    @classmethod
    def to_fits(cls, kerrfile, fitsfile, *, ring=False, overwrite=True):
        """ Translate Kerr format to FITS
        """
        km = PixelTable(kerrfile, ring=ring )
        cls(km).writeto(fitsfile, overwrite=overwrite)

def grouper(points, radius,):
    """Group a SkyCoord array of points into connected clusters using a separation threshold.

    Parameters
    ----------
    points : Skycoord array
    radius : float
        Maximum pairwise separation, in degrees, for two points to be considered neighbors.

    Returns
    -------
    list[np.ndarray]
        A list of clusters, where each cluster is an array of point indices.
    """

    if radius <= 0:
        raise ValueError("radius must be > 0")

    n_points = len(points)
    if n_points == 0:
        return []

    unvisited = np.ones(n_points, dtype=bool)
    clusters = []

    # Build connected components where edges join points within `radius`.
    for seed in range(n_points):
        if not unvisited[seed]:
            continue

        stack = [seed]
        unvisited[seed] = False
        cluster = []

        while stack:
            i = stack.pop()
            cluster.append(i)

            remaining = np.flatnonzero(unvisited)
            if len(remaining) == 0:
                continue

            dists = points[i].separation(points[remaining]).deg 
            
            neighbors = remaining[dists <= radius]
            if len(neighbors):
                unvisited[neighbors] = False
                stack.extend(neighbors.tolist())

        clusters.append(np.array(cluster, dtype=int))

    return clusters

def plot_residuals_for_given_energy(pixel_table, energy_index):
    """ Plot residuals vs model count predictions for a given energy index across all four PSF bands.
    """
    def mdplot(band,ax=None):
        d = band.photons; m = band.diffuse+band.ptsrc+band.sunmoon
        fig, ax = plt.subplots(figsize=(5,5)) if ax is None else (ax.figure, ax)
        ax.scatter(m.clip(1,1e4), ((d-m)/np.sqrt(m)).clip(-5,10), s=2);
        ax.set(xscale='log',yscale='linear',xlabel='model counts/pixel', ylabel=r'residual ($\sigma$ units)', )
        ax.text(1,8, f'{band.psf}\nnside {band.nside}', fontsize=14)
        
    fig, axx = plt.subplots(2,2, figsize=(12,8), sharey=True, sharex=True)
    for i,ax in enumerate(axx.flat):
        mdplot(pixel_table(i,energy_index),ax)
        if i<2: ax.set(xlabel='')
        if i%2==1: ax.set(ylabel='')
        ax.axhline(0, color='0.5', ls='--', lw=2)
    fig.suptitle(pixel_table(0,energy_index).energy, fontsize=16  )
    return fig

def histograms_of_residuals_for_given_energy(pixel_table, energy_index):
    fig, axx = plt.subplots(2,2, figsize=(8,6),sharey=True, sharex=True)
    for i, ax in enumerate(axx.flat):
        pt = pixel_table(i, energy_index)
        ResidualPlotter( pt, ).residual_hist(ax=ax,)
        ax.text(0.05, 0.9, f'PSF{i}', transform=ax.transAxes, fontsize=12, ha='left')
        if i%2>0: ax.set_ylabel('')
        if i<2: ax.set_xlabel('')
    fig.suptitle(f'Residual histograms for {pixel_table(0,energy_index).energy} ', fontsize=18)
    return fig

class ResidualPoints:
    
    def __init__(self, pixel_table, energy_index, sigma_min=5):
        
        self.bands = [pixel_table(i, energy_index) for i in range(4)]
        self.sigma_min = sigma_min

        dff = []
        for ie,band in enumerate(self.bands):


            df = band.get_outliers(self.sigma_min)
            skyc = band.healpix_to_skycoord(df.pixel)
            glon = skyc.galactic.l.deg; glon[glon>180] -= 360
            df['glon'] = glon
            df['glat'] = skyc.galactic.b.deg
            df['psf'] = ie 
            df['nside'] = band.nside
            dff.append(df)
        self.df = pd.concat(dff)
        self.skycoord = SkyCoord(self.df.glon, self.df.glat, unit='deg', frame='galactic')


    def ait_plot(self):
        from utilities.skymaps import AITfigure
        band=self.band
        afig = AITfigure(figsize=(12,6), 
                title=rf"""{band.psf}: {len(self.points)} {band.energy} residuals > {self.sigma_min} $\sigma$""")
        (afig.scatter(self.points, marker='o',s=5*np.sqrt(self.df.data-self.df.model), color='yellow')
        .show()
        )

    def clusterer(self, radius=1.5, ptmin=2):
        r""" Group the $\sigma>5$ residuals into clusters of nearby points
            Cluster separation threshold 1.5 deg. Select those with at least 2 points
        """
        self.cluster_idx = grouper(self.skycoord, radius)
        # select only clusters with more than 1 point
        clgt1 = list(filter(lambda x: len(x)>=ptmin, self.cluster_idx))
        # make a list of the skycoords of the clusters, choosing the point with largest flux as the cluster center
        cld = dict() 
        for idx, cluster in enumerate(clgt1):
            t = self.df.iloc[clgt1[idx]].sort_values('model', ascending=False).iloc[0]
            cld[idx] = dict(glon=round(t.glon,3), glat=round(t.glat,3), n=len(clgt1[idx]), 
                            sigma=round(t.sigma,1), data=t.data.astype(int), model=round(t.model,1), ids=clgt1[idx])  
            
        self.cldf = pd.DataFrame.from_dict(cld, orient='index')['glon glat data model sigma n ids'.split()]

    def zea_plot(self, center, size=5, **kwargs):
        """ Plot the residual clusters in a ZEA projection centered on the given coordinates"""
        from utilities.skymaps import ZEAfigure

        zfig = ZEAfigure(center, size=size, fig=None, figsize=(8,8), title='Residual clusters', frame='galactic')
        zfig.scatter(self.skycoord, s=self.df.sigma*10, c=self.df.sigma, cmap='jet', vmin=5, **kwargs)
        zfig.colorbar(label=r'$\sigma$', shrink=0.7)
        return zfig
    
    def ait_cluster_plot(self, **kwargs):
        """ Plot the residual clusters in an AIT projection"""
        from utilities.skymaps import AITfigure
        self.clpoints = SkyCoord(self.cldf.glon, self.cldf.glat, unit='deg', frame='galactic')
        kw = dict(figsize=(10,10), title=f"Residual clusters with >{self.sigma_min} sigma and >1 point")
        kw.update(kwargs)
        (AITfigure(**kw)
            .scatter(self.clpoints, s=self.cldf.n*20, c=self.cldf.sigma, cmap='jet', vmin=5)
            .colorbar(label=r'$\sigma$', shrink=0.4)
            .show())