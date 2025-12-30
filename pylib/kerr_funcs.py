import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from astropy_healpix import HEALPix 


class KerrModel(dict):

    class Band(HEALPix):

        def __init__(self, meta):
            self.psf, self.e0, self.e1, nside, self.nocc = meta
            self.counts=0 # 
            ekey = lambda energy:  (np.log10(energy)*4-8).astype(int) #energy bin index

            self.key = (int(self.psf[-1]), ekey(self.e0) ) # key is (psf index, energy index) tuple
            self.energy = f'{np.sqrt(self.e0*self.e1)*1e-3:.2f} GeV'
            super().__init__(nside, frame='galactic', order='nested')

        def __repr__(self) -> str:
            return f"Band{self.key}: {self.psf}@{self.energy} nside {self.nside} occ {self.nocc/(12*self.nside**2):.3f}"
        
        def cone_search(self, center, radius=5.0):
            """ Return mask for photons within radius (deg) of center (SkyCoord)"""            

            sc = self.healpix_to_skycoord(self.pix)
            return sc.separation(center) < Angle(radius, 'deg')

            # slower, more inclusive list
            # cone_pix = hp.cone_search_skycoord(center, radius=Angle(radius, 'deg'))
            # return  np.in1d(self.pix, cone_pix)
        
        def ring_map(self, nside=128, component='data'):
            """Return, for display purposes, a HEALPix RING map of the selected component or combination.
                nside: if set, and less than the Band's, combine
            """
            from astropy_healpix import healpy
            if component=='resid':
                values = self.photons - (self.diffuse + self.ptsrc)
            elif component=='model':
                values = self.diffuse + self.ptsrc
            else:
                values = dict(data=self.photons, diffuse=self.diffuse, ptsrc = self.ptsrc)[component]

            if nside is None or nside>self.nside:
                nside = self.nside
            ratio = (self.nside//nside)**2


            pix = self.pix
            pix = healpy.nest2ring(nside, pix//ratio )

            mp = np.zeros(12*nside**2) # RING sequence of values
            np.add.at(mp, pix, values)
            return mp
        
        def ait_plot(self, component, *, nside=128, figsize=(12,6), fig=None, colorbar=True, 
                     shrink=0.7, **kwargs):
            from utilities.skymaps import AITfigure

            mp = self.ring_map(nside, component=component)
            mp[mp==0] = np.nan

            afig = AITfigure(fig=fig, figsize=figsize, title=f'{component} for {self}')
            afig.imshow(np.log10(mp), cmap='viridis', **kwargs)
            if colorbar:
                afig.colorbar(label='log10(counts)', shrink=shrink)
            return afig   

        def zea_plot(self, component, center, *, nside=256, figsize=(8,8), size=5, fig=None, colorbar=True, **kwargs):
            from utilities.skymaps import ZEAfigure

            mp = self.ring_map(nside, component=component)

            zfig = ZEAfigure(center, size=size, fig=fig, figsize=figsize, title=f'{component} for {self}')
            zfig.imshow(np.log10(mp), cmap='viridis', **kwargs)
            if colorbar:
                zfig.colorbar(label='log10(counts)', shrink=0.7)
            return zfig    
         
        def hist_pix(self):

            fig, ax1 = plt.subplots(1,1, figsize=(6,4), sharex=True)
            nside = self.nside
            npix = 12*nside**2

            ax1.plot(self.pix, )
            ax1.axvline(512+256, ls='--', color='gray', label='boundary');
            ax1.set(ylim=(0,npix), xlim=(0,npix), ylabel='NESTED pixel index', xlabel='pixel sequence')
            ax1.set_title(str(self), fontsize=12)
            plt.show()
        

    def __init__(self, root='from-kerr/toby_v1'):

        import pickle
        filename, meta = root+'.npz', root+'.pickle'
        super().__init__()

        with np.load(filename) as f:
            self.diffuse = f['diffuse']
            self.ptsrc  = f['pointsources']
            self.photons = f['counts']
            self.pix = f['indices']
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
            b.pix     = self.pix[offset:offset+nocc]            
            offset += nocc
            b.totals = dict(diffuse=self.diffuse[-nbands+i], ptsrc=self.ptsrc[-nbands+i],)
        # the total pixel sums
        self.totals = dict(diffuse=self.diffuse[offset:], ptsrc=self.ptsrc[offset:],)
            
        print(f"""Loaded Kerr model from "{filename}":
            {len(self)} bands {self[(0,4)]} ... {self[(3,11)]}
            {self.photons.sum().astype(int):,d} photons""")
        
    def __call__(self, *pars):
        assert len(pars)==2, "Provide psf and energy bin index"
        return self[pars]

        
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

    def residual_hist(self, ax=None):
        from scipy.stats import norm

        fig, ax = plt.subplots(figsize=(4,3)) if ax is None else (ax.figure, ax)
        ylim=np.array([-5,5])
        rnorm = self.rnorm.clip(*ylim)

        nfit = norm.fit(rnorm[~np.isnan(rnorm)])
        ax.hist(rnorm, bins=25, range=ylim, density=True, histtype='stepfilled', alpha=0.5, )
        ax.plot((x:=np.linspace(*ylim,num=25)), norm.pdf(x, *nfit), 'r-', lw=4,
            label =rf'$\mu$={nfit[0]:.2f}'+'\n'+ rf'$\sigma$={nfit[1]:.2f}')
        ax.legend(fontsize=10, loc='lower center')
        ax.set(xlabel='sigma', ylabel='density', yscale='log',xlim=ylim)

    def plots(self):

        from utilities.skymaps import AITfigure

        fig = plt.figure(layout='constrained', figsize=(15,5))
        fig.suptitle(str(self.band), fontsize=18)
        fig1,fig2 = fig.subfigures(ncols=2, wspace=0.07)
        ap = self.band.ait_plot(component='data', nside=self.nside, fig=fig1,)
        ap.title( f'photons / nside {self.nside} pixel', ha='right')

        resid = self.resid 
        model = self.model
    
        afig = AITfigure(fig=fig2, )
        afig.imshow( resid/np.sqrt(model), 
                    cmap='coolwarm',  vmin=-2, vmax=2)#**kwargs)
        afig.colorbar(label='normalized residual', shrink=0.5)
        afig.title( f'residuals', ha='right')
        plt.show()
    
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,4), gridspec_kw={'width_ratios': [2.5, 1]})
        ylim=np.array([-5,5])
        ax1.scatter(model, self.rnorm.clip(*ylim),  s=10, )
        ax1.axhline(0, color='0.5', ls='--', lw=2)
        ax1.set(xlabel='model counts/pixel', ylabel='sigma', xscale='log', 
        ylim = ylim, yscale='linear') 
        ax1.axvline(0, color='0.5', ls='--', lw=2)   

        self.residual_hist(ax=ax2)
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
        rp.residual_hist(ax=ax) 
        ax.set(ylabel='', xlabel='', yticks=[])
    ax.set(ylim=(1e-4, 0.5)) 
    plt.show()