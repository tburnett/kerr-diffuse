""" PSF functions management"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylib.binned_data import BandList
from utilities.ipynb_docgen import show, show_fig


class PSFlist(list):
    """ Manage a list of PSF functions"""

    class PSF(dict):
        """ the PSF fumctor, in degrees, for a band
        Note that it is the density per square degree (180/pi)**2 = 3283 per sr)
        """
        et_name = ['FRONT', 'BACK', 'PSF0', 'PSF1', 'PSF2', 'PSF3',]
        
        def __init__(self, table, which):
            from scipy.interpolate import CubicSpline
            self.which = which # an index from input table

            x = table.x
            y = np.log(table.y)
            # make big linear extrapolation
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
            x = np.append(x, x[-1]+dx)
            y = np.append(y, y[-1]+dy)
            self.spline = CubicSpline(x, y,)# extrapolate=True)
 
            self['r68'] = round(table.r68,3)
            self['energy']=round(table.energy,0)
            self['event_type'] = table.event_type
            self.__dict__.update(self)
            self.max_x = 5*table.r68

        def __call__(self, angle):
            # note clip to avoid strange behavior at large angles
            return np.exp(self.spline(np.asarray(angle)))# .clip(0, self.max_x)))        
               
        def corresponding_sigma(self):
            # return value of corresponding sigma, from curvature at zero
            from findiff import Diff # needs to be imported
            d2df = Diff(0, (dx:=0.001))**2
            x = np.arange(0,0.5,dx)
            psf = self(x)/self(0)
            d = -d2df(psf)
            sigma = 1/np.sqrt(d[0])
            return sigma
        
        def plot_w_gaussian(self, maxr=0.5 ):
            fig,ax1= plt.subplots( figsize=(5,4))
            r = np.arange(0,maxr,1e-2)
            npsf = self(r)/self(0)
            sigma = self.corresponding_sigma()

            ax1.plot(r, npsf, label='PSF') 
            ax1.plot(r, np.exp(-(r/sigma)**2/2), ls=':', label='Gaussian\n'+fr' ($\sigma={sigma:.3f}$)')
            ax1.set(ylabel='Function', yscale='linear', ylim=(0,1), xlim=(0,None),xlabel='radius (deg)')
            ax1.legend(fontsize=12)
            return fig

    def __init__(self, event_type=None, table_path='files/loc/psf_table.pkl'):
        try:
            psf_table = pd.read_pickle(table_path)
        except Exception as msg:
            print(msg, file=sys.stderr)
            return
        for which,table in enumerate(psf_table.itertuples()):
            t = self.PSF(table, which)
            if event_type is None or t.event_type==event_type:
                self.append(t)

    @classmethod
    def example_plot(cls, *, title='',ids=None, default_ids=[0,4,8,10]):
        t = cls()
        plt.figure(figsize=(8,4))
        for i in ids or default_ids:
            plt.semilogy((x:=np.linspace(0,3,100)), t[i](x)/t[i](0), label=f'{i}')
        plt.legend(title='Band index')
        plt.gca().set(xlabel='angle (deg)', ylabel='PSF relative to 0', 
                      ylim=(1e-3,1), xlim=(0,3),)
        plt.title(title)
        plt.show()  

    @classmethod
    def demo_df(cls,):
        """
        Create a DataFrame of PSF functions for each band, to be used in BandList"""
        
        nsides = np.array([  16,   32,   64,  128,  256,  512,  512,  512, 1024, 2048, 2048, 2048])
        plist = cls(event_type=0)[:12]
        df = pd.DataFrame(plist)
        df.drop(['event_type','r68'], axis=1, inplace=True)
        df['psf'] = plist
        df['nside'] = nsides
        return df

    # @classmethod
    # def demo(cls, idx=8):
    #     p = cls()[idx]
    #     show(f"""###  PSF demo: compare with equivalent Gaussian
    #          Select PSF index {idx}: {p} """)
    #     # radial functions
    #     sigma = p.corresponding_sigma()
    #     norm = lambda r: 1/(2*np.pi*sigma**2) * np.exp(-(r/sigma)**2/2) # corresponding Gaussian
    #     psf = lambda r: Like(p).pdf(r).clip(1e-4) # normalized psf values
    #     disk = lambda r: np.where( r<1, 1/(np.pi) , 0) # unit disk radius 1

    #     # Set up grid for integration
    #     R,grid = 1.5,201 # maximum radius, number of bins
    #     x = y = np.linspace(-R,R,grid)
    #     dx = x[1]-x[0]
    #     xx,yy = np.meshgrid(x,y)
    #     rr = np.sqrt(xx**2 + yy**2)#.clip(0,R)

    #     radial_integral = lambda z : np.sum(z) * dx**2

    #     show(f"""Check Integrals: unit disk, {radial_integral(disk(rr)):.3f},
    #         PSF: {radial_integral(psf(rr)):.3f}, 
    #         Norm: {radial_integral(norm(rr)):.3f}""")

    #     show_fig(p.plot_w_gaussian, maxr=4*sigma);   
