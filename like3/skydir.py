from astropy.coordinates import SkyCoord
import numpy as np

class SkyDir(object):
    """ Replacement for SWIGified skymaps.SkyDir using SkyCoord
    wrap a astropy.coordinates.SkyCoord object
    
    May still need to implement: 
     'cross', 'dot', 'isValid', 'project', 'zenithCoords
    """
    GALACTIC = 'galactic'
    EQUATORIAL = 'fk5'
    
    def __init__(self,  *pars, **kwargs ):
        """
        """
        frame = kwargs.pop('frame', 'fk5')
        if len(pars)==2:
            self.coord = SkyCoord(*pars, unit='deg',frame=frame) .fk5
        elif len(pars)==3:
            if type(pars[2])==str:
                self.coord = SkyCoord(pars[0],pars[1], unit='deg', frame=pars[2]).fk5
            else:
                self.coord = SkyCoord(*pars, frame='fk5', representation='cartesian')
                self.coord.representation='spherical'
        elif len(pars)==1 and type(pars)==tuple:
            x,y,z = pars[0]
            self.coord = SkyCoord(x,y,z,  frame='fk5', representation='cartesian')
            self.coord.representation='spherical'
        else:
            raise ValueError('Unrecognized SkyDir pars: {} {}'.format(pars, type(pars)))
            
    def ra(self):   return self.coord.ra.deg
    def dec(self):  return self.coord.dec.deg
    def l(self):    return self.coord.galactic.l.deg
    def b(self):    return self.coord.galactic.b.deg
    def dir(self):  return self.coord.cartesian.xyz.value
    
    def difference(self, other):
        return other.coord.separation(self.coord).rad
    
    def __getitem__(self, i):
        """ return 3-vector components"""
        return self.coord.cartesian.xyz.value[i]
    
    def __str__(self):
        return 'SkyDir({:.3f}, {:.3f})'.format(self.ra(), self.dec())
    def __repr__(self): return self.__str__()

class WeightedSkyDir(SkyDir):
    
    def __init__(self, *pars, **kwargs):
        super(self.__class__, self).__init__(*pars, **kwargs)
        self._weight=0

    def set_weight(self, weight): self._weight = weight
    def weight(self): return self._weight
    
    def __repr__(self):
        return 'SkyDir({:.3f}, {:.3f}), weight: {})'.format(self.ra(), self.dec(),self._weight)


class WeightedSkyDirList(list):
    """ only invoked from bands.py from EnergyBand.load_data, that called from BandSet.load_data.
    
    Wraps data in a "cband"
    to implement: __call__, total_pix, counts
    """
    def init(self, cband, skydir, radius_in_rad, q=False):
        pass
        
