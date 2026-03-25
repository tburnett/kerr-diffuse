from astropy.coordinates import SkyCoord
import numpy as np

class SkyDir(object):
    """Compatibility wrapper around `astropy.coordinates.SkyCoord`.

    This class emulates the subset of legacy `skymaps.SkyDir` behavior used by
    this project while storing coordinates internally in ICRS.

    Still unimplemented from the old API: `cross`, `dot`, `isValid`,
    `project`, `zenithCoords`.
    """
    GALACTIC = 'galactic'
    EQUATORIAL = 'icrs'
    
    def __init__(self, *pars, **kwargs):
        """Create a sky direction from spherical or cartesian inputs.

        Supported forms:
        - `(lon_deg, lat_deg)` with optional `frame=...`
        - `(x, y, z)` cartesian values (assumed icrs)
        - `((x, y, z),)` single tuple/list cartesian input
        - `(lon_deg, lat_deg, frame_name)` explicit frame string
        """
        frame = kwargs.pop('frame', 'icrs')
        if len(pars)==2:
            self.coord = SkyCoord(*pars, unit='deg',frame=frame) .icrs
        elif len(pars)==3:
            if isinstance(pars[2], str):
                self.coord = SkyCoord(pars[0],pars[1], unit='deg', frame=pars[2]).icrs
            else:
                self.coord = SkyCoord(*pars, frame='icrs', representation='cartesian')
                self.coord.representation='spherical'
        elif len(pars)==1 and isinstance(pars[0], (tuple, list, np.ndarray)) and len(pars[0])==3:
            x,y,z = pars[0]
            self.coord = SkyCoord(x,y,z,  frame='icrs', representation='cartesian')
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
        """Return cartesian component `i` from the unit direction vector."""
        return self.coord.cartesian.xyz.value[i]
    
    def __str__(self):
        return 'SkyDir({:.3f}, {:.3f})'.format(self.ra(), self.dec())
    def __repr__(self): return self.__str__()

class WeightedSkyDir(SkyDir):
    """`SkyDir` with an attached scalar weight."""
    
    def __init__(self, *pars, **kwargs):
        super().__init__(*pars, **kwargs)
        self._weight=0

    def set_weight(self, weight): self._weight = weight
    def weight(self): return self._weight
    
    def __repr__(self):
        return 'SkyDir({:.3f}, {:.3f}), weight: {}'.format(self.ra(), self.dec(), self._weight)


class WeightedSkyDirList(list):
    """Placeholder container for weighted sky directions loaded from cband data.

    Only referenced from `EnergyBand.load_data` via `BandSet.load_data`.
    Future API planned: `__call__`, `total_pix`, `counts`.
    """
    def init(self, cband, skydir, radius_in_rad, q=False):
        pass
        
