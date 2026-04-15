"""Source abstractions and spectral-model adapters for like3.

This module defines:
- convenience constructors for common spectral model shapes,
- `Source` base class with model conversion and bounds setup,
- concrete `PointSource`, `ExtendedSource`, and `GlobalSource` variants.
"""
import numpy as np
from typing import Any, cast
# from . skydir import SkyDir
from astropy.coordinates import SkyCoord#, Angle
from . import spectral_models


class _BandResponseAdapter:
    """Compatibility adapter exposing ``evaluate`` for legacy call sites."""

    def __init__(self, source, band):
        self.source = source
        self.band = band

    def evaluate(self, pixels=None):
        return self.band.response(self.source, pixels)

# convenience adapters 
def LogParabola(*pars):
    """Create a LogParabola model with default free-mask settings."""
    model = spectral_models.LogParabola(p=pars, free=[True,True,False,False])
    return model
def PowerLaw(*pars):   
    """Create a PowerLaw model."""
    model = spectral_models.PowerLaw(p=pars)
    return model
def ExpCutoff(*pars):  
    """Create an ExpCutoff model with default free-mask settings."""
    model = spectral_models.ExpCutoff(p=pars, free=[True, True, False])
    return model
def PLSuperExpCutoff(*pars):  
    """Create a PLSuperExpCutoff model with default free-mask settings."""
    model = spectral_models.PLSuperExpCutoff(p=pars, free=[True,True,False,False])
    return model
def PLSuperExpCutoff4(*pars):  
    """Create a PLSuperExpCutoff4 model with default free-mask settings."""
    model = spectral_models.PLSuperExpCutoff4(p=pars, free=[True,True,False,False])
    return model

def Constant(*pars, **kw):
    """Create a Constant model."""
    return spectral_models.Constant(p=pars, **kw)

def FBconstant(f,b, **kw):
    """Create a FrontBackConstant model."""
    return spectral_models.FrontBackConstant(f,b, **kw)

def PSR_default():
    """Create default pulsar-like PLSuperExpCutoff4 configuration."""
    return spectral_models.PLSuperExpCutoff4(free=[True,True,True,False])
    
def ismodel(model):
    """ check that model is an instance of Models.Model"""
    return isinstance(model, spectral_models.Model)

def set_default_bounds( model, force=False):
    """
    Handy utility to set bounds for a model from like.Models
    force=True to override previously set bounds.
    """
    if not force and hasattr(model, 'bounds'):
        # model has bounds. Were they set? check to see if all are None
        notset =  np.all(np.array([np.all(b ==[None,None]) for b in model.bounds]))
        if not notset: return
    bounds=[]
    def to_internal(fun, values):
        return [fun(value) if value is not None else None for value in values]
    
    for pname, mp in zip(model.param_names, model.mappers):
        lim = model.default_limits.get(pname, None)
        if lim is not None:
            # print(f'Parameter: {pname}, limits: {lim}')
            bounds.append( to_internal(mp.tointernal, (lim.lower, lim.upper)) )
        else:
            bounds.append( (None, None) )

    # Convert to ndarray so the free-parameter mask can index bounds directly.
    model.bounds = np.array(bounds)


class Source(object):
    """Base class for all source types used by like3.

    Subclasses must implement `response(band, ...)` and provide source-type-
    specific behavior (point, extended, global/diffuse).
    """
    def __init__(self, **kwargs):
        """Initialize source metadata and normalize model representation."""
        self.__dict__.update(kwargs)
        self.changed = False # flag for bandlike
        assert self.name is not None, 'bad source name'
        self.name = str(self.name) # force to be a string
        if self.skydir is None:
            # Global source: keep original model setup and exit early.
            self.free = np.array(self.model.free).copy() if self.model is not None else None  # save copy of initial free array to restore
            return
        elif isinstance(self.skydir, SkyCoord):
            pass # already a SkyCoord, nothing to do
        elif hasattr(self.skydir, '__iter__'): #allow a tuple of (ra,dec)
            self.skydir = SkyCoord(*cast(tuple, self.skydir), unit='deg', frame=kwargs.get('frame', 'icrs'))
        if 'model' not in kwargs or self.model is None:
            self.model=LogParabola(1e-14, 2.2, 0, 1e3)
            self.model.free[2:]=False
        elif type(self.model)==str:
            try:
                t =eval(self.model)
            except Exception as exp:
                print('Failed to evaluate model expression, %s: %s' %(self.model, exp))
                raise
            self.model=t
                
        if self.model.name=='PowerLaw':
            # convert from PowerLaw to LogParabola
            stats = self.model.statistical()
            par, sig = stats[:2]
            free = self.model.free[:]
            self.model = LogParabola(*(list(par)+[0, self.model.e0]))
            self.model.free[:2] = free
            self.model.free[2:] = False
 
        elif self.model.name=='ExpCutoff':
            try:
                print('converting %s to PLSuperExpCutoff' %self.name)
                self.model = cast(spectral_models.ExpCutoff, self.model).create_super_cutoff()
            except FloatingPointError:
                print('Failed')
                
        elif self.model.name=='PowerLawFlux':
            f, gamma = self.model.get_all_parameters() #10**self.model.p
            emin = cast(spectral_models.PowerLawFlux, self.model).emin  # type: ignore[attr-defined]
            try:
                self.model=LogParabola(f*(gamma-1)/emin, gamma, 0, emin)
            except Exception as msg:
                print('Failed to create LogParabola for source %s, pars= %s'% (self.name, (f,gamma,emin)))
                raise
            self.model.free[2:]=False
        elif self.model.name=='LogParabola':
            #what was this for?
            #if hasattr(self, 'free') and len(self.free)>3: self.free[3]=False
            if sum(self.model.free)==4:
                # do not allow all parameters to be free: freeze E_break if so
                self.model.free[-1]=False
            elif sum(self.model.free)==2 and not self.model.free[1]:
                # undo freezing
                print('Unfreezing E_break for source %s' % self.name)
                self.model.free[-1]=True
        if self.model.name not in ['LogParabola','PLSuperExpCutoff','ExpCutoff', 'Constant','PLSuperExpCutoff4']:
            raise Exception('model %s not supported' % self.model.name)
        #self.free = self.model.free.copy()

        if not hasattr(self.model, 'npar'):
            raise Exception('model %s for source %s was not converted to new format'\
                    % (self.model.name, self.name))
        # finally, add bounds to the models object, ignoring similar capability in Models.
        set_default_bounds( self.model )
           
            
    def get_spectral_model(self):
        return self.model
    def set_spectral_model(self, newmodel):
        self.model = newmodel
        self.changed = True
    spectral_model = property(get_spectral_model, set_spectral_model)

    def freeze(self, parname, value=None):
        """Freeze one model parameter and optionally force its value."""
        self.model.freeze(parname)
        if value is not None: self.model.setp(parname, value)
        self.changed=True
        #assert sum(self.model.free)>0, 'cannot freeze all parameters this way'

    def thaw(self, parname):
        """Unfreeze one model parameter."""
        self.model.freeze(parname, freeze=False)
        self.changed = True

    def __str__(self):
        sdir = 'None' if self.skydir is None\
                    else f'({self.skydir.icrs.ra.deg:07.3f}, {self.skydir.icrs.dec.deg:+05.3f})'  # type: ignore[union-attr]
        return '\tname  : %s\n\ticrs  : %s\n\tmodel : %s\n\t\t%s' %\
    (self.name, sdir, self.model.name, self.model.__str__(indent='\t\t'))
    
    def __repr__(self):
        return '%s.%s: \n%s' % (self.__module__,self.__class__.__name__ , self.__str__())
        
    @property
    def e_ref(self):
        """Convenience property to access model reference energy, if it exists."""
        return self.model.e0 if hasattr(self.model, 'e0') else None

    @property
    def isextended(self):
        return hasattr(self, 'dmodel') and not self.isglobal

    @property
    def isglobal(self):
        return self.skydir is None

    def sed_plot(self, ax=None, title=None, label=None, emin=100, emax=1e5, npts=50):
        """Plot the SED (E² dN/dE vs E) for this source.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to draw into. A new figure is created when ``None``.
        title : str or None
            Axes title. Defaults to the source name.
        label : str or None
            Legend label. Defaults to the source name.
        emin, emax : float
            Energy range in MeV.
        npts : int
            Number of logarithmically-spaced evaluation points.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        model = self.model
        energies = np.logspace(np.log10(emin), np.log10(emax), npts)  # MeV
        dnde = model(energies)                                          # ph cm⁻² s⁻¹ MeV⁻¹
        e2dnde = energies**2 * dnde                                     # MeV cm⁻² s⁻¹

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        ax.loglog(energies, e2dnde, label=self.name.strip() if label is None else label)

        if model.has_errors():
            g = model.external_gradient(energies)   # shape (npar, npts)
            cov = model.get_cov_matrix()             # shape (npar, npar)
            var_dnde = np.sum((cov @ g) * g, axis=0)
            var_dnde = np.clip(var_dnde, 0, None)
            sigma_e2dnde = energies**2 * np.sqrt(var_dnde)
            ax.fill_between(energies, e2dnde - sigma_e2dnde,
                            e2dnde + sigma_e2dnde, alpha=0.3)

        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel(r'$E^2\,dN/dE\ [\mathrm{MeV\,cm^{-2}\,s^{-1}}]$')
        ax.set_title(self.name.strip() if title is None else title)
        ax.legend()
        return ax


class PointSource(Source):
    """Point-like source with PSF-based response construction."""
    skydir: SkyCoord

    def __init__(self, **kwargs):
        kwargs.update(spatial_model=None) # allow test for extent (no extent!)
        super(PointSource, self).__init__(**kwargs)

    def near(self, otherdir, distance=10):
        """Return True if separation from `otherdir` is below `distance` (deg)."""
        return float(self.skydir.separation(otherdir).deg) < distance  # type: ignore[arg-type]

    def copy(self, **kwargs):
        """ return a new PointSource object, with a copy of the model, others"""
        kw = self.__dict__.copy()
        kw.update(kwargs)
        ret = PointSource(**kw)
        ret.model = self.model.copy()
        return ret

    def response(self, band,  ):
        """Return an adapter providing ``evaluate(pixels)`` response calls."""
        return _BandResponseAdapter(self, band)


class ExtendedSource(Source):
    """Extended source with spatial model (`dmodel`) and convolved response."""
    skydir: SkyCoord
    dmodel: Any  # spatial model, set via kwargs

    #def __str__(self):
    #    return self.name + ' '+ self.model.name \
    #            +  (' (free)' if np.any(self.model.free) else ' (fixed)') 
    def __str__(self):
        return '\tname  : %s\n\tskydir: %s\n\tSpatial : %s\n\tmodel : %s\n\t\t%s' %\
    (self.name, self.skydir, self.dmodel.name, self.model.name, self.model.__str__(indent='\t\t'))
 
  
    def near(self, otherdir, distance=10):
        """Return True if separation from `otherdir` is below `distance` (deg)."""
        return float(self.skydir.separation(otherdir).deg) < distance  # type: ignore[arg-type]
        
    def copy(self):
        """ return a new ExtendSource object, with a copy of the model object"""
        ret = ExtendedSource(**self.__dict__)
        ret.model = self.model.copy()
        if ret.model.name=='LogParabola':
            ret.model.free[-1]=False # make sure Ebreak is frozen
        return ret
         
    def response(self, band, roi=None, **kwargs):
        """Return an adapter providing ``evaluate(pixels)`` response calls."""
        return _BandResponseAdapter(self, band)

        
