"""Source-list management utilities for like3 ROI-style models.

`SourceList` is a list-like container with helpers for:
- aggregating flux and gradients across sources,
- managing free-parameter views via `parameterset`,
- finding, adding, deleting, and updating source models.
"""

import numpy as np
import pandas as pd

from . import (sources, parameterset)


class SourceModelException(Exception):
    """Raised for source-lookup and source-list management errors."""

class SourceModel(list):
    """List-like container for model sources plus parameter-management helpers.

    Notes
    -----
    Parameter serialization/subselection is delegated to
    `parameterset.ParameterSet` and `parameterset.ParSubSet`.
    """

    def __init__(self, sources):
        """Initialize from an iterable of source objects."""

        for source in sources:
            self.add_source(source)
        
        self.initialize()

        # print(self.__repr__())
        self.selected_source = None
        self.selected_source_index = -1

    def __repr__(self):
        return f'SourceModel: {len(self)} sources with {len(self.parameters)} free parameters'

    def model_counts(self, band, pix):
        """Return predicted counts for pixels in one energy band.

        Parameters
        ----------
        band : object
            Band descriptor providing:
            - ``energy`` for model evaluation,
            - ``response(source, pix)`` returning per-pixel PSF weights,
            - ``exposure_map(pix)`` returning per-pixel exposure.
        pix : array-like
            Pixel indices (or IDs) for which counts are requested.

        Returns
        -------
        numpy.ndarray
            Predicted counts per input pixel, summed over all sources.
        """
        pixel_flux = np.zeros_like(pix)
        for source in self:
            source_flux = source.model(band.energy)
            _, source_pixel_psf = band.response(source, pix)
            pixel_flux += source_pixel_psf * source_flux
        # finally multiply by exposure to get counts 
        return pixel_flux * band.exposure_map(pix)
        
    def flux(self, energies):
        """Return summed model flux over all sources at `energies`."""
                
        r = np.zeros_like(energies)
        for source in self:
            r += source.model(energies)
        return r
    
    def gradient(self, energies):
        """Return flux gradient with respect to active free parameters."""
        energies = np.atleast_1d(energies)           
        g = np.vstack([source.model.gradient(energies)[source.model.free]*1.0
                          for source in self])
        return g [ self.parameters.mask]
       
    def initialize(self, **kw):
        """Rebuild flattened parameter view after any source/model changes."""
        self.parameters = parameterset.ParameterSet(self, **kw)
  
    def parsubset(self, *select):
        """Return a `ParSubSet` view with optional initial selection."""
        return parameterset.ParSubSet(self, *select)
        
    
    # note that the following properties are dynamic, in case sources or their models change interactively
    @property
    def source_names(self): return np.array([s.name for s in self])
    @property
    def models(self): return np.array([s.model for s in self])
    @property
    def free(self): 
        """Mask selecting sources with at least one free model parameter."""
        return np.array([ np.any(s.model.free) for s in self])
    
    @property 
    def bounds(self):
        """Concatenated fitter-space bounds for all free parameters."""
        return np.concatenate([m.bounds[m.free] for m in self.models])
    
    @property
    def parameter_names(self):
        """Array of free parameter names formatted as `source_parameter`."""
        names = []
        for source_name, model in zip(self.source_names, self.models):
            for pname in np.array(model.param_names)[np.array(model.free)]: # mod for future warning
                names.append(source_name.strip()+'_'+pname)
        return np.array(names)
    
    def find_source(self, source_name):
        """ Search for the source with the given name
        
        source_name : [string | None | sources.Source instance ]
            if the first or last character is '*', perform a wild card search, return first match
            if None, and a source has been selected, return it
            if an instance, and in the list, just select it and return it
        """
        if source_name is None:
            if self.selected_source is None:
                raise SourceModelException('No source is selected')
            return self.selected_source
        elif isinstance(source_name, sources.Source):
            if source_name in self:
                self.selected_source = source_name
                return self.selected_source
            self.selected_source = None
            raise SourceModelException('source %s not found' % source_name.name)
            
        names = [s.name for s in self]
        def not_found():
            self.selected_source_index =-1
            raise SourceModelException('source %s not found' %source_name)
        def found(s):
            self.selected_source=s
            self.selected_source_index = names.index(s.name)
            return s
        if isinstance(source_name, str) and len(source_name) > 0 and source_name[-1]=='*':
            for name in names:
                if name.startswith(source_name[:-1]): 
                    return found(self[names.index(name)])
            not_found()
        if isinstance(source_name, str) and len(source_name) > 0 and source_name[0]=='*':
            for name in names:
                if name.endswith(source_name[1:]): 
                    return found(self[names.index(name)])
            not_found()
        try:
            k = names.index(source_name)
            self.selected_source = self[k]
            #if self.selected_source is None or self.selected_source != selected_source:
            #    print 'selected source %s for analysis' % selected_source.name
            return found(self.selected_source)
        except:
            self.selected_source = None
            not_found()

    def add_source(self, newsource=None, **kw):
        """Add a source to the model and rebuild parameter indexing.
        
        parameters
        ----------
        newsource : Source object or None
            if None, expect source to be defined as a PointSource by the keywords
            
        keywords:
            name : string
            model : uw.like.Models object
            skydir : skymaps.SkyDir object | (ra,dec) 
        """
        if newsource is not None:
            assert isinstance(newsource, sources.Source)
        else:
            newsource = sources.PointSource(**kw)
            
        if len(self.source_names)>0 and newsource.name in self.source_names:
            print('Attempt to add source "{}: a source with that name already exists'.format(
                 newsource.name))
            return None
        self.append(newsource)
        self.initialize()
        return newsource
     
    def del_source(self, source_name):
        """Remove a source from the model and rebuild parameter indexing."""
        source = self.find_source(source_name) # first get it
        self.remove(source)
        self.initialize()
        return source
        
    def set_model(self, model, source_name=None):
        """Replace selected source model and return `(source, old_model)`.
        
        model : string, or like.Models.Model object
            if string, evaluate. Note that 'PowerLaw(1e-11,2.0)' will work. Also supported:
            ExpCutoff, PLSuperExpCutoff, LogParabola, each with all parameters required.
        source_name: None or string
            if None, use currently selected source
        """
        src = self.find_source(source_name)
        if src is None:
            raise SourceModelException(f'source {source_name} not found')
        old_model = src.model
        if isinstance(model, str):
            model = eval(model) 
        assert sources.ismodel(model), 'model must inherit from Model class'
        src.model = model
        src.changed = True
        sources.set_default_bounds(model)
        self.initialize()
        return src, old_model
        
    def list_sources(self):
        """Print all sources currently in the model."""
        for source in self:
            print(source)
        return
    
    
    def setposition(self, skydir):
        """Set the position of the selected source.
        Return self to allow chaining, e.g. ``with sl.localization_view('Blazar').setposition((ra,dec)) as view:``"""
        self.selected_source.skydir = skydir
   
    
    @classmethod
    def demo(cls,  src_key=2,) :
        """Create a toy source list for quick experiments.

            0 : PLSuperExpCutoff source
            1 : PowerLaw source
            2 : both sources
        """ 
        ps = sources.PointSource(name='Pulsar',  skydir=(0,0), frame='galactic',
                        model=sources.PLSuperExpCutoff4(1e-11, 2., 0.7, 0.69),)  
        
        pl = sources.PointSource(name='Blazar',skydir=(5,0), frame='galactic',
                        model=sources.LogParabola(4e-12, 2, 0, 1e3))
        
        pp = []
        if src_key==0:
            pp = [ps]
        elif src_key==1:
            pp = [pl]
        else:
            pp = [ps, pl]
        model = cls(pp)

        print(f'Model: {str(model)}')
        return model

    def view(self):
        """Return a context-manager view that restores all source model state on exit.

        Within the ``with`` block the caller receives the original ``SourceList``
        and may freely add, remove, or mutate sources and their models.  When the
        block exits (whether normally or via an exception) every source and its
        spectral model are rolled back to the state captured at block entry.

        Usage
        -----
        .. code-block:: python

            with source_list.view() as sl:
                sl.find_source('Blazar').model.free[0] = False
                # ... trial computation ...
            # source_list is fully restored here

        Returns
        -------
        SourceListView
        """
        ret = SourceListView(self)
        # print('Entering SourceList view context: snapshot taken, original list is available as "sl"', ret.__class__)
        return ret
    
    def localization_context(self, source_name=None):
        """Set up context-manager that restores only the position of the selected source on exit.

        Within the ``with`` block the caller receives the original ``SourceList``
        and may freely add, remove, or mutate sources and their models.  When the
        block exits (whether normally or via an exception) the position of the selected source is rolled back to the state captured at block entry.

        Usage
        -----
        .. code-block:: python

            with source_list.localization_view('Blazar') as sl:
                sl.find_source('Blazar').model.free[0] = False
                # ... trial computation ...
            # source_list is fully restored here

        Returns
        -------
        SourceListView
        """
        if self.selected_source is None or self.selected_source.name != source_name:
            self.find_source(source_name)
        if self.selected_source is None:
            raise SourceListException(f'source {source_name} not found')
        return SourceListContext(self,  position_only=True)


class SourceListContext:
    """Context-manager snapshot/restore wrapper for a :class:`SourceList`.

    On ``__enter__`` a lightweight snapshot of the source-list contents and
    every source's spectral-model state is captured.  On ``__exit__`` the
    original list is rewound to that snapshot, regardless of whether the block
    raised an exception.

    Do not instantiate directly; use :meth:`SourceList.view` instead.
    """

    def __init__(self, source_list,  position_only=False):
        """Bind to *source_list*; snapshot is taken lazily on ``__enter__``."""
        self._sl = source_list
        self._snapshot = None
        self._source_name = source_list.selected_source.name if source_list.selected_source else None
        self._position_only = position_only
        
        
    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _take_snapshot(self):
        """Capture the current source-list state.

        Stores which sources are present and a ``model.copy()`` for each so
        that parameter values, free masks, and bounds can all be restored.
        """
        if self._position_only:
            return self._sl.selected_source.skydir
        return {
            'sources': list(self._sl),  # ordered list of source objects
            'models': {src.name: src.model.copy() for src in self._sl},
            'selected_source': self._sl.selected_source,
            'selected_source_index': self._sl.selected_source_index,
        }

    def _restore(self, snapshot):
        """Rewind *self._sl* to the given snapshot."""
        if self._position_only:
            self._sl.selected_source.skydir = snapshot
            return
        
        # Restore the membership list (handles add_source / del_source calls).
        del self._sl[:]
        self._sl.extend(snapshot['sources'])

        # Restore each source's model to the saved copy.
        for src in self._sl:
            src.model = snapshot['models'][src.name]
            src.changed = True

        # Restore selection bookkeeping.
        self._sl.selected_source = snapshot['selected_source']
        self._sl.selected_source_index = snapshot['selected_source_index']

        # Rebuild the flattened parameter view.
        self._sl.initialize()

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        """Capture a snapshot and return the original ``SourceList``."""
        self._snapshot = self._take_snapshot()
        return self._sl

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the ``SourceList`` to its pre-block state.

        Always restores, even when the block raised an exception.
        Never suppresses exceptions (returns ``False``).
        """
        self._restore(self._snapshot)
        return False  # propagate any exception unchanged





