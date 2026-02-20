"""
Set up and manage the model for all the sources in an ROI

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/like2/roimodel.py,v 1.29 2018/01/27 15:37:17 burnett Exp $

"""

import numpy as np
import pandas as pd

from . import (sources, parameterset) 
class SourceListException(Exception):pass        

class SourceList(list):
    """ Manage the list of free sources, Note that it inherits from list.
    
    In particular, provide an interface to serialize the set of free parameters, or define a subset thereof.
    This is delegated to the classes ParameterSet and ParSubSet in the module parameterset
    
    Methods are provided to add or remove sources, and change the model associated with a source.
    """

    def __init__(self, sources):

        for source in sources:
            self.add_source(source)
        
        self.initialize()

        # print(self.__repr__())
        self.selected_source = None

    def __repr__(self):
        ns = len(self)
        n_free = len(self.parameters)
        return f'{len(self)} sources, {len(self.parameters)} free parameters'

    def flux(self, energies):
        """ Compute flux values for given parameter set """
                
        r = np.zeros_like(energies)
        for source in self:
            r += source.model(energies)
        return r
    
    def gradient(self, energies):
        """ Return the gradient of the flux with respect to free parameters
        """    
        energies = np.atleast_1d(energies)           
        g = np.vstack([source.model.gradient(energies)[source.model.free]*1.0
                          for source in self])
        return g [ self.parameters.mask]
       

    def initialize(self, **kw):
        """For fast parameter access: must be called if any source changes
        """
        self.parameters = parameterset.ParameterSet(self, **kw)
  
    def parsubset(self, *select):
        """ return a ParSubSet object with possible initial selection of a subset of the parameters
        """
        return parameterset.ParSubSet(self, *select)
        
    
    # note that the following properties are dynamic, in case sources or their models change interactively
    @property
    def source_names(self): return np.array([s.name for s in self])
    @property
    def models(self): return np.array([s.model for s in self])
    @property
    def free(self): 
        """ mask which defines variable sources: all global and local sources with at least one variable parameter 
        """
        return np.array([ np.any(s.model.free) for s in self])
    
    @property 
    def bounds(self):
        """ fitter representation of applied bounds """
        return np.concatenate([m.bounds[m.free] for m in self.models])
    
        
    @property
    def parameter_names(self):
        """ array of free parameter names """
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
                raise SourceListException('No source is selected')
            return self.selected_source
        elif isinstance(source_name, sources.Source):
            if source_name in self.sources:
                self.selected_source = source_name
                return self.selected_source
            not_found()
            
        names = [s.name for s in self]
        def not_found():
            self.selected_source_index =-1
            raise SourceListException('source %s not found' %source_name)
        def found(s):
            self.selected_source=s
            self.selected_source_index = names.index(s.name)
            return s
        if source_name[-1]=='*':
            for name in names:
                if name.startswith(source_name[:-1]): 
                    return found(self[names.index(name)])
            not_found()
        if source_name[0]=='*':
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
        """ add a source to the ROI
        
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
        """ remove a source from the model for this ROI
        """
        source = self.find_source(source_name) # first get it
        self.remove(source)
        self.initialize()
        return source
        
    def set_model(self, model, source_name=None):
        """ replace the current model, return reference to previous
        
        model : string, or like.Models.Model object
            if string, evaluate. Note that 'PowerLaw(1e-11,2.0)' will work. Also supported:
            ExpCutoff, PLSuperExpCutoff, LogParabola, each with all parameters required.
        source_name: None or string
            if None, use currently selected source
        """
        src = self.find_source(source_name)
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
        """ print the list of sources in the model, with indication of which are free
        """
        for source in self:
            print(source)
        return
    
    @classmethod
    def demo(cls,  src_key=2,) :
        """ Create a simple model with one (or two) point sources 
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





