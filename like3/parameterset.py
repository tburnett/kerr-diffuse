"""Parameter-vector views for source-model fitting.

This module exposes two helpers:
- `ParameterSet`: flattened virtual array over all free source parameters.
- `ParSubSet`: masked/subselected view over a `ParameterSet`.

Both classes propagate updates back to source models and mark sources as
`changed` when parameter values are modified.
"""
import numpy as np

class ParameterSet(object):
    """Virtual 1D array over free parameters from a collection of sources.

    Notes
    -----
    If a parameter in a source model is changed, the corresponding source is
    marked dirty by setting `source.changed = True`.
    Parameter values are model-internal coordinates.
    """
    def __init__(self, sources, **kw):
        """Build flattened indexing structures over free source parameters.

        Parameters
        ----------
        sources : iterable
            Collection of source objects with `model.free` masks.
        """
        self.sources = sources
        self.free_sources = [source for source in sources if np.any(source.model.free)]
        # dangerous? self.clear_changed()
        # make two indexing arrays:
        #  ms : list of (source, npar) for each source
        #  index :  (source, index within that source) for each parameter
        self.ms = t = [(source, sum(source.model.free)) for source in self.free_sources]
        ss=[]; ii=[]
        for (s,k) in t:
            for j in range(k):
                ss.append(s) 
                ii.append(j)
        # `index[:, i] -> (source, local_free_parameter_index)`
        self.index = np.array([ss, ii])
        # Global fit mask (all True by default).
        self.mask = np.ones(len(ss),bool)
    
    def __getitem__(self, i):
        """Return parameter `i`, or all parameters for `[:]`."""
        if isinstance(i, slice):
            if i==slice(None,None,None):
                return self.get_parameters()
            else:
                raise Exception('slice format not supported')
        else:
            source, k = self.index[:,i]
        return source.model.get_parameters()[k]
    
    def __setitem__(self,i,x):
        """Set parameter `i` and mark the source as changed when needed."""
        source, k = self.index[:,i]
        model = source.model
        pars = model.get_parameters()
        try:
            if x==pars[k]: return
        except ValueError as ve:
            print(f'ValueError in ParameterSet __setitem__:{ve}: {pars[k]} vs {x}')


        pars[k] = x
        source.changed=True
        model.set_parameters(pars)
    
    def setitems(self, set_dict, quiet=False):
        """Set multiple parameters by index or parameter name.

        Example
        -------
        `{1: 1e-14, 2: 2.1, 'Source_Index': 2.0}`
        """
        def par_index(self, i):
            npar = self.__len__()
            if isinstance(i,int):
                if i<0 or i>=npar:
                    raise Exception('Index, %d, out of range for %d parameters' % (i,npar ) )
                return i
            else:
                try:
                    return list(self.parameter_names).index(i)
                except:
                    raise Exception('Parameter name "%s" not found' % i)
        for key,value in list(set_dict.items()):
            i = par_index(self,key)
            if not quiet: print(key, i, self[i], '-->', value)
            self[i]=value
            
    def __len__(self):
        return self.index.shape[1]
        
    def get_parameters(self):
        """Return the concatenated free-parameter vector."""
        t = [s.model.get_parameters() for s in self.free_sources]
        return np.concatenate(t) if len(t)>0 else []
    
    def set_parameters(self, pars):
        """Set the full free-parameter vector and update dirty flags."""
        # print('Setting parameters:', pars)
        pars = np.atleast_1d(pars)
        i =0
        for source, n in self.ms:
            model = source.model
            oldpars = model.get_parameters()
            newpars = pars[i:i+n]
            if np.any(oldpars != newpars):
                source.model.set_parameters(newpars)
                source.changed=True
            i += n
    
    values = property(fget=get_parameters, fset=set_parameters)
        
    # def set_values(self, pars):
    #     """ set parameters, checking to see if changed"""
    #     self.set_parameters(pars)
        
    def get_covariance(self, nomask=False):
        """Assemble covariance matrix from source-model blocks.

        Parameters
        ----------
        nomask : bool, optional
            If true, return full covariance. Otherwise apply current mask.
        """
        na,nt =len(self.mask), sum(self.mask)
        # deprecated
        # cov = np.matrix( np.zeros(na*na).reshape(na,na))
        cov =  np.zeros(na*na).reshape(na,na)

        i = 0
        for source, n in self.ms:
            model = source.model
            mcov = model.internal_cov_matrix[np.outer(model.free,model.free)]
            cov[i:i+n,i:i+n] = mcov.reshape(n,n)
            i += n
        if nomask: return cov
        return cov[np.outer(self.mask, self.mask)].reshape(nt,nt)
    
    def set_covariance(self, cov):
        """Write covariance values back into source-model covariance blocks."""
        cnow = np.asarray(self.get_covariance(nomask=True)).flatten()
        # print('Setting covariance:', cov, 'mask=', self.mask)
        
        cnow[np.outer(self.mask, self.mask).flatten()] = np.array(cov).flatten()
        na = len(self.mask)
        cnew = cnow.reshape(na,na)
        i = 0
        for source, n in self.ms:
            model = source.model
            model.set_cov_matrix(cnew[i:i+n, i:i+n])
            i += n
    
    @property
    def model_parameters(self):
        """External/model-space parameters for all free sources."""
        if len(self.free_sources)==0: return []
        return np.concatenate([s.model.free_parameters for s in self.free_sources])
    
    @property
    def uncertainties(self):
        """ return relative uncertainties from diagonals of individual covariance matrices 
        """
        variances = np.concatenate([s.model.get_cov_matrix().diagonal()[s.model.free] \
            for s in self.free_sources])[self.mask]
        variances[variances<0]=0
        return np.sqrt(variances) / (np.abs(self.model_parameters) +1e-20) #avoid divide by zero

    @property 
    def bounds(self):
        """Fitter-space bounds for currently free parameters."""
        return np.concatenate([source.model.bounds[source.model.free] for source in self.free_sources])

    def __repr__(self):
        return '%d parameters from %d free sources' % (len(self), len(self.free_sources))
    def clear_changed(self):
        """Reset `changed` flags on all free sources."""
        for s in self.free_sources:
            s.changed=False
    @property
    def dirty(self):
        """Boolean array indicating which free sources are marked changed."""
        return np.array([s.changed for s in self.free_sources])
    @property
    def parameter_names(self):
        """Array of names formatted as `source_parameter`."""
        names = []
        for source in self.free_sources:
            for pname in np.array(source.model.param_names)[source.model.free]:
                names.append(source.name.strip()+'_'+pname)                
        return np.array(names)
        
    def parameter_summary(self, out=None):
        """Print formatted summary of names, values, and relative errors.

        Parameters
        ----------
        out : file-like or None
            Output stream passed to `print`.
        """
        if len(self.parameter_names)==0:
            print('No free parameters')
            return
        print('\n%-21s %8s %8s' % ('parameter', 'value', 'error(%)'), file=out)
        print('%-21s %8s %8s' % ('---------', '-----', '--------'), file=out)
        for u in zip(self.parameter_names, self.get_parameters(), self.uncertainties):
            print('%-21s %8.2f %8.1f' % u, file=out)

    def select_parameters(self, *select):
        """ Select fit parameters by number or name

        Parameters
        ----------
        *select : list of int, str 
            If int, select the parameter by its index.
            If str, select parameter by name or by source name.
                - Start with _: select parameters ending with the string following the _
                - End with *: select parameters containing the string before the *
                - Otherwise, select parameters starting with the source name.

        Returns
        -------
        selected : set
            Set of selected parameter indices.
        """
        selected= set()
        npars = len(self.parameter_names)
    
        # if not hasattr(select, '__iter__') or isinstance(select, (str, bytes)): select = [select]

        for item in select:

            if type(item)==int or type(item)==np.int64:
                selected.add(item)
                if item>=npars:
                    raise Exception('Selected parameter number, %d, not in range [0,%d)' %(item, npars))
            elif type(item)==bytes or type(item)==str: #np.string_:
                if (isinstance(item, bytes) and item.startswith(b'_')) or (isinstance(item, str) and item.startswith('_')):
                    # look for parameters
                    if item[-1] != ('*' if isinstance(item, str) else ord('*')):
                        toadd = [i for i in range(npars) if self.parameter_names[i].endswith(item)]
                    else:
                        def filt(i):
                            return self.parameter_names[i].find(item[:-1])!=-1
                        toadd = list(filter( filt, list(range(npars)) ))
                elif item in self.parameter_names:
                    toadd = [list(self.parameter_names).index(item)]
                    self.selection_description = 'parameter %s' % item
                else:
                    try:
                        src = self.sources.find_source(item)
                    except Exception:
                        raise Exception('fit parameter select list item %s not found as parameter name or source name' %item)
                    self.selection_description = 'source %s'%src.name
                    toadd = [i for i in range(npars) if self.parameter_names[i].startswith(src.name)]
                selected = selected.union(toadd )
            else:
                raise Exception('fit parameter select list item %s, type %s, must be either an integer or a string' %(item, type(item)))
        return selected


class ParSubSet(ParameterSet):
    """Masked/subselected view over a `ParameterSet`.

    Use `mask` directly or call `select(...)` to define the active subset.
    """
    def __init__(self, roimodel, *select, mask=None):
        """Create subset wrapper bound to a ROI model/source list.

        Parameters
        ----------
        roimodel : ROImodel object
        mask : array[bool] or None
        """
        self.roimodel=roimodel
        super(ParSubSet,self).__init__(roimodel)
        self.set_mask(mask)
        self.selection_description = None
        if len(select) > 0:
            self.select(*select,)
        
    def __repr__(self):
        return '%s.%s: subset of %d parameters' % (self.__module__, self.__class__.__name__, sum(self.mask))
    def set_mask(self, m=None):
        """Assign active boolean mask and cached subset indices."""
        if m is None:
            self._mask = np.ones(len(self),bool)
        else: 
            assert len(m)==len(self)
            assert sum(m)>0
            self._mask = m
        self.subsetindex = np.arange(len(self))[self._mask]
    def get_mask(self): return self._mask        
    mask = property(get_mask, set_mask) 
    
    def select(self, *select, ):
        """
        Parameters
        ----------
        select : None, item or list of items, where item is an int or a string
            if not None, it defines a subset of the parameter numbers to select
                    to define a projected function to fit
            int:  select the corresponding parameter number
            string: select parameters according to matching rules
                    The name of a source (with possible wild cards) to select for fitting
                    If initial character is '_', match the rest with parameter names
                    if initial character is not '_' and last character is '*', treat as wild card
            
        exclude : None, int, or list of int 
                if specified, will remove parameter numbers from selection
        """

        # select a list of parameter numbers, or None for all free parameters
        selected = self.select_parameters(*select)
        
        t = np.zeros(len(self.parameter_names), bool)
        t[list(selected)]=True
        self.set_mask( t )
        if self.selection_description is None:
            self.selection_description = 'parameters %s' % selected
        
    def __getitem__(self, i):
        """Return subset parameter `i` mapped to parent index."""
        return super(ParSubSet,self).__getitem__(self.subsetindex[i])
    
    def __setitem__(self,i,x):
        super(ParSubSet,self).__setitem__(self.subsetindex[i], x)
    def get_parameters(self):
        t = super(ParSubSet, self).get_parameters()
        return t[self._mask]
    def set_parameters(self, pars):
        t = super(ParSubSet, self).get_parameters()
        t[self._mask]=pars
        super(ParSubSet, self).set_parameters(t)    

    values = property(fget=get_parameters, fset=set_parameters)
    
    @property
    def spectral_model(self):
        """ access to the spectral model for the first parameter"""
        return self.index[0, self.subsetindex[0]].model
    @property
    def source(self):
        """ access to the source associated with the first parameter"""
        return self.index[0, self.subsetindex[0]]
        
    def get_model(self,i):
        """Spectral model object owning subset parameter `i`."""
        return self.index[0, self.subsetindex[i]].model
        
    @property
    def parameter_names(self):
        t = super(ParSubSet, self).parameter_names
        return t[self._mask]
    @property
    def bounds(self):
        t = super(ParSubSet, self).bounds
        return t[self.mask]
    @property
    def model_parameters(self):
        t = super(ParSubSet, self).model_parameters
        return t[self.mask]

