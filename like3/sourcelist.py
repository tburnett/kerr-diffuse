"""Source-list management utilities for like3 ROI-style models.

`SourceModel` is a list-like container with helpers for:
- aggregating flux and gradients across sources,
- managing free-parameter views via `parameterset`,
- finding, adding, deleting, and updating source models.
"""

import numpy as np
import pandas as pd
import re

from . import (sources, parameterset)
from utilities.catalogs import Fermi4FGL


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

        self.fermi_catalog = None
        self.name=''
        for source in sources:
            self.add_source(source)
        
        self.initialize()

        if len(self) > 0:
            self.selected_source = self[0]
            self.selected_source_index = 0
        else:
            self.selected_source = None
            self.selected_source_index = -1

    def clear(self):
            """Remove all sources from the model and reset selection."""
            super().clear()
            self.selected_source = None
            self.selected_source_index = -1
            self.initialize()

    def __repr__(self):
        return f'SourceModel: @{self.name} {len(self)} sources with {len(self.parameters)} free parameters'

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
       
    def reinitialize(self, **kw):
        """Rebuild parameter mappings from current sources.

        This reconstructs the flattened free-parameter bookkeeping used by
        fitting and parameter selection, and should be called after any
        source-model change that affects free parameters.
        """
        self.parameters = parameterset.ParameterSet(self, **kw)

    def initialize(self, **kw):
        """Backward-compatible alias for :meth:`reinitialize`."""
        self.reinitialize(**kw)
  
    def parsubset(self, *select):
        """Return a `ParSubSet` view with optional initial selection."""
        return parameterset.ParSubSet(self, *select)

    def summarize(self, out=None):
        """Print a summary of free parameters and their current values.

        If ``self.fit_info`` is present (set after fitting), also prints the
        log-likelihood and fit quality from that record.  Otherwise falls back
        to :meth:`parameterset.ParameterSet.parameter_summary`.

        Parameters
        ----------
        out : file-like or None
            Output stream passed to ``print``.
        """
        fit_info = getattr(self, 'fit_info', None)
        if fit_info is not None:
            print(f'loglike={fit_info.get("loglike", float("nan")):.3f}  '
                  f'qual={fit_info.get("qual", float("nan")):.3f}', file=out)
        self.parameters.parameter_summary(out=out)

    def sed_plot(self, source_name=None, ax=None, title=None, label=None, emin=100, emax=1e5, npts=50, ylim=(0.1, None)):
        """Plot the SED (E² dN/dE vs E) for the selected or named source.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector passed to :meth:`find_source`. Defaults to the
            currently selected source.
        ax : matplotlib.axes.Axes or None
            Axes to draw into. A new figure is created when ``None``.
        emin, emax : float
            Energy range in MeV.
        npts : int
            Number of logarithmically-spaced evaluation points.
        ylim : tuple[float | None, float | None]
            Y-axis limits in eV cm^-2 s^-1. Defaults to ``(0.1, None)``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return self.find_source(source_name).sed_plot(
            ax=ax, title=title, label=label, emin=emin, emax=emax, npts=npts, ylim=ylim, butterfly=True
        )


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

    @staticmethod
    def _canonical_source_name(name):
        """Return a normalized source-name token for alias-friendly matching.

        Normalization is case-insensitive, removes punctuation/whitespace,
        and maps common Markarian aliases (``mrk``/``markarian``) to ``mkn``.
        """
        text = str(name).strip().lower()
        text = re.sub(r'[^a-z0-9]+', '', text)
        if text.startswith('markarian'):
            text = 'mkn' + text[len('markarian'):]
        elif text.startswith('mrk'):
            text = 'mkn' + text[len('mrk'):]
        return text
    
    def find_source(self, source_name=None):
        """Return and select a source by name or object.

        Parameters
        ----------
        source_name : str, Source, or None
            Selector for the desired source.

            - ``None``: return the currently selected source.
            - ``Source`` instance: validate membership and select it.
            - ``str``: exact name match, or wildcard prefix/suffix match
              when the string begins or ends with ``*``.

              Examples: ``"Blazar"``, ``"Bla*"``, ``"*zar"``.

        Returns
        -------
        Source
            The matched source. On success, updates ``selected_source`` and
            ``selected_source_index``.

        Raises
        ------
        SourceModelException
            If no source is selected when ``source_name`` is ``None``, or if
            no matching source is found.
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
            self.selected_source_index = -1
            raise SourceModelException('source %s not found' % source_name)
        def found(s):
            self.selected_source = s
            self.selected_source_index = names.index(s.name)
            return s
        if isinstance(source_name, str) and source_name.endswith('*') and not source_name.startswith('*'):
            prefix = source_name[:-1]
            for name in names:
                if name.startswith(prefix):
                    return found(self[names.index(name)])
            not_found()
        if isinstance(source_name, str) and source_name.startswith('*') and not source_name.endswith('*'):
            suffix = source_name[1:]
            for name in names:
                if name.endswith(suffix):
                    return found(self[names.index(name)])
            not_found()
        try:
            k = names.index(source_name)
            return found(self[k])
        except ValueError:
            pass

        # Alias-friendly fallback: canonicalized string matching.
        if isinstance(source_name, str):
            target = self._canonical_source_name(source_name)
            canonical_names = [self._canonical_source_name(name) for name in names]
            matches = [i for i, cname in enumerate(canonical_names) if cname == target]
            if len(matches) > 0:
                return found(self[matches[0]])

            # Secondary alias match against per-source catalog aliases.
            alias_matches = []
            for i, src in enumerate(self):
                alias_values = [src.name]
                alias_values.extend(getattr(src, 'aliases', []))
                canonical_aliases = {self._canonical_source_name(value) for value in alias_values}
                if target in canonical_aliases:
                    alias_matches.append(i)
            if len(alias_matches) > 0:
                return found(self[alias_matches[0]])

            # Final fallback: resolve the requested string to coordinates and
            # choose the nearest source in the current source list.
            try:
                from astropy.coordinates import SkyCoord

                target_coord = self._coerce_catalog_skycoord(source_name, frame='icrs')
                ras = []
                decs = []
                for src in self:
                    sd = src.skydir
                    if hasattr(sd, 'ra') and hasattr(sd.ra, 'deg'):
                        ra = float(sd.ra.deg)
                    elif hasattr(sd, 'ra') and callable(sd.ra):
                        ra = float(sd.ra())
                    else:
                        ra = None

                    if hasattr(sd, 'dec') and hasattr(sd.dec, 'deg'):
                        dec = float(sd.dec.deg)
                    elif hasattr(sd, 'dec') and callable(sd.dec):
                        dec = float(sd.dec())
                    else:
                        dec = None
                    if ra is None or dec is None:
                        ras = []
                        decs = []
                        break
                    ras.append(ra)
                    decs.append(dec)
                if len(ras) > 0:
                    model_coords = SkyCoord(ras, decs, unit='deg', frame='icrs')
                    k = int(np.argmin(target_coord.separation(model_coords).deg))
                    return found(self[k])
            except Exception:
                pass

        self.selected_source = None
        not_found()

    def add_source(self, newsource=None, **kw):
        """Add a source to the model and rebuild parameter indexing.

        Parameters
        ----------
        newsource : Source or None
            Existing source object to append. If ``None``, this method builds
            a new ``sources.PointSource`` from ``**kw``.
        **kw : dict
            Keyword arguments forwarded to ``sources.PointSource`` when
            ``newsource`` is ``None``. Common keys include ``name``, ``model``,
            and ``skydir``.

        Returns
        -------
        Source or None
            The appended source on success. Returns ``None`` if a source with
            the same name already exists.
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
        self.selected_source = newsource
        return newsource
     
    def del_source(self, source_name):
        """Remove and return a source from the model.

        Parameters
        ----------
        source_name : str, Source, or None
            Source selector accepted by ``find_source``.

        Returns
        -------
        Source
            The removed source object.
        """
        source = self.find_source(source_name) # first get it
        self.remove(source)
        self.initialize()
        return source
        
    def set_model(self, model, source_name=None):
        """Replace selected source model and return `(source, old_model)`.

        Parameters
        ----------
        model : str or Model
            Replacement spectral model. If a string is provided, it is
            evaluated and must produce a valid model instance. Example:
            ``'PowerLaw(1e-11,2.0)'``.
        source_name : str, Source, or None
            Target source selector accepted by ``find_source``. If ``None``,
            applies to the currently selected source.

        Returns
        -------
        tuple
            ``(source, old_model)`` where ``source`` is the updated source and
            ``old_model`` is the model replaced.
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
        """Print every source in the model in current order.

        This is a convenience inspection helper and returns ``None``.
        """
        for source in self:
            print(source)
        return
    
    def setposition(self, skydir):
        """Set the position of the selected source.

        Parameters
        ----------
        skydir : object
            Position value compatible with the selected source's ``skydir``
            attribute.

        Notes
        -----
        Requires a currently selected source.
        """
        if self.selected_source is None:
            raise SourceModelException('No source is selected')
        self.selected_source.skydir = skydir

    
    @classmethod
    def demo(cls,  src_key=2,) :
        """Create a toy source list for quick experiments.

        Parameters
        ----------
        src_key : int, default=2
            Select which demo sources to include.

            - ``0``: Pulsar only
            - ``1``: Blazar only
            - ``2``: both sources

        Returns
        -------
        SourceModel
            A newly constructed toy model.
        """
        ps = sources.PointSource(name='Pulsar',  skydir=(0,0), frame='galactic',
            model=sources.PLSuperExpCutoff4(1e-11, 2., 0.7, 0.69))  
        
        pl = sources.PointSource(name='Blazar',skydir=(5,0), frame='galactic',
                model=sources.LogParabola(4e-12, 2, 0, 332))
        
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

    @staticmethod
    def _coerce_catalog_skycoord(skycoord, frame='icrs'):
        """Normalize a catalog cone-center specification to ``SkyCoord``."""
        from astropy.coordinates import SkyCoord

        if isinstance(skycoord, SkyCoord):
            return skycoord
        if isinstance(skycoord, str):
            return SkyCoord.from_name(skycoord, frame=frame)
        if hasattr(skycoord, '__iter__'):
            lon, lat = skycoord
            return SkyCoord(lon, lat, unit='deg', frame=frame)
        raise TypeError('skycoord must be a SkyCoord, string, or a (lon, lat) pair in degrees')

    @staticmethod
    def _load_fermi_catalog(catalog=None, *, version=None, path='$FERMI/catalog/', reset_index=False):
        """Return an existing Fermi catalog or construct one on demand."""
        if catalog is not None:
            return catalog
        from utilities.catalogs import Fermi4FGL
        fermi_catalog = Fermi4FGL(version=version, path=path, reset_index=reset_index)
        if not reset_index:
            fermi_catalog.index = pd.Index(
                [
                    name.replace('FL16Y', '').strip() if isinstance(name, str) else name
                    for name in fermi_catalog.index
                ],
                name=fermi_catalog.index.name,
            )
        return fermi_catalog

    @staticmethod
    def _apply_catalog_selection(catalog, select):
        """Apply explicit row selection to a catalog dataframe."""
        if callable(select):
            resolved = select(catalog)
            if isinstance(resolved, pd.DataFrame):
                return resolved
            return SourceModel._apply_catalog_selection(catalog, resolved)

        if isinstance(select, slice):
            return catalog.iloc[select]

        if isinstance(select, str):
            if select not in catalog.index:
                raise SourceModelException(f'source {select} not found in catalog')
            return catalog.loc[[select]]

        if np.isscalar(select):
            return catalog.iloc[[int(select)]]

        items = list(select)
        if len(items) == 0:
            return catalog.iloc[0:0].copy()

        if all(isinstance(item, (bool, np.bool_)) for item in items):
            if len(items) != len(catalog):
                raise SourceModelException('boolean catalog selector length mismatch')
            return catalog.loc[np.asarray(items, dtype=bool)]

        if all(isinstance(item, (int, np.integer)) for item in items):
            return catalog.iloc[list(map(int, items))]

        return catalog.loc[items]

    @classmethod
    def _subset_fermi_catalog(cls, catalog_id, *, select=None, query=None, skycoord=None, cone_size=1.0, frame='icrs'):
        """Return a filtered catalog dataframe based on subset arguments."""
        
        catalog =Fermi4FGL(catalog_id) if isinstance(catalog_id, str) else catalog_id
        subset = catalog

        if skycoord is not None:
            if not hasattr(catalog, 'select_cone'):
                raise TypeError('catalog does not support cone selection')
            subset = catalog.select_cone(
                cls._coerce_catalog_skycoord(skycoord, frame=frame),
                cone_size=cone_size,
            )
            if subset is None:
                subset = catalog.iloc[0:0].copy()

        if query is not None:
            subset = subset.query(query)

        if select is not None:
            subset = cls._apply_catalog_selection(subset, select)

        if len(subset) == 0:
            raise SourceModelException('no sources selected from the Fermi catalog')

        result = subset.copy()
        if skycoord is not None and 'sep' in result.columns:
            result = result.sort_values('sep')
        return result

    @staticmethod
    def _convert_fermi_model(specfunc):
        """Convert a Fermi catalog spectral function into a like3 model."""
        if sources.ismodel(specfunc):
            return specfunc.copy()

        model_name = specfunc.__class__.__name__
        raw_pars = getattr(specfunc, 'pars', None)
        if raw_pars is None:
            raise SourceModelException('catalog spectral function is missing parameter values')
        pars = np.asarray(raw_pars, dtype=float)
        e0 = getattr(specfunc, 'e0', None)

        spectral_models = sources.spectral_models
        builders = {
            'PowerLaw': lambda values, scale: spectral_models.PowerLaw(
                p=values,
                e0=1e3 if scale is None else float(scale),
            ),
            'LogParabola': lambda values, scale: spectral_models.LogParabola(
                p=values,
                free=[True, True, False, False],
            ),
            'PLSuperExpCutoff4': lambda values, scale: spectral_models.PLSuperExpCutoff4(
                p=values,
                free=[True, True, False, False],
                e0=1e3 if scale is None else float(scale),
            ),
        }
        if model_name not in builders:
            raise SourceModelException(f'unsupported catalog spectral model {model_name}')

        return builders[model_name](pars, e0)

    @classmethod
    def _source_from_fermi_row(cls, source_name, row):
        """Create a ``PointSource`` from one Fermi catalog row."""
        resolved_name = row['name'] if 'name' in row.index else source_name
        resolved_name = str(resolved_name)
        if resolved_name.startswith('FL16Y'):
            resolved_name = resolved_name[5:].lstrip()
        src = sources.PointSource(
            name=resolved_name,
            skydir=(float(row.ra), float(row.dec)),
            frame='icrs',
            model=cls._convert_fermi_model(row.specfunc),
        )

        # Preserve common catalog aliases for robust user-facing lookup.
        alias_fields = ('name', 'assoc1', 'assoc2', 'assoc_name', 'alt_name', 'common_name', 'source_name', 'assoc')
        aliases = [str(source_name)]
        for key in alias_fields:
            if key in row.index and pd.notna(row[key]):
                value = str(row[key]).strip()
                if value:
                    aliases.append(value)
        src.aliases = list(dict.fromkeys(aliases))
        return src

    @classmethod
    def from_fermi_catalog( cls,
        skycoord,  *,
        version='v40',
        catalog=None,
        path='$FERMI/catalog/',
        reset_index=False,
        select=None,
        query=None,
        cone_size=1.0,
        frame='icrs',
    ):
        """Build a ``SourceModel`` from a Fermi catalog subset.

        Parameters
        ----------
        skycoord : SkyCoord, str, or tuple, optional
            Cone center for spatial subset selection. Can be a SkyCoord object,
            a source name string (resolved via SkyCoord.from_name), or a
            (longitude, latitude) tuple in degrees -- see frame for coordinate frame, default 'icrs'.
        version : str or None, optional
            Catalog version forwarded to ``utilities.catalogs.Fermi4FGL`` when
            ``catalog`` is not provided.
        catalog : pandas.DataFrame-like, optional
            Existing Fermi catalog object. If omitted, one is loaded from
            ``version`` and ``path``.
        path : str, default='$FERMI/catalog/'
            Catalog directory or FITS path used when loading a catalog.
        reset_index : bool, default=False
            Forwarded to catalog construction.
        select : selector, optional
            Explicit subset selector applied after ``query``/cone filtering.
            Supported forms are source name, integer row index, slices, lists
            of names or indices, boolean masks, or a callable that returns one
            of those forms.
        query : str, optional
            Pandas query expression applied to the catalog before ``select``.
        cone_size : float, default=1.0
            Cone radius in degrees. Only used if ``skycoord`` is provided.
        frame : str, default='icrs'
            Coordinate frame used when ``skycoord`` is provided as a tuple.

        Returns
        -------
        SourceModel
            Source model containing one point source per selected catalog row.
        """
        cone_center = None
        if skycoord is not None:
            cone_center = cls._coerce_catalog_skycoord(skycoord, frame=frame)

        fermi_catalog = cls._load_fermi_catalog(
            catalog=catalog,
            version=version,
            path=path,
            reset_index=reset_index,
        )
        catalog_subset = cls._subset_fermi_catalog(
            fermi_catalog,
            select=select,
            query=query,
            skycoord=cone_center,
            cone_size=cone_size,
            frame=frame,
        )
        model = cls([
            cls._source_from_fermi_row(source_name, row)
            for source_name, row in catalog_subset.iterrows()
        ])
        model.fermi_catalog = fermi_catalog

        # Set a descriptive name for the SourceModel
        if isinstance(skycoord, str):
            model.name = f"{skycoord} ({version})"
        elif hasattr(cone_center, 'to_string'):
            model.name = f"ROI@{cone_center.to_string('hmsdms')} ({version})"
        else:
            model.name = f"Fermi4FGL ({version})"

        # If a target coordinate/name was given, ensure selected_source is
        # deterministic for downstream code that relies on the active source.
        if len(model) > 0 and cone_center is not None:
            selected = False
            if isinstance(skycoord, str):
                target_name = skycoord.strip().lower()
                for src in model:
                    if src.name.strip().lower() == target_name:
                        model.find_source(src.name)
                        selected = True
                        break
            if not selected:
                from astropy.coordinates import SkyCoord

                cat_coords = SkyCoord(
                    catalog_subset.ra.astype(float).values,
                    catalog_subset.dec.astype(float).values,
                    unit='deg',
                    frame='icrs',
                )
                k = int(np.argmin(cone_center.separation(cat_coords).deg))
                model.find_source(model[k].name)

        return model

    def view(self):
        """Return a context-manager view that restores all source model state on exit.

        Within the ``with`` block the caller receives the original ``SourceModel``
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
        SourceModelContext
        """
        ret = SourceModelContext(self)
        # print('Entering SourceModel view context: snapshot taken, original list is available as "sl"', ret.__class__)
        return ret
    
    def localization_view(self, source_name=None):
        """Set up context-manager that restores only the selected-source position on exit.

        Within the ``with`` block the caller receives a ``LocalizedSourceView``
        bound to the selected source. The view delegates attribute access to the
        original ``SourceModel`` and adds a ``delta_ts`` helper for localization
        scans. When the block exits (whether normally or via an exception), the
        selected source position is rolled back to the state captured at block
        entry.

        Usage
        -----
        .. code-block:: python

            with source_list.localization_view('Blazar') as loc:
                ts = loc.delta_ts(my_loglike_callable)
                # ... trial computation ...
            # selected source position is restored here

        Returns
        -------
        SourceModelContext
            Context manager whose ``__enter__`` returns ``LocalizedSourceView``.
        """
        # Accept either a source-name string or a source-like object.
        # We do object membership resolution here so it works even when the
        # instance originates from a reloaded module and fails isinstance checks.
        if source_name is not None and not isinstance(source_name, str) and hasattr(source_name, 'name'):
            if source_name in self:
                self.selected_source = source_name
                self.selected_source_index = self.index(source_name)
            else:
                self.find_source(source_name.name)
        elif self.selected_source is None or self.selected_source.name != source_name:
            self.find_source(source_name)
        if self.selected_source is None:
            raise SourceModelException(f'source {source_name} not found')
        return SourceModelContext(self,  position_only=True)

    def localization_context(self, source_name=None):
        """Backward-compatible alias for :meth:`localization_view`."""
        return self.localization_view(source_name)


class SourceModelContext:
    """Context-manager snapshot/restore wrapper for a :class:`SourceModel`.

    On ``__enter__`` a lightweight snapshot of the source-list contents and
    every source's spectral-model state is captured.  On ``__exit__`` the
    original list is rewound to that snapshot, regardless of whether the block
    raised an exception.

    Do not instantiate directly; use :meth:`SourceModel.view` instead.
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
        """Capture a snapshot and return a context view object.

        Returns
        -------
        SourceModel or LocalizedSourceView
            Returns the original ``SourceModel`` for full-state contexts.
            For position-only localization contexts, returns a
            ``LocalizedSourceView`` centered on the selected source.
        """
        self._snapshot = self._take_snapshot()
        if self._position_only:
            return LocalizedSourceView(self._sl)
        return self._sl

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the ``SourceModel`` to its pre-block state.

        Always restores, even when the block raised an exception.
        Never suppresses exceptions (returns ``False``).
        """
        self._restore(self._snapshot)
        return False  # propagate any exception unchanged


class LocalizedSourceView:
    """Selected-source helper view for localization workflows.

    Instances are returned by ``SourceModelContext`` when created via
    ``SourceModel.localization_view``. The object delegates unknown
    attributes to the underlying ``SourceModel`` and adds a ``delta_ts``
    helper for likelihood-ratio scans of the currently selected source.
    """

    def __init__(self, source_model):
        """Bind to a ``SourceModel`` with an already selected source.

        Parameters
        ----------
        source_model : SourceModel
            Backing model used for delegation and source-position updates.

        Raises
        ------
        SourceModelException
            If ``source_model`` has no selected source.
        """
        self.source_model = source_model
        self.source = source_model.selected_source
        if self.source is None:
            raise SourceModelException('No source is selected')

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped ``SourceModel``."""
        return getattr(self.source_model, name)

    def _evaluate_loglike(self, loglike, position=None):
        """Evaluate a log-likelihood callable at an optional trial position.

        Supports two callable styles:
        - ``loglike(position)`` taking the trial sky position directly.
        - ``loglike()`` where this helper sets ``self.source.skydir`` first.
        """
        if position is None:
            try:
                return float(loglike())
            except TypeError:
                return float(loglike(self.source.skydir))

        try:
            return float(loglike(position))
        except TypeError:
            saved = self.source.skydir
            self.source.skydir = position
            try:
                return float(loglike())
            finally:
                self.source.skydir = saved

    def delta_ts(self, loglike, position=None, baseline=None):
        """Evaluate or build a delta-TS function for the selected source.

        Parameters
        ----------
        loglike : callable
            Log-likelihood evaluator. It may accept a trial position argument
            (`loglike(position)`) or use current model state (`loglike()`).
        position : optional
            If provided, return the delta-TS evaluated at this position.
            If omitted, return a callable `f(position)`.
        baseline : float, optional
            Reference log-likelihood value. If omitted, uses the current
            selected-source position as the baseline.

        Returns
        -------
        float or callable
            `2 * (loglike(position) - baseline)` for one position, or a
            callable that computes this value for arbitrary trial positions.
        """
        l0 = self._evaluate_loglike(loglike) if baseline is None else float(baseline)

        def eval_delta(position_value):
            return 2.0 * (self._evaluate_loglike(loglike, position_value) - l0)

        if position is None:
            return eval_delta
        return eval_delta(position)





