# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for ROI analysis - Spectral Energy Distribution functions

This module provides utilities for measuring and analyzing the Spectral Energy Distribution
(SED) of gamma-ray sources. The primary interface is the SED class, which measures energy flux
vs. energy for a given source across a range of energy bands. Supporting functions provide
table generation, residual analysis, and comparative source analysis.

Main Components:
    - SED: Main class for SED measurement and analysis
    - sed_table: Generate DataFrame summary of SED measurements
    - norm_table: Generate DataFrame of normalization Poisson results
    - residual_tables: Batch processing of residual tables for multiple sources
    - normalization_poiss: Get Poisson fit objects for source normalizations
    - alternate_source: Create alternative source models for comparison
    - makesed_all: Batch SED processing with plotting and statistics
    - add_flat_sed: Add flat model SED results for comparison
"""
import os, pickle
import importlib
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import pandas as pd

# from uw.utilities import ( keyword_options)
_pkg = __package__ if __package__ else 'like3'
plotting = importlib.import_module(f'{_pkg}.plotting')
tools = importlib.import_module(f'{_pkg}.tools')
loglikelihood = importlib.import_module(f'{_pkg}.loglikelihood')
sources = importlib.import_module(f'{_pkg}.sources')
bands = importlib.import_module(f'{_pkg}.pixel_table')
# 2/decade above 31.6 GeV
energybins=np.concatenate( [np.logspace(2,4.25,10), np.logspace(4.5,6,4)])


def _poisson_fitter_with_retry(func, tol, *, scale=None, delta=None):
    """Run PoissonFitter with progressively looser fallback settings."""
    attempts = [
        dict(tol=float(tol)),
        dict(tol=float(max(1.0, 2.0 * tol))),
        dict(tol=float(max(2.0, 4.0 * tol)), delta=1e-3),
    ]
    if scale is not None:
        for a in attempts:
            a['scale'] = float(scale)
    if delta is not None:
        attempts[0]['delta'] = float(delta)

    last_exc = None
    for kw in attempts:
        try:
            return loglikelihood.PoissonFitter(func, **kw)
        except Exception as exc:
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def _poisson_fitter_robust(func, tol, *, scale_hint=None):
    """Run PoissonFitter with retries, unitless scaling, and scan-seeded fallback."""
    try:
        return _poisson_fitter_with_retry(func, tol), 'ok', ''
    except Exception as first_exc:
        base = float(scale_hint) if scale_hint is not None and np.isfinite(scale_hint) and scale_hint > 0 else 1.0

        # Retry in unitless coordinates u = flux / base and map fit back to flux.
        def unitless(u):
            return func(max(float(u), 0.0) * base)

        try:
            pf_u = _poisson_fitter_with_retry(unitless, tol, scale=1.0, delta=1e-4)
            sp, e, b = pf_u.poiss.p
            poiss_flux = loglikelihood.Poisson([sp * base, e / base, b * base])
            pf_scaled = SimpleNamespace(
                poiss=poiss_flux,
                maxdev=float(pf_u.maxdev),
                wprime=float(pf_u.wprime) / base,
            )
            return pf_scaled, 'scaled-unitless', f'initial retry failed: {first_exc}'
        except Exception as scaled_exc:
            scaled_reason = scaled_exc

        # Scan nearby scales to find a stable seed for difficult profiles.
        factors = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0], dtype=float)
        x_scan = np.clip(base * factors, 0.0, np.inf)

        vals = []
        good_x = []
        for x in x_scan:
            try:
                v = float(func(x))
            except Exception:
                continue
            if np.isfinite(v):
                good_x.append(float(x))
                vals.append(v)

        if len(vals) < 3:
            raise RuntimeError(
                f'Poisson fit failed after retries; scan fallback had only {len(vals)} finite samples. '
                f'first error: {first_exc}; scaled error: {scaled_reason}'
            )

        good_x = np.asarray(good_x, dtype=float)
        vals = np.asarray(vals, dtype=float)
        seed = float(max(good_x[int(np.argmax(vals))], 1e-8))
        seed_delta = float(max(1e-4, min(1.0, 1e-3 * seed)))

        try:
            pf = _poisson_fitter_with_retry(func, tol, scale=seed, delta=seed_delta)
            return pf, 'scan-seeded', f'initial retry failed: {first_exc}; scaled error: {scaled_reason}'
        except Exception as second_exc:
            raise RuntimeError(
                f'Poisson fit failed after retries and scan-seeded fallback. '
                f'first error: {first_exc}; scaled error: {scaled_reason}; scan error: {second_exc}'
            )
       
class SED(tools.WithMixin):
    """Measure the energy flux vs. energy for a given source.
    
    The SED class provides flexible energy flux measurement across energy bands,
    supporting selection of individual bands, energy ranges, or event types.
    Supports context manager protocol (with statement) for automatic state restoration.
    
    Attributes:
        rs: ROIstat object containing ROI information
        func: Energy flux view function for the source
        source_name (str): Name of the source being analyzed
        energies (list): Array of energies with data in the ROI
        energybins (np.ndarray): Energy bin edges for SED computation
    """
    def __init__(self, rstat, source_name, quiet=False):
        """Initialize SED analysis for a source.
        
        Parameters:
            rstat (ROIstat): ROI statistics object containing band information
            source_name (str): Name of source from the SourceList
            quiet (bool): If True, suppress status messages (default: False)
        """
        self.rs = rstat
        self.rs.quiet=quiet
        self.func = self.rs.energy_flux_view(source_name, bound=-20) # note very low bound
        self.source_name = source_name
        # make a list of energies with data; only have info if there is data in the ROI
        self.energies = list(set(np.array([b.band.energy for b in self.rs if b.pixels>0])))  

        # combines the bands above 100 GeV 
        emax = rstat[-1].band.emax
        global energybins
        self.energybins=[e for e in energybins if e<=emax]
    
    def full(self):
        """Fit Poisson distribution to all bands for the source.
        
        Returns:
            tuple: (poiss_obj, maxdev) where poiss_obj is the Poisson fitter result
                and maxdev is the maximum deviation of the fit
        
        Raises:
            Exception: If Poisson fitting fails
        """
        try:
            fp = self.select(None)
            if fp is None:
                raise RuntimeError('no data available for full-band Poisson fit')
            self.full_poiss = fp.poiss
        except Exception as msg:
            print('Failed poisson fit to source %s: "%s"' % (self.source_name, msg))
            raise
        return fp.poiss, fp.maxdev
        
    def __repr__(self):
        return '{}.{} : SED analysis for source "{}", using {:d} selected bands, {:.0f}-{:.0f} MeV'.format(
                    self.__module__, self.__class__.__name__,self.source_name,len(self.rs.selected), 
                    self.rs.emin, self.rs.emax)
    
    def select(self, index, event_type=None, poisson_tolerance=0.20, 
        elow=None, ehigh=None,**kwargs):
        """Select energy band(s) and fit Poisson distribution.
        
        Parameters:
            index (int or None): Index into energies list for single band selection.
                If None, selects all bands using current spectral model.
                For non-None, uses model-independent powerlaw for the band.
            event_type (int or None): Event type selection:
                None=all, 0=front, 1=back, 2-5=PSF0-3 (default: None)
            poisson_tolerance (float): Tolerance for Poisson fit convergence (default: 0.20)
            elow (float): Lower energy limit for band selection (MeV)
            ehigh (float): Upper energy limit for band selection (MeV)
            **kwargs: Additional arguments passed to PoissonFitter
        
        Returns:
            object: PoissonFitter result object with .poiss attribute, or None if no data
        
        Notes:
            - If elow/ehigh specified and no data exists, returns None
            - Automatically sets energy for flux function based on selection
        """
        if index is None and elow is None and event_type is None:
            self.rs.select()
            self.func.set_energy(None)# =func = self.rs.energy_flux_view(self.source_name)
        elif index is not None:
            self.rs.select(index, event_type)
            energies = self.rs.energies
            assert len(energies)==1
            energy = self.rs.energies[0]
            self.func.set_energy(energy)
            assert self.func(0) != self.func(1), 'Function not variable? energy %.0f' % energy
        else:
            # case to perhaps combine bands
            self.rs.select(event_type=event_type, elow=elow, ehigh=ehigh)
            has_data = np.any([b.band.has_pixels for b in self.rs.selected])
            if not has_data: return None
            if elow is None or ehigh is None:
                raise ValueError('elow and ehigh must both be set for combined-band selection')
            energy = np.sqrt(elow*ehigh)
            self.func.set_energy(energy)
        pf = loglikelihood.PoissonFitter(self.func, tol=poisson_tolerance, **kwargs)
        return pf

    def all_poiss(self, event_type=None, tol=0.1, debug=False):
        """Fit Poisson distribution for each energy band.
        
        Parameters:
            event_type (int or None): Event type (None=all, or 0/1 for front/back, 2-5 for PSF0-3)
            tol (float): Poisson tolerance for fitting (default: 0.1)
            debug (bool): If True, print progress information (default: False)
        
        Returns:
            np.ndarray: Array of Poisson objects (None for failed fits)
        """
        pp = []
        for i,e in enumerate(self.energies):
            if debug: print('%3i %8.0f' % (i,e), end=' ')
            try:
                pf = self.select(i, event_type=event_type,poisson_tolerance=tol)
                if pf is None:
                    pp.append(None)
                    continue
                pp.append(pf.poiss)
                if debug: print(pf)
            except Exception as msg:
                print('Fail poiss fit for %.0f MeV: %s ' % (e,msg))
                pp.append(None)
                
        self.restore()
        return np.array(pp)
        
    def sed_rec(self, event_type=None, tol=0.1):
        """Generate recarray of SED measurements for each energy band.
        
        Parameters:
            event_type (int or None): Event type selection (None=all, 0/1=front/back, 2-5=PSF0-3)
            tol (float): Poisson tolerance (default: 0.1)
        
        Returns:
            numpy.recarray: Record array with following columns:
                - elow, ehigh: Energy band limits (MeV)
                - flux, lflux, uflux: Fitted flux and 1-sigma lower/upper errors (ph cm⁻² s⁻¹ MeV⁻¹)
                - npred: Number of predicted photons at maximum likelihood
                - pindex: Photon index at band center computed by finite difference
                - ts: Test statistic (2× Δlog L) for the band
                - mflux: Model-predicted flux
                - delta_ts: TS difference between fit and model
                - pull: Signed sqrt(delta_ts), indicates significance of deviation
                - maxdev: Maximum deviation of fit
                - zero_fract: Predicted fraction of time with zero expected flux
        """
        names = 'elow ehigh flux lflux uflux npred pindex ts mflux  delta_ts pull maxdev zero_fract'.split()
        rec = tools.RecArray(names, dtype=dict(names=names, formats=['>f4']*len(names)) )
        
        ebins = self.energybins
        if event_type is not None:
            et_min = getattr(bands, 'event_type_min_energy', {})
            emin_et = et_min.get(event_type, 0)
            ebins = [e for e in ebins if e >= emin_et]
        for i,(elow,ehigh) in enumerate(zip(ebins[:-1], ebins[1:])):
        #for i,energy in enumerate(self.energies):
                
            try:
                pf = self.select(None, elow=elow,ehigh=ehigh, 
                    event_type=event_type, poisson_tolerance=tol)
                xlo,xhi = self.rs.emin, self.rs.emax
            except Exception as msg:
                print('Fail poiss fit for %.0f-%.0f MeV: %s ' % (elow,ehigh,msg))
                rec.append(elow,ehigh, 0, 0, np.nan, 0,0,0, np.nan, np.nan, np.nan, np.nan,     np.nan )
                continue
            if pf is None: # no data
                rec.append(elow,ehigh, 0, 0, np.nan, 0,0,0, np.nan, np.nan, np.nan, np.nan,     np.nan )
                continue
            elif np.isnan(pf.wprime):
                print('Fail poiss fit for %.0f-%.0f MeV: %s ' % (elow,ehigh,'bad poiss'))
                rec.append(elow,ehigh, 0, 0, np.nan, 0,0,0, np.nan, np.nan, np.nan, np.nan, np.nan )
                continue
            
            w = pf.poiss
            err = pf.maxdev
            lf,uf = w.errors
            maxl  = w.flux
            mf    = self.func.eflux
            self.func(maxl) # set to maxl for npred
            npred = sum([bs[self.source_name].counts for bs in self.rs.selected])
            
            # get spectral function evaluate exponential slope by finite difference
            m = self.rs.get_model(self.source_name)
            x = np.sqrt(xlo*xhi)
            delta=0.01 # 1%
            pindex= (1-m((1+delta)*x)/m(x))/delta
            
            delta_ts = 2.*(self(maxl) - self(mf) )
            zf =   w.zero_fraction()
            if lf>0 :
                pull = np.sign(maxl-mf) * np.sqrt(max(0, delta_ts))
                assert not np.isnan(pull), 'row {}: pull = {}'.format(i,pull)
                rec.append(xlo, xhi, maxl, lf, uf, npred, pindex, w.ts, mf, delta_ts, pull, err, zf)
            else:
                pull = -np.sqrt(max(0, delta_ts))
                rec.append(xlo, xhi, 0, 0, w.cdfcinv(0.05), 0,pindex, 0, mf, delta_ts, pull, err, zf )
            
        self.restore()
        return rec()

    def data_frame(self, event_type=None, tol=0.1):
        """Generate pandas DataFrame summary of SED measurements.
        
        Parameters:
            event_type (int or None): Event type selection (None=all, 0/1=front/back, 2-5=PSF0-3)
            tol (float): Poisson tolerance (default: 0.1)
        
        Returns:
            pd.DataFrame: SED summary with energy index and columns:
                elow, ehigh, flux, lflux, uflux, TS, mflux, npred, pindex, pull, zf
        """
        si = self.sed_rec(event_type,tol)
        r =pd.DataFrame(
            dict(elow=si.elow, ehigh=si.ehigh, 
                flux=si.flux.round(2), TS=si.ts.round(1), lflux=si.lflux.round(2),
                npred= si.npred.round(1),
                pindex=si.pindex.round(2),
                uflux=si.uflux.round(2), mflux=si.mflux.round(2), pull=si.pull.round(2), zf=si.zero_fract.round(3)), 
            index=np.array(np.sqrt(si.elow*si.ehigh),int), 
            columns='elow ehigh flux lflux uflux mflux npred pindex TS pull zf'.split())
        r.index.name='energy'
        return r

    def poisson_table(self, event_type=None, tol=0.1):
        """Generate a per-band SED table with one Poisson object per row.

        Parameters
        ----------
        event_type : int or None
            Event type selection (None=all, 0/1=front/back, 2-5=PSF0-3).
        tol : float
            Poisson tolerance (default: 0.1).

        Returns
        -------
        pd.DataFrame
            Table indexed by band-center energy with columns:
            ``elow``, ``ehigh``, ``poiss``, ``maxdev``, ``wprime``,
            ``flux``, ``lflux``, ``uflux``, and ``ts``.
            The ``poiss`` column contains ``loglikelihood.Poisson`` objects
            (or ``None`` when no fit is available).
        """
        ebins = self.energybins
        if event_type is not None:
            et_min = getattr(bands, 'event_type_min_energy', {})
            emin_et = et_min.get(event_type, 0)
            ebins = [e for e in ebins if e >= emin_et]

        rows = []
        try:
            for elow, ehigh in zip(ebins[:-1], ebins[1:]):
                row = dict(
                    energy=float(np.sqrt(elow * ehigh)),
                    elow=float(elow),
                    ehigh=float(ehigh),
                    poiss=None,
                    maxdev=np.nan,
                    wprime=np.nan,
                    flux=np.nan,
                    lflux=np.nan,
                    uflux=np.nan,
                    ts=np.nan,
                    fit_status='pending',
                    fail_reason='',
                )
                try:
                    # Configure band selection and energy explicitly so robust fitting
                    # can evaluate the function directly if needed.
                    self.rs.select(event_type=event_type, elow=elow, ehigh=ehigh)
                    has_data = np.any([b.band.has_pixels for b in self.rs.selected])
                    if not has_data:
                        row['fit_status'] = 'no-data'
                        rows.append(row)
                        continue
                    self.func.set_energy(np.sqrt(elow * ehigh))
                    scale_hint = float(self.func.eflux) if np.isfinite(self.func.eflux) else None
                    pf, status, reason = _poisson_fitter_robust(self.func, tol, scale_hint=scale_hint)
                except Exception as msg:
                    print('Fail poiss fit for %.0f-%.0f MeV: %s ' % (elow, ehigh, msg))
                    row['fit_status'] = 'failed'
                    row['fail_reason'] = str(msg)
                    rows.append(row)
                    continue

                if pf is None or np.isnan(pf.wprime):
                    row['fit_status'] = 'invalid-wprime'
                    row['fail_reason'] = 'wprime is NaN'
                    rows.append(row)
                    continue

                w = pf.poiss
                lf, uf = w.errors
                row.update(
                    poiss=w,
                    maxdev=float(pf.maxdev),
                    wprime=float(pf.wprime),
                    flux=float(w.flux),
                    lflux=float(lf),
                    uflux=float(uf),
                    ts=float(w.ts),
                    fit_status=status,
                    fail_reason=reason,
                )
                rows.append(row)
        finally:
            self.restore()

        ret = pd.DataFrame(rows)
        if len(ret) == 0:
            ret = pd.DataFrame(
                columns='elow ehigh poiss maxdev wprime flux lflux uflux ts fit_status fail_reason'.split()
            )
            ret.index.name = 'energy'
            return ret

        ret.index = np.array(ret.pop('energy'), int)
        ret.index.name = 'energy'
        return ret

        
    def restore(self):
        """Restore ROI and flux function to default selection state."""
        self.rs.select()
        self.func.restore()
        
    def __call__(self, eflux):
        """Evaluate log-likelihood at given energy flux value(s).
        
        Parameters:
            eflux (float or np.ndarray): Energy flux value(s) in eV units (integrated over band)
        
        Returns:
            float or np.ndarray: Log-likelihood value(s)
        """
        return self.func(eflux)
        
    def plots(self):
        """Generate grid of likelihood profiles for each energy band.
        
        Returns:
            matplotlib.figure.Figure: 4x4 grid figure with binned likelihood profiles
                filled with empty subplots if fewer bands exist
        """
        import matplotlib.pylab as plt
           
        fig, axx = plt.subplots(4,4, figsize=(12,12), sharey=True)
        for i, ax in enumerate(axx.flatten()):
            if i >= len(self.energies):
                ax.set_visible(False)
                continue
            pf = self.select(i)
            if pf is None:
                ax.set_visible(False)
                continue
            pf.plot(ax)
            ax.set_title('%d MeV' %( int(self.func.energy),), size=10)
            ax.set_ylim(0,1)
        self.restore()
        fig.suptitle('Binned likelihood plots for '+self.source_name, size=14)
        return fig
 
def sed_table(roi, source_name=None, event_type=None, tol=0.1):
    """Generate SED DataFrame for a source using context manager.
    
    Parameters:
        roi: ROI object
        source_name (str or None): Source name (if None, uses default)
        event_type (int, str, or None): Event type selection:
            None=all, 0=front, 1=back, 2-5=PSF0-3, or name string like 'front'/'back'
        tol (float): Poisson tolerance (default: 0.1)
    
    Returns:
        pd.DataFrame: SED summary table with columns for energy, flux, and statistics
    
    Raises:
        Exception: If event_type name is not recognized
    """
    finder = getattr(roi, 'sources', None)
    if finder is None:
        finder = getattr(roi, 'source_model', None)
    if finder is None or not hasattr(finder, 'find_source'):
        raise AttributeError('Object passed to sed_table must provide sources.find_source or source_model.find_source')
    source = finder.find_source(source_name)
    
    if isinstance(event_type,str):
        etname = event_type.lower()
        if etname=='all': event_type=None
        elif etname in roi.config.event_type_names:
            event_type = roi.config.event_type_names.index(etname)
        else:
            raise Exception('event type name %s not recognized' % event_type)
            
    with SED(roi, source.name) as sf:
        return sf.data_frame(event_type=event_type, tol=tol)


def sed_poisson_table(roi, source_name=None, event_type=None, tol=0.1):
    """Generate an SED table with a Poisson object in each energy-bin row.

    Parameters
    ----------
    roi: ROI object
    source_name : str or None
        Source name (if None, uses default).
    event_type : int, str, or None
        Event type selection:
        None=all, 0=front, 1=back, 2-5=PSF0-3, or a name string.
    tol : float
        Poisson tolerance (default: 0.1).

    Returns
    -------
    pd.DataFrame
        Per-band table indexed by energy. Includes a ``poiss`` column whose
        entries are ``loglikelihood.Poisson`` objects.
    """
    finder = getattr(roi, 'sources', None)
    if finder is None:
        finder = getattr(roi, 'source_model', None)
    if finder is None or not hasattr(finder, 'find_source'):
        raise AttributeError('Object passed to sed_poisson_table must provide sources.find_source or source_model.find_source')
    source = finder.find_source(source_name)

    if isinstance(event_type, str):
        etname = event_type.lower()
        if etname == 'all':
            event_type = None
        elif etname.startswith('psf') and etname[3:].isdigit():
            # 'PSF0'–'PSF3' → integer event-type codes 2–5
            event_type = int(etname[3:]) + 2
        elif hasattr(roi, 'config') and etname in roi.config.event_type_names:
            event_type = roi.config.event_type_names.index(etname)
        else:
            raise Exception('event type name %s not recognized' % event_type)

    # FermiFit-compatible path: operate directly on its PixelTable selections
    # and likelihood view without requiring the legacy ROIstat interface.
    if hasattr(roi, 'pixel_table') and hasattr(roi, 'energy_flux_view'):
        pt = roi.pixel_table
        func = roi.energy_flux_view(source.name, bound=-20)

        emax = max(b.e1 for b in pt.values()) if len(pt) > 0 else 0
        ebins = [e for e in energybins if e <= emax]
        candidate_bands = list(pt.values())
        if event_type is not None:
            candidate_bands = [b for b in candidate_bands if int(b.event_type) == int(event_type)]

        rows = []
        try:
            for elow, ehigh in zip(ebins[:-1], ebins[1:]):
                row = dict(
                    energy=float(np.sqrt(elow * ehigh)),
                    elow=float(elow),
                    ehigh=float(ehigh),
                    poiss=None,
                    maxdev=np.nan,
                    wprime=np.nan,
                    flux=np.nan,
                    lflux=np.nan,
                    uflux=np.nan,
                    ts=np.nan,
                    fit_status='pending',
                    fail_reason='',
                )

                # Include bands that overlap the energy bin, not only those fully contained.
                selected = [
                    b for b in candidate_bands
                    if float(b.e1) > float(elow) and float(b.e0) < float(ehigh)
                ]
                has_data = np.any([
                    (getattr(b, 'nocc', 0) > 0) or (len(getattr(b, 'pix', [])) > 0)
                    for b in selected
                ])
                if len(selected) == 0 or not has_data:
                    row['fit_status'] = 'no-data'
                    rows.append(row)
                    continue

                pt.select(keys=[b.key for b in selected])
                func.set_energy(np.sqrt(elow * ehigh))
                try:
                    scale_hint = float(func.eflux) if np.isfinite(func.eflux) else None
                    pf, status, reason = _poisson_fitter_robust(func, tol, scale_hint=scale_hint)
                except Exception as msg:
                    print('Fail poiss fit for %.0f-%.0f MeV: %s ' % (elow, ehigh, msg))
                    row['fit_status'] = 'failed'
                    row['fail_reason'] = str(msg)
                    rows.append(row)
                    continue

                if np.isnan(pf.wprime):
                    row['fit_status'] = 'invalid-wprime'
                    row['fail_reason'] = 'wprime is NaN'
                    rows.append(row)
                    continue

                w = pf.poiss
                lf, uf = w.errors
                row.update(
                    poiss=w,
                    maxdev=float(pf.maxdev),
                    wprime=float(pf.wprime),
                    flux=float(w.flux),
                    lflux=float(lf),
                    uflux=float(uf),
                    ts=float(w.ts),
                    fit_status=status,
                    fail_reason=reason,
                )
                rows.append(row)
        finally:
            pt.select()
            if hasattr(func, 'restore'):
                func.restore()

        ret = pd.DataFrame(rows)
        if len(ret) == 0:
            ret = pd.DataFrame(
                columns='elow ehigh poiss maxdev wprime flux lflux uflux ts fit_status fail_reason'.split()
            )
            ret.index.name = 'energy'
            return ret

        ret.index = np.array(ret.pop('energy'), int)
        ret.index.name = 'energy'
        return ret

    with SED(roi, source.name) as sf:
        return sf.poisson_table(event_type=event_type, tol=tol)


def norm_table(roi, source_name=None, event_type=None, tol=0.25, ignore_exception=True):
    """Generate normalization Poisson results table for a source.
    
    Fits the normalization (flux scaling factor) independently in each energy band,
    providing model-independent constraints on source flux variations.
    
    Parameters:
        roi: ROI object
        source_name (str or None): Source name (if None, uses default)
        event_type (int or None): Event type selection (None=all, 0=front, 1=back, 2-5=PSF0-3)
        tol (float): Poisson tolerance (default: 0.25)
        ignore_exception (bool): If True, return empty dict for failed fits; if False, raise (default: True)
    
    Returns:
        pd.DataFrame: Normalization table with columns:
            - maxl: Maximum likelihood normalization (ratio to model)
            - lower, upper: 1-sigma confidence interval
            - ts: Test statistic
            - err: Fit error/uncertainty
            - pull: (maxl - 1) / err (deviation from unity in units of sigma)
    """
    source = roi.sources.find_source(source_name)
    #print 'table for {}'.format(source.name)
    roi.select()
    energies = roi.energies
    poiss_list = dict()
    with roi.normalization_view(source.name) as nv:
        for i,energy  in enumerate(energies):
            roi.select(i, event_type)
            try:
                p = loglikelihood.PoissonFitter(nv, tol=tol)
                poiss_list[int(energy)] = p.normalization_summary()
            except Exception as msg:
                print('Fail for %.f: %s' % (energy, msg))
                if not ignore_exception: raise
                poiss_list[int(energy)]= {}
    roi.select()
    ret = pd.DataFrame(poiss_list, index='maxl lower upper ts err'.split() ).T
    ret['pull'] = (ret.maxl-1)/ret.err
    ret.index.name='energy'
    return ret
                
def residual_tables(roi, tol=0.3, types=None, globals=None, locals=None):
    """Generate residual analysis tables for global and local sources.
    
    Computes normalization tables (global) and SED tables (local) across event types,
    allowing residual analysis of both extended models and point sources.
    
    Parameters:
        roi: ROI object
        tol (float): Poisson tolerance (default: 0.3)
        types (list or None): Event type names to process (default: ['all'] + event_type_names)
        globals (list or None): List of global sources (default: sources with isglobal=True)
        locals (list or None): List of local sources (default: free, non-global sources)
    
    Returns:
        dict: Dictionary with source names as keys; values are dicts containing:
            - 'model': The spectral model object
            - Event type strings: Either nom_table (global) or sed_table (local) results
    """
    if types is None: types = ['all']+ list(roi.config.event_type_names)
    if globals is None: globals = [s for s in roi.sources if s.isglobal]
    residuals = dict()
    for source in globals:
        yy = residuals[source.name] = dict()
        yy['model'] = source.model    
        for et in types:
            print(source.name, et)
            yy[et] = norm_table(roi, source.name,et, tol)
            
    if locals is None: locals = [s for s in roi.sources if np.any(s.model.free) and not s.isglobal]
    for source in locals:
        yy = residuals[source.name] = dict()
        yy['model'] = source.model    
        for et in ('all', 'front', 'back'):
            print(source.name, et)
            yy[et] = sed_table(roi, source.name, et, tol)
    return residuals



def print_sed(roi, source_name=None):
    """Print formatted SED table for a source.
    
    Temporarily sets pandas float formatting to 1 decimal for readability.
    
    Parameters:
        roi: ROI object
        source_name (str or None): Source name
    """
    source = roi.get_source(source_name)
    t = pd.get_option('display.float_format')
    pd.set_option('display.float_format', lambda x: '%.1f'%x)
    print(sed_table(roi, source_name))
    pd.set_option('display.float_format', t)
               

def makesed_all(roi, source_name='all', **kwargs):
    """Compute SED and statistical quality for source(s) with optional plotting.
    
    Adds sedrec (SED record array) and ts attributes to each source, computing
    chi-squared goodness-of-fit and corresponding p-value. Optionally generates
    diagnostic SED plots.
    
    Parameters:
        roi: ROI object
        source_name (str): 'all' to process all sources with free parameters,
            or specific source name (default: 'all')
        sedfig_dir (str or None): Directory for output figures; if starts with $,
            expands environment variables (default: None, no plotting)
        showts (bool): If True, annotate plots with TS and p-value (default: True)
        ndf (int): Degrees of freedom for chi-squared fit quality test (default: 10)
        poisson_tolerance (float): Tolerance for Poisson fitting (default: 0.50)
        **kwargs: Additional arguments passed to plotting functions
    
    Side Effects:
        - Modifies source.sedrec and source.ts attributes
        - Creates output directory if sedfig_dir specified and doesn't exist
        - Asserts no net change to ROI log-likelihood after processing
    """
    from scipy import stats # for chi2 
    sedfig_dir = kwargs.pop('sedfig_dir', None)
    if sedfig_dir is not None and sedfig_dir[0]=='$':
        sedfig_dir = os.path.expandvars(sedfig_dir)
    ndf = kwargs.pop('ndf', 10) 
    if sedfig_dir is not None and not os.path.exists(sedfig_dir): os.mkdir(sedfig_dir)
    showts = kwargs.pop('showts', True)
    poisson_tolerance = kwargs.pop('poisson_tolerance', 0.50)
    initw = roi.log_like()

    if source_name=='all':
        sources = [s for s in roi.sources if s.skydir is not None and np.any(s.spectral_model.free)]
    else:
        sources = [roi.get_source(source_name)]
    print('sources:', [s.name for s in sources])
    for source in sources:
        with SED(roi, source.name, ) as sf:
            print(source.name,':', end=' ')
            try:
                source.sedrec = sf.sed_rec( tol=poisson_tolerance)
                source.ts = roi.TS(source.name)
                qual = sum(source.sedrec.pull**2)
                pval = 1.- stats.chi2.cdf(qual, ndf)
                if sedfig_dir is not None:
                    annotation =(0.04,0.88, 'TS=%.0f\npvalue %.1f%%'% (source.ts,pval*100.)) if showts else None 
                    plotting.sed.stacked_plots(sf,  #gev_scale=True, energy_flux_unit='eV',
                         galmap=source.skydir, outdir=sedfig_dir, 
                            annotate=annotation, **kwargs)
                        
            except Exception as e:
                print('***Warning: source %s failed flux measurement: %s' % (source.name, e))
                #raise
                source.sedrec=None
    curw= roi.log_like()
    assert abs(initw-curw)<0.1, \
        'makesed_all: unexpected change in roi state after spectral analysis, from %.1f to %.1f' %(initw, curw)

def add_flat_sed(roi, source_name=None, cols='flux lflux uflux ts'.split()):
    """Measure excess flux by adding flat model at source position.
    
    Compares fitted source model with flat (LogParabola) model at the same position
    as a test for unmodeled photons. Adds _flat columns to sedrec for comparison.
    
    Parameters:
        roi: ROI object
        source_name (str or 'ALL'): Source name or 'ALL' for all free local sources
        cols (list): Column names to add from flat model (default: ['flux', 'lflux', 'uflux', 'ts'])
    
    Returns:
        float or list: For single source, returns sum of TS values;
            for 'ALL', returns list of (name, ts_sum) tuples
    
    Side Effects:
        - Modifies source.sedrec with new *_flat columns
        - Temporarily adds/removes 'temp' source during measurement
    """

    def do_one(s):
        roi.add_source(name='temp', skydir=s.skydir, model='LogParabola(1e-12, 2, 0, 1e4)')
        ss = roi.get_source('temp')
        t = roi.get_sed()
        roi.del_source('temp')
        df = pd.DataFrame(s.sedrec)
        df_flat=pd.DataFrame(OrderedDict( [(n, t[n].astype(float)) for n in cols]))
        for col in cols:
            df[col+'_flat'] = df_flat[col]
        s.sedrec = rec = df.to_records(index=False)
        rec.dtype.names = list(map(str, rec.dtype.names)) # numpy dtype names must be str 
        return sum(df_flat.ts)

    if source_name!='ALL':
        s = roi.get_source(source_name)
        return do_one(s)
    else:
        sources = [s for s in roi.sources if s.skydir is not None and np.any(s.spectral_model.free)]
        return [(s.name, do_one(s)) for s in sources]


def normalization_poiss(roi, source_name, event_type=None):
    """Get list of Poisson fit objects for source normalization in each band.
    
    Parameters:
        roi: ROI object
        source_name (str): Name of source in ROI
        event_type (int or None): Event type selection:
            None=all, 0=front, 1=back, 2-5=PSF0-3 (default: None)
    
    Returns:
        list: Poisson fit objects (one per energy band), each with .flux, .errors, .ts properties
    """
    roi.select()
    energies = roi.energies
    poiss_list = []
    with roi.normalization_view(source_name) as nv:
        for i,energy  in enumerate(energies):
            roi.select(i, event_type)
            p = loglikelihood.PoissonFitter(nv, tol=0.25).poiss
            poiss_list.append(p)
    roi.select()
    return poiss_list

def alternate_source(roi, source, name, skydir, model):
    """Create alternative source model for comparison analysis.
    
    Temporarily replaces source spectral model with alternative, measures SED,
    then restores original model. Useful for model comparison or source disambiguation.
    
    Parameters:
        roi: ROI object
        source: Current Source object with sedrec attribute
        name (str): Name for the alternative source
        skydir: Sky position for alternative source (typically near source.skydir)
        model: Alternative spectral model object
    
    Returns:
        PointSource: New source object with alternative model and corresponding sedrec
    
    Side Effects:
        - Temporarily modifies source.spectral_model during measurement
        - Preserves original source.sedrec after measurement
    """
    roi.get_source(source.name) # make sure selected
    altsrc = sources.PointSource(name=name, skydir=skydir, model=model)
    saved_model = source.spectral_model
    saved_sed = source.sedrec.copy()
    source.spectral_model = model
    altsrc.sedrec = roi.get_sed(update=True)
    source.spectral_model = saved_model
    source.sedrec = saved_sed
    return altsrc
    