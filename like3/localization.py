"""Localization helpers for point sources.

This module contains utilities used to estimate and visualize best-fit source
positions from TS-based likelihood surfaces.

Main entry points
-----------------
- :func:`full_localization`: run one-source localization, optional association,
    and optional TS-map plotting.
- :class:`Localization`: iterative ellipse fit around a source using
    :mod:`uw.like.quadform`.
- :func:`moment_analysis`: fallback/diagnostic ellipse estimate from TS map
    moments.
- :func:`localize_all`: batch localization over suitable sources in an ROI.

Notes
-----
This file is historically derived from earlier pointlike code and keeps that
runtime behavior for compatibility.
"""
import os,sys
import numpy as np
from . skydir import SkyDir
from . import quadform
from . utilities import keyword_options
from . import (sources, plotting )

def moment_analysis(tsmap, wcs, fudge=1.44):
    """Estimate localization ellipse parameters from TS-map moments.

    Parameters
    ----------
    tsmap : array-like
        Square 2D array of TS-like values sampled on a grid.
    wcs : object
        Projection helper implementing ``pix2sph(x, y) -> (ra, dec)``.
    fudge : float, default=1.44
        Empirical multiplicative factor applied to the derived ellipse radii.

    Returns
    -------
    tuple
        ``(ra, dec, a, b, ang)`` where ``a`` and ``b`` are semi-axis sizes in
        degrees and ``ang`` is position angle in degrees.
    """
    vals = np.exp(-0.5 * tsmap**2).flatten()
    peak_fraction = vals.max()/sum(vals)
    

    n = len(vals)
    nx = ny =int(np.sqrt(n))
    # Pixel centers are at integer index + 0.5.
    ix = np.array([i % nx for i in range(n)]) + 0.5
    iy = np.array([i // nx for i in range(n)]) + 0.5
    norm = 1./sum(vals)
    t = [sum(u*vals)*norm for u in  (ix,iy, ix**2, ix*iy, iy**2)]
    center = (t[0],t[1])
    C = np.matrix(center)
    variance = (np.matrix(((t[2], t[3]),(t[3], t[4]))) - C.T * C)
    ra,dec = wcs.pix2sph(center[0]+0.5,center[1]+0.5)
    peak = SkyDir(ra,dec)
    # Coordinate scale in degrees per pixel around map center.
    nc = (nx+1)/2
    rac, decc = wcs.pix2sph(nc, nc)
    scale = wcs.pix2sph(nc, nc+1)[1] - decc
    size = nx*scale
    # adjust variance
    variance = scale**2 * variance
    offset = np.degrees(peak.difference(SkyDir(rac,decc)))

    # Keep current variance model (historical behavior).
    var = variance
    # Eigen-decomposition yields ellipse axes/orientation.
    u,v =np.linalg.eigh(var)
    ang =np.degrees(np.arctan2(v[1,1], -v[1,0]))
    if min(u)< 0.5* max(u): 
        print('Too elliptical : %s, setting circular' % u)
        u[0]=u[1] = max(u)
    tt = np.sqrt(u) * fudge
    if u[1]>u[0]:
        ax,bx = tt[1], tt[0]
        ang = 90-ang
    else:
        ax,bx = tt
    return ra, dec, ax,bx, ang

class MomentAnalysis(object):
    """Small wrapper that runs and stores a moment-based localization.

    The class is mostly convenience for plotting and introspection around
    :func:`moment_analysis`.
    """
    def __init__(self, tsplot, fudge=1.44):
        """Build from a TS-plot object.

        Parameters
        ----------
        tsplot : object
            Plot object expected to expose ``zea.projector`` and ``zea.image``.
        fudge : float, default=1.44
            Radius scale factor forwarded to :func:`moment_analysis`.
        """
        self.tsp=tsplot
        zea = tsplot.zea
        wcs, tsmap = zea.projector, zea.image
        self.ellipse = moment_analysis(tsmap, wcs, fudge)
        
    def moments(self):
        """Return raw first/second moments used by :func:`moment_analysis`."""
        tsmap = self.tsp.zea.image
        vals = np.exp(-0.5 * tsmap**2).flatten()
        peak_fraction = vals.max()/sum(vals)
        n = len(vals)
        nx = ny =int(np.sqrt(n))
        ix = np.array([i % nx for i in range(n)]) + 0.5
        iy = np.array([i // nx for i in range(n)]) + 0.5
        norm = 1./sum(vals)
        t = [sum(u*vals)*norm for u in  (ix,iy, ix**2, ix*iy, iy**2)]
        return t

    def drawit(self):
        """Overlay the fitted moment ellipse and center on the TS plot figure."""
        
        self.tsp.overplot(self.ellipse, color='w', lw=2, ls='-', contours=[2.45])
        self.tsp.plot(SkyDir(*self.ellipse[:2]), color='w', symbol='o' )
        return self.tsp.zea.axes.figure

        
def full_localization(roi, source_name=None, ignore_exception=False, 
            update=False, associator=None, tsmap_dir='tsmap_fail', tsfits=False, delta_ts_bad=10):
    """Run full localization workflow for one source.

    This function orchestrates:
    1. iterative localization via :class:`Localization`,
    2. optional source-position update when quality is good,
    3. optional association,
    4. optional TS-map plot generation, and
    5. moment-analysis fallback/overlay for suspicious fits.

    Parameters
    ----------
    roi : object
        ROI-like object exposing ``sources`` and ``tsmap_view``.
    source_name : str, optional
        Source name to localize.
    ignore_exception : bool, default=False
        If True, swallow localization exceptions and continue.
    update : bool, default=False
        If True, commit a good fitted position to the selected source.
    associator : object, optional
        Association helper passed to ``make_association`` when provided.
    tsmap_dir : str or None, default='tsmap_fail'
        Directory for TS-map outputs. If suffix is ``'fail'``, only problematic
        localizations are plotted.
    tsfits : bool, default=False
        Forwarded to plotting helper.
    delta_ts_bad : float, default=10
        Threshold for flagging suspect localizations.

    Returns
    -------
    object or None
        TS plot object when generated, otherwise ``None``.
    """
    import pylab as plt

    source = roi.sources.find_source(source_name)
    source.ellipsex = None  # In case a moment-analysis result already exists.
    tsp=None
    with roi.tsmap_view(source.name) as tsm:

        loc = Localization(tsm)
        try:
            if not loc.localize():
                print('Failed')
            if hasattr(loc, 'ellipse') and  (update or loc['qual']<1.0 and loc['a']<0.1):
                # Automatically update position if good fit.
                t = loc.ellipse
                prev = tsm.saved_skydir
                tsm.saved_skydir = SkyDir(t['ra'], t['dec'])
                print('updated position: %s --> %s' % (prev, tsm.saved_skydir))
            else:
                print('Failed localization')
        except Exception as msg:
            print('Localization of %s failed: %s' % (source.name, msg))
            if not ignore_exception:
                raise

        if not roi.quiet and hasattr(loc, 'niter') and loc.niter > 0:
            print('Localized %s: %d iterations, moved %.3f deg, deltaTS: %.1f' % \
                (source.name, loc.niter, loc.delt, loc.delta_ts))
            labels = 'ra dec a b ang qual'.split()
            print((len(labels)*'%10s') % tuple(labels))
            p = loc.qform.par[0:2] + loc.qform.par[3:7]
            print(len(p)*'%10.4f' % tuple(p))
        if associator is not None:
            try:
                make_association(source, loc.TSmap, associator, quiet=roi.quiet)
            except Exception as msg:
                print('Exception raised associating %s: %s' % (source.name, msg))
        
        if tsmap_dir is not None:
            if hasattr(loc, 'ellipse'):
                a, qual, delta_ts = loc.ellipse['a'], loc.ellipse['qual'], loc.delta_ts
                tsize = min(a * 15., 2.0)
                bad = a > 0.25 or qual > 5 or abs(delta_ts) > delta_ts_bad
                if bad:
                    print(
                        'Flagged as possibly bad: a=%.2f>0.25 or qual=%.1f>5 '
                        'or abs(delta_ts=%.1f)>%f:'
                        % (a, qual, delta_ts, delta_ts_bad)
                    )
            else:
                print('no localization')
                bad = True
                tsize = 2.0

            if tsmap_dir.endswith('fail') and not bad:
                return

            # Make a TS map and apply moment-analysis overlay when fit quality is poor.
            done = False
            while not done:
                try:
                    tsp = plotting.tsmap.plot(
                        loc,
                        source.name,
                        center=tsm.saved_skydir,
                        outdir=tsmap_dir,
                        catsig=0,
                        size=tsize,
                        pixelsize=tsize / 15,  # was 14: desire to have central pixel
                        assoc=source.__dict__.get('adict', None),  # either None or a dictionary
                        notitle=True,  # don't do title
                        markersize=10,
                        primary_markersize=12,
                        tsfits=tsfits,
                    )
                    zea = tsp.zea
                    wcs = zea.projector
                    tsmap = zea.image
                    vals = np.exp(-0.5 * tsmap**2).flatten()
                    peak_fraction = vals.max()/sum(vals)

                except Exception as msg:
                    print('Plot of %s failed: %s' % (source.name, msg))
                    return None
                if peak_fraction < 0.8:
                    done = True
                else:
                    # Scale is too large: reduce it and retry.
                    tsize /= 2.
                    print('peak fraction= %0.2f: setting size to %.2f' % (peak_fraction, tsize))
            ellipsex = moment_analysis(zea.image, wcs)
            source.ellipsex = list(ellipsex) + [tsize, peak_fraction]  # Copy to source for diagnostics.
            print('moment analysis ellipse:', np.array(ellipsex))
            rax, decx, ax, bx, phi = ellipsex
            tsp.overplot([rax, decx, ax, bx, phi], color='w', lw=2, ls='-', contours=[2.45])
            tsp.plot(SkyDir(rax, decx), color='w', symbol='o')
            filename = source.name.replace(' ', '_').replace('+', 'p')
            fout = os.path.join(tsmap_dir, '%s_tsmap.jpg' % filename)
            print('saving updated tsplot with moment analysis ellipse to %s...' % fout)
            sys.stdout.flush()
            plt.savefig(fout, bbox_inches='tight', padinches=0.2)  # Avoid clipping.
            
        return tsp
        
        
class Localization(object):
    """Fit a local quadratic approximation to source-position likelihood.

    The object presents a minimizer-friendly interface consumed by
    :class:`uw.like.quadform.Localize` and stores final ellipse diagnostics on
    success.
    """
    defaults = (
        ('tolerance',1e-4),
        ('verbose',False),
        ('update',False,"Update the source position after localization"),
        ('max_iteration',15,"Number of iterations"),
        #('bandfits',True,"Default use bandfits"),
        ('maxdist',1,"fail if try to move further than this"),
        ('seedpos', None, 'if set, start from this position instead of the source position'),
        ('factor', 1.0,  'factor to divide the likelihood for systmatics'),
        ('quiet', False, 'set to suppress output'),
    )

    @keyword_options.decorate(defaults)
    def __init__(self, tsm, **kwargs):
        """Create localization state for a selected TS-map source.

        Parameters
        ----------
        tsm : object
            TS-map view object with a selected source. Callable as ``tsm(skydir)``
            and expected to return a TS-like value (twice log-likelihood ratio)
            relative to the nominal source position.
        **kwargs
            Keyword overrides for :attr:`defaults`.
        """
        keyword_options.process(self, kwargs)
        
        self.tsm = tsm # roistat.tsmap_view(source_name)
        self.maxlike = self.log_like()
        self.skydir  = self.tsm.skydir
        if self.seedpos is not None: 
            if not isinstance(self.seedpos, SkyDir):
                self.seedpos = SkyDir(*self.seedpos)
            self.skydir = self.seedpos
        self.name = self.tsm.source.name
        if self.factor!=1.0: 
            print('Applying factor {:.2f}'.format(self.factor))
    
    def log_like(self, skydir=None):
        """Return log-likelihood proxy at ``skydir``.

        The TS-map callable returns approximately ``2*logL`` differences; this
        method converts by dividing by 2 for compatibility with legacy math.
        """
        return self.tsm(skydir)/2
   
    def TSmap(self, skydir):
        """Return TS difference at ``skydir`` relative to nominal maximum."""
        val= 2*(self.log_like(skydir)-self.maxlike)
        return val / self.factor

    # Minimizer interface expected by quadform.Localize.
    def get_parameters(self):
        return np.array([self.tsm.skydir.ra(), self.tsm.skydir.dec()])
    
    def set_parameters(self, par):
        self.skydir = SkyDir(par[0],par[1])
        self.tsm.skydir = self.tsm.set_dir(self.skydir)
        
    def __call__(self, par):
        # Negative sign because minimizers perform minimization.
        return -self.TSmap(SkyDir(par[0],par[1]))
    
    def reset(self):
        """Restore source/TS-map state modified during localization."""
        self.tsm.reset()
      
    def dir(self):
        return self.skydir

    def errorCircle(self):
        """Return initial guess for isotropic error-circle radius in degrees."""
        return 0.05 #initial guess

    def spatialLikelihood(self, sd):  # Negative sign kept for legacy fitter logic.
        """Legacy-sign convention adapter used by the historical fitter path."""
        return -self.log_like(sd)
        
    def localize(self):
        """Localize source position with an elliptic likelihood approximation.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on an early quality/motion failure.
        """
        #roi    = self.roi
        #bandfits = self.bandfits
        verbose = self.verbose
        tolerance = self.tolerance
        l = quadform.Localize(self, verbose=verbose)
        ld = l.dir

        ll0 = self.spatialLikelihood(self.skydir)

        if not self.quiet:
            fmt ='Localizing source %s, tolerance=%.1e...\n\t'+7*'%10s'
            tup = (self.name, tolerance,)+tuple('moved delta ra     dec    a     b  qual'.split())
            print(fmt % tup)
            print(('\t' + 4 * '%10.4f') % (0, 0, self.skydir.ra(), self.skydir.dec()))
            diff = np.degrees(l.dir.difference(self.skydir))
            print(('\t' + 7 * '%10.4f') % (diff, diff, l.par[0], l.par[1], l.par[3], l.par[4], l.par[6]))
        
        old_sigma = 1.0
        for i in range(self.max_iteration):
            try:
                l.fit(update=True)
            except:
                #raise
                l.recenter()
                if not self.quiet:
                    print('trying a recenter...')
                continue
            diff = np.degrees(l.dir.difference(ld))
            delt = np.degrees(l.dir.difference(self.skydir))
            sigma = l.par[3]
            if not self.quiet:
                print(('\t' + 7 * '%10.4f') % (diff, delt, l.par[0], l.par[1], l.par[3], l.par[4], l.par[6]))
            if delt > self.maxdist:
                l.par[6] = 99  # Flag very bad quality and reset position.
                l.sigma = 1.0
                l.par[0] = self.skydir.ra()
                l.par[1] = self.skydir.dec()
                if not self.quiet:
                    print('\t -attempt to move beyond maxdist=%.1f' % self.maxdist)
                break
                #self.tsm.source.ellipse = self.qform.par[0:2]+self.qform.par[3:7]
                return False # hope this does not screw things up
                #raise Exception('localize failure: -attempt to move beyond maxdist=%.1f' % self.maxdist)
            if (diff < tolerance) and (abs(sigma - old_sigma) < tolerance):
                break  # converge
            ld = l.dir
            old_sigma = sigma

        self.qform    = l
        self.lsigma   = l.sigma
        q = l.par
        self.ellipse = dict(ra=float(q[0]), dec=float(q[1]),
                a=float(q[3]), b=float(q[4]),
                ang=float(q[5]), qual=float(q[6]),
                lsigma = l.sigma)

        ll1 = self.spatialLikelihood(l.dir)
        if not self.quiet:
            print('TS change: %.2f' % (2 * (ll0 - ll1)))

        #roi.delta_loc_logl = (ll0 - ll1)
        # Keep these diagnostics even if fit quality is poor.
        delt = np.degrees(l.dir.difference(self.skydir))
        self.delta_ts = 2 * (ll0 - ll1)
        self.delt = delt
        self.niter = i
        # Persist ellipse parameters on the source for downstream consumers.
        self.tsm.source.ellipse = self.qform.par[0:2] + self.qform.par[3:7] + [self.delta_ts]
        return True  # success
        
    def summary(self):
        """Print a concise post-fit localization summary if available."""
        if hasattr(self, 'niter') and self.niter > 0:
            print('Localized %s: %d iterations, moved %.3f deg, deltaTS: %.1f' % \
                (self.name, self.niter, self.delt, self.delta_ts))
            labels = 'ra dec a b ang qual'.split()
            print((len(labels)*'%10s') % tuple(labels))
            p = self.qform.par[0:2] + self.qform.par[3:7]
            print(len(p)*'%10.4f' % tuple(p))


       
def localize_all(roi, ignore_exception=True, **kwargs):
    """Batch-localize eligible point sources in an ROI.

    Source selection defaults to variable/free point sources above a TS
    threshold. Optional filters and output controls are supplied via ``kwargs``.

    Recognized ``kwargs``
    ---------------------
    tsmin : float, default=10
        Minimum TS to include a source.
    prefix : str, optional
        Restrict to source names starting with this prefix.
    source_name : str, optional
        Localize only this source.
    update : bool, default=False
        Whether to update source positions in place.
    tsmap_dir : str or None, optional
        Directory for TS-map outputs.
    associator : object, optional
        Association helper object.
    tsfits : bool, default=True
        Whether to also write TS FITS products.
    """
    tsmin = kwargs.pop('tsmin', 10)
    prefix = kwargs.pop('prefix', None)
    source_name = kwargs.pop('source_name', None)
    update = kwargs.pop('update', False)

    def filt(s):
        ok = s.skydir is not None \
            and isinstance(s, sources.PointSource) \
            and np.any(s.spectral_model.free)
        if not ok:
            return False
        if not hasattr(s, 'ts'):
            s.ts = roi.TS(s.name)
        return ok and s.ts > tsmin

    if source_name is not None:
        vpsources = [roi.get_source(source_name)]
    else:
        vpsources = list(filter(filt, roi.sources))
    tsmap_dir = kwargs.pop('tsmap_dir', None)
    if tsmap_dir is not None:
        if tsmap_dir[0] == '$':
            tsmap_dir = os.path.expandvars(tsmap_dir)
        if not os.path.exists(tsmap_dir):
            os.makedirs(tsmap_dir)
    associator = kwargs.pop('associator', None)
    tsfits = kwargs.pop('tsfits', True)
    if len(list(kwargs.keys())) > 0:
        print('Warning: unrecognized args to localize_all: %s' % kwargs)
    initw = roi.log_like()
    
    for source in vpsources:
        if prefix is not None and not source.name.startswith(prefix):
            continue
        
        full_localization(roi, source.name, ignore_exception=ignore_exception,
            update=update, associator=associator, tsmap_dir=tsmap_dir, tsfits=tsfits)
        

    curw = roi.log_like()
    if abs(initw - curw) > 1.0 and not update:
        print(
            'localize_all: unexpected change in roi state after localization, '
            'from %.1f to %.1f (%+.1f)'
            % (initw, curw, curw - initw)
        )
        return False
    else:
        return True

class TS_function(object):
    """Context-manager helper exposing a temporary TS function.

    Example
    -------
    ``with TS_function(roi, 'src') as tsfun: tsfun(skydir)``
    """
    def __init__(self, roi, source_name):
        self.loc = Localization(roi, source_name)

    def __enter__(self):
        """Return callable TS-map function for use inside the context."""
        return self.loc.TSmap

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Reset localization state on context exit."""
        self.loc.reset()
    