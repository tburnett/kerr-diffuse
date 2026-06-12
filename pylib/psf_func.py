""" PSF functions management"""

import sys
import types
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylib.binned_data import BandList
from utilities.ipynb_docgen import show, show_fig


def _ensure_pandas_indexes_compat():
    """Provide a lightweight shim for legacy pickles expecting pandas.indexes."""
    if 'pandas.indexes' in sys.modules:
        return
    try:
        from pandas import (
            CategoricalIndex,
            DatetimeIndex,
            Index,
            MultiIndex,
            PeriodIndex,
            RangeIndex,
            TimedeltaIndex,
        )
    except Exception:
        return

    pkg = types.ModuleType('pandas.indexes')
    pkg.__path__ = []

    def _new_Index(cls, values=None, **kwargs):
        name = kwargs.get('name', None)
        data = values
        if isinstance(values, dict):
            name = values.get('name', name)
            data = values.get('data', values.get('_data', values.get('values', None)))
        if cls is RangeIndex and isinstance(values, dict):
            return RangeIndex(
                start=values.get('start', 0),
                stop=values.get('stop', 0),
                step=values.get('step', 1),
                name=name,
            )
        try:
            return cls(data, name=name)
        except Exception:
            try:
                return Index(data if data is not None else [], name=name)
            except Exception:
                return Index([])

    module_defs = {
        'pandas.indexes.base': {'Index': Index, '_new_Index': _new_Index},
        'pandas.indexes.multi': {'MultiIndex': MultiIndex},
        'pandas.indexes.range': {'RangeIndex': RangeIndex},
        'pandas.indexes.datetimes': {'DatetimeIndex': DatetimeIndex},
        'pandas.indexes.timedeltas': {'TimedeltaIndex': TimedeltaIndex},
        'pandas.indexes.period': {'PeriodIndex': PeriodIndex},
        'pandas.indexes.category': {'CategoricalIndex': CategoricalIndex},
    }

    for mod_name, attrs in module_defs.items():
        mod = types.ModuleType(mod_name)
        for attr, value in attrs.items():
            setattr(mod, attr, value)
        sys.modules[mod_name] = mod
        setattr(pkg, mod_name.rsplit('.', 1)[-1], mod)

    pkg.Index = Index
    pkg.MultiIndex = MultiIndex
    pkg.RangeIndex = RangeIndex
    pkg.DatetimeIndex = DatetimeIndex
    pkg.TimedeltaIndex = TimedeltaIndex
    pkg.PeriodIndex = PeriodIndex
    pkg.CategoricalIndex = CategoricalIndex
    sys.modules['pandas.indexes'] = pkg


def _read_pickle_compat(path):
    """Load a pickle with a latin1 fallback for legacy python2-encoded files."""
    try:
        return pd.read_pickle(path)
    except Exception as first_error:
        if 'ascii' not in str(first_error).lower() or 'decode' not in str(first_error).lower():
            raise
        with open(path, 'rb') as stream:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message=r'dtype\(\): align should be passed as Python or NumPy boolean',
                    category=Warning,
                )
                return pickle.load(stream, encoding='latin1')


class PSFlist(list):
    """ Manage a list of PSF functions"""

    @classmethod
    def _event_type_to_int(cls, event_type):
        if isinstance(event_type, str):
            label = event_type.strip().upper()
            aliases = {'FB': 0}
            if label in aliases:
                return aliases[label]
            if label in cls.PSF.et_name:
                return cls.PSF.et_name.index(label)
            if label.isdigit():
                return int(label)
            raise ValueError(f'Unknown event type label: {event_type!r}')
        return int(event_type)

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
            self['energy']= e =round(table.energy,0)
            self['event_type'] = table.event_type

            # a kluge to make delta E 1
            self['e0'] = e-0.5
            self['e1'] = e+0.5
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

    def __init__(self, event_type=None, table_path='files/loc'):
        from pathlib import Path
        path = Path(table_path)
        _ensure_pandas_indexes_compat()
        if path.is_dir():
            et = None if event_type is None else int(event_type)
            load_fb  = et is None or et < 2
            load_psf = et is None or et >= 2
            frames = []
            for fname, do_load in [('fb_psf_table.pkl', load_fb),
                                   ('psf_psf_table.pkl', load_psf)]:
                if not do_load:
                    continue
                try:
                    frames.append(_read_pickle_compat(path / fname))
                except Exception as msg:
                    print(msg, file=sys.stderr)
            if not frames:
                return
            psf_table = pd.concat(frames, ignore_index=True)
        else:
            try:
                psf_table = _read_pickle_compat(path)
            except Exception as msg:
                print(msg, file=sys.stderr)
                return
        for which, table in enumerate(psf_table.itertuples()):
            t = self.PSF(table, which)
            if event_type is None or t.event_type == event_type:
                self.append(t)

    def get_psf(self, event_type, energy, tol_mev=1.0):
        """Return the nearest-energy PSF for *event_type* within *tol_mev* MeV."""
        et = self._event_type_to_int(event_type)
        target_energy = float(energy)
        tol_mev = float(tol_mev)

        matches = [psf for psf in self if int(psf.event_type) == et]
        if not matches:
            raise ValueError(f'No PSF entries for event_type={event_type!r} ({et})')

        nearest = min(matches, key=lambda psf: abs(float(psf.energy) - target_energy))
        delta_mev = abs(float(nearest.energy) - target_energy)
        if delta_mev > tol_mev:
            raise ValueError(
                f'No PSF within {tol_mev:.3g} MeV for event_type={event_type!r} '
                f'and energy={target_energy:.3f} MeV (nearest: {nearest.energy:.3f} MeV)'
            )
        return nearest

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

