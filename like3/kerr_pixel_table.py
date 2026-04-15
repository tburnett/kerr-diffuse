
"""
Load and export Kerr-format pixel tables stored as .npz / .pickle file pairs.

The primary public class is `KerrPixelTable`, a dict-subclass keyed by
``(psf_index, energy_key)`` tuples whose values are `Band` objects.  Each
`Band` carries the per-band pixel indices and any additional data columns
(e.g. photon counts, weights) that were present in the source .npz file.

FITS export is handled by the internal `_ToFITS` helper and exposed via
`KerrPixelTable.to_fits`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from astropy_healpix import HEALPix 
from pathlib import Path
from typing import Any, cast

_energy_index = lambda energy: (np.log10(energy) * 4 - 8).astype(int)


def _event_type_to_int(value: str | int | np.integer[Any]) -> int:
    """Normalize event-type labels/codes to the FITS integer convention."""
    if isinstance(value, str):
        label = value.strip().upper()
        if label == 'FRONT':
            return 0
        if label == 'BACK':
            return 1
        if label.startswith('PSF'):
            return int(label[3:]) + 2
        if label.isdigit():
            return int(label)
    if isinstance(value, (int, np.integer)):
        ivalue = int(value)
        if 0 <= ivalue <= 5:
            return ivalue
    raise ValueError(f'Unsupported event type: {value!r}')

class KerrPixelTable(dict):
    """Sparse HEALPix pixel table loaded from a Kerr-format .npz/.pickle pair.

    Inherits from ``dict``; keys are ``(psf_index, energy_key)`` tuples and
    values are `Band` instances.  All data columns from the .npz file are also
    accessible as attributes on the table and on each individual `Band`.

    Parameters
    ----------
    *pars : str or Path
        One or more file path stems.  The first is the primary file; any
        additional stems are appended (columns merged in order).
    ring : bool, optional
        If ``True``, convert NESTED pixel indices to RING ordering after
        loading.  Default is ``False`` (NESTED).

    Attributes
    ----------
    name : str or Path
        Stem of the primary input file.
    columns : list[str]
        Names of data columns loaded from the .npz file(s).
    indices : np.ndarray
        Flat array of all pixel indices across all bands.
    ring : bool
        ``True`` if pixel indices are in RING ordering.
    meta_df : pandas.DataFrame
        Per-band metadata with columns
        ``event_type emin emax nside nocc occupancy``.
    """

    class Band(HEALPix):
        """Single energy/event-type channel of a `KerrPixelTable`.

        Extends `~astropy_healpix.HEALPix` with energy range, event-type, and
        per-band data column attributes set dynamically by the parent table.

        Attributes
        ----------
        psf_name : str
            Event-type label (e.g. ``'FRONT'``, ``'PSF2'``).
        e0, e1 : float
            Lower and upper energy bounds in MeV.
        nocc : int
            Number of occupied pixels in this band.
        event_type : int
            Integer event-type code (FRONT=0, BACK=1, PSF0-3=2-5).
        key : tuple[int, int]
            ``(psf_index, energy_key)`` used as dict key in the parent table.
        energy : str
            Geometric-mean energy formatted as ``'X.XX GeV'``.
        slice : slice or None
            Slice into the parent table's flat arrays for this band's rows.
        """

        def __init__(self, meta: tuple[str, float, float, int, int]) -> None:
            """Construct a Band from a metadata tuple.

            Parameters
            ----------
            meta : tuple
                ``(event_type_label, emin_MeV, emax_MeV, nside, nocc)``
            """

            self.psf_name, self.e0, self.e1, nside, self.nocc = meta
            self.event_type = _event_type_to_int(self.psf_name)
            self.slice: slice | None = None
  
            psf_index = self.event_type if self.event_type < 2 else self.event_type - 2
            self.key = (int(psf_index), _energy_index(self.e0))
            self.energy = f'{np.sqrt(self.e0 * self.e1) * 1e-3:.2f} GeV'
            super().__init__(nside=nside, frame='galactic', order='nested')

        def __repr__(self) -> str:
            return f"Band{self.key}: {self.psf_name}@{self.energy} nside {self.nside} occ {self.nocc/(12*self.nside**2):.3f}"
        
            
    def __init__(self, *pars: str | Path, toring: bool = False) -> None:
        """Load a Kerr pixel table from .npz/.pickle file pair(s).

        Parameters
        ----------
        *pars : str or Path
            Path stem(s).  First is the primary file; extras are appended.
        ring : bool, optional
            Convert pixel indices from NESTED to RING ordering. Default False.

        Raises
        ------
        FileNotFoundError
            If any .npz file does not exist.
        KeyError
            If the required ``'indices'`` column is absent.
        """

        import pickle
        root = pars[0]
        toappend = pars[1:] if len(pars) > 1 else []
        meta_file =  Path(root).with_suffix('.pickle')
        
        self.name = root

        npzfiles = [root] + list(toappend)

        self.columns = []
        for npzfile in npzfiles:
            file = Path(npzfile).with_suffix('.npz')
            if not file.exists():
                raise FileNotFoundError(f"File {file} does not exist")

            with np.load(file) as data:
                self.columns += list(data.keys())
                for key, value in data.items():
                    setattr(self, key, value)
            print(f"Loaded columns {list(data.keys())} from {file}")

        if 'indices' not in self.columns:
            raise KeyError(f"Required 'indices' column not found in {npzfiles}")
        self.indices: np.ndarray  # set dynamically from npz 'indices' column via setattr

        # create the band objects from the metadata
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)

        offset = 0
        for i, m in enumerate(meta):
            b = self.Band(m, )
            self[b.key] = b
            nocc = int(m[-1])
            sl = slice(offset, offset + nocc)
            b.slice = sl
            for column in self.columns:
                setattr(b, column, getattr(self, column)[sl])
            offset += nocc

        # optionally convert NESTED pixel indices to RING
        if toring:
            import healpy as hp
            for b in self.values():
                b.indices = hp.nest2ring(b.nside, b.indices.astype(int)).astype(np.uint32)
            self.indices = np.concatenate([b.indices for b in self.values()])

        self.ring = toring  # flag for the FITS fits
        self.meta_df = pd.DataFrame(meta, columns='event_type emin emax nside nocc'.split())
        self.meta_df['occupancy'] = (self.meta_df.nocc / (12 * self.meta_df.nside**2)).round(3)

    def to_fits(self, filename: str | Path) -> None:
        """Write the pixel table to a FITS file using the Kerr layout.

        Parameters
        ----------
        filename : str or Path
            Output FITS file path.
        """
        tf = _ToFITS(self)
        hdul = fits.HDUList([fits.PrimaryHDU(), tf.skymap_hdu(), tf.band_hdu()])
        hdul.writeto(filename, overwrite=True)
        print(f"Wrote FITS file to {filename}")



from astropy.io import fits

class _ToFITS:
    """Serialize `PixelTable` content into the FITS layout used by Kerr files.

    The generated FITS file contains a sparse `SKYMAP` table holding pixel
    counts and a `BANDS` table describing the energy/event-type metadata for
    each channel.
    """
    def __init__(self, kerrmodel: KerrPixelTable) -> None:
        """Wrap a PixelTable and expose FITS export utilities.

        Parameters
        ----------
        kerrmodel : PixelTable
            Pixel table instance to serialize.
        Attributes
        -----------
        pixeltable : PixelTable
            Reference to source pixel table.

        """
        self.pixeltable = kerrmodel
       
    def __repr__(self) -> str:
        return f'KerrDataFile for {self.pixeltable}'
    

    def skymap_hdu(self) -> fits.BinTableHDU:
        """Create sparse SKYMAP HDU with PIX/CHANNEL/VALUE columns.

        Returns
        -------
        astropy.io.fits.BinTableHDU
            Binary table with columns:
                - PIX (uint32): NESTED pixel indices
                - CHANNEL (uint32): Band/channel index into BANDS HDU
                - VALUE (uint32): Photon counts per pixel
            Includes HEALPix metadata in header (ORDERING, COORDSYS, etc.).
        """
        km = self.pixeltable

        nocc = km.meta_df.nocc.to_numpy()
        # channels: index of BANDS entry for each pixel
        chn = np.repeat(np.arange(len(nocc), dtype=np.uint32), nocc.astype(np.uint32))

        cols = [
            fits.Column(name='PIX', format='J',    array=km.indices),
            fits.Column(name='CHANNEL', format='I',array=chn),
        ]
        
        for colname in km.columns:
            # all other columns besides 'indices' and 'photons' are stored as single-precision
            # floats in the FITS table
            if colname not in ('indices', 'photons'):
                cols.append(fits.Column(name=colname.upper(), format='E', array=getattr(km, colname)))
                
        hdu=fits.BinTableHDU.from_columns(cols, name='SKYMAP')
        hdu.header.update(
            PIXTYPE='HEALPIX',
            INDXSCHM='SPARSE',
            ORDERING='RING' if self.pixeltable.ring else 'NESTED',
            COORDSYS='GAL',
            BANDSHDU='BANDS',
            AXCOLS='E_MIN,E_MAX',
            )
        return hdu  

    def band_hdu(self, version: int = 5) -> fits.BinTableHDU:
        """Create BANDS HDU containing NSIDE/energy/event-type metadata.

        Parameters
        ----------
        version : int, optional
            FITS version number stored in HDU header. Default is 5.

        Returns
        -------
        astropy.io.fits.BinTableHDU
            Binary table with columns:
                - NSIDE (int64): HEALPix nside per band
                - E_MIN (float64): Minimum energy in keV
                - E_MAX (float64): Maximum energy in keV
                - EVENT_TYPE (int64): Event type code
        """
        df = self.pixeltable.meta_df
        band_cols = [
            fits.Column(name='NSIDE', format='J', array=df.nside),
            fits.Column(name='E_MIN', format='D', array=df.emin*1e+3, unit='keV'),
            fits.Column(name='E_MAX', format='D', array=df.emax*1e+3, unit='keV'),
            fits.Column(name='EVENT_TYPE', format='J', array=df.event_type.apply(_event_type_to_int)),
        ]
        if 'exposure' in df.columns:
            band_cols.append(fits.Column(name='EXPOSURE', format='D', array=np.asarray(df.exposure, dtype=float)))
        hdu=fits.BinTableHDU.from_columns(band_cols, name='BANDS')
        hdu.header.update(VERSION=version)
        return hdu

    def writeto(self, filename: str | Path, overwrite: bool = True) -> None:
        """Write FITS file with PrimaryHDU, SKYMAP, and BANDS extensions.

        Parameters
        ----------
        filename : str or Path
            Output FITS filename.
        overwrite : bool, optional
            Overwrite existing file. Default is True.

        Prints
        ------
        Status message indicating successful write and ring/nested ordering.
        """

        hdus=[fits.PrimaryHDU(), 
              self.skymap_hdu(), 
              self.band_hdu()] 
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
        print(f'wrote file {filename}' + (f' (ring={self.pixeltable.ring})' if self.pixeltable.ring else ''))

    @classmethod
    def readfrom(cls, filename: str | Path, kerrmodel: KerrPixelTable) -> _ToFITS:
        """Open and print a FITS file summary, then return a wrapper instance."""
        hdus = fits.open(filename)
        print(f'Read KerrDataFile from {filename}:')
        hdus.info()
        return cls(kerrmodel)

    
    # @classmethod
    # def to_fits(cls, kerrfile: str | Path, fitsfile: str | Path, *, ring: bool = False, overwrite: bool = True) -> None:
    #     """Translate a Kerr `.npz/.pickle` pair into FITS representation.

    #     Parameters
    #     ----------
    #     kerrfile : str or Path
    #         Path stem for input .npz/.pickle files.
    #     fitsfile : str or Path
    #         Output FITS filename.
    #     ring : bool, optional
    #         If True, convert pixels to RING ordering before export. Default is False.
    #     overwrite : bool, optional
    #         Overwrite existing FITS file. Default is True.

    #     Returns
    #     -------
    #     None

    #     Notes
    #     -----
    #     This is a convenience classmethod that loads the pixel table and
    #     calls writeto() in a single operation.
    #     """
    #     km = PixelTable(kerrfile, ring=ring )
    #     cls(km).writeto(fitsfile, overwrite=overwrite)
