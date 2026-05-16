
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
from typing import Any, Sequence, cast

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
            self.key = (int(psf_index), int(_energy_index(self.e0)))
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


class KerrPtsrcInfo:

    class PtBand:
        """Pixels associated with one or more sources in a specific band."""

        def __init__(self, source_indices: Sequence[int], band_index: int, meta_row: pd.Series, table: pd.DataFrame) -> None:
            self.source_indices = [int(v) for v in source_indices]
            # Backward-compatible alias for single-source PtBand instances.
            self.source_index = self.source_indices[0] if len(self.source_indices) == 1 else None
            self.band_index = int(band_index)
            self.meta = meta_row
            self.table = table

        def __repr__(self) -> str:
            src_repr = self.source_index if self.source_index is not None else self.source_indices
            return (
                f"PtBand(source={src_repr}, band={self.band_index}, "
                f"pixels={len(self.table)})"
            )
    
    def __init__(self, root = 'files/kerr/toby_v5'):    
        import pickle    
        
        r, v = root.split('_')
        if not Path((ptsrc_file:=r+'_ptsrcs_' + v + '.npz')).exists():
            raise FileNotFoundError(f"ptsrcs file {ptsrc_file} not found")
        meta_file =  Path(root).with_suffix('.pickle')
        if not meta_file.exists():
            raise FileNotFoundError(f"meta file {meta_file} not found ")
        
        self.columns = []
        with np.load(ptsrc_file) as data:
            self.columns += list(data.keys())
            for key, value in data.items():
                setattr(self, key, value)
        print(f"Loaded columns {list(data.keys())} from {ptsrc_file}")

        npix = sum(self.entries_per_band)
        print(f"Total number of pixels: {npix:,d}")
        if npix != len(self.nameidx):
            raise ValueError(f"Total number of pixels {npix} does not match length of nameidx {len(self.nameidx)}")
            
        # read the meta file and make it a dataframe
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        self.meta = pd.DataFrame(meta, columns='event_type emin emax nside entries'.split())
        # replace "entries" with the actual number of entries per band
        self.meta['entries'] = self.entries_per_band
        self.meta['energy'] = np.sqrt(self.meta['emin'] * self.meta['emax']).astype(int)
        self.meta = self.meta.astype({'event_type': 'category', 'emin':int, 'emax':int, 'nside':int, 'entries':int})

        # make a slice column that gives the slice of the pixel table for each band
        sl = []
        offset = 0
        for i, row in self.meta.iterrows():
            sl.append(slice(offset, offset + row.entries))
            offset += row.entries
        self.meta['slice'] = sl
        print(f"Loaded meta data describing {len(self.meta)} bands from {meta_file}")

    def create_ptband(self, source_index: int | Sequence[int], band: int | pd.Series) -> PtBand:
        """Collect per-pixel counts for one or more sources in one band.

        Parameters
        ----------
        source_index : int or sequence of int
            Source index (or source indices) matching entries in
            ``self.nameidx``.
        band : int or pandas.Series
            Band selector. Either an integer row index into ``self.meta`` or
            a row from ``self.meta``.

        Returns
        -------
        PtBand
            For a single source, object holding ``pixel_id`` and ``counts``.
            For multiple sources, object holding ``pixel_id`` and one
            ``counts_<source_index>`` column per source.
            In both cases, ``other_counts`` gives the sum from all non-
            requested sources for the same pixels.
        """
        if isinstance(band, pd.Series):
            if 'slice' not in band:
                raise KeyError("Band row must contain a 'slice' entry")
            meta_row = band
            band_index = int(meta_row.name) if isinstance(meta_row.name, (int, np.integer)) else -1
        else:
            band_index = int(band)
            if band_index < 0 or band_index >= len(self.meta):
                raise IndexError(f"Band index {band_index} out of range for {len(self.meta)} bands")
            meta_row = self.meta.iloc[band_index]

        if isinstance(source_index, (int, np.integer)):
            source_indices = [int(source_index)]
            single_source = True
        else:
            source_indices = [int(v) for v in source_index]
            if len(source_indices) == 0:
                raise ValueError("source_index sequence must not be empty")
            # Remove duplicates while preserving caller-provided order.
            source_indices = list(dict.fromkeys(source_indices))
            single_source = len(source_indices) == 1

        sl = meta_row['slice']
        band_pixels = self.healpixidx[sl]
        band_counts = self.pscounts[sl]#.round(1)  # round to 0.1 photons for cleaner output
        band_nameidx = self.nameidx[sl]
        total_by_pixel = pd.DataFrame(
            {
                'pixel_id': band_pixels,
                '_total_counts': band_counts,
            }
        ).groupby('pixel_id', as_index=False)['_total_counts'].sum()

        if single_source:
            src_index = source_indices[0]
            mask = band_nameidx == src_index
            table = pd.DataFrame(
                {
                    'pixel_id': band_pixels[mask],
                    'counts': band_counts[mask],
                }
            )
            table = table.groupby('pixel_id', as_index=False)['counts'].sum()
            requested_sum = table['counts']
        else:
            selected = np.isin(band_nameidx, np.asarray(source_indices, dtype=band_nameidx.dtype))
            pixel_id = np.asarray(np.unique(band_pixels[selected]))
            table = pd.DataFrame({'pixel_id': pixel_id})
            for src_index in source_indices:
                mask = band_nameidx == src_index
                col_name = f'counts_{src_index}'
                src_table = pd.DataFrame(
                    {
                        'pixel_id': band_pixels[mask],
                        col_name: band_counts[mask],
                    }
                ).groupby('pixel_id', as_index=False)[col_name].sum()
                table = table.merge(src_table, on='pixel_id', how='left')
            fill_cols = [c for c in table.columns if c != 'pixel_id']
            table[fill_cols] = table[fill_cols].fillna(0.0)
            requested_sum = table[fill_cols].sum(axis=1)

        table = table.merge(total_by_pixel, on='pixel_id', how='left')
        table['_total_counts'] = table['_total_counts'].fillna(0.0)
        table['other_counts'] = np.maximum(table['_total_counts'] - requested_sum, 0.0)
        table = table.drop(columns=['_total_counts'])

        return self.PtBand(source_indices, band_index, meta_row, table)

    @classmethod
    def test_demo(
        cls,
        root: str = 'files/kerr/toby_v5',
        band_index: int = 30,
        n_sources: int = 3,
        verbose: bool = True,
    ) -> tuple['KerrPtsrcInfo', 'KerrPtsrcInfo.PtBand', 'KerrPtsrcInfo.PtBand']:
        """Run a demonstration test of the PtBand creation for one band and multiple sources.

        Parameters
        ----------
        root : str, optional
            Root file stem passed to ``KerrPtsrcInfo``.
        band_index : int, optional
            Band row index in ``self.meta`` to test.
        n_sources : int, optional
            Number of sources to include in the multi-source PtBand test.
        verbose : bool, optional
            If True, print a short summary matching the notebook output style.

        Returns
        -------
        tuple
            ``(ptsrc_info, ptband_multi, ptband_single)``.
        """
        if n_sources < 2:
            raise ValueError('n_sources must be at least 2')

        ptsrc_info = cls(root)
        band_row = ptsrc_info.meta.iloc[int(band_index)]
        sl = band_row['slice']

        if int(band_row['entries']) == 0:
            raise RuntimeError(f'Band {band_index} has no entries to test')

        source_indices = [int(v) for v in np.unique(ptsrc_info.nameidx[sl])[:n_sources]]
        if len(source_indices) < 2:
            raise RuntimeError('Need at least two sources in this band to test multi-source PtBand')

        ptband = ptsrc_info.create_ptband(source_indices, int(band_index))

        expected_cols = ['pixel_id'] + [f'counts_{s}' for s in source_indices] + ['other_counts']
        if list(ptband.table.columns) != expected_cols:
            raise AssertionError(
                f'Unexpected columns: {list(ptband.table.columns)} != {expected_cols}'
            )
        if len(ptband.table) == 0:
            raise AssertionError('Multi-source PtBand table is empty')
        if not (ptband.table[expected_cols[1:]] >= 0).all().all():
            raise AssertionError('Multi-source PtBand has negative values in count columns')

        ptband_single = ptsrc_info.create_ptband(source_indices[0], int(band_index))
        expected_single = ['pixel_id', 'counts', 'other_counts']
        if list(ptband_single.table.columns) != expected_single:
            raise AssertionError(
                f'Unexpected single-source columns: {list(ptband_single.table.columns)} != {expected_single}'
            )

        if verbose:
            print(ptband)
            print('columns:', list(ptband.table.columns))
            print('rows:', len(ptband.table))
            print(ptband.table.head())

        return ptsrc_info, ptband, ptband_single
