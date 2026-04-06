"""Exposure map utilities for HEALPix workflows."""

from __future__ import annotations

import os
import numpy as np
from astropy.io import fits
from astropy_healpix import HEALPix, npix_to_nside, nside_to_pixel_area
from typing import Any, cast

__all__ = [
    "EffectiveAreaIRF",
    "ExposureMap",
    "make_aeff_costheta_function",
    "make_exposure_map_healpix",
    "build_pixel_table_exposure",
]


class EffectiveAreaIRF(object):
    """Manage effective area from CALDB FITS files for FB or PSF partitions."""

    VALID_PARTITIONS = ("FB", "PSF0", "PSF1", "PSF2", "PSF3")

    def __init__(self,
                 irf="P8R2_SOURCE_V6",
                 file_path=None,
                 CALDB=None,
                 partition="FB",
                 use_phidep=False):
        """Initialize an effective-area reader for a given IRF and event partition.

        Parameters
        ----------
        irf : str
            IRF name, e.g. ``"P8R2_SOURCE_V6"`` or ``"P8R3_SOURCE_V3"``.
        file_path : str or None
            Directory containing the CALDB ``bcf/ea`` effective-area FITS files.
            If *None*, derived from the ``CALDB`` environment variable.
        CALDB : str or None
            Root of the CALDB tree.  Overrides the ``$CALDB`` environment
            variable when provided.
        partition : str
            Event partition: ``"FB"`` (front+back combined), ``"PSF0"``,
            ``"PSF1"``, ``"PSF2"``, or ``"PSF3"``.
        use_phidep : bool
            If *True*, also load the azimuthal phi-dependence correction tables.
        """
        if partition not in self.VALID_PARTITIONS:
            raise ValueError(
                f"partition must be one of {self.VALID_PARTITIONS}, got {partition!r}"
            )

        if CALDB is None:
            CALDB = os.environ.get("CALDB", None)
        self.CALDB = CALDB
        assert (file_path or self.CALDB), "No path given for effective area"

        if not file_path:
            caldb = str(self.CALDB)
            if os.path.exists(f"{caldb}/data"):
                caldb += "/data"
            self.CALDB = caldb
            file_path = f"{caldb}/glast/lat/bcf/ea"

        self.irf = irf
        self.partition = partition
        self.file_path = os.path.expandvars(str(file_path))
        self._psf_from_combined_file = False
        self._partition_cache: dict[str, EffectiveAreaIRF] = {partition: self}

        self.aeff_file = self._resolve_aeff_file()

        self._read_aeff(self.aeff_file, self.aeff_file)
        if use_phidep:
            self._read_phi(self.aeff_file, self.aeff_file)

    def _resolve_aeff_file(self):
        """Locate the CALDB FITS file that contains the effective-area tables.

        Tries a partition-specific file first; for PSF partitions falls back to
        the combined ``aeff_<IRF>_PSF.fits`` file.  Raises ``FileNotFoundError``
        when no candidate is found.

        Returns
        -------
        str
            Absolute path to the effective-area FITS file.
        """
        if self.partition == "FB":
            filename = os.path.join(self.file_path, f"aeff_{self.irf}_FB.fits")
            if os.path.exists(filename):
                return filename
            raise FileNotFoundError(f"Effective area file {filename} not found")

        partition_file = os.path.join(self.file_path, f"aeff_{self.irf}_{self.partition}_FB.fits")
        if os.path.exists(partition_file):
            self._psf_from_combined_file = False
            return partition_file

        combined_file = os.path.join(self.file_path, f"aeff_{self.irf}_PSF.fits")
        if os.path.exists(combined_file):
            self._psf_from_combined_file = True
            return combined_file

        raise FileNotFoundError(
            f"Could not find PSF effective area file for {self.irf} {self.partition}. "
            f"Tried {partition_file} and {combined_file}"
        )

    def _read_file(self, filename, tablename, columns):
        """Read energy/cos-theta bins and 2-D image columns from a FITS table.

        Parameters
        ----------
        filename : str
            Path to the CALDB FITS file.
        tablename : str
            Name of the binary-table HDU (e.g. ``"EFFECTIVE AREA_FRONT"``).
        columns : list of str
            Data column names to extract (e.g. ``["EFFAREA"]``).

        Returns
        -------
        ebins : ndarray
            Energy bin edges in MeV, shape ``(n_energy + 1,)``.
        cbins : ndarray
            cos(theta) bin edges, shape ``(n_costh + 1,)``.
        images : list of ndarray
            One ``(n_costh, n_energy)`` array per requested column.
        """
        with fits.open(filename) as hdu:
            table = cast(Any, hdu[tablename])
            table_data = table.data
            if table_data is None:
                raise ValueError(f"No data payload found in {tablename} of {filename}")
            cbins = np.append(table_data.field("CTHETA_LO")[0], table_data.field("CTHETA_HI")[0][-1])
            ebins = np.append(table_data.field("ENERG_LO")[0], table_data.field("ENERG_HI")[0][-1])
            images = [
                np.asarray(table_data.field(column)[0], dtype=float)
                .reshape(len(cbins) - 1, len(ebins) - 1)
                for column in columns
            ]
        return ebins, cbins, images

    def _read_aeff(self, ct0_file, ct1_file):
        """Populate the effective-area arrays and interpolation table from FITS.

        Sets ``self.feffarea`` and ``self.beffarea`` (in cm²), ``self.ebins``,
        ``self.cbins``, ``self.aeff``, ``self.faeff_aug``, and
        ``self.baeff_aug``.

        Parameters
        ----------
        ct0_file : str
            FITS file used for FRONT (or combined PSF) tables.
        ct1_file : str
            FITS file used for BACK tables (may equal *ct0_file*).
        """
        if self.partition.startswith("PSF") and self._psf_from_combined_file:
            tablename = f"EFFECTIVE AREA_{self.partition}"
            ebins, cbins, effarea = self._read_file(ct0_file, tablename, ["EFFAREA"])
            total = effarea[0] * 1e4
            self.feffarea = 0.5 * total
            self.beffarea = 0.5 * total
            self.ebins, self.cbins = ebins, cbins
        elif self.partition == "FB":
            try:
                ebins, cbins, feffarea = self._read_file(ct0_file, "EFFECTIVE AREA_FRONT", ["EFFAREA"])
                ebins, cbins, beffarea = self._read_file(ct1_file, "EFFECTIVE AREA_BACK", ["EFFAREA"])
                self.ebins, self.cbins = ebins, cbins
                self.feffarea = feffarea[0] * 1e4
                self.beffarea = beffarea[0] * 1e4
            except KeyError as exc:
                raise KeyError(
                    f"FB partition requires EFFECTIVE AREA_FRONT/BACK tables in {ct0_file}"
                ) from exc
        else:
            try:
                ebins, cbins, feffarea = self._read_file(ct0_file, "EFFECTIVE AREA_FRONT", ["EFFAREA"])
                ebins, cbins, beffarea = self._read_file(ct1_file, "EFFECTIVE AREA_BACK", ["EFFAREA"])
                self.feffarea = feffarea[0] * 1e4
                self.beffarea = beffarea[0] * 1e4
                self.ebins, self.cbins = ebins, cbins
            except KeyError:
                ebins, cbins, effarea = self._read_file(ct0_file, "EFFECTIVE AREA", ["EFFAREA"])
                total = effarea[0] * 1e4
                self.feffarea = 0.5 * total
                self.beffarea = 0.5 * total
                self.ebins, self.cbins = ebins, cbins

        self.aeff = _InterpTable(np.log10(self.ebins), self.cbins)
        self.faeff_aug = self.aeff.augment_data(self.feffarea)
        self.baeff_aug = self.aeff.augment_data(self.beffarea)

    def _read_phi(self, ct0_file, ct1_file):
        """Load azimuthal phi-dependence correction tables from FITS.

        Populates ``self.fphis``, ``self.bphis``, and ``self.phi`` (the
        interpolation grid).  Only called when ``use_phidep=True``.

        Parameters
        ----------
        ct0_file : str
            FITS file used for FRONT (or combined PSF) phi tables.
        ct1_file : str
            FITS file used for BACK phi tables.
        """
        if self.partition.startswith("PSF") and self._psf_from_combined_file:
            tablename = f"PHI_DEPENDENCE_{self.partition}"
            ebins, cbins, phis = self._read_file(ct0_file, tablename, ["PHIDEP0", "PHIDEP1"])
            self.fphis = phis
            self.bphis = phis
            self.phi = _InterpTable(np.log10(ebins), cbins, augment=False)
            return

        if self.partition == "FB":
            ebins, cbins, fphis = self._read_file(ct0_file, "PHI_DEPENDENCE_FRONT", ["PHIDEP0", "PHIDEP1"])
            ebins, cbins, bphis = self._read_file(ct1_file, "PHI_DEPENDENCE_BACK", ["PHIDEP0", "PHIDEP1"])
        else:
            try:
                ebins, cbins, fphis = self._read_file(ct0_file, "PHI_DEPENDENCE_FRONT", ["PHIDEP0", "PHIDEP1"])
                ebins, cbins, bphis = self._read_file(ct1_file, "PHI_DEPENDENCE_BACK", ["PHIDEP0", "PHIDEP1"])
            except KeyError:
                ebins, cbins, phis = self._read_file(ct0_file, "PHI_DEPENDENCE", ["PHIDEP0", "PHIDEP1"])
                fphis = phis
                bphis = phis

        self.fphis = fphis
        self.bphis = bphis
        self.phi = _InterpTable(np.log10(ebins), cbins, augment=False)

    def _phi_mod(self, e, c, event_class, phi):
        """Compute the azimuthal modulation correction factor.

        Implements the parametric LAT phi-dependence model
        ``(1 + p0 * phi_sym^p1) / norm``, where ``phi_sym`` is a symmetrised
        azimuthal angle.  Returns 1 when *phi* is *None*.

        Parameters
        ----------
        e : float
            log10(energy / MeV).
        c : float
            cos(theta).
        event_class : int
            0 for FRONT, 1 for BACK.
        phi : float or None
            Azimuthal angle in radians; pass *None* to skip the correction.

        Returns
        -------
        float
            Multiplicative phi-dependence correction factor.
        """
        if phi is None:
            return 1
        tables = self.fphis if event_class == 0 else self.bphis
        par0 = self.phi(e, c, tables[0], bilinear=False)
        par1 = self.phi(e, c, tables[1], bilinear=False, reset_indices=False)
        norm = 1.0 + par0 / (1.0 + par1)
        phi = 2 * abs((2.0 / np.pi) * phi - 0.5)
        return (1.0 + par0 * phi**par1) / norm

    def _eval_front_back(self, e, c, phi=None, bilinear=True):
        """Return the (front, back) effective area at energy *e* and cos(theta) *c*.

        Parameters
        ----------
        e : float
            Energy in MeV.
        c : float or ndarray
            cos(theta) value(s).
        phi : float or None
            Azimuthal angle in radians for phi-dependence correction.
        bilinear : bool
            Use bilinear interpolation when *True*, nearest-neighbour otherwise.

        Returns
        -------
        front : float or ndarray
            FRONT effective area in cm².
        back : float or ndarray
            BACK effective area in cm².
        """
        e = np.log10(e)
        at = self.aeff
        front = at(e, c, self.faeff_aug, bilinear=bilinear) * self._phi_mod(e, c, 0, phi)
        back = at(e, c, self.baeff_aug, bilinear=bilinear, reset_indices=False) * self._phi_mod(e, c, 1, phi)
        return front, back

    def _get_partition_evaluator(self, partition):
        """Return a cached ``EffectiveAreaIRF`` for the requested partition.

        Instantiates a new reader on first access and stores it in
        ``self._partition_cache`` for reuse.

        Parameters
        ----------
        partition : str
            Partition name, e.g. ``"PSF0"``.

        Returns
        -------
        EffectiveAreaIRF
        """
        if partition in self._partition_cache:
            return self._partition_cache[partition]
        self._partition_cache[partition] = EffectiveAreaIRF(
            irf=self.irf,
            file_path=self.file_path,
            partition=partition,
        )
        return self._partition_cache[partition]

    def __call__(self, e, c, phi=None, event_class=-1, bilinear=True):
        """Evaluate effective area using event_class mapping.

        event_class mapping:
        -1 -> (FRONT, BACK) tuple for the current partition
         0 -> FRONT
         1 -> BACK
         2 -> PSF0 total
         3 -> PSF1 total
         4 -> PSF2 total
         5 -> PSF3 total
        """
        if event_class == -1:
            return self._eval_front_back(e, c, phi=phi, bilinear=bilinear)

        if event_class == 0:
            front, _ = self._eval_front_back(e, c, phi=phi, bilinear=bilinear)
            return front

        if event_class == 1:
            _, back = self._eval_front_back(e, c, phi=phi, bilinear=bilinear)
            return back

        if event_class in (2, 3, 4, 5):
            target_partition = f"PSF{event_class - 2}"
            psf_eval = self._get_partition_evaluator(target_partition)
            front, back = psf_eval._eval_front_back(e, c, phi=phi, bilinear=bilinear)
            return np.asarray(front) + np.asarray(back)

        raise ValueError("event_class must be one of -1, 0, 1, 2, 3, 4, 5")


class ExposureMap:
    """Callable HEALPix exposure map interpolator using astropy-healpix."""

    def __init__(self, values, nside=None, nest=False, frame="icrs"):
        """Wrap a full-sky HEALPix exposure map for sky-coordinate lookup.

        Parameters
        ----------
        values : array-like
            Full-sky HEALPix map values, one entry per pixel.
        nside : int or None
            HEALPix ``nside`` resolution parameter; inferred from
            ``len(values)`` when *None*.
        nest : bool
            *True* for NESTED pixel ordering; *False* (default) for RING.
        frame : str
            Coordinate frame of the map, ``"icrs"`` or ``"galactic"``.
        """
        arr = np.asarray(values, dtype=float).ravel()
        if nside is None:
            nside = npix_to_nside(arr.size)
        nside = int(nside)
        expected_npix = 12 * nside**2
        if arr.size != expected_npix:
            raise ValueError(
                f"HEALPix map length {arr.size} does not match nside={nside} "
                f"(expected {expected_npix})"
            )

        frame = frame.lower()
        if frame not in ("icrs", "galactic"):
            raise ValueError("frame must be 'icrs' or 'galactic'")

        self.values = arr
        self.nside = nside
        self.nest = bool(nest)
        self.frame = frame
        self.hpx = HEALPix(
            nside=self.nside,
            order="nested" if self.nest else "ring",
            frame=self.frame,
        )

    def __call__(self, skycoord):
        """Return bilinearly interpolated exposure at one or many sky coordinates."""
        from astropy.coordinates import SkyCoord

        if not isinstance(skycoord, SkyCoord):
            raise TypeError("skycoord must be an astropy.coordinates.SkyCoord")

        c = cast(Any, skycoord.transform_to(self.frame))
        lon = c.spherical.lon
        lat = c.spherical.lat

        return self.hpx.interpolate_bilinear_lonlat(lon, lat, self.values)

    def ait_plot(
        self,
        *,
        cmap="coolwarm",
        pixelsize=1.0,
        grid_color="0.45",
        figsize=(12, 6),
        title=None,
        colorbar=True,
        **healpix_fill_kwargs,
    ):
        """Display the map with utilities.skymaps.AITfigure and return the figure."""
        from utilities.skymaps import AITfigure

        vals = np.asarray(self.values, dtype=float)
        finite_vals = np.isfinite(vals)
        mean_val = float(np.nanmean(vals[finite_vals])) if np.any(finite_vals) else np.nan
        plot_vals = 100 * (vals / mean_val - 1) if np.isfinite(mean_val) and mean_val != 0 else np.full_like(vals, np.nan, dtype=float)
        
        if title is None:
            title = "Exposure ratio to mean" + (" (log10)" if log10 else "")

        ait = AITfigure(figsize=figsize, grid_color=grid_color)
        ait.healpix_fill(
            plot_vals,
            pixelsize=pixelsize,
            cmap=cmap,
            **healpix_fill_kwargs,
        )
        if colorbar:
            ait.colorbar(label=r'Deviation from mean (%)')
        ait.title(title)
        ait.axes_text( 0.0, 0.0,
            f"mean = {mean_val:.2e}"+ r" $\mathrm{cm^2 \, s}$",
            ha="left", va="bottom", fontsize=12,
         )
        ait.show()
        


class _InterpTable(object):
    """Helper class for 2D interpolation in log10(E), cos(theta)."""

    def __init__(self, xbins, ybins, augment=True):
        """Set up bin edges for 2-D bilinear interpolation in (x, y).

        Parameters
        ----------
        xbins : ndarray
            Bin edges along the x-axis (typically log10(E / MeV)).
        ybins : ndarray
            Bin edges along the y-axis (typically cos(theta)).
        augment : bool
            If *True*, pad the grid by one half-cell on every side so that the
            bilinear stencil never requires out-of-range indexing.
        """
        self.xbins_0, self.ybins_0 = xbins, ybins
        self.augment = augment
        if augment:
            x0 = xbins[0] - (xbins[1] - xbins[0]) / 2
            x1 = xbins[-1] + (xbins[-1] - xbins[-2]) / 2
            y0 = ybins[0] - (ybins[1] - ybins[0]) / 2
            y1 = ybins[-1] + (ybins[-1] - ybins[-2]) / 2
            self.xbins = np.concatenate(([x0], xbins, [x1]))
            self.ybins = np.concatenate(([y0], ybins, [y1]))
        else:
            self.xbins = xbins
            self.ybins = ybins
        self.xbins_s = (self.xbins[:-1] + self.xbins[1:]) / 2
        self.ybins_s = (self.ybins[:-1] + self.ybins[1:]) / 2

    def augment_data(self, data):
        """Pad a 2-D data array to match the augmented bin grid.

        The original data occupies the interior ``[1:-1, 1:-1]`` slice; border
        cells are filled by replicating the nearest edge values.

        Parameters
        ----------
        data : ndarray, shape (ny, nx)
            Original data on the un-augmented grid.

        Returns
        -------
        ndarray, shape (ny + 2, nx + 2)
            Padded array ready for bilinear lookup.
        """
        augmented = np.empty([data.shape[0] + 2, data.shape[1] + 2])
        augmented[1:-1, 1:-1] = data
        augmented[0, 1:-1] = data[0, :]
        augmented[1:-1, 0] = data[:, 0]
        augmented[-1, 1:-1] = data[-1, :]
        augmented[1:-1, -1] = data[:, -1]
        augmented[0, 0] = data[0, 0]
        augmented[-1, -1] = data[-1, -1]
        augmented[0, -1] = data[0, -1]
        augmented[-1, 0] = data[-1, 0]
        return augmented

    def set_indices(self, x, y, bilinear=True):
        """Compute and cache lower-left bin indices for interpolation at (x, y).

        Uses midpoint grids for bilinear mode and edge grids for
        nearest-neighbour mode.  Results are stored in ``self.indices`` and
        consumed by :meth:`value`.

        Parameters
        ----------
        x : float
            Query point on the x-axis (log10 energy).
        y : float or ndarray
            Query point(s) on the y-axis (cos theta).
        bilinear : bool
            Select bilinear (*True*) or nearest-neighbour (*False*) mode.
        """
        if bilinear and (not self.augment):
            print("Not equipped for bilinear, going to nearest neighbor.")
            bilinear = False
        self.bilinear = bilinear
        if not bilinear:
            i = np.searchsorted(self.xbins, x) - 1
            j = np.searchsorted(self.ybins, y) - 1
        else:
            i = np.searchsorted(self.xbins_s, x) - 1
            j = np.searchsorted(self.ybins_s, y) - 1
        self.indices = i, j

    def value(self, x, y, data):
        """Return the interpolated value at (x, y) using cached indices.

        Must be preceded by a call to :meth:`set_indices` for the same (x, y).

        Parameters
        ----------
        x : float
            Query x coordinate.
        y : float or ndarray
            Query y coordinate(s).
        data : ndarray
            Augmented data array produced by :meth:`augment_data`.

        Returns
        -------
        float or ndarray
            Interpolated value(s) at (x, y).
        """
        i, j = self.indices
        if not self.bilinear:
            return data[j, i]
        x2, x1 = self.xbins_s[i + 1], self.xbins_s[i]
        y2, y1 = self.ybins_s[j + 1], self.ybins_s[j]
        f00 = data[j, i]
        f11 = data[j + 1, i + 1]
        f01 = data[j + 1, i]
        f10 = data[j, i + 1]
        norm = (x2 - x1) * (y2 - y1)
        return ((x2 - x) * (f00 * (y2 - y) + f01 * (y - y1)) +
                (x - x1) * (f10 * (y2 - y) + f11 * (y - y1))) / norm

    def __call__(self, x, y, data, bilinear=True, reset_indices=True):
        """Interpolate *data* at (x, y), optionally reusing cached indices.

        Parameters
        ----------
        x : float
            Query x coordinate.
        y : float or ndarray
            Query y coordinate(s).
        data : ndarray
            Augmented data array.
        bilinear : bool
            Use bilinear interpolation when *True*.
        reset_indices : bool
            Re-run :meth:`set_indices` before evaluating.  Set to *False* when
            evaluating multiple datasets at the same (x, y) to avoid redundant
            index computation.

        Returns
        -------
        float or ndarray
            Interpolated value(s).
        """
        if reset_indices:
            self.set_indices(x, y, bilinear=bilinear)
        return self.value(x, y, data)


def _resolve_aeff_for_event_type(
    Aeff,
    *,
    event_type,
    irf,
    file_path,
    CALDB,
    use_phidep,
):
    """Resolve an effective-area evaluator and low-level event class selector.

    Event types ``0`` and ``1`` map to FRONT/BACK in the base IRF.
    Event types ``2``-``5`` map to PSF0-PSF3 IRFs, summing front+back within
    the selected PSF partition.
    """
    event_type = int(event_type)

    if isinstance(Aeff, dict):
        if event_type not in Aeff:
            raise KeyError(f"No effective-area object provided for event_type={event_type}")
        return Aeff[event_type], -1

    if Aeff is not None:
        low_level_event_class = event_type if event_type in (-1, 0, 1) else -1
        return Aeff, low_level_event_class

    if event_type in (-1, 0, 1):
        return EffectiveAreaIRF(
            irf=irf,
            file_path=file_path,
            CALDB=CALDB,
            partition="FB",
            use_phidep=use_phidep,
        ), event_type

    if 2 <= event_type <= 5:
        try:
            return EffectiveAreaIRF(
                irf=irf,
                file_path=file_path,
                CALDB=CALDB,
                partition=f"PSF{event_type - 2}",
                use_phidep=use_phidep,
            ), -1
        except (AssertionError, FileNotFoundError) as exc:
            raise FileNotFoundError(
                f"event_type={event_type} requires PSF-specific IRF 'PSF{event_type - 2}'. "
                "The requested effective-area files were not found in the configured path."
            ) from exc

    raise ValueError("Supported event_type values are -1, 0, 1, 2, 3, 4, 5")


def make_aeff_costheta_function(
    emin,
    emax,
    *,
    Aeff=None,
    event_type=-1,
    irf="P8R2_SOURCE_V6",
    file_path=None,
    CALDB=None,
    use_phidep=False,
    n_energy=32,
    spectrum=None,
    ctmin=0.4,
    ctmax=1.0,
    n_costh=96,
):
    """Return a callable band-averaged effective area as a function of cos(theta).

    The returned function evaluates A_eff(cos(theta)) for a fixed
    event class, averaged over the energy interval ``[emin, emax]`` using the
    supplied spectral weighting.

    Parameters
    ----------
    emin, emax : float
        Energy band boundaries in MeV.  Must satisfy ``0 < emin < emax``.
    Aeff : callable, dict, or None
        Effective-area evaluator with signature ``Aeff(E, cos_theta, event_class=...)``.
        If a dict, it should be keyed by ``event_type``.
        If *None*, an :class:`EffectiveAreaIRF` is instantiated using the
        provided IRF/path settings, selecting PSF-specific IRFs for event
        types 2–5.
    event_type : int
        Event-type code: ``-1`` returns front+back sum, ``0`` FRONT only,
        ``1`` BACK only, ``2``–``5`` PSF0–PSF3 total.
    irf : str
        IRF name passed to :class:`EffectiveAreaIRF` when *Aeff* is *None*.
    file_path : str or None
        Path to the CALDB ``bcf/ea`` directory.
    CALDB : str or None
        CALDB root; falls back to ``$CALDB`` environment variable.
    use_phidep : bool
        Load phi-dependence tables when constructing a new IRF object.
    n_energy : int
        Number of logarithmically-spaced energy sample points for averaging.
    spectrum : callable or None
        Spectral weighting function ``S(E)``; defaults to ``E^{-2}``.
    ctmin, ctmax : float
        cos(theta) integration limits.  Must satisfy ``0 <= ctmin < ctmax <= 1``.
    n_costh : int
        Number of cos(theta) sample points.

    Returns
    -------
    callable
        Function ``aeff_costheta(costheta)`` returning the energy-averaged
        effective area in cm² at each requested cos(theta) value.
    """
    if not (emin > 0 and emax > emin):
        raise ValueError("Expect 0 < emin < emax")
    ctmin = float(ctmin)
    ctmax = float(ctmax)
    if not (0.0 <= ctmin < ctmax <= 1.0):
        raise ValueError("Expect 0 <= ctmin < ctmax <= 1")
    if int(n_energy) < 2:
        raise ValueError("n_energy must be >= 2")
    if int(n_costh) < 2:
        raise ValueError("n_costh must be >= 2")

    Aeff_eval, low_level_event_class = _resolve_aeff_for_event_type(
        Aeff,
        event_type=event_type,
        irf=irf,
        file_path=file_path,
        CALDB=CALDB,
        use_phidep=use_phidep,
    )

    energies = np.logspace(np.log10(emin), np.log10(emax), int(n_energy))
    wE = spectrum(energies) if spectrum is not None else energies ** (-2.0)
    wE = np.asarray(wE, dtype=np.float64)
    if wE.shape != energies.shape:
        raise ValueError("spectrum must return array with same shape as energies")
    den_E = np.trapz(wE, energies)
    if den_E <= 0 or not np.isfinite(den_E):
        raise ValueError("Invalid energy weight normalization")

    cth = np.linspace(ctmin, ctmax, int(n_costh))
    aeff_e = np.empty((energies.size, cth.size), dtype=np.float64)
    for i, e in enumerate(energies):
        vals = Aeff_eval(e, cth, event_class=low_level_event_class)
        if isinstance(vals, tuple):
            vals = vals[0] + vals[1]
        vals = np.asarray(vals, dtype=np.float64)
        if vals.shape != cth.shape:
            raise ValueError("Aeff(E, cos_theta, ...) must return shape matching cos(theta) grid")
        aeff_e[i] = vals

    aeff_band_cth = np.trapz(aeff_e * wE[:, None], energies, axis=0) / den_E

    def aeff_costheta(costheta):
        """Evaluate energy-averaged effective area at cos(theta)."""
        costheta = np.asarray(costheta, dtype=np.float64)
        out = np.zeros_like(costheta, dtype=np.float64)
        mask = (costheta >= ctmin) & (costheta <= ctmax)
        if np.any(mask):
            out[mask] = np.interp(costheta[mask], cth, aeff_band_cth)
        return out

    return aeff_costheta


def _convolve_axisymmetric_kernel_healpy(livetime_density, kernel_theta, theta, nside, lmax):
    """Convolve a map with an axisymmetric kernel in harmonic space via healpy.

    This backend is isolated here because astropy_healpix does not yet provide
    spherical-harmonic map transforms required for this convolution path.
    """
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError(
            "make_exposure_map_healpix requires healpy for harmonic convolution "
            "(beam2bl/map2alm/smoothalm/alm2map). Install healpy or replace "
            "the convolution backend."
        ) from exc

    bl = hp.sphtfunc.beam2bl(kernel_theta, theta, lmax=int(lmax))
    alm = hp.map2alm(livetime_density, lmax=int(lmax), pol=False, iter=0)
    alm_conv = hp.smoothalm(alm, beam_window=bl, pol=False)
    return hp.alm2map(alm_conv, nside=int(nside), lmax=int(lmax), pol=False)


def make_exposure_map_healpix(
    Aeff,
    emin,
    emax,
    livetime,
    nside=None,
    event_type=-1,
    irf="P8R2_SOURCE_V6",
    file_path=None,
    CALDB=None,
    use_phidep=False,
    n_energy=32,
    spectrum=None,
    ctmin=0.4,
    ctmax=1.0,
    n_costh=96,
    n_theta=512,
    lmax=None,
    costh_weight=None,
    zenith_deg=None,
    zmax_deg=None,
):
    """Create a HEALPix exposure map via spherical harmonic convolution.

    Computes the band-averaged effective area as a function of cos(theta),
    builds an axisymmetric kernel, and convolves it with the livetime-density
    map in harmonic space using healpy.

    Parameters
    ----------
    Aeff : callable, dict, or None
        Effective-area evaluator; see :func:`make_aeff_costheta_function`.
    emin, emax : float
        Energy band boundaries in MeV.
    livetime : float or ndarray
        Per-pixel livetime in seconds.  A scalar is broadcast to a full-sky
        map of ``12 * nside**2`` pixels (requires *nside*); a 1-D array
        infers ``nside`` from its length.
    nside : int or None
        HEALPix resolution; required only when *livetime* is a scalar.
    event_type : int
        Event-type code (see :func:`make_aeff_costheta_function`).
    irf : str
        IRF name.
    file_path, CALDB, use_phidep : see :func:`make_aeff_costheta_function`.
    n_energy : int
        Energy integration points.
    spectrum : callable or None
        Spectral weighting; defaults to ``E^{-2}``.
    ctmin, ctmax : float
        cos(theta) acceptance limits.
    n_costh : int
        cos(theta) sample points.
    n_theta : int
        Number of polar-angle points for the kernel (>= 16).
    lmax : int or None
        Maximum spherical-harmonic order; defaults to ``3 * nside - 1``.
    costh_weight : callable or None
        Optional additional weighting over cos(theta).
    zenith_deg : ndarray or None
        Per-pixel mean zenith angle in degrees; used with *zmax_deg* to mask
        pixels above the zenith-angle cut.
    zmax_deg : float or None
        Maximum zenith angle cut in degrees.

    Returns
    -------
    ndarray, shape (12 * nside²,), dtype float32
        Full-sky HEALPix exposure map in cm² s.
    """
    if not (emin > 0 and emax > emin):
        raise ValueError("Expect 0 < emin < emax")
    ctmin = float(ctmin)
    ctmax = float(ctmax)
    if not (0.0 <= ctmin < ctmax <= 1.0):
        raise ValueError("Expect 0 <= ctmin < ctmax <= 1")
    if int(n_energy) < 2:
        raise ValueError("n_energy must be >= 2")
    if int(n_costh) < 2:
        raise ValueError("n_costh must be >= 2")
    if int(n_theta) < 16:
        raise ValueError("n_theta must be >= 16")
    if (zenith_deg is None) ^ (zmax_deg is None):
        raise ValueError("Provide both zenith_deg and zmax_deg, or neither")

    if np.isscalar(livetime):
        if nside is None:
            raise ValueError("Provide nside when livetime is scalar")
        npix = 12 * int(nside) ** 2
        lt_scalar = np.asarray(livetime, dtype=np.float64).item()
        lt = np.full(npix, lt_scalar, dtype=np.float64)
    else:
        lt = np.asarray(livetime, dtype=np.float64)
        if lt.ndim != 1:
            raise ValueError("livetime array must be 1D")
        npix = lt.size
        nside = npix_to_nside(npix)

    nside = int(nside)
    pix_area = nside_to_pixel_area(nside).value
    if lmax is None:
        lmax = 3 * nside - 1

    if zenith_deg is not None and zmax_deg is not None:
        zmap = np.asarray(zenith_deg, dtype=np.float64)
        if zmap.shape != (npix,):
            raise ValueError("zenith_deg must be a 1D array matching livetime map length")
        zmax = float(zmax_deg)
        if not np.isfinite(zmax):
            raise ValueError("zmax_deg must be finite")
        lt = np.where(zmap <= zmax, lt, 0.0)

    cth = np.linspace(ctmin, ctmax, int(n_costh))
    wc = costh_weight(cth) if costh_weight is not None else np.ones_like(cth)
    wc = np.asarray(wc, dtype=np.float64)
    if wc.shape != cth.shape:
        raise ValueError("costh_weight must return array with same shape as cos(theta) grid")
    den_cth = np.trapz(wc, cth)
    if den_cth <= 0 or not np.isfinite(den_cth):
        raise ValueError("Invalid cos(theta) weight normalization")

    aeff_costheta = make_aeff_costheta_function(
        emin,
        emax,
        Aeff=Aeff,
        event_type=event_type,
        irf=irf,
        file_path=file_path,
        CALDB=CALDB,
        use_phidep=use_phidep,
        n_energy=n_energy,
        spectrum=spectrum,
        ctmin=ctmin,
        ctmax=ctmax,
        n_costh=n_costh,
    )
    aeff_band_cth = aeff_costheta(cth)
    aeff_band_cth *= wc / den_cth

    theta = np.linspace(0.0, np.pi, int(n_theta))
    ctheta = np.cos(theta)
    kernel_theta = np.zeros_like(theta, dtype=np.float64)
    mask = (ctheta >= ctmin) & (ctheta <= ctmax)
    kernel_theta[mask] = np.interp(ctheta[mask], cth, aeff_band_cth)

    livetime_density = lt / pix_area
    exposure_map = _convolve_axisymmetric_kernel_healpy(
        livetime_density=livetime_density,
        kernel_theta=kernel_theta,
        theta=theta,
        nside=nside,
        lmax=lmax,
    )

    np.maximum(exposure_map, 0.0, out=exposure_map)
    return exposure_map.astype(np.float32)


def _pixel_table_band_event_type(band):
    """Infer the FITS-style event-type code for a PixelTable band."""
    event_type = getattr(band, "event_type", None)
    if event_type is not None:
        return int(event_type)

    label = str(getattr(band, "psf", "")).strip().upper()
    if label == "FRONT":
        return 0
    if label == "BACK":
        return 1
    if label.startswith("PSF"):
        return int(label[3:]) + 2
    raise ValueError(f"Could not infer event type for band {band!r}")


def build_pixel_table_exposure(
    pixel_table,
    livetime,
    *,
    Aeff=None,
    irf="P8R3_SOURCE_V3",
    file_path=None,
    CALDB=None,
    use_phidep=False,
    frame="galactic",
    nest=False,
    n_energy=32,
    spectrum=None,
    ctmin=0.4,
    ctmax=1.0,
    n_costh=96,
    n_theta=512,
    lmax=None,
    costh_weight=None,
    zenith_deg=None,
    zmax_deg=None,
):
    """Compute and attach band exposure products to a PixelTable.

    Iterates over every band in *pixel_table*, builds a full-sky HEALPix
    exposure map for each band's energy range and event type, and attaches
    the results via :meth:`PixelTable.attach_exposure`.

    For each band the following attributes are set:

    - ``band.aeff_costheta`` — band-averaged effective area vs. cos(theta)
    - ``band.exposure_map``  — callable :class:`ExposureMap` instance
    - ``band.pixel_exposure`` — exposure sampled at each sparse pixel
    - ``band.exposure``       — mean sparse-pixel exposure for the band

    Parameters
    ----------
    pixel_table : PixelTable
        Sparse pixel-data object whose bands supply ``e0``, ``e1``, ``pix``,
        and either ``event_type`` or ``psf`` attributes.
    livetime : float or ndarray
        Per-pixel livetime in seconds passed to :func:`make_exposure_map_healpix`.
    Aeff : callable, dict, or None
        Effective-area evaluator; see :func:`make_aeff_costheta_function`.
    irf : str
        IRF name (default ``"P8R3_SOURCE_V3"``).
    file_path, CALDB, use_phidep : see :func:`make_aeff_costheta_function`.
    frame : str
        Coordinate frame for the exposure maps, ``"galactic"`` or ``"icrs"``.
    nest : bool
        HEALPix pixel ordering; *False* (default) for RING.
    n_energy, spectrum, ctmin, ctmax, n_costh, n_theta, lmax, costh_weight,
    zenith_deg, zmax_deg : see :func:`make_exposure_map_healpix`.

    Returns
    -------
    PixelTable
        The same *pixel_table* object, modified in-place.
    """
    exposure_by_band = {}

    for band in pixel_table.values():
        event_type = _pixel_table_band_event_type(band)
        band.aeff_costheta = make_aeff_costheta_function(
            band.e0,
            band.e1,
            Aeff=Aeff,
            event_type=event_type,
            irf=irf,
            file_path=file_path,
            CALDB=CALDB,
            use_phidep=use_phidep,
            n_energy=n_energy,
            spectrum=spectrum,
            ctmin=ctmin,
            ctmax=ctmax,
            n_costh=n_costh,
        )

        map_values = make_exposure_map_healpix(
            Aeff=Aeff,
            emin=band.e0,
            emax=band.e1,
            livetime=livetime,
            event_type=event_type,
            irf=irf,
            file_path=file_path,
            CALDB=CALDB,
            use_phidep=use_phidep,
            n_energy=n_energy,
            spectrum=spectrum,
            ctmin=ctmin,
            ctmax=ctmax,
            n_costh=n_costh,
            n_theta=n_theta,
            lmax=lmax,
            costh_weight=costh_weight,
            zenith_deg=zenith_deg,
            zmax_deg=zmax_deg,
        )
        exposure_by_band[band.key] = ExposureMap(map_values, nest=nest, frame=frame)

    pixel_table.attach_exposure(exposure_by_band, frame=frame, nest=nest)
    return pixel_table
