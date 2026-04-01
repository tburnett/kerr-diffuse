"""Livetime map utilities used by exposure workflows.

This module collects reusable functions from the notebook-based livetime
exploration in ``exposure.ipynb`` so they can be imported from code.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
import pickle
from typing import Callable, Iterable, Sequence

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix, npix_to_nside


__all__ = [
    "merge_exclusions",
    "build_livetime_map",
    "zenith_angle_map_from_sc",
    "read_livetime_map",
    "write_healpix_map",
]


def _ensure_scipy_trapz_compat() -> None:
    """Provide scipy.integrate.trapz for older healpy compatibility."""
    try:
        import scipy.integrate as sp_integrate
    except Exception:
        return
    if not hasattr(sp_integrate, "trapz"):
        sp_integrate.trapz = np.trapz


def _write_livetime_output(path: str | Path, livetime_map: np.ndarray, overwrite: bool) -> Path:
    """Write livetime output to FITS or NPY based on file extension."""
    outpath = Path(path)
    suffix = outpath.suffix.lower()
    if suffix == ".fits":
        return write_healpix_map(outpath, livetime_map.astype(np.float32), overwrite=overwrite)
    if suffix == ".npy":
        if outpath.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {outpath}")
        np.save(outpath, livetime_map.astype(np.float32))
        return outpath
    raise ValueError("Output filename must end with '.fits' or '.npy'.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a GTI-filtered livetime map file for an MJD interval."
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output livetime file path (.fits or .npy)",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=128,
        help="HEALPix NSIDE for the output map (default: 128)",
    )
    parser.add_argument(
        "--mjd-min",
        type=float,
        required=False,
        default=0.0,
        help="Minimum MJD (inclusive), default: 0.0 (no lower limit)",
    )
    parser.add_argument(
        "--mjd-max",
        type=float,
        required=True,
        help="Maximum MJD (inclusive)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=None,
        help="Worker count for weekly GTI preprocessing (default: auto)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.nside <= 0:
        parser.error("--nside must be a positive integer")
    if args.mjd_min >= args.mjd_max:
        parser.error("Require --mjd-min < --mjd-max")

    result = build_livetime_map(
        nside=int(args.nside),
        save=False,
        mjd_range=(float(args.mjd_min), float(args.mjd_max)),
        preprocess_workers=args.preprocess_workers,
    )
    livetime_map = np.asarray(result["livetime_map"], dtype=np.float32)
    outpath = _write_livetime_output(args.output, livetime_map, overwrite=bool(args.overwrite))

    total_livetime = float(result["total_livetime"])
    print(f"Wrote livetime map to {outpath} ({livetime_map.size} pixels, total livetime={total_livetime:.3f})")
    return 0


def write_healpix_map(
    path: str | Path,
    data: np.ndarray,
    overwrite: bool = True,
    coord: str = "G",
) -> Path:
    """Write a 1D RING HEALPix map to FITS without extra HEALPix IO dependencies."""
    arr = np.asarray(data, dtype=np.float32).ravel()
    nside = int(npix_to_nside(arr.size))
    column = fits.Column(name="T", format=f"{arr.size}E", array=[arr])
    table_hdu = fits.BinTableHDU.from_columns([column])
    table_hdu.header["PIXTYPE"] = "HEALPIX"
    table_hdu.header["ORDERING"] = "RING"
    table_hdu.header["INDXSCHM"] = "IMPLICIT"
    table_hdu.header["NSIDE"] = nside
    table_hdu.header["FIRSTPIX"] = 0
    table_hdu.header["LASTPIX"] = arr.size - 1
    table_hdu.header["COORDSYS"] = str(coord)
    hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
    outpath = Path(path)
    hdul.writeto(outpath, overwrite=bool(overwrite))
    return outpath


def _read_healpix_fits(path: str | Path, dtype=np.float32) -> np.ndarray:
    """Read a 1D HEALPix FITS map written by compatible FITS writers."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            if isinstance(hdu, fits.BinTableHDU):
                if "T" in data.names:
                    return np.asarray(data["T"][0], dtype=dtype).ravel()
                first_name = data.names[0]
                return np.asarray(data[first_name][0], dtype=dtype).ravel()
            return np.asarray(data, dtype=dtype).ravel()
    raise ValueError(f"No readable HEALPix map data found in {path}")


def merge_exclusions(*interval_sets: Iterable[float]) -> np.ndarray:
    """Merge one or more GTI exclusion arrays of interleaved start/stop times.

    Parameters
    ----------
    *interval_sets : iterable of float
        One or more arrays/lists containing interleaved start/stop times.

    Returns
    -------
    np.ndarray
        Flat array of merged start/stop times.
    """
    merged: list[list[float]] = []
    for times in interval_sets:
        arr = np.asarray(list(times), dtype=float).ravel()
        if arr.size == 0:
            continue
        if arr.size % 2 != 0:
            raise ValueError("Expected interleaved start/stop times.")
        pairs = arr.reshape(-1, 2)
        pairs = pairs[np.argsort(pairs[:, 0])]
        for start, stop in pairs:
            if not merged or start > merged[-1][1]:
                merged.append([float(start), float(stop)])
            else:
                merged[-1][1] = max(merged[-1][1], float(stop))

    if not merged:
        return np.array([], dtype=float)
    return np.asarray(merged, dtype=float).ravel()


def _extract_week_exclusion_edges(
    week_file: str | Path,
    mjd_converter: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray | None:
    """Extract MJD exclusion edges from one weekly pickle file."""
    with open(week_file, "rb") as inp:
        wk = pickle.load(inp)
    gti_times = wk.get("gti_times", None)
    if gti_times is None or len(gti_times) < 4:
        return None
    # Week files store good intervals in MET; GTI expects excluded in MJD.
    return np.asarray(mjd_converter(gti_times)[1:-1], dtype=float)


def build_livetime_map(
    config=None,
    nside: int = 64,
    extra_gti_files: Sequence[str] = ("nogrb.gti", "nosolarflares.gti"),
    output_dir: str | Path = "files",
    save: bool = True,
    overwrite: bool = True,
    mjd_range: tuple[float, float] | None = None,
    preprocess_workers: int | None = None,
) -> dict:
    """Build a GTI-filtered livetime map and optionally save it.

    This mirrors the notebook flow: combine week GTIs with extra exclusions,
    generate the full-sky livetime map via ``DataView``, then optionally save
    to ``.npy`` and ``.fits``.

    Parameters
    ----------
    config : wtlike.config.Config or None
        Optional ``Config`` instance. Uses default ``Config()`` if omitted.
    nside : int
        HEALPix nside of output map.
    extra_gti_files : sequence[str]
        Additional FITS GTI files to merge with week exclusions.
    output_dir : str or Path
        Directory used when ``save=True``.
    save : bool
        If True, save map to ``.npy`` and ``.fits``.
    overwrite : bool
        Passed to FITS writer.
    mjd_range : tuple[float, float] or None
        Optional MJD interval ``(mjd_min, mjd_max)`` used to limit spacecraft
        data in ``DataView``. If None, uses all available times.
    preprocess_workers : int or None
        Worker count for parallel weekly GTI preprocessing. ``None`` uses an
        automatic value and ``1`` disables parallel preprocessing.

    Returns
    -------
    dict
        Dictionary with map and metadata, plus output paths if saved.
    """
    _ensure_scipy_trapz_compat()
    from wtlike.config import Config, MJD
    from wtlike.data_man import GTI, DataView, get_week_files

    cfg = config or Config()
    week_files_all = get_week_files(cfg)

    if mjd_range is None:
        interval = (0, 0)
    else:
        if len(mjd_range) != 2:
            raise ValueError("mjd_range must be a 2-tuple: (mjd_min, mjd_max)")
        mjd_min, mjd_max = float(mjd_range[0]), float(mjd_range[1])
        if not np.isfinite(mjd_min) or not np.isfinite(mjd_max):
            raise ValueError("mjd_range values must be finite")
        if mjd_min >= mjd_max:
            raise ValueError("Require mjd_min < mjd_max")
        interval = (mjd_min, mjd_max)

    if preprocess_workers is not None and int(preprocess_workers) < 1:
        raise ValueError("preprocess_workers must be >= 1 or None")

    if preprocess_workers is None:
        auto_workers = min(32, (os.cpu_count() or 1) + 4)
        worker_count = min(auto_workers, max(1, len(week_files_all)))
    else:
        worker_count = min(int(preprocess_workers), max(1, len(week_files_all)))

    if worker_count == 1 or len(week_files_all) < 2:
        extracted = [_extract_week_exclusion_edges(wf, MJD) for wf in week_files_all]
    else:
        extract_fn = partial(_extract_week_exclusion_edges, mjd_converter=MJD)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            extracted = list(executor.map(extract_fn, week_files_all))

    exclusion_edges = [arr for arr in extracted if arr is not None]

    if len(exclusion_edges) == 0:
        raise RuntimeError("No GTI information found in weekly files.")

    weekly_exclusions = np.sort(np.concatenate(exclusion_edges))
    extra_gtis = [GTI.from_FITS(name, config=cfg) for name in extra_gti_files]
    combined_exclusions = merge_exclusions(
        weekly_exclusions, *(gti.times for gti in extra_gtis)
    )
    gti_all = GTI(combined_exclusions, name="all-weeks + extra exclusions")

    dv = DataView(interval=interval, config=cfg, gti=gti_all, nside=int(nside))
    ltmap_all = np.asarray(dv.livetime_map(nside=int(nside)), dtype=np.float32)

    result: dict[str, object] = dict(
        livetime_map=ltmap_all,
        nside=int(nside),
        npix=ltmap_all.size,
        weeks_processed=len(week_files_all),
        combined_exclusions=combined_exclusions,
        mjd_range=interval,
        total_livetime=float(ltmap_all.sum()),
    )

    if save:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        npy_file = outdir / f"livetime_allweeks_gti_nside{int(nside)}.npy"
        fits_file = outdir / f"livetime_allweeks_gti_nside{int(nside)}.fits"

        np.save(npy_file, ltmap_all.astype(np.float32))
        write_healpix_map(
            fits_file,
            ltmap_all.astype(np.float32),
            overwrite=bool(overwrite),
            coord="G",
        )
        result["npy_file"] = npy_file
        result["fits_file"] = fits_file

    return result


def zenith_angle_map_from_sc(
    weeks: Sequence[int] | None = None,
    nside: int = 64,
    config=None,
    frame: str = "galactic",
    chunk_size: int = 512,
    max_weeks: int | None = None,
) -> np.ndarray:
    """Compute a livetime-weighted mean zenith-angle map from spacecraft data.

    Parameters
    ----------
    weeks : sequence[int] or None
        Weeks to include. If None, include all available week files.
    nside : int
        Target HEALPix nside for output map.
    config : wtlike.config.Config or None
        Optional ``Config`` instance. Uses default ``Config()`` if omitted.
    frame : str
        Output frame for the returned map: ``'galactic'`` or ``'fk5'``.
    chunk_size : int
        Number of spacecraft intervals to process per matrix chunk.
    max_weeks : int or None
        Optional limit to most recent N weeks for faster testing.

    Returns
    -------
    np.ndarray
        Livetime-weighted mean zenith angle per pixel in degrees.
    """
    _ensure_scipy_trapz_compat()
    from wtlike.config import Config
    from wtlike.data_man import get_week_files

    cfg = config or Config()
    week_files = get_week_files(cfg)

    if weeks is not None:
        week_set = {int(w) for w in weeks}
        week_files = [wf for wf in week_files if int(wf.name[-7:-4]) in week_set]

    if max_weeks is not None:
        week_files = week_files[-int(max_weeks) :]

    if len(week_files) == 0:
        raise ValueError("No week files selected.")

    npix = 12 * int(nside) ** 2
    pix = np.arange(npix)
    hpix = HEALPix(nside=int(nside), order="ring")
    lon_pix, lat_pix = hpix.healpix_to_lonlat(pix)
    lon_pix = lon_pix.to_value(u.rad)
    lat_pix = lat_pix.to_value(u.rad)
    v_pix = np.vstack(
        [
            np.cos(lat_pix) * np.cos(lon_pix),
            np.cos(lat_pix) * np.sin(lon_pix),
            np.sin(lat_pix),
        ]
    ).astype(np.float64)

    weighted_cosz_sum = np.zeros(npix, dtype=np.float64)
    total_livetime = 0.0

    def _get_zenith_radec(sc_data: dict) -> tuple[np.ndarray, np.ndarray]:
        candidates = (
            ("ra_zenith", "dec_zenith"),
            ("ra_zen", "dec_zen"),
            ("RA_ZENITH", "DEC_ZENITH"),
            ("RA_ZEN", "DEC_ZEN"),
        )
        for ra_key, dec_key in candidates:
            if ra_key in sc_data and dec_key in sc_data:
                return np.asarray(sc_data[ra_key]), np.asarray(sc_data[dec_key])
        raise KeyError("Could not find zenith RA/Dec fields in sc_data.")

    for wf in week_files:
        with open(Path(wf), "rb") as inp:
            week_data = pickle.load(inp)

        sc_data = week_data.get("sc_data", None)
        if sc_data is None:
            continue

        lt = np.asarray(sc_data.get("livetime", []), dtype=np.float64)
        if lt.size == 0:
            continue

        ra_zen, dec_zen = _get_zenith_radec(sc_data)
        zenith = SkyCoord(ra_zen, dec_zen, unit="deg", frame="fk5")
        if frame.lower() == "galactic":
            zframe = zenith.galactic
            lon = np.asarray(getattr(zframe, "l").deg)
            lat = np.asarray(getattr(zframe, "b").deg)
        elif frame.lower() == "fk5":
            lon = np.asarray(getattr(zenith, "ra").deg)
            lat = np.asarray(getattr(zenith, "dec").deg)
        else:
            raise ValueError("frame must be 'galactic' or 'fk5'")

        th_z = np.radians(90.0 - lat)
        ph_z = np.radians(lon)
        v_z = np.vstack(
            [
                np.sin(th_z) * np.cos(ph_z),
                np.sin(th_z) * np.sin(ph_z),
                np.cos(th_z),
            ]
        ).astype(np.float64)

        n = lt.size
        for i0 in range(0, n, int(chunk_size)):
            i1 = min(i0 + int(chunk_size), n)
            vz = v_z[:, i0:i1]
            ltc = lt[i0:i1]

            cosz_chunk = np.dot(v_pix.T, vz)
            np.clip(cosz_chunk, -1.0, 1.0, out=cosz_chunk)
            weighted_cosz_sum += np.dot(cosz_chunk, ltc)
            total_livetime += ltc.sum()

    if total_livetime <= 0:
        raise ValueError("No livetime found in selected spacecraft data.")

    mean_cosz = np.clip(weighted_cosz_sum / total_livetime, -1.0, 1.0)
    zenith_deg_map = np.degrees(np.arccos(mean_cosz))
    return zenith_deg_map.astype(np.float32)


def read_livetime_map(path: str | Path, dtype=np.float32) -> np.ndarray:
    """Read a saved livetime HEALPix map from FITS."""
    inpath = Path(path)
    if not inpath.exists():
        raise FileNotFoundError(f"Could not find {inpath}")
    return _read_healpix_fits(inpath, dtype=dtype)


if __name__ == "__main__":
    raise SystemExit(main())
