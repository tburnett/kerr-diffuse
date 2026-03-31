"""Exposure map utilities for HEALPix workflows."""

from __future__ import annotations

import numpy as np
from astropy_healpix import npix_to_nside, nside_to_pixel_area

__all__ = ["make_exposure_map_healpix"]


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
    event_class=-1,
    n_energy=32,
    spectrum=None,
    ctmin=0.64,
    ctmax=1.0,
    n_costh=96,
    n_theta=512,
    lmax=None,
    costh_weight=None,
    zenith_deg=None,
    zmax_deg=None,
):
    """Create a HEALPix exposure map via spherical convolution."""
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

    energies = np.logspace(np.log10(emin), np.log10(emax), int(n_energy))
    wE = spectrum(energies) if spectrum is not None else energies ** (-2.0)
    wE = np.asarray(wE, dtype=np.float64)
    if wE.shape != energies.shape:
        raise ValueError("spectrum must return array with same shape as energies")
    den_E = np.trapz(wE, energies)
    if den_E <= 0 or not np.isfinite(den_E):
        raise ValueError("Invalid energy weight normalization")

    cth = np.linspace(ctmin, ctmax, int(n_costh))
    wc = costh_weight(cth) if costh_weight is not None else np.ones_like(cth)
    wc = np.asarray(wc, dtype=np.float64)
    if wc.shape != cth.shape:
        raise ValueError("costh_weight must return array with same shape as cos(theta) grid")
    den_cth = np.trapz(wc, cth)
    if den_cth <= 0 or not np.isfinite(den_cth):
        raise ValueError("Invalid cos(theta) weight normalization")

    aeff_e = np.empty((energies.size, cth.size), dtype=np.float64)
    for i, e in enumerate(energies):
        vals = Aeff(e, cth, event_class=event_class)
        if isinstance(vals, tuple):
            vals = vals[0] + vals[1]
        vals = np.asarray(vals, dtype=np.float64)
        if vals.shape != cth.shape:
            raise ValueError("Aeff(E, cos_theta, ...) must return shape matching cos(theta) grid")
        aeff_e[i] = vals

    aeff_band_cth = np.trapz(aeff_e * wE[:, None], energies, axis=0) / den_E
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
