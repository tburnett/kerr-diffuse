"""Helpers for source-centric PSF studies using like3/FermiFit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from like3.main import FermiFit
from like3.pixel_table import PixelTable
from like3.sourcelist import SourceModel

DEFAULT_PSF_NAMES = ("PSF0", "PSF1", "PSF2", "PSF3")


def chdir_to_git_root(start_path: str | Path | None = None) -> Path:
    """Walk up parent folders, chdir to the first one containing ``.git``.

    Parameters
    ----------
    start_path : str, Path, or None
        Directory to start searching from. Defaults to current working directory.

    Returns
    -------
    pathlib.Path
        Resolved path that contains ``.git`` and is now the current directory.

    Raises
    ------
    FileNotFoundError
        If no parent directory contains a ``.git`` entry.
    """
    start = Path.cwd() if start_path is None else Path(start_path)
    start = start.resolve()
    if start.is_file():
        start = start.parent

    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            os.chdir(candidate)
            return candidate

    raise FileNotFoundError(f"No parent folder with .git found from {start}")


@dataclass
class PSFStudyContext:
    """Container for objects produced when setting up a PSF study."""

    source_name: str
    roi_center: SkyCoord
    roi_sources: SourceModel
    selected_name: str
    pixel_table: PixelTable
    ff: FermiFit


def setup_psf_study(
    source_name: str,
    *,
    cone_size: float = 1.0,
    catalog: str = "v40",
    query: str = "significance >= 25",
    pixel_table_path: str = "files/kerr/toby_v4.fits",
    psf_table_path: str = "files/loc",
) -> PSFStudyContext:
    """Build and return a PSF-study analysis context for a source."""
    roi_center = SkyCoord.from_name(source_name)
    roi_sources = SourceModel.from_fermi_catalog(
        catalog,
        skydir=roi_center,
        cone_size=cone_size,
        query=query,
    )
    selected_name = roi_sources.source_names[0]
    pt = PixelTable(pixel_table_path, source_model=roi_sources)
    # pt.set_psf(table_path=psf_table_path)
    ff = FermiFit(pt)
    return PSFStudyContext(
        source_name=source_name,
        roi_center=roi_center,
        roi_sources=roi_sources,
        selected_name=selected_name,
        pixel_table=pt,
        ff=ff,
    )


def get_sed_table(ff: FermiFit, source_name: str, *, tol: float = 0.1) -> pd.DataFrame:
    """Return all-event SED Poisson table for a source."""
    return ff.get_sed_poisson_table(source_name=source_name, tol=tol)


def get_psf_sed_table(
    ff: FermiFit,
    source_name: str,
    *,
    psf_names: tuple[str, ...] | list[str] = DEFAULT_PSF_NAMES,
    tol: float = 0.1,
) -> pd.DataFrame:
    """Return concatenated PSF-resolved SED table indexed by (psf, energy)."""
    sed_by_psf = {
        psf_name: ff.get_sed_poisson_table(source_name=source_name, event_type=psf_name, tol=tol)
        for psf_name in psf_names
    }
    return pd.concat(sed_by_psf, names=["psf", "energy"])


def make_flux_ratio_table(sed_all: pd.DataFrame, sed_table: pd.DataFrame) -> pd.DataFrame:
    """Normalize PSF-resolved flux columns by all-event SED flux in each band."""
    sed_flux_by_energy = sed_table["flux"].rename("sed_flux")
    ratio_table = sed_all.join(sed_flux_by_energy, on="energy")
    for col in ("flux", "lflux", "uflux"):
        ratio_table[f"{col}_ratio"] = ratio_table[col] / ratio_table["sed_flux"]
    return ratio_table


def plot_source_sed(
    ff: FermiFit,
    source_name: str,
    *,
    sed_table: pd.DataFrame | None = None,
    ylim: tuple[float, float] = (1.0, 3e3),
    ax=None,
):
    """Plot source SED with per-band points and return the axes."""
    src = ff.source_model.find_source(source_name)
    return ff.plot_sed_with_band_points(
        src,
        sed_table=sed_table,
        set_kwargs=dict(ylim=ylim),
        ax=ax,
    )


def plot_flux_ratio_errorbars(
    ratio_table: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10, 8),
    ylim: tuple[float, float] | None = (0.8, 1.2),
    yscale: str = "linear",
    linestyle: str = ":",
    marker: str = "o",
    markersize: float = 10,
    capsize: float = 2,
    ts_min: float | None = None,
    title: str = "PSF Flux Ratio by Energy",
    ax=None,
):
    """Plot flux ratios vs energy for each PSF with asymmetric error bars."""
    plot_df = ratio_table.reset_index().copy()
    plot_df["energy_mev"] = np.sqrt(plot_df["elow"] * plot_df["ehigh"])

    if ts_min is not None and "ts" in plot_df.columns:
        plot_df = plot_df.loc[plot_df["ts"] >= float(ts_min)]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for psf, grp in plot_df.groupby("psf"):
        grp = grp.sort_values("energy_mev")
        ok = grp["flux_ratio"].notna() & grp["lflux_ratio"].notna() & grp["uflux_ratio"].notna()
        g = grp.loc[ok].copy()
        if len(g) == 0:
            continue

        y = g["flux_ratio"].to_numpy(dtype=float)
        yerr_lo = np.clip(y - g["lflux_ratio"].to_numpy(dtype=float), 0, np.inf)
        yerr_hi = np.clip(g["uflux_ratio"].to_numpy(dtype=float) - y, 0, np.inf)

        ax.errorbar(
            g["energy_mev"],
            y,
            yerr=np.vstack([yerr_lo, yerr_hi]),
            marker=marker,
            ms=markersize,
            ls=linestyle,
            capsize=capsize,
            label=psf,
        )

    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.set(
        xscale="log",
        yscale=yscale,
        xlabel="Energy (MeV)",
        ylabel="Flux / SED flux",
        title=title,
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(alpha=0.2)
    return ax


def main(
    source_name: str,
    *,
    cone_size: float = 1.0,
    catalog: str = "v40",
    query: str = "significance >= 25",
    pixel_table_path: str = "files/kerr/toby_v4.fits",
    psf_table_path: str = "files/loc",
    tol: float = 0.1,
    sed_ylim: tuple[float, float] = (1.0, 3e3),
    ratio_figsize: tuple[float, float] = (10, 8),
    ratio_ylim: tuple[float, float] | None = (0.8, 1.2),
    ratio_yscale: str = "linear",
    ratio_linestyle: str = ":",
    ratio_marker: str = "o",
    ratio_markersize: float = 10,
    ratio_capsize: float = 2,
    ratio_ts_min: float | None = None,
):
    """Run the full PSF study workflow and return key results.

    Returns
    -------
    dict
        Keys: ``study``, ``ff``, ``selected_name``, ``sedp``, ``sed_all``,
        ``ratio_table``, ``sed_ax``, ``ratio_ax``.
    """
    study = setup_psf_study(
        source_name=source_name,
        cone_size=cone_size,
        catalog=catalog,
        query=query,
        pixel_table_path=pixel_table_path,
        psf_table_path=psf_table_path,
    )
    ff = study.ff
    selected_name = study.selected_name

    sedp = get_sed_table(ff, selected_name, tol=tol)
    sed_all = get_psf_sed_table(ff, selected_name, tol=tol)
    ratio_table = make_flux_ratio_table(sed_all, sedp)

    sed_ax = plot_source_sed(ff, selected_name, sed_table=sedp, ylim=sed_ylim)
    ratio_ax = plot_flux_ratio_errorbars(
        ratio_table,
        figsize=ratio_figsize,
        ylim=ratio_ylim,
        yscale=ratio_yscale,
        linestyle=ratio_linestyle,
        marker=ratio_marker,
        markersize=ratio_markersize,
        capsize=ratio_capsize,
        ts_min=ratio_ts_min,
        title=f"{source_name} PSF Flux Ratio by Energy",
    )

    return dict(
        study=study,
        ff=ff,
        selected_name=selected_name,
        sedp=sedp,
        sed_all=sed_all,
        ratio_table=ratio_table,
        sed_ax=sed_ax,
        ratio_ax=ratio_ax,
    )
