import pytest
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd

sourcelist_mod = pytest.importorskip("like3.sourcelist")

SourceModel = sourcelist_mod.SourceModel
LocalizedSourceView = sourcelist_mod.LocalizedSourceView


class FakeCatalog(pd.DataFrame):
    @property
    def _constructor(self):
        return FakeCatalog

    @property
    def skycoord(self):
        return SkyCoord(self.ra.values, self.dec.values, unit="deg", frame="icrs")

    def select_cone(self, other, cone_size=0.5):
        sep = self.skycoord.separation(other).deg
        mask = sep < cone_size
        subset = self.loc[mask].copy()
        subset["sep"] = sep[mask]
        return subset


class LogParabola:
    def __init__(self, pars, e0=1000.0):
        self.pars = np.asarray(pars, dtype=float)
        self.e0 = float(e0)


class PowerLaw:
    def __init__(self, pars, e0=1000.0):
        self.pars = np.asarray(pars, dtype=float)
        self.e0 = float(e0)


class PLSuperExpCutoff4:
    def __init__(self, pars, e0=1000.0):
        self.pars = np.asarray(pars, dtype=float)
        self.e0 = float(e0)


def make_fake_catalog():
    return FakeCatalog(
        {
            "ra": [10.0, 11.0, 40.0],
            "dec": [0.0, 0.2, 5.0],
            "significance": [50.0, 15.0, 40.0],
            "specfunc": [
                LogParabola([1e-11, 2.1, 0.2, 1500.0]),
                PowerLaw([3e-12, 2.4], e0=2000.0),
                PLSuperExpCutoff4([2e-12, 1.8, 0.5, 1.0], e0=1200.0),
            ],
        },
        index=pd.Index(["SrcA", "SrcB", "SrcC"], name="4FGL"),
    )


def test_localization_view_returns_localized_source_view():
    source_model = SourceModel.demo(src_key=2)

    with source_model.localization_view("Blazar") as loc:
        assert isinstance(loc, LocalizedSourceView)
        assert loc.source.name == "Blazar"


def test_localization_view_accepts_source_object():
    source_model = SourceModel.demo(src_key=2)
    src = source_model.find_source("Blazar")

    with source_model.localization_view(src) as loc:
        assert isinstance(loc, LocalizedSourceView)
        assert loc.source is src


def test_localized_source_view_delta_ts_for_position_and_noarg_loglike():
    source_model = SourceModel.demo(src_key=2)

    with source_model.localization_view("Blazar") as loc:
        l0 = float(loc.source.skydir.galactic.l.deg)
        b0 = float(loc.source.skydir.galactic.b.deg)
        trial = SkyCoord(l0 + 1.0, b0, unit="deg", frame="galactic")

        # Style 1: loglike callable accepts trial position directly.
        def loglike_position(position):
            return float(position.galactic.l.deg)

        ts_direct = loc.delta_ts(loglike_position, position=trial)
        assert ts_direct == pytest.approx(2.0)

        # Style 2: loglike callable uses current source state only.
        def loglike_noarg():
            return float(loc.source.skydir.galactic.l.deg)

        ts_noarg = loc.delta_ts(loglike_noarg, position=trial)
        assert ts_noarg == pytest.approx(2.0)

        # No-arg fallback should restore source position after evaluation.
        assert float(loc.source.skydir.galactic.l.deg) == pytest.approx(l0)

    # Context manager should restore original selected-source position on exit.
    restored = source_model.find_source("Blazar").skydir
    assert float(restored.galactic.l.deg) == pytest.approx(l0)
    assert float(restored.galactic.b.deg) == pytest.approx(b0)


def test_from_fermi_catalog_selects_named_subset():
    source_model = SourceModel.from_fermi_catalog(
        catalog=make_fake_catalog(),
        select=["SrcA", "SrcC"],
    )

    assert list(source_model.source_names) == ["SrcA", "SrcC"]
    assert source_model[0].model.name == "LogParabola"
    assert source_model[1].model.name == "PLSuperExpCutoff4"


def test_from_fermi_catalog_applies_query_before_building_sources():
    source_model = SourceModel.from_fermi_catalog(
        catalog=make_fake_catalog(),
        query="significance >= 40",
    )

    assert list(source_model.source_names) == ["SrcA", "SrcC"]


def test_from_fermi_catalog_supports_cone_subset():
    source_model = SourceModel.from_fermi_catalog(
        catalog=make_fake_catalog(),
        skydir=(10.1, 0.0),
        cone_size=1.0,
    )

    assert list(source_model.source_names) == ["SrcA", "SrcB"]


def test_from_fermi_catalog_raises_when_subset_is_empty():
    with pytest.raises(sourcelist_mod.SourceModelException, match="no sources selected"):
        SourceModel.from_fermi_catalog(
            catalog=make_fake_catalog(),
            query="significance > 1000",
        )
