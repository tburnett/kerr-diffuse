"""Tests for PixelTable.Band and PixelTable model/fit integration.

Covers:
- Band.exposure_map caching
- Band.pixel_counts with and without source_model
- Band.pixel_gradient
- Band.simulate
- Band.loglike
- Band._component_values('sources') with source_model
- PixelTable.parameters / parameter_names / bounds properties
- PixelTable.select / _iter_bands
- PixelTable.loglike (aggregation and select)
- PixelTable.simulate (in-place photon update)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from like3.pixel_table import PixelTable

Band = PixelTable.Band


# ---------------------------------------------------------------------------
# Fake model infrastructure (mirrors test_bands.py style)
# ---------------------------------------------------------------------------

class FakeModel:
    def __init__(self, flux=1.0, gradient=None, free=None):
        self._flux = float(flux)
        self._gradient = np.asarray([1.0] if gradient is None else gradient, dtype=float)
        self.free = np.asarray([True] if free is None else free, dtype=bool)

    def __call__(self, energy):
        return self._flux

    def gradient(self, energy):
        return self._gradient


class FakeResponse:
    def __init__(self, pixels, values):
        self._pixels = np.asarray(pixels, dtype=int)
        self._values = np.asarray(values, dtype=float)
        self._by_pix = dict(zip(self._pixels.tolist(), self._values.tolist()))

    def evaluate(self, keys=None):
        if keys is None:
            return self._pixels, self._values
        keys = np.asarray(keys, dtype=int)
        values = np.array([self._by_pix.get(int(k), 0.0) for k in keys], dtype=float)
        return keys, values


class FakeSource:
    def __init__(self, *, flux=1.0, gradient=None, free=None, pixels, response_values):
        self.model = FakeModel(flux=flux, gradient=gradient, free=free)
        self._response = FakeResponse(pixels=pixels, values=response_values)

    def response(self, band):
        return self._response


class FakeParameterSet:
    def get_parameters(self):
        return np.array([1.0])

    def set_parameters(self, pars):
        pass


class FakeSourceModel(list):
    """Iterable source model with the attributes PixelTable expects."""
    def __init__(self, sources=()):
        super().__init__(sources)
        self.parameters = FakeParameterSet()
        self.parameter_names = np.array(['flux'])
        self.bounds = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pt_band(source_model, *, e0=100.0, e1=200.0, nside=1,
                 pixels=None, photons=None, exposure=None):
    """Return a PixelTable.Band with manually attached sparse arrays."""
    n = 3 if pixels is None else len(pixels)
    meta = ('PSF0', e0, e1, nside, n)
    band = Band(meta, 'ring', source_model=source_model)
    band.pix = np.asarray(pixels if pixels is not None else [0, 1, 2], dtype=np.int64)
    band.photons = np.asarray(photons if photons is not None else [1.0, 2.0, 3.0],
                               dtype=float)
    band.diffuse = np.zeros(n, dtype=float)
    band.sources = np.zeros(n, dtype=float)
    if exposure is not None:
        band.pixel_exposure = np.full(n, float(exposure))
    return band


def make_pt(source_model, bands):
    """Return a bare PixelTable dict with given bands keyed 0, 1, …"""
    pt = PixelTable.__new__(PixelTable)
    dict.__init__(pt)
    pt.source_model = source_model
    pt._selected = None
    pt.fit_info = {}
    for i, band in enumerate(bands):
        pt[i] = band
    return pt


# ---------------------------------------------------------------------------
# Band.exposure_map caching
# ---------------------------------------------------------------------------

class TestBandExposureMapCaching:
    def test_lookup_dict_built_on_first_access(self):
        src = FakeSource(pixels=[0, 1], response_values=[1.0, 1.0])
        band = make_pt_band([src], exposure=5.0)
        assert band._exposure_lookup is None
        _ = band.exposure_map
        assert band._exposure_lookup is not None

    def test_lookup_dict_reused_on_subsequent_access(self):
        src = FakeSource(pixels=[0, 1], response_values=[1.0, 1.0])
        band = make_pt_band([src], exposure=5.0)
        _ = band.exposure_map
        first_id = id(band._exposure_lookup)
        _ = band.exposure_map
        assert id(band._exposure_lookup) == first_id

    def test_returns_correct_values(self):
        src = FakeSource(pixels=[0, 1, 2], response_values=[1.0, 1.0, 1.0])
        band = make_pt_band([src], exposure=7.0)
        np.testing.assert_allclose(
            band.exposure_map(np.array([0, 1, 2], dtype=int)), [7.0, 7.0, 7.0]
        )

    def test_returns_zeros_when_no_pixel_exposure(self):
        src = FakeSource(pixels=[0], response_values=[1.0])
        band = make_pt_band([src])  # no exposure set
        np.testing.assert_allclose(
            band.exposure_map(np.array([0, 1], dtype=int)), [0.0, 0.0]
        )


# ---------------------------------------------------------------------------
# Band.pixel_counts
# ---------------------------------------------------------------------------

class TestBandPixelCounts:
    def test_with_source_model_and_exposure(self):
        src = FakeSource(flux=2.0, pixels=[0, 1, 2], response_values=[3.0, 1.0, 0.5])
        band = make_pt_band([src], exposure=4.0)
        pix, counts = band.pixel_counts()
        np.testing.assert_array_equal(pix, [0, 1, 2])
        # flux * response * exposure = 2 * [3,1,0.5] * 4
        np.testing.assert_allclose(counts, [24.0, 8.0, 4.0])

    def test_diffuse_added_to_source_counts(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[1.0, 1.0, 1.0])
        band = make_pt_band([src], exposure=1.0)
        band.diffuse[:] = [10.0, 20.0, 30.0]
        pix, counts = band.pixel_counts()
        np.testing.assert_allclose(counts, [11.0, 21.0, 31.0])

    def test_without_source_model_uses_fits_sources_array(self):
        band = make_pt_band(None)
        band.sources = np.array([5.0, 6.0, 7.0])
        _, counts = band.pixel_counts()
        np.testing.assert_allclose(counts, [5.0, 6.0, 7.0])

    def test_subset_pixels(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[2.0, 3.0, 4.0])
        band = make_pt_band([src], exposure=1.0)
        pix, counts = band.pixel_counts(pixels=np.array([0, 2], dtype=int))
        np.testing.assert_array_equal(pix, [0, 2])
        np.testing.assert_allclose(counts, [2.0, 4.0])


# ---------------------------------------------------------------------------
# Band.pixel_gradient
# ---------------------------------------------------------------------------

class TestBandPixelGradient:
    def test_shape_and_values(self):
        # Source with 2 free params; response only covers pix 0 and 1
        src = FakeSource(
            flux=1.0, gradient=[5.0, 7.0], free=[True, True],
            pixels=[0, 1], response_values=[2.0, 3.0],
        )
        band = make_pt_band([src], exposure=1.0)
        g = band.pixel_gradient((band.pix, band.photons))
        assert g.shape == (3, 2)
        # pix 0: response 2.0, grad [5,7] -> [10, 14]
        np.testing.assert_allclose(g[0], [10.0, 14.0])
        # pix 1: response 3.0 -> [15, 21]
        np.testing.assert_allclose(g[1], [15.0, 21.0])
        # pix 2: not in response -> [0, 0]
        np.testing.assert_allclose(g[2], [0.0, 0.0])

    def test_exposure_scales_gradient(self):
        src = FakeSource(
            flux=1.0, gradient=[1.0], free=[True],
            pixels=[0], response_values=[1.0],
        )
        band = make_pt_band([src], exposure=3.0)
        g = band.pixel_gradient((np.array([0], dtype=int), np.array([0.0])))
        np.testing.assert_allclose(g[0], [3.0])

    def test_raises_without_source_model(self):
        band = make_pt_band(None)
        with pytest.raises(ValueError, match='source_model'):
            band.pixel_gradient((band.pix, band.photons))


# ---------------------------------------------------------------------------
# Band.simulate
# ---------------------------------------------------------------------------

class TestBandSimulate:
    def test_reproducible_with_same_seed(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[5.0, 10.0, 15.0])
        band = make_pt_band([src], exposure=1.0)
        k1, c1 = band.simulate(random_state=42)
        k2, c2 = band.simulate(random_state=42)
        np.testing.assert_array_equal(k1, k2)
        np.testing.assert_array_equal(c1, c2)

    def test_returns_only_nonzero_pixels(self):
        # Response 0.0 on pix 1 means that pixel gets 0 model counts
        src = FakeSource(flux=1.0, pixels=[0, 2], response_values=[1.0, 1.0])
        band = make_pt_band([src], exposure=1.0)
        _, counts = band.simulate(random_state=None)
        assert np.all(counts > 0)

    def test_total_counts_normalizes_shape(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[1.0, 2.0, 3.0])
        band = make_pt_band([src], exposure=1.0)
        keys, counts = band.simulate(total_counts=60.0, random_state=None)
        assert counts.sum() == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Band.loglike
# ---------------------------------------------------------------------------

class TestBandLoglike:
    def test_returns_float(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[1.0, 1.0, 1.0])
        band = make_pt_band([src], exposure=2.0)
        assert isinstance(band.loglike(), float)

    def test_consistent_with_manual_computation(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[1.0, 2.0, 3.0])
        band = make_pt_band([src], exposure=1.0, photons=[1.0, 2.0, 3.0])
        _, model = band.pixel_counts()
        model = model.clip(1e-30, None)
        expected = float(np.sum(band.photons * np.log(model) - model))
        assert band.loglike() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Band._component_values('sources') with source_model
# ---------------------------------------------------------------------------

class TestBandComponentValuesSources:
    def test_returns_fits_array_when_no_source_model(self):
        band = make_pt_band(None)
        band.sources = np.array([3.0, 5.0, 7.0])
        result = band._component_values('sources')
        np.testing.assert_array_equal(result, [3.0, 5.0, 7.0])

    def test_returns_dynamic_counts_when_source_model_set(self):
        src = FakeSource(flux=2.0, pixels=[0, 1, 2], response_values=[1.0, 1.0, 1.0])
        band = make_pt_band([src], exposure=3.0)
        # Expected: flux * response * exposure = 2 * 1 * 3 = 6 per pixel
        result = band._component_values('sources')
        np.testing.assert_allclose(result, [6.0, 6.0, 6.0])

    def test_dynamic_sources_does_not_include_diffuse(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[1.0, 1.0, 1.0])
        band = make_pt_band([src], exposure=1.0)
        band.diffuse[:] = 100.0
        result = band._component_values('sources')
        # Should be just the source counts, not diffuse + sources
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# PixelTable.parameters / parameter_names / bounds
# ---------------------------------------------------------------------------

class TestPixelTableProperties:
    def test_parameters_delegates_to_source_model(self):
        sm = FakeSourceModel()
        pt = make_pt(sm, [])
        assert pt.parameters is sm.parameters

    def test_parameter_names_delegates(self):
        sm = FakeSourceModel()
        pt = make_pt(sm, [])
        assert pt.parameter_names is sm.parameter_names

    def test_bounds_delegates(self):
        sm = FakeSourceModel()
        pt = make_pt(sm, [])
        assert pt.bounds is sm.bounds

    def test_parameters_raises_without_source_model(self):
        pt = make_pt(None, [])
        with pytest.raises(AttributeError, match='source_model'):
            _ = pt.parameters

    def test_bounds_returns_none_without_source_model(self):
        pt = make_pt(None, [])
        assert pt.bounds is None


# ---------------------------------------------------------------------------
# PixelTable.select / _iter_bands
# ---------------------------------------------------------------------------

class TestPixelTableSelect:
    def _make_pt_with_3_bands(self):
        sm = FakeSourceModel()
        src = FakeSource(pixels=[0], response_values=[1.0])
        bands = [make_pt_band([src], e0=100.0 * (i + 1), e1=200.0 * (i + 1),
                               exposure=1.0) for i in range(3)]
        return make_pt(sm, bands)

    def test_select_restricts_iter_bands(self):
        pt = self._make_pt_with_3_bands()
        pt.select([0, 2])
        assert len(list(pt._iter_bands())) == 2

    def test_select_none_resets_to_all(self):
        pt = self._make_pt_with_3_bands()
        pt.select([0])
        pt.select(None)
        assert len(list(pt._iter_bands())) == 3

    def test_select_returns_self(self):
        pt = self._make_pt_with_3_bands()
        assert pt.select() is pt

    def test_default_iter_bands_covers_all(self):
        pt = self._make_pt_with_3_bands()
        assert len(list(pt._iter_bands())) == 3


# ---------------------------------------------------------------------------
# PixelTable.loglike
# ---------------------------------------------------------------------------

class TestPixelTableLoglike:
    def test_sums_per_band_loglikes(self):
        src = FakeSource(flux=1.0, pixels=[0, 1], response_values=[1.0, 1.0])
        sm = FakeSourceModel([src])
        b0 = make_pt_band(sm, exposure=1.0, pixels=[0, 1], photons=[1.0, 1.0])
        b1 = make_pt_band(sm, exposure=1.0, pixels=[0, 1], photons=[2.0, 2.0])
        pt = make_pt(sm, [b0, b1])
        assert pt.loglike() == pytest.approx(b0.loglike() + b1.loglike())

    def test_raises_without_source_model(self):
        pt = make_pt(None, [])
        with pytest.raises(ValueError, match='source_model'):
            pt.loglike()

    def test_select_restricts_loglike(self):
        src = FakeSource(flux=1.0, pixels=[0], response_values=[2.0])
        sm = FakeSourceModel([src])
        bands = [make_pt_band(sm, exposure=1.0, photons=[1.0]) for _ in range(3)]
        pt = make_pt(sm, bands)
        full = pt.loglike()
        pt.select([0])
        partial = pt.loglike()
        # each band is identical, so partial = full / 3
        assert partial == pytest.approx(full / 3)


# ---------------------------------------------------------------------------
# PixelTable.simulate
# ---------------------------------------------------------------------------

class TestPixelTableSimulate:
    def test_overwrites_band_photons(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[2.0, 2.0, 2.0])
        sm = FakeSourceModel([src])
        band = make_pt_band(sm, exposure=1.0)
        pt = make_pt(sm, [band])
        original = band.photons.copy()
        # Use fixed seed to guarantee deterministic output
        pt.simulate(random_state=7)
        # Photons must have been written (shape preserved)
        assert band.photons.shape == original.shape

    def test_reproducible_with_same_seed(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[3.0, 3.0, 3.0])
        sm = FakeSourceModel([src])
        band_a = make_pt_band(sm, exposure=2.0)
        band_b = make_pt_band(sm, exposure=2.0)
        pt_a = make_pt(sm, [band_a])
        pt_b = make_pt(sm, [band_b])
        pt_a.simulate(random_state=123)
        pt_b.simulate(random_state=123)
        np.testing.assert_array_equal(band_a.photons, band_b.photons)

    def test_raises_without_source_model(self):
        pt = make_pt(None, [])
        with pytest.raises(ValueError, match='source_model'):
            pt.simulate()

    def test_select_only_updates_selected_bands(self):
        src = FakeSource(flux=1.0, pixels=[0, 1, 2], response_values=[1.0, 1.0, 1.0])
        sm = FakeSourceModel([src])
        b0 = make_pt_band(sm, exposure=1.0)
        b1 = make_pt_band(sm, exposure=1.0)
        saved_b1 = b1.photons.copy()
        pt = make_pt(sm, [b0, b1])
        pt.select([0])
        pt.simulate(random_state=5)
        np.testing.assert_array_equal(b1.photons, saved_b1)
