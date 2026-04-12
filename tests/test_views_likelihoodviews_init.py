import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from like3.views import LikelihoodViews


class DummyParameters:
    pass


class DummySourceModel:
    def __init__(self):
        self.parameters = DummyParameters()


class PixelTableLike:
    def __init__(self):
        self.source_model = DummySourceModel()

    def _iter_bands(self):
        return iter(())


def test_init_accepts_pixel_table_like_instance():
    pt = PixelTableLike()

    lv = LikelihoodViews(pt)

    assert lv.pixel_table is pt
    assert lv.bands is pt
    assert lv.sources is pt.source_model
    assert lv.parameterset is pt.source_model.parameters


def test_init_accepts_legacy_bands_and_sources_signature():
    bands = object()
    sources = DummySourceModel()

    lv = LikelihoodViews(bands, sources)

    assert lv.bands is bands
    assert lv.sources is sources
    assert lv.parameterset is sources.parameters


def test_init_rejects_invalid_single_argument():
    try:
        LikelihoodViews(object())
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for invalid constructor input")
