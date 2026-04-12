"""Compatibility shim for legacy like3.bands imports.

The PixelTable implementation has moved to like3.pixel_table.
This module re-exports legacy names so older imports keep working.
"""

from .pixel_table import (
    PixelTable,
    PixelTableLocalizationView,
    _PixelTableLocalizationContext,
)

# Legacy aliases
BandList = PixelTable
Band = PixelTable.Band

# Optional compatibility export for older callers.
from .likelihood import BandModel

__all__ = [
    "PixelTable",
    "BandList",
    "Band",
    "PixelTableLocalizationView",
    "_PixelTableLocalizationContext",
    "BandModel",
]
