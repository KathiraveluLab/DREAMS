"""Compatibility shim for legacy imports.

This module re-exports the implementation that moved to
`dreams_app.core.extra.location_extractor`.
"""

from dreams_app.core.extra import location_extractor as _impl
import sys

sys.modules[__name__] = _impl
