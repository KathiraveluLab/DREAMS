"""Compatibility shim for legacy imports.

This module re-exports the implementation that moved to
`dreamsApp.core.extra.location_extractor`.
"""

from dreamsApp.core.extra import location_extractor as _impl
import sys

sys.modules[__name__] = _impl
