"""
Core algorithms and pipelines for the DREAMS ecosystem.
"""
__all__ = ["DreamsPipeline"]


def __getattr__(name):
	if name == "DreamsPipeline":
		from .pipeline import DreamsPipeline
		return DreamsPipeline
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
