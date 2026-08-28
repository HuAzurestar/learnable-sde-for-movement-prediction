"""Parameter-capacity preregistration helpers for NEX-381.

Import functions from ``count_parameters`` explicitly.  Keeping package import
side-effect free also lets ``python -m ...count_parameters`` run without loading
the target module twice.
"""
