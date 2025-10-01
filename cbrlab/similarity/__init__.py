"""Similarity utilities for CBR Lab."""

from importlib import import_module

from . import local as _local

_global_mod = import_module("cbrlab.similarity.global")
aggregate = _global_mod.aggregate
sim_bool = _local.sim_bool
sim_cat = _local.sim_cat
sim_numeric = _local.sim_numeric

__all__ = ["aggregate", "sim_bool", "sim_cat", "sim_numeric"]
