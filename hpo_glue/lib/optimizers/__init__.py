from __future__ import annotations

from typing import TYPE_CHECKING

from hpo_glue.lib.optimizers.random_search import RandomSearch

if TYPE_CHECKING:
    from hpo_glue.optimizer import Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    RandomSearch.name: RandomSearch,
}

MF_OPTIMIZERS: dict[str, type[Optimizer]] = {}
BB_OPTIMIZERS: dict[str, type[Optimizer]] = {}
MO_OPTIMIZERS: dict[str, type[Optimizer]] = {}
SO_OPTIMIZERS: dict[str, type[Optimizer]] = {}

for name, opt in OPTIMIZERS.items():
    if "single" in opt.support.fidelities:
        MF_OPTIMIZERS[name] = opt
    else:
        BB_OPTIMIZERS[name] = opt
    if "many" in opt.support.objectives:
        MO_OPTIMIZERS[name] = opt
    if "single" in opt.support.objectives:
        SO_OPTIMIZERS[name] = opt
__all__ = [
    "OPTIMIZERS",
    "MF_OPTIMIZERS",
    "BB_OPTIMIZERS",
    "MO_OPTIMIZERS",
    "SO_OPTIMIZERS",
]