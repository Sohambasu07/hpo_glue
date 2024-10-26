from __future__ import annotations

from typing import TYPE_CHECKING

from lib.benchmarks.ackley import ackley_desc
from lib.benchmarks.ackley2 import ackley_bench
from lib.benchmarks.branin import branin_desc

if TYPE_CHECKING:
    from hpo_glue.benchmark import BenchmarkDescription

BENCHMARKS: dict[str, BenchmarkDescription] = {}
for desc in ackley_desc():
    BENCHMARKS[desc.name] = desc
for desc in branin_desc():
    BENCHMARKS[desc.name] = desc
BENCHMARKS["ackley2"] = ackley_bench().description

__all__ = [
    "BENCHMARKS"
]
