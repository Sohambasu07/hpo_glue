from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace, Float

from hpo_glue.benchmark import BenchmarkDescription, SurrogateBenchmark
from hpo_glue.env import Env
from hpo_glue.measure import Measure
from hpo_glue.result import Result

if TYPE_CHECKING:

    from hpo_glue.query import Query

def ackley_desc() -> Iterator[BenchmarkDescription]:

    env = Env(
        name="ackley",
        python_version="3.10",
        requirements=(),
        post_install=(),
    )
    name = "ackley"
    yield BenchmarkDescription(
        name=name,
        load=partial(_ackley_surrogate),
        costs=None,
        fidelities=None,
        metrics={
            "value": Measure.metric((0.0, np.inf), minimize=True),
        },
        has_conditionals=False,
        is_tabular=False,
        env=env,
        mem_req_MB = 100,
    )


def _ackley_surrogate(
    description: BenchmarkDescription
) -> SurrogateBenchmark:
    ackley_space = ConfigurationSpace()
    for i in range(2):
        ackley_space.add(Float(name=f"x{i}", bounds=[-32.768, 32.768]))
    return SurrogateBenchmark(
        desc=description,
        benchmark=ackley,
        config_space=ackley_space,
        query=partial(ackley),
    )


def ackley(query: Query) -> Result:

    n_var=2
    a=20
    b=1/5
    c=2 * np.pi
    x = np.array(query.config.to_tuple())
    part1 = -1. * a * np.exp(-1. * b * np.sqrt((1. / n_var) * np.sum(x * x)))
    part2 = -1. * np.exp((1. / n_var) * np.sum(np.cos(c * x)))
    out = part1 + part2 + a + np.exp(1)

    return Result(
        query=query,
        fidelity=None,
        values={"value": out},
    )
