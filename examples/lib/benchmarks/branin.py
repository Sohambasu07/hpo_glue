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

def branin_desc() -> Iterator[BenchmarkDescription]:

    env = Env(
        name="branin",
        python_version="3.10",
        requirements=(),
        post_install=(),
    )
    name = "branin"
    yield BenchmarkDescription(
        name=name,
        load=partial(_branin_surrogate),
        costs=None,
        fidelities=None,
        metrics={
            "value": Measure.metric((0.0, np.inf), minimize=True),
        },
        has_conditionals=False,
        is_tabular=False,
        env=env,
        mem_req_mb = 100,
    )


def _branin_surrogate(
    description: BenchmarkDescription
) -> SurrogateBenchmark:
    branin_space = ConfigurationSpace()
    for i in range(2):
        branin_space.add(Float(name=f"x{i}", bounds=[-32.768, 32.768]))
    return SurrogateBenchmark(
        desc=description,
        benchmark=branin,
        config_space=branin_space,
        query=partial(branin),
    )


def branin(query: Query) -> Result:

    x = np.array(query.config.to_tuple())
    x1 = x[0]
    x2 = x[1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    out = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    return Result(
        query=query,
        fidelity=None,
        values={"value": out},
    )
