from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from hpo_glue.budget import CostBudget, TrialBudget
from hpo_glue.config import Config
from hpo_glue.constants import DEFAULT_RELATIVE_EXP_DIR
from hpo_glue.fidelity import Fidelity, ListFidelity, RangeFidelity
from hpo_glue.measure import Measure
from hpo_glue.optimizer import Optimizer
from hpo_glue.query import Query
from hpo_glue.result import Result

if TYPE_CHECKING:
    from hpo_glue.benchmark import BenchmarkDescription
    from hpo_glue.budget import BudgetType
    from hpo_glue.run import Run

logger = logging.getLogger(__name__)

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]


@dataclass(kw_only=True, unsafe_hash=True)
class Problem:
    """A problem to optimize over."""

    # NOTE: These are mainly for consumers who need to interact beyond forward facing API
    Config: TypeAlias = Config
    Query: TypeAlias = Query
    Result: TypeAlias = Result
    Measure: TypeAlias = Measure
    TrialBudget: TypeAlias = TrialBudget
    CostBudget: TypeAlias = CostBudget
    RangeFidelity: TypeAlias = RangeFidelity
    ListFidelity: TypeAlias = ListFidelity

    objective: tuple[str, Measure] | Mapping[str, Measure] = field(hash=False)
    """The metrics to optimize for this problem, with a specific order.

    If only one metric is specified, this is considered single objective and
    not multiobjective.
    """

    fidelity: tuple[str, Fidelity] | Mapping[str, Fidelity] | None = field(hash=False)
    """Fidelities to use from the Benchmark.

    When `None`, the problem is considered a black-box problem with no fidelity.

    When a single fidelity is specified, the problem is considered a _multi-fidelity_ problem.

    When many fidelities are specified, the problem is considered a _many-fidelity_ problem.
    """

    cost: tuple[str, Measure] | Mapping[str, Measure] | None = field(hash=False)
    """The cost metric to use for this proble.

    When `None`, the problem is considered a black-box problem with no cost.

    When a single cost is specified, the problem is considered a _cost-sensitive_ problem.

    When many costs are specified, the problem is considered a _multi-cost_ problem.
    """

    budget: BudgetType
    """The type of budget to use for the optimizer."""

    benchmark: BenchmarkDescription
    """The benchmark to use for this problem statement"""

    is_tabular: bool = field(init=False)
    """Whether the benchmark is tabular"""

    is_multiobjective: bool = field(init=False)
    """Whether the problem has multiple objectives"""

    is_multifidelity: bool = field(init=False)
    """Whether the problem has a fidelity parameter"""

    is_manyfidelity: bool = field(init=False)
    """Whether the problem has many fidelities"""

    supports_trajectory: bool = field(init=False)
    """Whether the problem setup allows for trajectories to be queried."""

    name: str = field(init=False)
    """The name of the problem statement.

    This is used to identify the problem statement.
    """

    precision: int = field(default=12) #TODO: Set default

    mem_req_MB: int = field(init=False)

    def __post_init__(self) -> None:
        self.mem_req_MB = self.benchmark.mem_req_MB
        self.is_tabular = self.benchmark.is_tabular
        self.is_manyfidelity: bool
        self.is_multifidelity: bool
        self.supports_trajectory: bool

        name_parts: list[str] = [
            f"benchmark={self.benchmark.name}",
            self.budget.path_str,
        ]

        self.is_multiobjective: bool
        match self.objective:
            case tuple():
                self.is_multiobjective = False
                # name_parts.append(f"objective={self.objective[0]}")
            case Mapping():
                if len(self.objective) == 1:
                    raise ValueError("Single objective should be a tuple, not a mapping")

                self.is_multiobjective = True
                # name_parts.append("objective=" + ",".join(self.objective.keys()))
            case _:
                raise TypeError("Objective must be a tuple (name, measure) or a mapping")

        match self.fidelity:
            case None:
                self.is_multifidelity = False
                self.is_manyfidelity = False
                self.supports_trajectory = False
            case (_name, _fidelity):
                self.is_multifidelity = True
                self.is_manyfidelity = False
                if _fidelity.supports_continuation:
                    self.supports_trajectory = True
                else:
                    self.supports_trajectory = False
                # name_parts.append(f"fidelity={_name}")
            case Mapping():
                if len(self.fidelity) == 1:
                    raise ValueError("Single fidelity should be a tuple, not a mapping")

                self.is_multifidelity = False
                self.is_manyfidelity = True
                self.supports_trajectory = False
                # name_parts.append("fidelity=" + ",".join(self.fidelity.keys()))
            case _:
                raise TypeError("Fidelity must be a tuple (name, fidelity) or a mapping")

        match self.cost:
            case None:
                pass
            case (_name, _measure):
                # name_parts.append(f"cost={_name}")
                pass
            case Mapping():
                if len(self.cost) == 1:
                    raise ValueError("Single cost should be a tuple, not a mapping")

                # name_parts.append("cost=" + ",".join(self.cost.keys()))

        self.name = ".".join(name_parts)

    def generate_runs(
        self,
        optimizers: (
            type[Optimizer]
            | OptWithHps
            | list[type[Optimizer]]
            | list[OptWithHps | type[Optimizer]]
        ),
        *,
        expdir: Path | str = DEFAULT_RELATIVE_EXP_DIR,
        seeds: Iterable[int],
        continuations: bool = False
    ) -> list[Run]:
        """Generate a set of problems for the given optimizer and benchmark.

        If there is some incompatibility between the optimizer, the benchmark and the requested
        amount of objectives, fidelities or costs, a ValueError will be raised.

        Args:
            optimizers: The optimizer class to generate problems for.
                Can provide a single optimizer or a list of optimizers.
                If you wish to provide hyperparameters for the optimizer, provide a tuple with the
                optimizer.
            expdir: Which directory to store experiment results into.
            seeds: The seed or seeds to use for the problems.

        Returns:
            An experiment object with the generated problems.
        """
        from hpo_glue.run import Run

        _seeds = seeds
        # if not isinstance(seeds, Iterable):
        #     _seeds = [seeds]
        _optimizers: list[OptWithHps]
        match optimizers:
            case tuple():
                _opt, hps = optimizers
                _optimizers = [(_opt, hps)]
            case list():
                _optimizers = [o if isinstance(o, tuple) else (o, {}) for o in optimizers]
            case _:
                _optimizers = [(optimizers, {})]

        for opt, _ in _optimizers:
            support: Problem.Support = opt.support
            support.check_opt_support(who=opt.name, problem=self)

        _runs_list = []
        for _seed, (opt, hps) in product(_seeds, _optimizers):
            if "single" not in opt.support.fidelities:
                continuations = False
            _runs_list.append(
                Run(
                problem=self,
                optimizer=opt,
                optimizer_hyperparameters=hps,
                seed=_seed,
                expdir=Path(expdir),
                continuations=continuations
                )
            )
        return _runs_list

    def group_for_optimizer_comparison(
        self,
    ) -> tuple[
        str,
        BudgetType,
        tuple[tuple[str, Measure], ...],
        None | tuple[tuple[str, Fidelity], ...],
        None | tuple[tuple[str, Measure], ...],
    ]:
        match self.objective:
            case (name, measure):
                _obj = ((name, measure),)
            case Mapping():
                _obj = tuple(self.objective.items())

        match self.fidelity:
            case None:
                _fid = None
            case (name, fid):
                _fid = ((name, fid),)
            case Mapping():
                _fid = tuple(self.fidelity.items())

        match self.cost:
            case None:
                _cost = None
            case (name, measure):
                _cost = ((name, measure),)
            case Mapping():
                _cost = tuple(self.cost.items())

        return (self.benchmark.name, self.budget, _obj, _fid, _cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": (
                list(self.objective.keys())
                if isinstance(self.objective, Mapping)
                else self.objective[0]
            ),
            "fidelity": (
                None
                if self.fidelity is None
                else (
                    list(self.fidelity.keys())
                    if isinstance(self.fidelity, Mapping)
                    else self.fidelity[0]
                )
            ),
            "cost": (
                None
                if self.cost is None
                else (list(self.cost.keys()) if isinstance(self.cost, Mapping) else self.cost[0])
            ),
            "budget_type": self.budget.name,
            "budget": self.budget.to_dict(),
            "benchmark": self.benchmark.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Problem:
        from hpo_glue.benchmarks import BENCHMARKS

        if data["benchmark"] not in BENCHMARKS:
            raise ValueError(
                f"Benchmark {data['benchmark']} not found in benchmarks!"
                " Please make sure your benchmark is registed in `BENCHMARKS`"
                " before loading/parsing."
            )

        benchmark = BENCHMARKS[data["benchmark"]]
        _obj = data["objective"]
        match _obj:
            case str():
                objective = (_obj, benchmark.metrics[_obj])
            case list():
                objective = {name: benchmark.metrics[name] for name in _obj}
            case _:
                raise ValueError("Objective must be a string or a list of strings")

        _fid = data["fidelity"]
        match _fid:
            case None:
                fidelity = None
            case str():
                assert benchmark.fidelities is not None
                fidelity = (_fid, benchmark.fidelities[_fid])
            case list():
                assert benchmark.fidelities is not None
                fidelity = {name: benchmark.fidelities[name] for name in _fid}
            case _:
                raise ValueError("Fidelity must be a string or a list of strings")

        _cost = data["cost"]
        match _cost:
            case None:
                cost = None
            case str():
                assert benchmark.costs is not None
                cost = (_cost, benchmark.costs[_cost])
            case list():
                assert benchmark.costs is not None
                cost = {name: benchmark.costs[name] for name in _cost}
            case _:
                raise ValueError("Cost must be a string or a list of strings")

        _budget_type = data["budget_type"]
        match _budget_type:
            case "trial_budget":
                budget = TrialBudget.from_dict(data["budget"])
            case "cost_budget":
                budget = CostBudget.from_dict(data["budget"])
            case _:
                raise ValueError("Budget type must be 'trial_budget' or 'cost_budget'")

        return cls(
            objective=objective,
            fidelity=fidelity,
            cost=cost,
            budget=budget,
            benchmark=benchmark,
        )

    @dataclass(kw_only=True)
    class Support:
        """The support of an optimizer for a problem."""

        objectives: tuple[Literal["single", "many"], ...]
        fidelities: tuple[Literal[None, "single", "many"], ...]
        cost_awareness: tuple[Literal[None, "single", "many"], ...]
        tabular: bool

        def check_opt_support(self, who: str, *, problem: Problem) -> None:
            """Check if the problem is supported by the support."""
            match problem.fidelity:
                case None if None not in self.fidelities:
                    raise ValueError(
                        f"Optimizer {who} does not support having no fidelties for {problem.name}!"
                    )
                case tuple() if "single" not in self.fidelities:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-fidelty for {problem.name}!"
                    )
                case Mapping() if "many" not in self.fidelities:
                    raise ValueError(
                        f"Optimizer {who} does not support many-fidelty for {problem.name}!"
                    )

            match problem.objective:
                case tuple() if "single" not in self.objectives:
                    raise ValueError(
                        f"Optimizer {who} does not support single-objective for {problem.name}!"
                    )
                case Mapping() if "many" not in self.objectives:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-objective for {problem.name}!"
                    )

            match problem.cost:
                case None if None not in self.cost_awareness:
                    raise ValueError(
                        f"Optimizer {who} does not support having no cost for {problem.name}!"
                    )
                case tuple() if "single" not in self.cost_awareness:
                    raise ValueError(
                        f"Optimizer {who} does not support single-cost for {problem.name}!"
                    )
                case Mapping() if "many" not in self.cost_awareness:
                    raise ValueError(
                        f"Optimizer {who} does not support multi-cost for {problem.name}!"
                    )

            match problem.is_tabular:
                case True if not self.tabular:
                    raise ValueError(
                        f"Optimizer {who} does not support tabular benchmarks for {problem.name}!"
                    )
