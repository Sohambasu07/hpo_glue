from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

from hpo_glue.benchmark import BenchmarkDescription
from hpo_glue.constants import DEFAULT_RELATIVE_EXP_DIR
from hpo_glue.env import (
    GLUE_PYPI,
    get_current_installed_hpo_glue_version,
)
from hpo_glue.lib.benchmarks import BENCHMARKS
from hpo_glue.lib.optimizers import OPTIMIZERS
from hpo_glue.optimizer import Optimizer

if TYPE_CHECKING:
    from hpo_glue.budget import BudgetType
    from hpo_glue.problem import Problem
    from hpo_glue.run import Run

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

smac_logger = logging.getLogger("smac")
smac_logger.setLevel(logging.ERROR)


OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]

GLOBAL_SEED = 42



class Study:
    """Represents a Glue study for hyperparameter optimization."""
    def __init__(
        self,
        name: str,
        output_dir: Path | None = None,
    ):
        """Initialize a Study object with a name and a results directory.

        Args:
            name: The name of the study.
            output_dir: The directory to store the experiment results.
        """
        self.name = name
        if output_dir is None:
            output_dir = Path.cwd().absolute().parent / "hpo-glue-output"
        self.output_dir = output_dir

    @classmethod
    def generate_seeds(
        cls,
        num_seeds: int,
    ):
        """Generate a set of seeds using a Global Seed."""
        cls._rng = np.random.default_rng(GLOBAL_SEED)
        return cls._rng.integers(0, 2 ** 30 - 1, size=num_seeds)


    @classmethod
    def generate(  # noqa: C901, PLR0912, PLR0913
        cls,
        optimizers: (
            type[Optimizer]
            | OptWithHps
            | list[type[Optimizer]]
            | list[OptWithHps | type[Optimizer]]
        ),
        benchmarks: BenchmarkDescription | Iterable[BenchmarkDescription],
        *,
        expdir: Path | str = DEFAULT_RELATIVE_EXP_DIR,
        budget: BudgetType | int,
        seeds: Iterable[int] | None = None,
        num_seeds: int = 1,
        fidelities: int = 0,
        objectives: int = 1,
        costs: int = 0,
        multi_objective_generation: Literal["mix_metric_cost", "metric_only"] = "mix_metric_cost",
        on_error: Literal["warn", "raise", "ignore"] = "warn",
        continuations: bool = False,
        precision: int | None = None
    ) -> list[Run]:
        """Generate a set of problems for the given optimizer and benchmark.

        If there is some incompatibility between the optimizer, the benchmark and the requested
        amount of objectives, fidelities or costs, a ValueError will be raised.

        Args:
            optimizers: The optimizer class to generate problems for.
                Can provide a single optimizer or a list of optimizers.
                If you wish to provide hyperparameters for the optimizer, provide a tuple with the
                optimizer.
            benchmarks: The benchmark to generate problems for.
                Can provide a single benchmark or a list of benchmarks.
            expdir: Which directory to store experiment results into.
            budget: The budget to use for the problems. Budget defaults to a n_trials budget
                where when multifidelty is enabled, fractional budget can be used and 1 is
                equivalent a full fidelity trial.
            seeds: The seed or seeds to use for the problems.
            num_seeds: The number of seeds to generate. Only used if seeds is None.
            fidelities: The number of fidelities to generate problems for.
            objectives: The number of objectives to generate problems for.
            costs: The number of costs to generate problems for.
            multi_objective_generation: The method to generate multiple objectives.
            on_error: The method to handle errors.

                * "warn": Log a warning and continue.
                * "raise": Raise an error.
                * "ignore": Ignore the error and continue.
        """
        # Generate seeds
        match seeds:
            case None:
                seeds = cls.generate_seeds(num_seeds).tolist()
            case Iterable():
                pass
            case int():
                seeds = [seeds]

        _benchmarks: list[BenchmarkDescription] = []
        match benchmarks:
            case BenchmarkDescription():
                _benchmarks = [benchmarks]
            case Iterable():
                _benchmarks = list(benchmarks)
            case _:
                raise TypeError(
                    "Expected BenchmarkDescription or Iterable[BenchmarkDescription],"
                    f" got {type(benchmarks)}"
                )

        _problems: list[Problem] = []
        for _benchmark in _benchmarks:
            try:
                _problem = _benchmark.problem(
                    objectives=objectives,
                    budget=budget,
                    fidelities=fidelities,
                    costs=costs,
                    multi_objective_generation=multi_objective_generation,
                    precision=precision
                )
                _problems.append(_problem)
            except ValueError as e:
                match on_error:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case "warn":
                        warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
                        continue

        _runs_per_problem: list[Run] = []
        for _problem in _problems:
            try:
                _runs = _problem.generate_runs(
                    optimizers=optimizers,
                    seeds=seeds,
                    expdir=expdir,
                    continuations = continuations
                )
                _runs_per_problem.extend(_runs)
            except ValueError as e:
                match on_error:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case "warn":
                        warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
                        continue

        return _runs_per_problem


    def create_env(             #TODO: This is not called for now. Fix this.
        self,
        *,
        how: Literal["venv", "conda"] = "venv",
        hpo_glue: Literal["current_version"] | str,
    ) -> None:
        """Set up the isolation for the experiment."""
        if hpo_glue == "current_version":
            raise NotImplementedError("Not implemented yet.")

        match hpo_glue:
            case "current_version":
                _version = get_current_installed_hpo_glue_version()
                req = f"{GLUE_PYPI}=={_version}"
            case str():
                req = hpo_glue
            case _:
                raise ValueError(f"Invalid value for `hpo_glue`: {hpo_glue}")

        requirements = [req, *self.env.requirements]

        if self.env_path.exists():
            logger.info(f"Environment already exists: {self.env.identifier}")
            return

        logger.info(f"Installing deps: {self.env.identifier}")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        with self.venv_requirements_file.open("w") as f:
            f.write("\n".join(requirements))

        # if self.env_path.exists(): TODO: Does this need to be after install step?
        #     return

        self.env_path.parent.mkdir(parents=True, exist_ok=True)

        env_dict = self.env.to_dict()
        env_dict.update({"env_path": str(self.env_path), "hpo_glue_source": req})

        logger.info(f"Installing env: {self.env.identifier}")
        match how:
            case "venv":
                logger.info(f"Creating environment {self.env.identifier} at {self.env_path}")
                self.venv.create(
                    path=self.env_path,
                    python_version=self.env.python_version,
                    requirements_file=self.venv_requirements_file,
                    exists_ok=False,
                )
                if self.env.post_install:
                    logger.info(f"Running post install for {self.env.identifier}")
                    with self.post_install_steps.open("w") as f:
                        f.write("\n".join(self.env.post_install))
                    self.venv.run(self.env.post_install)
            case "conda":
                raise NotImplementedError("Conda not implemented yet.")
            case _:
                raise ValueError(f"Invalid value for `how`: {how}")


    def optimize(
        self,
        optimizers: (
            str
            | tuple[str, Mapping[str, Any]]
            | type[Optimizer]
            | OptWithHps
            | list[tuple[str, Mapping[str, Any]]]
            | list[str]
            | list[OptWithHps]
            | list[type[Optimizer]]
        ),
        benchmarks: list[str],
        *,
        seeds: Iterable[int] | int | None = None,
        num_seeds: int = 1,
        budget: int = 50,
        precision: int | None = None,
        overwrite: bool = False,
        continuations: bool = False,
        on_error: Literal["warn", "raise", "ignore"] = "warn",
    ) -> None:
        """Execute multiple atomic runs using a list of Optimizers and a list of Benchmarks.

        Args:
            optimizers: The list of optimizers to use.

            benchmarks: The list of benchmarks to use.

            seeds: The seed or seeds to use for the experiment.

            num_seeds: The number of seeds to generate.

            budget: The budget for the experiment.

            overwrite: Whether to overwrite existing results.

            continuations: Whether to calculate continuations cost.
            Note: Only works for Multi-fidelity Optimizers.

            precision: The precision of the optimization run(s).

            on_error: The method to handle errors.
            Available options are:
                * "warn": Log a warning and continue.
                * "raise": Raise an error.
                * "ignore": Ignore the error and continue.
        """
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        if not isinstance(benchmarks, list):
            benchmarks = [benchmarks]

        _optimizers = []
        match optimizers[0]:
            case str():
                for optimizer in optimizers:
                    assert optimizer in OPTIMIZERS, f"Optimizer must be one of {OPTIMIZERS.keys()}"
                    _optimizers.append(OPTIMIZERS[optimizer])
            case tuple():
                for optimizer, opt_hps in optimizers:
                    assert optimizer in OPTIMIZERS, f"Optimizer must be one of {OPTIMIZERS.keys()}"
                    _optimizers.append((OPTIMIZERS[optimizer], opt_hps))
            case type():
                _optimizers = optimizers
            case _:
                raise TypeError(f"Unknown Optimizer type {type(optimizers[0])}")

        _benchmarks = []
        for benchmark in benchmarks:
            assert benchmark in BENCHMARKS, f"Benchmark must be one of {BENCHMARKS.keys()}"
            _benchmarks.append(BENCHMARKS[benchmark])

        exp_dir = self.output_dir

        self.experiments = Study.generate(
            optimizers=_optimizers,
            benchmarks=_benchmarks,
            expdir=exp_dir,
            budget=budget,
            seeds=seeds,
            num_seeds=num_seeds,
            on_error=on_error,
            precision=precision,
            continuations=continuations
        )
        for run in self.experiments:
            run.write_yaml()
            run.create_env(hpo_glue=f"-e {Path.cwd()}")
            run.run(
                overwrite=overwrite,
                progress_bar=False,
            )


def create_study(
        output_dir: Path | None = None,
        name: str | None = None,
    ) -> Study:
    """Create a Study object."""
    if output_dir is None:
        output_dir = Path.cwd().absolute().parent / "hpo-glue-output"
    """Create a Study object."""
    if name is None:
        name = f"glue_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return Study(name, output_dir)