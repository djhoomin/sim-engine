"""Replications, confidence intervals and the experiment record.

A run is an experiment. Every result carries the config hash, the base seed, the engine version
and the KPIs of each replication, so any number reported can be traced back and reproduced.
"""

import dataclasses
import hashlib
import json
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from . import __version__
from .rng import Streams

Model = Callable[[Any, Streams], dict[str, float]]


def _config_dict(config: Any) -> dict:
    if dataclasses.is_dataclass(config):
        return {"type": type(config).__name__, **dataclasses.asdict(config)}
    return dict(config)


def config_hash(config: Any) -> str:
    canonical = json.dumps(_config_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


@dataclasses.dataclass(frozen=True)
class Interval:
    mean: float
    half_width: float
    n: int
    level: float

    @property
    def low(self) -> float:
        return self.mean - self.half_width

    @property
    def high(self) -> float:
        return self.mean + self.half_width

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def confidence_interval(values: np.ndarray | list[float], level: float = 0.95) -> Interval:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("a confidence interval needs at least two replications")
    t = stats.t.ppf(0.5 + level / 2, df=n - 1)
    return Interval(float(x.mean()), float(t * x.std(ddof=1) / np.sqrt(n)), n, level)


@dataclasses.dataclass
class ExperimentRecord:
    config: dict
    config_hash: str
    base_seed: int
    replications: list[dict[str, float]]
    engine_version: str
    git: dict[str, Any]
    created: str

    def kpi(self, name: str) -> np.ndarray:
        return np.array([r[name] for r in self.replications])

    def interval(self, name: str, level: float = 0.95) -> Interval:
        return confidence_interval(self.kpi(name), level)

    def save(self, directory: str | Path = "runs") -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{self.config_hash}-seed{self.base_seed}.json"
        out.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        return out


def run_experiment(model: Model, config: Any, replications: int, base_seed: int) -> ExperimentRecord:
    if replications < 2:
        raise ValueError("never report a single unreplicated run: use at least two replications")
    results = [model(config, Streams(base_seed, r)) for r in range(replications)]
    return ExperimentRecord(
        config=_config_dict(config),
        config_hash=config_hash(config),
        base_seed=base_seed,
        replications=results,
        engine_version=__version__,
        git=_git_state(),
        created=datetime.now(timezone.utc).isoformat(),
    )


def paired_difference(
    a: ExperimentRecord, b: ExperimentRecord, kpi: str, level: float = 0.95
) -> Interval:
    """Interval on mean(b - a) for one KPI, pairing replications by index.

    Pairing is only valid when both experiments share a base seed, which is what gives the pairs
    common random numbers.
    """
    if a.base_seed != b.base_seed or len(a.replications) != len(b.replications):
        raise ValueError("paired comparison needs the same base seed and replication count")
    return confidence_interval(b.kpi(kpi) - a.kpi(kpi), level)
