"""Closed-form queueing results the engine is validated against."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SteadyState:
    rho: float  # utilisation per server
    p_wait: float  # probability an arrival has to queue
    Lq: float  # mean number in queue
    L: float  # mean number in system
    Wq: float  # mean wait in queue
    W: float  # mean time in system


def mm1(lam: float, mu: float) -> SteadyState:
    if lam >= mu:
        raise ValueError(f"unstable M/M/1: lambda={lam} >= mu={mu}")
    rho = lam / mu
    W = 1 / (mu - lam)
    Wq = rho / (mu - lam)
    return SteadyState(rho=rho, p_wait=rho, Lq=lam * Wq, L=rho / (1 - rho), Wq=Wq, W=W)


def erlang_c(c: int, a: float) -> float:
    """Probability of waiting in M/M/c with offered load a = lambda/mu, via the Erlang B recursion."""
    if a >= c:
        raise ValueError(f"unstable M/M/c: offered load {a} >= servers {c}")
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    rho = a / c
    return b / (1 - rho * (1 - b))


def mmc(lam: float, mu: float, c: int) -> SteadyState:
    a = lam / mu
    p_wait = erlang_c(c, a)
    Wq = p_wait / (c * mu - lam)
    W = Wq + 1 / mu
    return SteadyState(rho=a / c, p_wait=p_wait, Lq=lam * Wq, L=lam * W, Wq=Wq, W=W)


def lindley_waits(interarrivals: np.ndarray, services: np.ndarray) -> np.ndarray:
    """Queue waits of a single FIFO server, W[n+1] = max(0, W[n] + S[n] - A[n+1]).

    interarrivals[n] is the gap before customer n arrives, so interarrivals[0] is ignored.
    Holds for any interarrival and service distributions, which makes it an exact cross-check.
    """
    n = len(services)
    waits = np.zeros(n)
    for i in range(1, n):
        waits[i] = max(0.0, waits[i - 1] + services[i - 1] - interarrivals[i])
    return waits

