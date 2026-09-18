"""Exact 24-hour schedule optimization as a linear program (SciPy / HiGHS).

Variables per hour h: grid g_h, solar used s_h, charge c_h, discharge d_h, energy after E_h.

    minimize    sum_h tariff_h * g_h
    subject to  g_h + s_h + d_h - c_h = demand_h                 (energy balance)
                E_h - E_{h-1} - c_h + d_h = 0,  E_{-1} = initial  (battery transition)
                E_23 = initial                                    (end-of-day neutrality)
                0 <= g_h <= grid_cap_h, 0 <= s_h <= effective_solar_h,
                0 <= c_h <= max_charge_h, 0 <= d_h <= max_discharge_h,
                min_energy_h <= E_h <= capacity

Directives enter only through the bounds (see directives.model.build_hourly_limits).
A second LP keeps the optimal cost and minimizes total battery throughput, which removes
pointless charge/discharge cycling among equally cheap schedules.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from app.directives.model import HourlyLimits
from app.errors import InfeasibleScenarioError, OptimizerError
from app.models.schemas import HORIZON_HOURS

logger = logging.getLogger(__name__)

N = HORIZON_HOURS
G, S, C, D, E = (i * N for i in range(5))
NUM_VARS = 5 * N

# Status codes from scipy.optimize.linprog.
_INFEASIBLE = 2
_UNBOUNDED = 3

# Relative slack allowed on the optimal cost when re-solving for minimal cycling.
_COST_SLACK_REL = 0.0
_COST_SLACK_ABS = 1e-9


@dataclass(frozen=True)
class RawSchedule:
    grid: tuple[float, ...]
    solar_used: tuple[float, ...]
    charge: tuple[float, ...]
    discharge: tuple[float, ...]
    energy_after: tuple[float, ...]
    objective_cost: float


def _equality_system(limits: HourlyLimits) -> tuple[lil_matrix, np.ndarray]:
    rows = 2 * N + 1
    a_eq = lil_matrix((rows, NUM_VARS))
    b_eq = np.zeros(rows)
    for h in range(N):
        # Balance.
        a_eq[h, G + h] = 1.0
        a_eq[h, S + h] = 1.0
        a_eq[h, D + h] = 1.0
        a_eq[h, C + h] = -1.0
        b_eq[h] = limits.demand[h]
        # Transition.
        r = N + h
        a_eq[r, E + h] = 1.0
        a_eq[r, C + h] = -1.0
        a_eq[r, D + h] = 1.0
        if h == 0:
            b_eq[r] = limits.initial_energy
        else:
            a_eq[r, E + h - 1] = -1.0
    # Neutrality.
    a_eq[2 * N, E + N - 1] = 1.0
    b_eq[2 * N] = limits.initial_energy
    return a_eq, b_eq


def _bounds(limits: HourlyLimits) -> list[tuple[float, float | None]]:
    bounds: list[tuple[float, float | None]] = [(0.0, 0.0)] * NUM_VARS
    for h in range(N):
        cap = limits.grid_cap[h]
        bounds[G + h] = (0.0, None if math.isinf(cap) else cap)
        bounds[S + h] = (0.0, limits.effective_solar[h])
        bounds[C + h] = (0.0, limits.max_charge[h])
        bounds[D + h] = (0.0, limits.max_discharge[h])
        bounds[E + h] = (limits.min_energy[h], limits.capacity)
    return bounds


def _cost_vector(limits: HourlyLimits) -> np.ndarray:
    cost = np.zeros(NUM_VARS)
    cost[G : G + N] = limits.tariff
    return cost


def optimize(limits: HourlyLimits) -> RawSchedule:
    if any(m > limits.capacity for m in limits.min_energy):
        raise InfeasibleScenarioError("The active minimum battery energy exceeds battery capacity.")

    a_eq, b_eq = _equality_system(limits)
    a_eq = a_eq.tocsr()
    bounds = _bounds(limits)
    cost = _cost_vector(limits)

    try:
        first = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.error("linprog raised %s", type(exc).__name__)
        raise OptimizerError("The optimizer could not process this scenario.") from exc

    if first.status == _INFEASIBLE:
        raise InfeasibleScenarioError(
            "No 24-hour schedule satisfies the energy rules together with the interpreted directives."
        )
    if first.status == _UNBOUNDED or not first.success:
        logger.error("linprog failed status=%s message=%s", first.status, first.message)
        raise OptimizerError("The optimizer did not find an optimal schedule.")

    optimal_cost = float(first.fun)
    solution = first.x

    # Phase 2: same optimal cost, minimal battery throughput.
    throughput = np.zeros(NUM_VARS)
    throughput[C : C + N] = 1.0
    throughput[D : D + N] = 1.0
    slack = max(_COST_SLACK_ABS, abs(optimal_cost) * _COST_SLACK_REL)
    second = linprog(
        throughput,
        A_ub=cost.reshape(1, -1),
        b_ub=np.array([optimal_cost + slack]),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if second.success:
        solution = second.x
    else:
        logger.info("cycling-minimization pass skipped: %s", second.message)

    return RawSchedule(
        grid=tuple(float(v) for v in solution[G : G + N]),
        solar_used=tuple(float(v) for v in solution[S : S + N]),
        charge=tuple(float(v) for v in solution[C : C + N]),
        discharge=tuple(float(v) for v in solution[D : D + N]),
        energy_after=tuple(float(v) for v in solution[E : E + N]),
        objective_cost=optimal_cost,
    )
