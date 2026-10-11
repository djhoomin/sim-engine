# Simulation engine, airport terminal first

Started 11 October 2026. This file is the thinking, so a fresh session can pick it up cold. The
Python spike is scaffolded and its validation suite passes. See Status at the bottom.

## Why this exists

DJ has an MSc in Operations Research (cum laude), a BCom Hons in OR and a BSc in Mathematical
Sciences, all Stellenbosch, and two peer-reviewed papers on agent-based resistance modelling in the
Journal of Economic Entomology. That background has been dormant for about a decade while the work
was machine learning, platform and research leadership. The aim here is to put it back to use on
something current, and to end up with a portfolio piece rather than a hobby.

The master's simulation engine is not on this machine. It is agent-based and the domain is pest
resistance, so it is worth retrieving as reference and as a re-entry exercise, but it is not the
thing to extend. Airports need discrete-event semantics, which is a different paradigm.

## The strategic read

Do not build a general-purpose simulation engine. AnyLogic, Simio, FlexSim and Arena own the
commercial end and SimPy owns the free Python end. The gap sits between them, and it is specific:

1. **Speed at scale.** SimPy is single-threaded pure Python. It is comfortable at a few hundred
   thousand events and unhappy at tens of millions, which is what a realistic terminal or line model
   wants once you run enough replications.
2. **Calibration from operational data.** Fitting arrival processes and service-time distributions
   to real data, then validating against observed KPIs, is what actually kills simulation projects.
   Almost nothing open source helps with it.
3. **Experiment rigour.** Replications, common random numbers, variance reduction, confidence
   intervals on KPI differences, versioned scenarios. Simulation practice is mostly terrible at this.
   People run a model once, eyeball a number and present it.

## The differentiator

Number 3 is the unfair advantage and the reason this is worth doing at all. DJ built the model
evaluation stack at Raw Power Labs: seeds, scoring, prompt-version tracking, and the ship-or-not
standard the team used. Treating every simulation run as a tracked experiment with a config hash, a
seed, a version and KPIs out is the same discipline applied to a field that lacks it. That is the
two halves of the career joining up, rather than picking up a cold subfield.

## Why an airport, and not a logistics line

The system decomposes cleanly into arrivals, check-in, security, gates and baggage. The KPIs are
unarguable: queue time, throughput, missed connections, resource utilisation. There is published
literature to validate against. Copenhagen Airport is local and a plausible reference. And it demos
well, which matters if this doubles as a portfolio piece. A logistics line is structurally similar
but real data is harder to come by.

## Architecture

Spike in plain Python first. The hard part is modelling judgement, not throughput, and the first
version gets thrown away. Once the semantics have settled, move the event queue, entity state and
resource pools into Rust behind PyO3 bindings (maturin for the build) and keep scenario definition,
analysis and plotting in Python.

Put deterministic seeding and common random numbers in from the first commit. Retrofitting them is
miserable, and without CRN the scenario comparisons are noise.

## Validation strategy

Validate against theory before validating against data. A simulation is unusually good at being
confidently wrong, so the engine should be proved correct against closed-form results first:

- **M/M/1**: utilisation rho = lambda/mu, mean number in system L = rho/(1-rho), mean wait in system
  W = 1/(mu-lambda), mean queue wait Wq = rho/(mu-lambda).
- **Little's Law**: L = lambda * W. Should hold on every instrumented resource, and it is a good
  continuous assertion rather than a one-off test.
- **M/M/c via Erlang C**: directly applicable to parallel security lanes and check-in desks, which
  is the first real staffing question the model should be able to answer.
- **Lindley recursion** as a cross-check for single-server waiting times under general distributions.

Only once those pass should the model meet real or synthetic operational data.

## First milestone

A single terminal, with:

- a non-stationary arrival process, because flight schedules make arrivals peaky and a Poisson
  process with a constant rate will flatter the model
- check-in and security as multi-server resources with configurable lane counts
- per-entity traces and per-resource statistics
- replications with common random numbers across scenarios
- confidence intervals on the KPI differences between two staffing scenarios
- validation tests against M/M/1, M/M/c and Little's Law, run in CI

If that is in place and the numbers are defensible, the question of whether this holds interest is
answered, and the Rust port is the next decision.

## Explicitly out of scope for now

Visual animation, a GUI, baggage handling, gate allocation, multi-terminal, airside and aircraft
turnaround, anything optimisation-driven. Simulation-optimisation loops are the obvious second act
but they are worthless on top of a model nobody trusts yet.

## Commercial relevance, if it matters later

The Danish industrial base is full of problems where the value sits in the decision rather than the
prediction: Mærsk and DSV on networks and scheduling, Danish Crown and Arla on production planning,
Vestas, Ørsted and Energinet on capacity and maintenance, Grundfos and Danfoss on manufacturing.

## Prior art to read before writing code

SimPy and salabim for discrete-event in Python, Mesa for agent-based, AnyLogic and Simio for what
the commercial tools actually give people, OR-Tools CP-SAT and HiGHS for the optimisation side when
it arrives. The hybrid pattern worth knowing is agent-based actors making decisions on top of
discrete-event resources, which is where the old agent-based work and this project meet.

## Decisions

Settled on 11 October 2026, replacing the earlier open questions.

- **End state is a library.** That makes the calibration layer first-class rather than a nice to
  have, since fitting to operational data is a large part of the gap the library is meant to fill.
- **Real data is preferred, synthetic is acceptable.** Try to get a real schedule and real service
  times. Build against synthetic data until that happens, and keep the data interface the same so
  swapping is cheap.
- **The master's engine is not needed.** It was agent-based, so it does not inform the
  discrete-event core.

## Status

The spike lives in `src/simengine` and runs on Python 3.12 via uv.

- `rng.py` gives named streams keyed by base seed, replication and stream name. Scenario
  parameters never enter the seed, which is what gives common random numbers.
- `kernel.py` is the event list and clock.
- `resource.py` is a multi-server FIFO resource with per-visit traces, time-weighted statistics
  and an exact sample-path Little's Law check.
- `queueing.py` holds the closed forms: M/M/1, Erlang C for M/M/c, and the Lindley recursion.
- `station.py` is a single station with renewal arrivals. Entities draw their service requirement
  at arrival so it stays synchronised across scenarios.
- `experiment.py` runs replications and records config hash, base seed, engine version, git state
  and per-replication KPIs. It refuses a single replication and gives paired confidence intervals
  on scenario differences.

Run the suite with `uv run pytest`. It checks M/M/1 and M/M/c against theory at 99% confidence
with a precision guard, the engine against the Lindley recursion customer by customer, Little's
Law exactly and statistically, and that common random numbers cut the variance of a scenario
difference.

The terminal model sits on top of that.

- `schedule.py` holds the flight schedule. A synthetic short-haul day with three banks is the
  default, and `load_schedule` reads a real one from CSV with `departure` (HH:MM) and `seats`.
- `terminal.py` generates passengers from the schedule: binomial load per flight, a gamma show-up
  lead time before departure, a bag drop choice and lognormal service times. All of it is drawn
  before any resource exists, so staffing scenarios with the same seed see identical passengers.
  Passengers with bags go through check-in and then security, the rest go straight to security.
  The day starts and ends empty, so there is no warm-up to choose and Little's Law holds exactly.
- `roster.py` holds hourly staffing rosters. Desks and lanes change on the hour, and a closing desk
  or lane finishes its current passenger before it shuts. `staff_to_load` sizes each hour to the
  expected load at a target utilisation, which `terminal.offered_load` computes straight from the
  schedule.
- Hourly KPIs (`security_queue_hHH` and `security_wait_hHH`) sit in every replication record next
  to the day-level ones.
- `scripts/compare_staffing.py` runs a two-roster comparison, prints day KPIs with paired
  differences and the hourly profile, and saves both experiment records to `runs/`.

First result, synthetic day of about 9,100 passengers, security open 03:00 to 21:00, 30
replications with common random numbers, 95% intervals. Flat is 7 lanes all day. Load-led sizes
each hour to 75% expected utilisation.

| KPI | Flat 7 | Load-led | Paired difference |
| --- | --- | --- | --- |
| Lane hours | 126.0 | 90.0 | -36.0 |
| Mean security wait (min) | 2.32 ± 0.16 | 0.15 ± 0.01 | -2.17 ± 0.16 |
| P95 security wait (min) | 11.31 ± 0.68 | 0.71 ± 0.05 | -10.60 ± 0.67 |
| Peak security queue (pax) | 223 ± 11 | 28 ± 3 | -195 ± 11 |

The load-led roster uses 29% fewer lane hours and nearly removes queueing. The hourly profile
shows where the heuristic is weak. At 12:00 it drops from 8 lanes to 3 while the 11:00 peak is
still clearing, and the mean wait that hour rises to 0.9 minutes against well under 0.2 elsewhere.
The heuristic treats hours as independent and the simulation shows they are not.

All service and show-up parameters are plausible placeholders, not calibrated values, so these
numbers demonstrate the machinery rather than say anything about a real airport.

Next steps, roughly in order:

- Validate the non-stationary arrivals against the M(t)/G/infinity result, where the number in
  system is Poisson with a known time-varying mean. The expected hourly arrivals already match the
  simulated average, and this would extend that to the queue itself.
- A roster heuristic that looks back one hour, or a lag-adjusted offered load, and a comparison
  against the independent-hours version.
- Real data. A published schedule for a real airport day would replace the synthetic one through
  `load_schedule` with no other change. Show-up profiles and service times are the parameters that
  most need real values.
