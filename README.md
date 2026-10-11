# Simulation engine, airport terminal first

Started 11 October 2026. Nothing is built yet. This file is the thinking, so a fresh session can
pick it up cold.

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

## Open questions

- Retrieve the master's engine. Which language, and is any of it worth porting as reference?
- Synthetic data to start, or approach someone for a real schedule and real service times?
- Is the end state a library, a portfolio piece, or something that gets used on a client engagement?
  The answer changes how much the calibration layer matters.
