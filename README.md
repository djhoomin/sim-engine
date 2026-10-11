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
- **M(t)/G/infinity** for the non-stationary arrivals. With unlimited lanes the number at security
  at time t has a known mean function. Because flight loads here are binomial rather than Poisson,
  the exact distribution is a sum of binomials, one per seat, with that same mean and a smaller
  variance than Poisson.

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
shows where it is weak. At 12:00 it drops from 8 lanes to 3 and the mean wait that hour rises to
0.9 minutes against well under 0.2 elsewhere.

That dip is not a queue carried over from the peak. The queue at 12:00 is empty. Demand falls
steeply inside the hour, so in the first ten minutes arrivals run at 114% of three lanes'
capacity while the hourly average looks comfortable. The fix is to size each hour to its busiest
ten minutes rather than its mean (`offered_load(config, 10)` fed to `staff_to_load`). At equal
cost it wins, 30 replications, 95% intervals:

| KPI | Hourly mean at 70% | Busiest 10 min at 90% | Paired difference |
| --- | --- | --- | --- |
| Lane hours | 95 | 94 | -1 |
| Mean security wait (min) | 0.120 ± 0.007 | 0.092 ± 0.005 | -0.027 ± 0.006 |
| P95 security wait (min) | 0.57 ± 0.04 | 0.41 ± 0.02 | -0.16 ± 0.04 |
| Worst open hour, mean wait (min) | 1.00 ± 0.16 | 0.42 ± 0.08 | -0.58 ± 0.18 |

Those rosters swing hard, from 12 lanes to 3 within two hours, which real shifts may not allow.
`cover_with_shifts` builds the cheapest roster from whole shifts that meets a requirement every
hour, with a minimum and maximum shift length and a cap on lanes opening or closing per hour. It
is the classic shift covering integer program, small enough that SciPy's HiGHS solves it exactly
in milliseconds. With 4 to 8 hour shifts and at most 3 lanes changing per hour, 30 replications:

| Comparison | Lane hours | P95 wait (min) | Worst open hour (min) |
| --- | --- | --- | --- |
| Unconstrained, busiest 10 min at 90% | 94 | 0.41 ± 0.02 | 0.42 ± 0.08 |
| Same requirement built from shifts | 105 | 0.34 ± 0.02 | 0.42 ± 0.08 |
| Shifts at equal cost (target 100%) | 95 | 0.97 ± 0.15 | 0.97 ± 0.20 |

Shift rules cost about 12% more lane hours to keep service no worse in any hour. Spent at equal
cost instead, they more than double the tail wait, because shifts hold lanes open through quiet
hours such as 07:00 and 08:00 and leave too few for the 16:00 ramp.

A side finding: about 13 passengers a day arrive before security opens at 03:00 and wait for
it, around 25 minutes on average. No roster during open hours changes that. Only the opening
time does, which is why the worst-hour KPI covers open hours only.

All service and show-up parameters are plausible placeholders, not calibrated values, so these
numbers demonstrate the machinery rather than say anything about a real airport.

The non-stationary arrivals are validated in `tests/test_nonstationary.py`. Over 300
replications with unlimited lanes, the number at security matches the M(t)/G/infinity mean at
every quarter hour of the day within a Bonferroni bound, the pooled variance is within 5% of the
exact value and clearly below Poisson, and the full distribution at the peak passes a chi-square
fit. The same fit rejects a 5% error in the mean at p below 1e-18.

Next steps, roughly in order:

- Copenhagen waits. Use the Copenhagen wait history as a second check, once a departure schedule
  for those dates is available (see Real data below).
- Check-in rosters, using the same machinery as security.
- Real day-to-day variability in demand. See the replay results under Real data.

## Real data

Searched on 11 October 2026. What is publicly available:

| Source | What it gives | Access | Gap |
| --- | --- | --- | --- |
| TSA FOIA reading room, weekly throughput | Passengers screened per hour per checkpoint, every US airport | Public PDFs, current to early October 2026 | US only, no waiting times |
| BTS reporting carrier on-time data | Every scheduled departure with tail number | Public monthly files, current to July 2026 | Large US carriers only, no international |
| FAA aircraft registry | Seats per registered aircraft | Public download | None for this use |
| Copenhagen wait history, github.com/janschill/cph-ventetid-history | Terminal 2 and 3 security waits in 2 minute bands, December 2024 to February 2026 | Public repository, no licence file | No schedule. Lane staffing unknown |
| OpenSky Network historical flights | Departures by airport, which would give the Copenhagen schedule | Needs a free account | Not yet requested |
| Schiphol Flight API | Departures with aircraft type | Free key | Recent dates only, no queue or throughput data |
| OR Tambo | Schedules through paid APIs only | Paid | No queue data |

The US sources together calibrate demand without needing any staffing information, so they come
first. `scripts/ingest_us.py` turns them into `data/us/msy_schedule_2026_07.csv` and
`data/us/msy_tsa_2026_07.csv` for New Orleans (MSY). MSY has one terminal with one consolidated
checkpoint and mostly origin and destination traffic, so the passengers screened there are close to
the passengers departing.

`src/simengine/calibration.py` fits the gamma show-up profile and an effective load factor by
Poisson maximum likelihood, with an option for a separate profile per time-of-day band. Tests
check it recovers known parameters from simulated passengers before it is used on real data.
`scripts/calibrate_us.py` fits on one week and scores on another.

Results, fitting on 12 to 18 July 2026 and testing on 19 to 25 July, then the reverse:

| Model | Parameters (minutes before departure, load factor) | Out-of-sample R2 |
| --- | --- | --- |
| Placeholder 110 ± 35, load fitted | | 0.71 |
| One profile | 85 ± 55, 0.83 | 0.81 and 0.85 |
| Morning and afternoon | 74 ± 37, 0.77 to 0.80, and 144 ± 137, 0.86 to 0.87 | 0.87 and 0.87 |
| Three bands, split at 10:00 and 15:00 | 67 ± 32, 94 to 105 ± 50, about 197 ± 189 | 0.91 and 0.91 |

The parameters are stable when the weeks swap, and a separate profile by time of day clearly helps.
Morning passengers reach the checkpoint about 70 minutes before departure. The late band is not
believable as behaviour, though. A load factor of 1 with a mean lead of over three hours looks like
the fit absorbing passengers for departures the BTS file does not contain, which at MSY means
international flights and carriers that do not report. Its gain in fit is real, but its parameters
should not be read as a show-up profile.

What the TSA counts do not settle is waiting time. They count passengers screened, so the fitted
lead time is time of screening before departure and includes any queue.

### Replaying real days

`scripts/replay_us.py` runs the terminal model on each day of the test week from that day's real
MSY schedule, with morning and afternoon show-up profiles fitted on the other week. Everyone goes
straight to security, since the fitted lead time is already time of screening. The checkpoint is
closed before 03:00 in the roster, so the overnight pile-up at opening comes out of the simulation
rather than a binning rule. The model's passengers screened per hour are compared with TSA's
counts. 30 replications per day:

| Day | Observed | Simulated | Hourly R2 | Observed hours inside the model's 95% band |
| --- | --- | --- | --- | --- |
| Sun 19 July | 20,354 | 17,097 ± 19 | 0.86 | 10% |
| Mon 20 July | 17,979 | 18,253 ± 16 | 0.94 | 20% |
| Tue 21 July | 15,056 | 14,938 ± 19 | 0.90 | 26% |
| Wed 22 July | 15,135 | 15,733 ± 17 | 0.83 | 5% |
| Thu 23 July | 17,503 | 18,189 ± 23 | 0.92 | 40% |
| Fri 24 July | 17,547 | 17,913 ± 19 | 0.91 | 15% |
| Sat 25 July | 13,762 | 13,603 ± 19 | 0.89 | 21% |

Three findings.

- The hourly shape is right out of sample, with R2 between 0.83 and 0.94 on every day, and daily
  totals within 4% except Sunday.
- The model is far too sure of itself. A well calibrated model would have about 95% of observed
  hours inside its 95% band, and this one manages 5% to 40%. Its only randomness is independent
  passengers, binomial per seat. Real demand also moves with the day (Sunday's load factor was 0.98
  against about 0.8 midweek), with each flight, and with groups travelling together. The point
  forecast is usable, the uncertainty around it is not yet. This matters for staffing, which is
  decided against the bad days rather than the average one.
- Every day of both weeks screens far fewer passengers at 10:00 than the departures predict, 431 to
  902 against 1,000 to 1,300 in the hours either side, with no lull in the schedule to explain it.
  It could be a TSA shift change, lanes closed for a break or a quirk in how the hour is recorded.
  The data cannot tell which, and the model should not be bent to fit it until it can.

Waiting times from the replay are what a hypothetical roster would give, since real MSY staffing
is not public, so they say nothing about MSY.
