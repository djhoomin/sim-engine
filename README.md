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
- What moves passengers between hours at a real checkpoint. See Demand variation under Real data.

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

### Demand variation

The terminal model now has two optional layers of demand variation on top of independent
passengers, each on its own random stream so switching them on leaves every other draw alone. A
day-level multiplier on all load factors, lognormal with log standard deviation `day_sd`, drawn once
per replication. A flight-level load factor drawn from a beta with concentration
`flight_concentration` around the band's mean. `calibration.fit_dispersion` estimates both by
the method of moments, with tests that it recovers known values and that it does not mistake timing
noise for load variation.

On the MSY fit week the day multiplier has a log standard deviation of 0.074, with daily factors
from 0.91 to 1.10. With it switched on, the replay of the test week improves:

| | Independent passengers | With day variation |
| --- | --- | --- |
| Daily totals inside the 95% band | 0 of 7 | 6 of 7 |
| Hours inside the 95% band, mean over days | 20% | 38% |

Sunday stays outside. It was the busiest day in both weeks, which hints at a day-of-week effect
that one week of fitting cannot estimate.

The flight layer could not be estimated, and the reason is the more useful finding. Of the squared
hourly residuals on the fit week, 61% repeat at the same hour every day, an error in the mean
model's shape rather than randomness. The remaining day-to-day part is still larger than any load
variation could produce, even with every flight either full or empty. So hourly counts move for
reasons other than how many people fly. The repeating residuals point at one. From 09:00 to 12:00
they run -146, -406, +198 and +141, and at 14:00 and 15:00 -169 and +204, a dip followed by a
catch-up with passengers roughly conserved. TSA counts passengers screened, not passengers arriving,
so a drop in lanes open, at a shift change or for breaks, would hold passengers in the queue and
push them into the next hour's count. That is a capacity explanation, not a demand one, and these
data cannot confirm it. Observed waiting times at the same checkpoint could.

### Copenhagen

`scripts/ingest_cph.py history` replays the git-scraped wait history into `data/cph/waits.csv`,
which stays out of git because the source repository has no licence. It holds 22,990 readings
from 4 December 2024 to 3 February 2026, one per change, about every 22 minutes. The feed format
changed on 23 June 2025 from "9-11" to "9-11 min" and both are parsed. Kastrup has one central
security checkpoint for Terminals 2 and 3, and the two fields have always been equal. They should
only diverge once the new Terminal 3 building opens with its own security, expected around 2028.

The median wait is 5 minutes, the 90th percentile 13.5 and the maximum 39.5. By Copenhagen hour the
time-weighted mean wait peaks at 04:00 (9.3 minutes), around 07:00 and 11:00, and at 16:00 to 17:00
(9.4), and sits near 1 to 2 minutes overnight.

The public feed is still live without a key, so `ingest_cph.py poll` can extend the series from now
on. Departures come from the OpenSky Network through `ingest_cph.py schedule`, which needs an OpenSky
API client. OpenSky reports when an aircraft was first seen airborne rather than the scheduled
time, so departures run late by taxi time and any delay. Aircraft types for seat counts come from
OpenSky's public aircraft database, the August 2025 snapshot being downloadable without an account.

Departures for September 2025 and January 2026 are pulled into `data/cph/`, which stays out of git
like the waits. OpenSky has no data at all for 2, 3 and 24 January 2026, so those days are left
out. `ingest_cph.py seats` joins aircraft to types from OpenSky's aircraft database and seats from
`data/reference/aircraft_seats.csv`, typical European configurations written for this project. It
drops freight operators, freighters, business jets and helicopters. About 5% of departures fly
aircraft missing from the August 2025 database, 670 of them SAS, mostly newer deliveries. Dropping
them would undercount SAS, so they take their airline's median seats and are marked as imputed.
That leaves 19,412 passenger departures, 381 a day with about 63,000 seats in September and 285 a
day with about 47,000 seats in January.

Unlike MSY there are no passenger counts at Kastrup, only waits, and lane staffing is not public.
So Kastrup cannot calibrate demand the way MSY did. What it can do is test the capacity explanation
from MSY: given demand from the schedule, which capacity pattern through the day reproduces the
observed waits, and is it plausible.

### Inferring Kastrup's capacity from waits

`scripts/infer_capacity_cph.py` runs the terminal model on each day's real Kastrup departures and
infers lanes hour by hour, forward from 03:00, so the simulated mean wait matches the observed one.
Assumptions, since Kastrup publishes no checkpoint counts or staffing: an effective load factor of
0.68 (CPH's reported 80% seat occupancy in September 2025, less about 15% transfer passengers who
skip central security), MSY's fitted show-up profile shifted 20 minutes for takeoff rather than
scheduled time (105 ± 55 minutes before takeoff), and 150 passengers per lane hour. The September
demand level checks out against CPH's own figure of 3.1 million passengers for the month.

Fitted on 1 to 14 September 2025, tested on 15 to 29 September.

- Plausible in scale. Inferred peak capacity is 21 to 25 lanes, against about 23 physical lanes
  after the 2015 and 2018 expansions.
- Poor out of sample. The mean daily wait profile scores R2 0.13 with a mean absolute error of 2
  minutes, biased low, and single days correlate at a median of 0.45.
- Unstable from hour to hour. Inferred lanes jump, for example 18, 13, 16, 18, 25 and 17 between
  08:00 and 13:00, with no operational reason.

The reason is in the utilisation column. In most hours the inferred capacity sits almost exactly at
demand, utilisation close to 1. That is where waits are most sensitive to small changes in demand
or lanes, so inverting waits to lanes is badly conditioned. It also says something about Kastrup.
Waits held between 5 and 10 minutes from early morning to evening look like a checkpoint run to a
wait target, with supervisors opening and closing lanes as the queue grows and shrinks, rather than
a fixed roster. Under that kind of feedback the waits mostly reflect the target, and a single
open-loop roster cannot reproduce days that differ. The next model to try is staffing as a
controller, opening a lane when the queue passes a threshold and closing one when it falls below
another, with the thresholds fitted to the observed waits.

The run takes about 20 minutes, almost all of it bisection over truncated day simulations.

### Queue-responsive staffing

`src/simengine/control.py` adds a `QueuePolicy`. Every few minutes a supervisor estimates the wait
as queue length times mean service time over lanes open. Above one threshold they open lanes, which
take a few minutes to start, and below another they close one, within the lanes available that
hour. In the terminal model the roster then caps available lanes rather than setting them. Tests
check that it opens under a surge and closes after, respects the cap and the minimum, keeps
Little's Law exact, and that a tighter threshold buys shorter waits with more lane hours.

`scripts/fit_policy_cph.py` fits the two thresholds and the lanes available to Kastrup's waits,
scoring every hour of every fit day, and tests on held-out days. Lanes available is fitted because
the checkpoint was being rebuilt through both periods. CT lanes replaced conventional X-ray lanes in
phases from April 2025, and the new 20-lane CT checkpoint opened on 19 May 2026. January uses an
effective load factor of 0.65, from CPH's 2.226 million passengers that month against the schedule's
seats, less the same 15% transfer share.

| Period, model | Best fit | Held-out mean profile R2 | Mean error | Single-day correlation, median |
| --- | --- | --- | --- | --- |
| September 2025, fixed hourly roster | lanes inferred per hour | 0.13 | 2.05 min | 0.45 |
| September 2025, queue-responsive | open above 8 min, close below 4, 24 lanes | 0.15 | 1.78 min | 0.31 |
| January 2026, queue-responsive | open above 4 min, close below 2, 16 lanes | 0.38 | 1.50 min | 0.47 |

The policy model helps in January and barely in September, and both best fits sit at an edge of
the grid, September at the most lanes tried and January at the fewest lanes and lowest threshold.

What the waits identify is capacity in passengers per hour, not lanes. At 150 passengers per lane
hour, 16 lanes is about 2,400 passengers an hour, the same as 8 CT lanes at 300. Operational
experience is that Kastrup rarely if ever staffs more than about 10 lanes, whatever the physical
count. That is consistent with these fits if each lane handles 250 to 360 passengers an hour, which
is within reach for CT lanes where laptops and liquids stay in the bag, but these data cannot
distinguish it from more, slower lanes. The capacity estimate also moves one for one with the
assumed transfer share, which is a guess.

The limit here is data, not modelling. Three measurements Kastrup almost certainly holds would
settle most of it: a log of lanes open over time, passengers screened per lane per hour, and
checkpoint entry counts from the queue sensors. With those the model stops inferring staffing and
starts testing staffing policies, which is where it becomes useful to an operator.
