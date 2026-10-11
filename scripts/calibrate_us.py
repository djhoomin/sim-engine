"""Fit the show-up profile to one week of TSA checkpoint counts and test it on another.

Reads the CSVs that ingest_us.py writes. The checkpoint closes overnight, so each day's hours up
to the first busy one fold into a single bin, since early arrivals are screened when it opens.

    uv run python scripts/calibrate_us.py data/us/msy_schedule_2026_07.csv data/us/msy_tsa_2026_07.csv \\
        --fit 2026-07-12 2026-07-18 --test 2026-07-19 2026-07-25
"""

import argparse
import csv
import datetime as dt
from collections import defaultdict

import numpy as np

from simengine.calibration import ShowUpFit, expected_showups, fit_showup

FIRST_OPEN_HOUR = 4  # hours 00 to 03 fold into one bin ending at 04:00
PLACEHOLDER = (110.0, 35.0)  # the synthetic model's guess, for comparison


def day_range(first: str, last: str) -> list[dt.date]:
    a, b = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def load(schedule_csv: str, tsa_csv: str, days: list[dt.date]):
    """Departures and seats on one minute clock, plus bin edges and observed counts."""
    origin = days[0]
    minute = lambda d, hhmm: (d - origin).days * 1440 + 60 * int(hhmm[:2]) + int(hhmm[3:5])

    departures, seats = [], []
    for r in csv.DictReader(open(schedule_csv)):
        d = dt.date.fromisoformat(r["date"])
        if days[0] <= d <= days[-1] + dt.timedelta(days=1):  # next morning's flights draw tonight's arrivals
            departures.append(minute(d, r["departure"]))
            seats.append(int(r["seats"]))

    hourly = defaultdict(int)
    for r in csv.DictReader(open(tsa_csv)):
        d = dt.datetime.strptime(r["date"], "%m/%d/%Y").date()
        if days[0] <= d <= days[-1]:
            hourly[(d, int(r["hour"][:2]))] += int(r["passengers"])

    edges, observed, hour_of_bin = [0], [], []
    for d in days:
        base = (d - origin).days * 1440
        observed.append(sum(hourly[(d, h)] for h in range(FIRST_OPEN_HOUR)))
        hour_of_bin.append(FIRST_OPEN_HOUR - 1)
        edges.append(base + 60 * FIRST_OPEN_HOUR)
        for h in range(FIRST_OPEN_HOUR, 24):
            observed.append(hourly[(d, h)])
            hour_of_bin.append(h)
            edges.append(base + 60 * (h + 1))
    return np.array(departures, float), np.array(seats, float), np.array(edges, float), np.array(observed, float), np.array(hour_of_bin)


def scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - observed
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / np.sum((observed - observed.mean()) ** 2)),
        "total_error_pct": float(100 * err.sum() / observed.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("schedule")
    parser.add_argument("tsa")
    parser.add_argument("--fit", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--test", nargs=2, required=True, metavar=("FIRST", "LAST"))
    args = parser.parse_args()

    fit_data = load(args.schedule, args.tsa, day_range(*args.fit))
    test_data = load(args.schedule, args.tsa, day_range(*args.test))
    fit = fit_showup(*fit_data[:4])
    print(f"fitted on {args.fit[0]} to {args.fit[1]}: mean {fit.mean:.1f} min, sd {fit.sd:.1f} min, load factor {fit.load_factor:.3f}")

    # The placeholder gets its own best load factor on the fit week, so only the shape differs.
    dep, seats, edges, obs, _ = fit_data
    unit = expected_showups(dep, seats, edges, *PLACEHOLDER)
    placeholder = ShowUpFit(*PLACEHOLDER, obs.sum() / unit.sum(), float("nan"), len(obs))

    dep, seats, edges, obs, hours = test_data
    print(f"\nout of sample, {args.test[0]} to {args.test[1]}, {len(obs)} bins:")
    for name, model in [("fitted", fit), ("placeholder 110/35", placeholder)]:
        s = scores(obs, model.predict(dep, seats, edges))
        print(f"  {name:20} " + "  ".join(f"{k} {v:8.2f}" for k, v in s.items()))

    pred = fit.predict(dep, seats, edges)
    print("\nmean by hour of day, test week (03 holds everything up to 04:00)")
    print(" hour  observed  fitted  error")
    for h in range(FIRST_OPEN_HOUR - 1, 24):
        m = hours == h
        print(f"   {h:02d}  {obs[m].mean():8.0f} {pred[m].mean():7.0f} {pred[m].mean() - obs[m].mean():+6.0f}")


if __name__ == "__main__":
    main()
