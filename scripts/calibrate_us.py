"""Fit the show-up profile to one week of TSA checkpoint counts and test it on another.

Reads the CSVs that ingest_us.py writes. The checkpoint closes overnight, so each day's hours up
to the first busy one fold into a single bin, since early arrivals are screened when it opens.

    uv run python scripts/calibrate_us.py data/us/msy_schedule_2026_07.csv data/us/msy_tsa_2026_07.csv \\
        --fit 2026-07-12 2026-07-18 --test 2026-07-19 2026-07-25
"""

import argparse

import numpy as np

from simengine.calibration import ShowUpFit, expected_showups, fit_scores, fit_showup, fit_showup_grouped
from simengine.sources import calibration_inputs, day_range

FIRST_OPEN_HOUR = 4  # hours 00 to 03 fold into one bin ending at 04:00
PLACEHOLDER = (110.0, 35.0)  # the synthetic model's guess, for comparison
BANDS = {"one profile": [], "morning and afternoon": [12], "three bands": [10, 15]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("schedule")
    parser.add_argument("tsa")
    parser.add_argument("--fit", nargs=2, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("--test", nargs=2, required=True, metavar=("FIRST", "LAST"))
    args = parser.parse_args()

    fd = calibration_inputs(args.schedule, args.tsa, day_range(*args.fit), FIRST_OPEN_HOUR)
    td = calibration_inputs(args.schedule, args.tsa, day_range(*args.test), FIRST_OPEN_HOUR)
    print(f"fit {args.fit[0]} to {args.fit[1]}, test {args.test[0]} to {args.test[1]}, {len(td.observed)} test bins\n")

    # The placeholder gets its own best load factor on the fit week, so only the shape differs.
    unit = expected_showups(fd.departures, fd.seats, fd.edges, *PLACEHOLDER)
    placeholder = ShowUpFit(*PLACEHOLDER, fd.observed.sum() / unit.sum(), float("nan"), len(fd.observed))
    s = fit_scores(td.observed, placeholder.predict(td.departures, td.seats, td.edges))
    print(f"{'placeholder 110/35':22} test r2 {s['r2']:.3f}  rmse {s['rmse']:6.1f}")

    for name, cuts in BANDS.items():
        band = lambda dep: np.searchsorted(np.array(cuts) * 60, dep % 1440, side="right")
        fit = fit_showup_grouped(fd.departures, fd.seats, band(fd.departures), fd.edges, fd.observed)
        s = fit_scores(td.observed, fit.predict(td.departures, td.seats, band(td.departures), td.edges))
        params = " | ".join(f"{g.mean:.0f}±{g.sd:.0f} load {g.load_factor:.3f}" for g in fit.groups)
        print(f"{name:22} test r2 {s['r2']:.3f}  rmse {s['rmse']:6.1f}  {params}")


if __name__ == "__main__":
    main()
