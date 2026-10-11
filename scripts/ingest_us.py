"""Turn public US sources into the inputs calibration needs, for one airport.

  tsa       TSA FOIA weekly throughput PDFs to hourly checkpoint counts
  schedule  BTS on-time CSV plus the FAA aircraft registry to departures with seats

Raw downloads stay outside the repository. Only the small per-airport CSVs this writes go into
data/. All three sources are US government works.

    uv run python -I scripts/ingest_us.py tsa MSY data/us/msy_tsa.csv week1.pdf week2.pdf
    uv run python -I scripts/ingest_us.py schedule MSY data/us/msy_schedule.csv ontime.csv faa_dir
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
HOUR = re.compile(r"^\d{2}:\d{2}$")


def ingest_tsa(airport: str, out: Path, pdfs: list[Path]) -> None:
    import pdfplumber

    rows = []
    for path in pdfs:
        date = hour = code = None
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for cells in page.extract_table() or []:
                    # Columns: date, hour, code, name, city, state, checkpoint, passengers. Some
                    # pages add an empty ninth column. Merged cells come back empty, so date, hour
                    # and airport carry forward, and only values that look right are taken, which
                    # skips headers and page-number footers.
                    if len(cells) < 8:
                        continue
                    date = cells[0] if cells[0] and DATE.match(cells[0]) else date
                    hour = cells[1] if cells[1] and HOUR.match(cells[1]) else hour
                    code = cells[2] or code
                    count = (cells[7] or "").replace(",", "")
                    if code == airport and cells[6] and count.isdigit():
                        rows.append((date, hour, cells[6], int(count)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "hour", "checkpoint", "passengers"])
        w.writerows(rows)
    print(f"{len(rows)} hourly rows for {airport}", Counter(r[2] for r in rows))


def faa_seats(faa_dir: Path) -> dict[str, int]:
    """Seats by N-number, from the registry's model reference table."""
    with open(faa_dir / "ACFTREF.txt", newline="", encoding="utf-8-sig") as f:
        seats_by_model = {r["CODE"].strip(): int(r["NO-SEATS"]) for r in csv.DictReader(f)}
    with open(faa_dir / "MASTER.txt", newline="", encoding="utf-8-sig") as f:
        return {
            "N" + r["N-NUMBER"].strip(): seats_by_model.get(r["MFR MDL CODE"].strip(), 0)
            for r in csv.DictReader(f)
        }


def ingest_schedule(airport: str, out: Path, ontime: Path, faa_dir: Path) -> None:
    seats = faa_seats(faa_dir)
    rows, missing = [], Counter()
    with open(ontime, newline="", encoding="latin-1") as f:
        for r in csv.DictReader(f):
            if r["Origin"] != airport or r["Cancelled"].startswith("1"):
                continue
            n = seats.get(r["Tail_Number"], 0)
            if n == 0:
                missing[r["Reporting_Airline"]] += 1
            hhmm = r["CRSDepTime"].zfill(4)
            rows.append((r["FlightDate"], f"{hhmm[:2]}:{hhmm[2:]}", r["Reporting_Airline"], r["Tail_Number"], n))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "departure", "carrier", "tail", "seats"])
        w.writerows(sorted(rows))
    print(f"{len(rows)} departures from {airport}, seats unknown for {sum(missing.values())}", missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tsa")
    t.add_argument("airport")
    t.add_argument("out", type=Path)
    t.add_argument("pdfs", type=Path, nargs="+")
    s = sub.add_parser("schedule")
    s.add_argument("airport")
    s.add_argument("out", type=Path)
    s.add_argument("ontime", type=Path)
    s.add_argument("faa_dir", type=Path)
    args = parser.parse_args()
    if args.cmd == "tsa":
        ingest_tsa(args.airport, args.out, args.pdfs)
    else:
        ingest_schedule(args.airport, args.out, args.ontime, args.faa_dir)


if __name__ == "__main__":
    main()
