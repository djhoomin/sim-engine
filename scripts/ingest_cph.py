"""Turn Copenhagen Airport security waiting times into a CSV time series.

  history   replay a git-scraped history of the public wait feed, one row per change
  poll      fetch the live feed once and append a row, for collecting from now on
  schedule  departures from Kastrup (EKCH) per day from the OpenSky Network, which needs an
            OpenSky API client. Pass its credentials.json or set OPENSKY_CLIENT_ID and
            OPENSKY_CLIENT_SECRET. Credentials never go into the repository.
  seats     join departures to aircraft types from OpenSky's aircraft database and seats from
            data/reference/aircraft_seats.csv, keeping passenger flights only

The public feed is https://cphwaitingtime.z6.web.core.windows.net/waitingtime.json. It gives
Terminal 2 and Terminal 3 security waits as bands such as "2-4 min". Kastrup has one central
security checkpoint for both terminals, and the two fields have always carried the same value. They
should only diverge once the new Terminal 3 building opens with its own security, expected around
2028. The history comes from
github.com/janschill/cph-ventetid-history, which committed the feed on every change from December
2024 to February 2026. That repository has no licence, so the CSV this writes stays out of git.

    uv run python -I scripts/ingest_cph.py history path/to/cph-ventetid-history data/cph/waits.csv
    uv run python -I scripts/ingest_cph.py poll data/cph/waits.csv
    uv run python -I scripts/ingest_cph.py schedule 2025-09-01 2025-09-07 data/cph/departures.csv \
        --credentials ~/.config/opensky/credentials.json

OpenSky reports when an aircraft was first seen airborne, not the scheduled departure, so the
departure times here run late by taxi time plus any delay.
"""

import argparse
import csv
from collections import Counter
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

FEED = "https://cphwaitingtime.z6.web.core.windows.net/waitingtime.json"
OPENSKY_TOKEN = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
OPENSKY_DEPARTURES = "https://opensky-network.org/api/flights/departure"
COPENHAGEN = ZoneInfo("Europe/Copenhagen")
# Callsign prefixes of freight airlines seen at Kastrup. Passenger types they fly carry no passengers.
FREIGHT_CALLSIGNS = {"BCS", "DHK", "DHX", "FDX", "UPS", "CLX", "GEC", "BOX", "ABR", "SWN", "NPT", "MSX"}
FIELDS = ["observed_utc", "delivery_id", "t2_from", "t2_to", "t3_from", "t3_to"]
# "9-11" until June 2025, "9-11 min" and "14 - 17 min" after.
BAND = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*(min)?\s*$")


def band(text: str) -> tuple[str, str]:
    """'2-4 min' to ('2', '4'). Anything else, such as a closed checkpoint, gives empty values."""
    m = BAND.match(text or "")
    return (m.group(1), m.group(2)) if m else ("", "")


def row(observed_utc: str, feed: dict) -> list[str]:
    return [
        observed_utc,
        feed.get("deliveryId", ""),
        *band(feed.get("t2WaitingTimeInterval", "")),
        *band(feed.get("t3WaitingTimeInterval", "")),
    ]


def history(repo: Path, out: Path) -> None:
    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--reverse", "--format=%H %cI", "--", "cph-ventetid.json"],
        capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    commits = [line.split(" ", 1) for line in log if line]
    # One git process streams every version of the file, rather than one process per commit.
    batch = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input="".join(f"{sha}:cph-ventetid.json\n" for sha, _ in commits),
        capture_output=True, text=True, check=True,
    ).stdout
    rows, skipped, pos = [], 0, 0
    for _, committed in commits:
        header_end = batch.index("\n", pos)
        header = batch[pos:header_end].split()
        if header[-1] == "missing":
            pos = header_end + 1
            skipped += 1
            continue
        size = int(header[2])
        body = batch[header_end + 1 : header_end + 1 + size]
        pos = header_end + 1 + size + 1
        try:
            feed = json.loads(body)
        except json.JSONDecodeError:
            skipped += 1
            continue
        observed = datetime.fromisoformat(committed).astimezone(timezone.utc).isoformat(timespec="seconds")
        rows.append(row(observed, feed))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f"{len(rows)} observations from {rows[0][0]} to {rows[-1][0]}, {skipped} commits skipped")


def poll(out: Path) -> None:
    with urllib.request.urlopen(FEED, timeout=30) as response:
        feed = json.load(response)
    new = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerow(row(datetime.now(timezone.utc).isoformat(timespec="seconds"), feed))


def opensky_credentials(path: Path | None) -> tuple[str, str]:
    if path is not None:
        creds = json.loads(path.expanduser().read_text())
        get = lambda *keys: next(creds[k] for k in keys if k in creds)
        return get("clientId", "client_id"), get("clientSecret", "client_secret")
    return os.environ["OPENSKY_CLIENT_ID"], os.environ["OPENSKY_CLIENT_SECRET"]


def opensky_token(client_id: str, client_secret: str) -> str:
    form = urllib.parse.urlencode(
        {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    ).encode()
    with urllib.request.urlopen(urllib.request.Request(OPENSKY_TOKEN, data=form), timeout=30) as r:
        return json.load(r)["access_token"]


def opensky_departures(token: str, day: date) -> list[dict]:
    """One Copenhagen calendar day of EKCH departures, retrying on rate limits and server errors."""
    start = datetime(day.year, day.month, day.day, tzinfo=COPENHAGEN)
    query = urllib.parse.urlencode({
        "airport": "EKCH",
        "begin": int(start.timestamp()),
        "end": int((start + timedelta(days=1)).timestamp()) - 1,
    })
    request = urllib.request.Request(f"{OPENSKY_DEPARTURES}?{query}", headers={"Authorization": f"Bearer {token}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # OpenSky answers 404 for a day with no flights
                return []
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            time.sleep(2**attempt * 5)
    return []


def schedule(first: str, last: str, out: Path, credentials: Path | None) -> None:
    token, issued = None, 0.0
    rows = []
    day = date.fromisoformat(first)
    while day <= date.fromisoformat(last):
        if token is None or time.time() - issued > 25 * 60:  # tokens last 30 minutes
            token, issued = opensky_token(*opensky_credentials(credentials)), time.time()
        flights = opensky_departures(token, day)
        for f in flights:
            seen = datetime.fromtimestamp(f["firstSeen"], tz=timezone.utc).astimezone(COPENHAGEN)
            rows.append([
                seen.date().isoformat(), seen.strftime("%H:%M"), f["icao24"],
                (f.get("callsign") or "").strip(), f.get("estArrivalAirport") or "",
            ])
        # A day with no departures at Kastrup is a gap in OpenSky's data, not a quiet day.
        print(f"{day} {len(flights)} departures" + ("  WARNING: no data from OpenSky" if not flights else ""))
        day += timedelta(days=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "first_seen", "icao24", "callsign", "destination"])
        w.writerows(sorted(rows))


def aircraft_types(database: Path, wanted: set[str]) -> dict[str, tuple[str, str]]:
    """Type code and operator ICAO code for the wanted ICAO24 addresses."""
    found = {}
    with open(database, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f, quotechar="'"):
            if r["icao24"] in wanted:
                found[r["icao24"]] = (r["typecode"].strip().upper(), r["operatorIcao"].strip().upper())
    return found


def seats(departures: list[Path], database: Path, seat_table: Path, out: Path) -> None:
    """Passenger departures with seats. Freight and non-passenger types are dropped and counted.

    Aircraft missing from the database, mostly newer SAS deliveries, take their airline's median
    seats over the flights whose type is known, marked `airline_median` in `seats_source`.
    """
    with open(seat_table, newline="") as f:
        table = {r["typecode"]: r for r in csv.DictReader(f)}
    flights = [r for path in departures for r in csv.DictReader(open(path, newline=""))]
    types = aircraft_types(database, {r["icao24"] for r in flights})
    kept, untyped, dropped = [], [], Counter()
    for r in flights:
        typecode, operator = types.get(r["icao24"], ("", ""))
        entry = table.get(typecode)
        airline = r["callsign"][:3]
        if airline in FREIGHT_CALLSIGNS:
            dropped[f"freight operator {airline}"] += 1
        elif not typecode:
            untyped.append(r)
        elif entry is None:
            dropped[f"type {typecode} not in seat table"] += 1
        elif entry["seats"] == "0":
            dropped[f"non-passenger type {typecode}"] += 1
        else:
            kept.append([r["date"], r["first_seen"], r["callsign"], typecode, airline, entry["seats"], "type"])

    by_airline: dict[str, list[int]] = {}
    for row in kept:
        by_airline.setdefault(row[4], []).append(int(row[5]))
    for r in untyped:
        airline = r["callsign"][:3]
        if airline in by_airline:
            median = sorted(by_airline[airline])[len(by_airline[airline]) // 2]
            kept.append([r["date"], r["first_seen"], r["callsign"], "", airline, str(median), "airline_median"])
        else:
            dropped[f"no type and no known airline {airline or 'none'}"] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "departure", "callsign", "typecode", "airline", "seats", "seats_source"])
        w.writerows(sorted(kept))
    imputed = sum(row[6] == "airline_median" for row in kept)
    print(f"{len(kept)} passenger departures kept of {len(flights)}, {imputed} with seats from the airline median")
    for reason, n in dropped.most_common(25):
        print(f"  dropped {n:5d}  {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("history")
    h.add_argument("repo", type=Path)
    h.add_argument("out", type=Path)
    p = sub.add_parser("poll")
    p.add_argument("out", type=Path)
    s = sub.add_parser("schedule")
    s.add_argument("first")
    s.add_argument("last")
    s.add_argument("out", type=Path)
    s.add_argument("--credentials", type=Path)
    j = sub.add_parser("seats")
    j.add_argument("out", type=Path)
    j.add_argument("database", type=Path)
    j.add_argument("seat_table", type=Path)
    j.add_argument("departures", type=Path, nargs="+")
    args = parser.parse_args()
    if args.cmd == "history":
        history(args.repo, args.out)
    elif args.cmd == "poll":
        poll(args.out)
    elif args.cmd == "schedule":
        schedule(args.first, args.last, args.out, args.credentials)
    else:
        seats(args.departures, args.database, args.seat_table, args.out)


if __name__ == "__main__":
    main()
