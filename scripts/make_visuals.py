"""Render the project's visuals: an animation of a simulated Kastrup morning and a validation figure.

  animation  a morning at Kastrup security, written as MP4 and GIF
  figure     model against reality in three panels, written as PNG and SVG

The animated day is not picked by eye. Of January 2026's held-out days, it is the one whose
hourly simulated waits differ from the observed ones by the median mean absolute error, so it is
a typical day for the model rather than its best or worst. Passengers come from that day's real departures with the demand,
staffing policy and shift-shaped staff limit fitted on 4 to 16 January, plus rail pulses.

Needs the Kastrup data in data/cph/ (see scripts/ingest_cph.py) and the viz dependency group.

    uv run --group viz python scripts/make_visuals.py animation figure
"""

import argparse
import csv
import datetime as dt
import importlib.util
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from simengine import kastrup  # noqa: E402
from simengine.control import QueuePolicy  # noqa: E402
from simengine.experiment import run_experiment  # noqa: E402
from simengine.queueing import mmc  # noqa: E402
from simengine.rng import Streams  # noqa: E402
from simengine.sources import day_flights, day_range, read_departures_by_day, read_wait_bands_hourly  # noqa: E402
from simengine.station import MMcConfig, mmc_model  # noqa: E402
from simengine.terminal import TerminalConfig, generate_demand, simulate_terminal  # noqa: E402

OUT = Path("docs/media")
SCHEDULE, WAITS = "data/cph/schedule_cph.csv", "data/cph/waits.csv"

# Palette, validated with the dataviz skill's checker: slot 1 blue and slot 2 orange on the light surface.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

# What was fitted for January 2026 (README, A staff limit by shift).
JANUARY = dict(load_factor=0.65, policy=QueuePolicy(8.0, 2.0, max_step=2, min_servers=2), ct_speed=0.6)
JANUARY_LIMIT = {3: 6, 5: 9, 7: 10, 9: 8, 11: 10, 13: 10, 15: 8, 17: 12, 19: 7, 21: 6, 23: 4}
TEST_DAYS = ("2026-01-17", "2026-01-30")
CREDIT = "Simulated with github.com/djhoomin/sim-engine. Departures: OpenSky Network. Observed waits: Copenhagen Airport's public wait feed."


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 11,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-", "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False,
    })


def staff_limit() -> tuple[int, ...]:
    limit = [10] * 24
    for start, cap in JANUARY_LIMIT.items():
        for h in range(start, min(start + 2, 24)):
            limit[h] = cap
    return tuple(limit)


def january_config(day: dt.date, schedule, transit: bool = True, limit=None) -> TerminalConfig:
    base = kastrup.day_config(day_flights(schedule, day), JANUARY["load_factor"], transit=transit)
    return TerminalConfig(**{
        **base.__dict__,
        "security_lanes": limit or staff_limit(),
        "security_policy": JANUARY["policy"],
        "security_layout": kastrup.transition_layout(JANUARY["ct_speed"]),
    })


def hourly_waits(run, hours) -> np.ndarray:
    arrive = np.array([v.arrive for v in run.security.visits])
    wait = np.array([v.wait for v in run.security.visits])
    return np.array([wait[(arrive >= 60 * h) & (arrive < 60 * (h + 1))].mean() for h in hours])


def observed_queue_wait(waits, day, hours) -> np.ndarray:
    return np.array([max(waits[(day, h)] - kastrup.NO_QUEUE, 0.0) for h in hours])


def median_day(schedule, waits) -> dt.date:
    hours = list(range(kastrup.DAY_START, 24))
    scored = []
    for day in day_range(*TEST_DAYS):
        if not (schedule[day] and schedule[day + dt.timedelta(days=1)]):
            continue
        config = january_config(day, schedule, transit=False)
        sims = [hourly_waits(simulate_terminal(config, generate_demand(config, Streams(7, r))), hours) for r in range(3)]
        scored.append((float(np.abs(np.mean(sims, axis=0) - observed_queue_wait(waits, day, hours)).mean()), day))
    scored.sort()
    mae, day = scored[len(scored) // 2]
    print(f"median held-out day {day} (mean absolute error {mae:.2f} min, range {scored[0][0]:.2f} to {scored[-1][0]:.2f})")
    return day


def observed_bands(day: dt.date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The feed's wait bands for one local day as a step series: minutes after midnight, low, high."""
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(kastrup.TIMEZONE)
    t, lo, hi = [], [], []
    for r in csv.DictReader(open(WAITS, newline="")):
        if not r["t2_from"]:
            continue
        local = dt.datetime.fromisoformat(r["observed_utc"]).astimezone(zone)
        minutes = (local.date() - day).days * 1440 + local.hour * 60 + local.minute + local.second / 60
        if -120 <= minutes <= 1560:
            t.append(minutes)
            lo.append(max(int(r["t2_from"]) - kastrup.NO_QUEUE, 0.0))
            hi.append(max(int(r["t2_to"]) - kastrup.NO_QUEUE, 0.0))
    return np.array(t), np.array(lo), np.array(hi)


def clock(minutes: float) -> str:
    m = int(round(minutes)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def make_animation(schedule, waits, start=4 * 60, end=10 * 60, step=2.0, fps=12) -> None:
    day = median_day(schedule, waits)
    config = january_config(day, schedule, transit=True)
    demand = generate_demand(config, Streams(11, 0))
    run = simulate_terminal(config, demand)
    cp = run.security
    names = [g.name for g in config.security_layout.groups]
    sizes = [g.lanes for g in config.security_layout.groups]
    kind = ["X-ray" if n == "xray" else "CT" for n in names]

    times = np.arange(start, end + step / 2, step)
    queue = np.zeros((len(names), len(times)))
    lanes = np.zeros((len(names), len(times)))
    for i, group in enumerate(cp.groups):
        arrive = np.array([v.arrive for v in group.visits])
        begin = np.array([v.start for v in group.visits])
        if len(arrive):
            queue[i] = ((arrive[:, None] <= times) & (begin[:, None] > times)).sum(axis=0)
        log_t = np.array([t for t, _ in group.capacity_log])
        log_n = np.array([n for _, n in group.capacity_log])
        lanes[i] = log_n[np.searchsorted(log_t, times, side="right") - 1]

    arrive_all = np.array([v.arrive for v in cp.visits])
    wait_all = np.array([v.wait for v in cp.visits])
    per_minute = np.histogram(arrive_all, np.arange(start, end + 1, 1.0))[0]
    minute_t = np.arange(start, end, 1.0) + 0.5
    grid = np.arange(start, end + 0.1, 5.0)
    sim_wait = np.array([wait_all[(arrive_all >= t - 10) & (arrive_all < t)].mean() if ((arrive_all >= t - 10) & (arrive_all < t)).any() else 0 for t in grid])
    obs_t, obs_lo, obs_hi = observed_bands(day)
    limit = config.security_lanes

    style()
    fig = plt.figure(figsize=(12, 6.75), dpi=100)
    fig.text(0.04, 0.94, "A morning at Kastrup security, simulated", fontsize=20, color=INK, weight="bold")
    fig.text(0.04, 0.905, f"{day.strftime('%A %-d %B %Y')}: that day's real departures, rail pulses, and lanes opened by a simulated supervisor", fontsize=11.5, color=INK_2)
    fig.text(0.04, 0.025, CREDIT, fontsize=8.5, color=MUTED)

    floor = fig.add_axes([0.04, 0.47, 0.6, 0.38])
    floor.set_xlim(-1.2, 12)
    floor.set_ylim(-0.7, len(names) - 0.3)
    floor.axis("off")
    floor.invert_yaxis()
    max_queue = max(queue.max(), 1)
    bar_scale = 8.5 / max(40, max_queue)
    lane_patches, queue_bars, queue_labels = [], [], []
    for i, (name, size) in enumerate(zip(names, sizes)):
        label = "X-ray" if name == "xray" else f"CT {name[2:]}"
        floor.text(-1.2, i, label, va="center", ha="left", fontsize=10.5, color=INK_2)
        row = []
        for k in range(size):
            p = FancyBboxPatch((0.25 + k * 0.62, i - 0.24), 0.48, 0.48, boxstyle="round,pad=0,rounding_size=0.08", linewidth=1, edgecolor=AXIS, facecolor=SURFACE)
            floor.add_patch(p)
            row.append(p)
        lane_patches.append(row)
        bar = FancyBboxPatch((1.75, i - 0.2), 0.0, 0.4, boxstyle="round,pad=0,rounding_size=0.06", linewidth=0, facecolor=BLUE if kind[i] == "CT" else ORANGE, alpha=0.85)
        floor.add_patch(bar)
        queue_bars.append(bar)
        queue_labels.append(floor.text(1.85, i, "", va="center", ha="left", fontsize=10, color=INK_2))
    floor.text(0.25, -0.62, "lanes", fontsize=9.5, color=MUTED)
    floor.text(1.75, -0.62, "queue, passengers", fontsize=9.5, color=MUTED)

    hero = fig.add_axes([0.68, 0.47, 0.28, 0.38])
    hero.axis("off")
    hero_clock = hero.text(0.0, 0.78, "", fontsize=52, color=INK, weight="bold", va="center")
    hero_wait = hero.text(0.0, 0.46, "", fontsize=14, color=INK, va="center")
    hero_lanes = hero.text(0.0, 0.3, "", fontsize=14, color=INK, va="center")
    hero.text(0.0, 0.1, "Lanes fill CT pairs first, X-ray only as overflow", fontsize=10, color=MUTED, va="center")
    for y, colour, text in ((0.0, BLUE, "CT lanes"), (-0.12, ORANGE, "X-ray lanes")):
        hero.add_patch(FancyBboxPatch((0.0, y - 0.035), 0.05, 0.07, boxstyle="round,pad=0,rounding_size=0.01", facecolor=colour, linewidth=0, transform=hero.transAxes, clip_on=False))
        hero.text(0.08, y, text, fontsize=10, color=INK_2, va="center", clip_on=False)

    hours = np.arange(start, end + 1, 60)
    arr_ax = fig.add_axes([0.06, 0.1, 0.4, 0.27])
    arr_ax.set_xlim(start, end)
    arr_ax.set_ylim(0, per_minute.max() * 1.15)
    arr_ax.set_xticks(hours, [clock(h) for h in hours])
    arr_ax.set_title("Passengers joining the queue, per minute", loc="left", fontsize=11, color=INK)
    arr_line, = arr_ax.plot([], [], color=BLUE, linewidth=1.5)
    arr_now = arr_ax.axvline(start, color=AXIS, linewidth=1)

    wait_ax = fig.add_axes([0.55, 0.1, 0.41, 0.27])
    wait_ax.set_xlim(start, end)
    wait_ax.set_ylim(0, max(16, obs_hi[(obs_t >= start) & (obs_t <= end)].max() + 2))
    wait_ax.set_xticks(hours, [clock(h) for h in hours])
    wait_ax.set_title("Wait in the queue, minutes", loc="left", fontsize=11, color=INK)
    sim_line, = wait_ax.plot([], [], color=BLUE, label="Simulated, last 10 minutes")
    obs_fill = [None]
    wait_now = wait_ax.axvline(start, color=AXIS, linewidth=1)
    wait_ax.fill_between([], [], [], color=INK_2, alpha=0.18, linewidth=0, label="Observed, Kastrup feed band")
    wait_ax.legend(loc="upper left", fontsize=9.5, labelcolor=INK_2, handlelength=1.6)

    def draw(frame: int):
        t = times[frame]
        open_now = 0
        for i in range(len(names)):
            n_open = int(lanes[i, frame])
            open_now += n_open
            colour = BLUE if kind[i] == "CT" else ORANGE
            for k, p in enumerate(lane_patches[i]):
                p.set_facecolor(colour if k < n_open else SURFACE)
                p.set_edgecolor(colour if k < n_open else AXIS)
            q = queue[i, frame]
            queue_bars[i].set_width(q * bar_scale)
            queue_bars[i].set_visible(q > 0)
            queue_labels[i].set_position((1.85 + q * bar_scale + 0.05, i))
            queue_labels[i].set_text(f"{int(q)}" if q > 0 else "")
        hero_clock.set_text(clock(t))
        recent = (arrive_all >= t - 10) & (arrive_all < t)
        current = wait_all[recent].mean() if recent.any() else 0.0
        hero_wait.set_text(f"Wait now  {current:.1f} min")
        hero_lanes.set_text(f"Lanes open  {open_now} of {limit[int(t // 60) % 24]} staffed")
        shown = minute_t <= t
        arr_line.set_data(minute_t[shown], per_minute[shown])
        arr_now.set_xdata([t, t])
        g = grid <= t
        sim_line.set_data(grid[g], sim_wait[g])
        if obs_fill[0] is not None:
            obs_fill[0].remove()
        o = (obs_t >= start - 30) & (obs_t <= t)
        obs_fill[0] = wait_ax.fill_between(obs_t[o], obs_lo[o], obs_hi[o], step="post", color=INK_2, alpha=0.18, linewidth=0)
        wait_now.set_xdata([t, t])
        return []

    OUT.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, draw, frames=len(times), interval=1000 / fps)
    anim.save(OUT / "kastrup_morning.mp4", writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    draw(len(times) - 1)
    fig.savefig(OUT / "kastrup_morning_final.png", dpi=100)
    plt.close(fig)
    # A GIF from the MP4 through one shared palette, 900 pixels wide, keeps it small and clean.
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT / "kastrup_morning.mp4"), "-vf",
        f"fps={fps},scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=none",
        str(OUT / "kastrup_morning.gif"),
    ], check=True)
    for name in ("kastrup_morning.mp4", "kastrup_morning.gif", "kastrup_morning_final.png"):
        print(f"wrote {OUT / name} ({(OUT / name).stat().st_size / 1e6:.1f} MB)")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel_theory(ax) -> None:
    rhos = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    curve = np.linspace(0.45, 0.92, 100)
    ax.plot(curve, [mmc(3 * r, 1.0, 3).Wq for r in curve], color=INK_2, linewidth=2, label="Erlang C, exact")
    means, halves = [], []
    for r in rhos:
        record = run_experiment(mmc_model, MMcConfig(3 * r, 1.0, 3, horizon=20_000, warmup=2_000), 10, base_seed=20261011)
        ci = record.interval("Wq")
        means.append(ci.mean)
        halves.append(ci.half_width)
    ax.errorbar(rhos, means, yerr=halves, fmt="o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, elinewidth=2, capsize=0, label="Simulated, 95% interval")
    ax.set_xlabel("Utilisation per lane")
    ax.set_ylabel("Mean wait, multiples of service time")
    ax.legend(loc="upper left", fontsize=9.5, labelcolor=INK_2)


def panel_msy(ax) -> None:
    replay = load_script("replay_us")
    args = argparse.Namespace(
        schedule="data/us/msy_schedule_2026_07.csv", tsa="data/us/msy_tsa_2026_07.csv",
        fit=["2026-07-12", "2026-07-18"], bands=[12], service_mean=0.4,
    )
    bands, dispersion = replay.fit_demand(args)
    day = dt.date(2026, 7, 23)
    config = replay.day_config(args, day.isoformat(), bands, dispersion)
    from simengine.terminal import terminal_model
    from simengine.sources import read_checkpoint_hourly

    record = run_experiment(terminal_model, config, 30, base_seed=20261011)
    hours = np.arange(3, 23)
    sims = np.array([record.kpi(f"security_screened_h{h:02d}") for h in hours])
    observed = read_checkpoint_hourly(args.tsa)
    obs = np.array([observed[(day, int(h))] for h in hours])
    low, high = np.percentile(sims, [2.5, 97.5], axis=1)
    ax.fill_between(hours, low, high, color=BLUE, alpha=0.12, linewidth=0, label="Simulated, 95% band")
    ax.plot(hours, sims.mean(axis=1), color=BLUE, label="Simulated, mean")
    ax.plot(hours, obs, color=INK_2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label="TSA count")
    ax.set_xticks(hours[::3], [f"{h:02d}:00" for h in hours[::3]])
    ax.set_ylabel("Passengers screened per hour")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", bbox_to_anchor=(0.1, 0.0), fontsize=9.5, labelcolor=INK_2)


def panel_kastrup(ax, schedule, waits) -> float:
    hours = list(range(kastrup.DAY_START, 24))
    days = [d for d in day_range(*TEST_DAYS) if schedule[d] and schedule[d + dt.timedelta(days=1)]]
    flat, shift, obs = [], [], []
    for day in days:
        obs.append(observed_queue_wait(waits, day, hours))
        for store, limit in ((flat, (10,) * 24), (shift, None)):
            config = january_config(day, schedule, transit=False, limit=limit)
            # The same seed and replications as fit_staff_cap_cph.py's held-out test, so R2 matches it.
            store.append(np.mean([hourly_waits(simulate_terminal(config, generate_demand(config, Streams(20261012, r))), hours) for r in range(10)], axis=0))
    obs_m, flat_m, shift_m = (np.mean(x, axis=0) for x in (obs, flat, shift))
    r2 = lambda sim: 1 - ((sim - obs_m) ** 2).sum() / ((obs_m - obs_m.mean()) ** 2).sum()
    ax.plot(hours, obs_m, color=INK_2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label="Observed, Kastrup feed")
    ax.plot(hours, flat_m, color=ORANGE, label=f"Staff limit 10 all day, R\u00b2 {r2(flat_m):.2f}")
    ax.plot(hours, shift_m, color=BLUE, label=f"Staff limit by shift, R\u00b2 {r2(shift_m):.2f}")
    ax.set_xticks(hours[::3], [f"{h:02d}:00" for h in hours[::3]])
    ax.set_ylabel("Mean wait in the queue, minutes")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=9.5, labelcolor=INK_2)
    print(f"Kastrup panel: {len(days)} held-out days, R2 flat {r2(flat_m):.2f}, by shift {r2(shift_m):.2f}")
    return r2(shift_m)


def make_figure(schedule, waits) -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=100)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.74, bottom=0.16, wspace=0.28)
    fig.text(0.05, 0.94, "Checked against theory, then against real airports", fontsize=18, color=INK, weight="bold")
    fig.text(0.05, 0.895, "Every number is replicated with common random numbers, and every comparison with real data is on days the model was not fitted to.", fontsize=11, color=INK_2)
    titles = [
        ("1. The engine matches queueing theory", "Three lanes, Poisson arrivals: simulated mean wait against Erlang C"),
        ("2. Demand, New Orleans (MSY)", "Thursday 23 July 2026, profiles fitted on the week before"),
        ("3. Waits, Copenhagen (Kastrup)", "Mean by hour over 12 held-out days, January 2026"),
    ]
    for ax, (title, sub) in zip(axes, titles):
        ax.set_title(f"{title}\n", loc="left", fontsize=12.5, color=INK, weight="bold")
        ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=9.5, color=INK_2)
    panel_theory(axes[0])
    panel_msy(axes[1])
    panel_kastrup(axes[2], schedule, waits)
    fig.text(0.05, 0.02, "Simulated with github.com/djhoomin/sim-engine. MSY: TSA throughput, BTS schedules, FAA seats. Kastrup: OpenSky Network departures, Copenhagen Airport's public wait feed.", fontsize=8.5, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "model_against_reality.png", dpi=200)
    fig.savefig(OUT / "model_against_reality.svg")
    plt.close(fig)
    for name in ("model_against_reality.png", "model_against_reality.svg"):
        print(f"wrote {OUT / name} ({(OUT / name).stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", nargs="+", choices=["animation", "figure"])
    args = parser.parse_args()
    schedule = read_departures_by_day(SCHEDULE)
    waits = read_wait_bands_hourly(WAITS, kastrup.TIMEZONE)
    if "animation" in args.what:
        make_animation(schedule, waits)
    if "figure" in args.what:
        make_figure(schedule, waits)


if __name__ == "__main__":
    main()
