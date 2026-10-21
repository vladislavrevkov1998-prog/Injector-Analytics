"""Reproduce conditional confined-flow figures; no measured-flow claim.

Run with PYTHONPATH=src python tools/create_v07_schematics.py.
The diagram geometry matches the v0.6.3 teaching schemes, but fuel is
tracked from A to B and wall capture retains every inlet parcel weight.
This small deterministic illustration is independent of runtime fuel tracing.
"""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.tri as mtri
import numpy as np

from fuelfield.carrier import build_carrier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/images/v0.7"
COLORS = ("#247eae", "#9460a8", "#cd7b22")
U_CARRIER = 20.0
U_INJECTION = 40.0
TAU = .005
FILM_SPEED = 1.0
PARCELS_PER_BAND = 9
STEP_MM = .2


def geometry(before, after):
    return dict(schema_version=1, enabled=True, base_z_mm=10., outlet_z_mm=90.,
                primitives=[dict(id=wall, kind="line", wall=wall, unit="mm",
                                 start=[10., before[k]], end=[90., after[k]])
                            for wall, k in (("inner", 0), ("outer", -1))],
                objects=[], groups=[], sections=[dict(id="S1", z_mm=90., is_outlet=True)])


def bounds(z, before, after):
    t = (z-10.)/80.
    return ((1-t)*before[0]+t*after[0], (1-t)*before[-1]+t*after[-1])


def trace(carrier, r0, before, after):
    """Frozen-carrier exponential drag update, first-contact wall capture.

    Weights are fixed; the independent z-step illustration assumes forward
    motion. It has zero stochastic dispersion and no evaporation/breakup.
    """
    pos = np.array([10., r0])
    velocity = np.array([U_INJECTION, 0.])
    path = [pos.copy()]
    film_path = []
    film_wall = None
    for target_z in np.arange(10.+STEP_MM, 90.+STEP_MM/2, STEP_MM):
        target_z = min(float(target_z), 90.)
        if film_wall is not None:
            pos = np.array([target_z, bounds(target_z, before, after)[film_wall]])
            film_path.append(pos.copy())
            continue
        gas = np.asarray(carrier.meridional_velocity(pos[0], pos[1]), dtype=float)
        if min(gas[0], velocity[0]) <= 0:
            raise RuntimeError("Teaching tracer requires forward axial motion")
        dz_m = (target_z-pos[0])/1000.
        # Solve exact axial displacement with carrier frozen over this step.
        lo, hi = 0., 2*dz_m/min(gas[0], velocity[0])
        for _ in range(35):
            dt = (lo+hi)/2
            delta = gas*dt+(velocity-gas)*TAU*(-np.expm1(-dt/TAU))
            if delta[0] < dz_m:
                lo = dt
            else:
                hi = dt
        dt = (lo+hi)/2
        delta = gas*dt+(velocity-gas)*TAU*(-np.expm1(-dt/TAU))
        nxt = pos+1000*delta
        nxt[0] = target_z
        inner, outer = bounds(target_z, before, after)
        if nxt[1] < inner or nxt[1] > outer:
            film_wall = 0 if nxt[1] < inner else 1
            old_wall = bounds(pos[0], before, after)[film_wall]
            new_wall = (inner, outer)[film_wall]
            gap_old, gap_new = pos[1]-old_wall, nxt[1]-new_wall
            fraction = gap_old/(gap_old-gap_new)
            contact = pos+fraction*(nxt-pos)
            path.append(contact)
            film_path.append(contact.copy())
            pos = np.array([target_z, (inner, outer)[film_wall]])
            film_path.append(pos.copy())
        else:
            pos = nxt
            velocity = gas+(velocity-gas)*math.exp(-dt/TAU)
            path.append(pos.copy())
    category = (3+film_wall) if film_wall is not None else int(np.clip(np.searchsorted(after, pos[1], side="right")-1, 0, 2))
    return np.asarray(path), np.asarray(film_path), category


def add_arrow(ax, path, fraction, color, width=1.1):
    if len(path) < 3:
        return
    k = min(len(path)-2, max(1, round(fraction*(len(path)-1))))
    a, b = path[max(0, k-3)], path[min(len(path)-1, k+3)]
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=width, mutation_scale=8))


def diagram(name, before, after, title):
    before, after = np.asarray(before, float), np.asarray(after, float)
    carrier = build_carrier(geometry(before, after), U_CARRIER, max_step_mm=.5, radial_cells=40)
    z_grid = np.broadcast_to(carrier.z_mm[:, None], carrier.r_mm.shape)
    triangulation = mtri.Triangulation(z_grid.ravel(), carrier.r_mm.ravel(), carrier.triangles)
    psi = carrier.streamfunction.ravel()
    fig = plt.figure(figsize=(12.2, 8.4), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[4.3, 1.45, .75])
    ax = fig.add_subplot(grid[0])
    for z in (10., 30., 50., 70., 90.):
        inner, outer = bounds(z, before, after)
        ax.plot([z, z], [inner, outer], color="#c1cad5", lw=.8, zorder=0)
    for j in range(4):
        ax.plot([10, 90], [before[j], after[j]], color="#b5bfca", ls=":", lw=.8, zorder=0)
    ax.tricontour(triangulation, psi, levels=np.linspace(psi.min(), psi.max(), 11)[1:-1],
                  colors="#7b9dac", linewidths=.75, linestyles="dashed", zorder=1)
    values = np.zeros((3, 5))
    tracks = []
    for band, color in enumerate(COLORS):
        starts = np.sqrt(before[band]**2+(np.arange(PARCELS_PER_BAND)+.5)/PARCELS_PER_BAND
                         *(before[band+1]**2-before[band]**2))
        for r0 in starts:
            path, film, category = trace(carrier, float(r0), before, after)
            values[band, category] += 1/PARCELS_PER_BAND
            ax.plot(path[:, 0], path[:, 1], color=color, lw=1.25, alpha=.85, zorder=3)
            add_arrow(ax, path, .55, color)
            if len(film):
                ax.plot(film[:, 0], film[:, 1], color="#e14f31", lw=2.5, zorder=4)
                add_arrow(ax, film, .7, "#b83622", 1.5)
            tracks.append(dict(source_band=band+1, inlet_radius_mm=float(r0),
                               relative_band_weight=1/PARCELS_PER_BAND,
                               output_category=category, wall_capture=bool(len(film))))
        ax.text(7.5, (before[band]+before[band+1])/2, f"A{band+1}",
                ha="right", va="center", color=color, weight="bold")
        ax.text(92.5, (after[band]+after[band+1])/2, f"B{band+1}",
                va="center", color=color, weight="bold")
    for k in (0, -1):
        ax.plot([10, 90], [before[k], after[k]], color="#293d4c", lw=1.7, zorder=2)
    ax.set(xlim=(2, 98), ylim=(96, 164), ylabel="R, мм", xlabel="z, мм",
           title=title)
    ax.set_xticks([10, 30, 50, 70, 90], ["A: 10", "30", "50", "70", "B: 90"])
    ax.grid(axis="y", alpha=.15)
    handles = [Line2D([], [], color="#7b9dac", ls="--", label="носитель: ψ = const"),
               Line2D([], [], color=COLORS[0], label="капли: инерция + сопротивление"),
               Line2D([], [], color="#e14f31", lw=2.5, label="захваченное топливо: плёнка")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=3,
              fontsize=8, framealpha=.94)
    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")
    display = np.column_stack((values, values.sum(axis=1)))*100
    table = table_ax.table(cellText=[[f"{v:.1f}%" for v in row] for row in display],
                           rowLabels=["вклад A1", "вклад A2", "вклад A3"],
                           colLabels=["взвесь B1", "взвесь B2", "взвесь B3", "плёнка внутри", "плёнка снаружи", "вся масса"],
                           cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(.94, 1.55)
    for j, color in enumerate(COLORS):
        table[j+1, -1].set_facecolor(color)
        table[j+1, -1].get_text().set_color("white")
    note = fig.add_subplot(grid[2])
    note.axis("off")
    note.text(.5, .9,
              "Синтетический пример: Uнос = 20 м/с, Uкап = 40 м/с вдоль z; τ = 5 мс; u′ = 0; Uпл = 1 м/с.\n"
              "9 равных по расходу пакетов на каждый входной пояс; расходы пакетов сохраняются.\n"
              "Захват при первом контакте; плёнка идёт по стенке. Таблица иллюстративная, шаг доли 1/9.\n"
              "Скольжение носителя у стенки; нет расчёта турбулентности, отрыва, обратного влияния и дробления.",
              ha="center", va="top", fontsize=9, color="#344658")
    fig.suptitle("Условная модель в замкнутом тракте: топливо остаётся во взвеси или в плёнке", fontsize=13, weight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/f"{name}.png", dpi=145)
    vector=OUT/f"{name}.svg"
    fig.savefig(vector)
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')
    plt.close(fig)
    return dict(geometry=geometry(before.tolist(), after.tolist()),
                prescribed_inputs=dict(carrier_inlet_velocity_m_s=U_CARRIER,
                                       injection_velocity_m_s=U_INJECTION,
                                       relaxation_time_s=TAU, film_speed_m_s=FILM_SPEED,
                                       transverse_rms_m_s=0., axial_step_mm=STEP_MM,
                                       parcels_per_band=PARCELS_PER_BAND),
                relative_band_transfers=values.tolist(),
                max_row_mass_error=float(np.max(np.abs(values.sum(axis=1)-1))),
                carrier_diagnostics=dict(carrier.diagnostics), parcels=tracks)


if __name__ == "__main__":
    results = {
        "narrowing": diagram("narrowing-transport", [100, 120, 140, 160], [115, 125, 135, 145],
                             "Сужение: вклад каждого входного пояса сохраняется до B"),
        "widening": diagram("widening-transport", [115, 125, 135, 145], [100, 120, 140, 160],
                            "Расширение: инерция капель отличается от движения носителя"),
    }
    (OUT/"schematic-inputs-and-ledger.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: dict(max_row_mass_error=row["max_row_mass_error"],
                                 relative_band_transfers=row["relative_band_transfers"])
                      for name, row in results.items()}, ensure_ascii=False, indent=2))
