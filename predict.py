"""Predict SM 2026 8+ outcomes with a Plackett–Luce rank-based model.

Why Plackett–Luce: rowing race times vary wildly with wind, water, and
timing-rig issues (see SM25 men's heat 1). A rank-based model uses only
finishing orders, which are condition-invariant. Sandbagging by heat winners
matters less too — only "who finished ahead of whom" enters the fit.

Approach:
  1. Extract finishing rankings for every SM24 + SM25 race.
  2. Fit P-L strengths with choix.ilsr_rankings (regularised, handles partial
     rankings natively). Weights: 2025 × 2 vs 2024 (student-crew turnover);
     A-final × 2.5, rep × 1.6, heat × 1.0.
  3. Fit a seconds-per-strength-unit scale α from within-race time gaps, so
     we can display "expected" times.
  4. Bracket Monte Carlo: each race samples a P-L ordering via the Gumbel-max
     trick. Reps are the canonical SM format: top-3 rep times → A-final.
  5. Render the *most likely* path in the flow diagram (deterministic by
     strength) with Monte-Carlo probabilities as badges.
"""
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import choix

random.seed(0xC0FFEE)
np.random.seed(0xC0FFEE)

ROOT = Path(__file__).parent
NSIM = 50_000

# 2026 offisiell heat-trekning (låst fra arrangør).
OFFICIAL_HEATS = {
    "M": [
        ["NTNUIH1", "BSIH2", "MRKH1", "OSIH2", "CREW"],       # Heat 1
        ["NTNUIH2", "JUSTITIAH1", "BSIH1", "RNNABC2"],         # Heat 2
        ["OSIH1", "RNNABC1", "MRKH2", "JRC"],                  # Heat 3
    ],
    "W": [
        ["NTNUID1", "JUSTITIAD1", "SKKRD2", "MRKD1", "BSID2", "OMRD2"],      # Heat 1
        ["OSID1", "JRCD1", "OKTAGOND2", "OMRD1", "NTNUID3", "NTNUID2"],      # Heat 2
        ["OKTAGOND1", "SKKRD1", "JUSTITIAD2", "BSID1", "OSID2", "MRKD2"],    # Heat 3
    ],
}
TEAMS_2026 = {g: [t for h in heats for t in h] for g, heats in OFFICIAL_HEATS.items()}

GENDERS = ("M", "W")

# Each +/- step shifts perceived strength by this many seconds over 1600m.
SEC_PER_STEP = 2.0

# 2026 prior-knowledge strength shifts (integer steps).
PRIOR_ADJUSTMENTS = {
    "M": {"OSIH1": 2, "OSIH2": 1},
}

STAGE_WEIGHT = {"heat": 1.0, "rep": 1.6, "A": 2.5, "B": 1.5}
YEAR_WEIGHT = {2024: 1.0, 2025: 2.0}


def load():
    rows = []
    with (ROOT / "results.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "year": int(r["year"]),
                "gender": r["gender"],
                "stage": r["stage"],
                "race_key": (int(r["year"]), r["gender"], r["stage"], r["race_no"]),
                "team": r["team"],
                "time_sec": float(r["time_sec"]) if r["time_sec"] else None,
                "place": int(r["place"]) if r["place"] else None,
            })
    return rows


def fit_plackett_luce(rows, gender):
    """Fit P-L strengths. Returns (teams_sorted, strengths_dict, alpha_sec_per_unit)."""
    sub = [r for r in rows if r["gender"] == gender and r["place"] is not None]
    teams = sorted({r["team"] for r in sub})
    idx = {t: i for i, t in enumerate(teams)}

    by_race = defaultdict(list)
    for r in sub:
        by_race[r["race_key"]].append(r)

    rankings = []
    for key, boats in by_race.items():
        placed = sorted(boats, key=lambda b: b["place"])
        if len(placed) < 2:
            continue
        ranking = [idx[b["team"]] for b in placed]
        year, _, stage, _ = key
        w = STAGE_WEIGHT.get(stage, 1.0) * YEAR_WEIGHT.get(year, 1.0)
        reps = max(1, int(round(w)))
        for _ in range(reps):
            rankings.append(ranking)

    # alpha regularises toward zero (prevents blow-up when a team always wins/loses).
    params = choix.ilsr_rankings(len(teams), rankings, alpha=0.05)
    params = params - params.mean()

    # Fit α (seconds per P-L strength unit) via within-race time gaps. Only use
    # A-final and repechage races — heats contain weak teams whose massive
    # time gaps stretch the slope, making top-of-field predictions too fast.
    # Competitive races are where realistic time ⇄ strength scaling lives.
    gaps_t, gaps_s = [], []
    for key, boats in by_race.items():
        if key[2] not in ("A", "B", "rep"):
            continue
        timed = [b for b in boats if b["time_sec"] is not None]
        if len(timed) < 2:
            continue
        tmean = sum(b["time_sec"] for b in timed) / len(timed)
        smean = sum(params[idx[b["team"]]] for b in timed) / len(timed)
        for b in timed:
            gaps_t.append(b["time_sec"] - tmean)
            gaps_s.append(params[idx[b["team"]]] - smean)
    num = sum(t * s for t, s in zip(gaps_t, gaps_s))
    den = sum(s * s for s in gaps_s)
    alpha = -num / den if den > 1e-6 else 10.0
    # Sanity cap: realistic top-to-bottom A-final spread is ~25-30 s; with
    # typical strength spread of ~3-5 units between A-finalists this means
    # α ≈ 6–8 s. Cap to prevent 4:25 "best-ever" predictions.
    alpha = min(alpha, 7.5)

    strengths = {t: float(params[idx[t]]) for t in teams}
    return teams, strengths, alpha


def club_of(code):
    """Strip trailing [HD]?\\d+ to get the club part (e.g. NTNUIH1 → NTNUI)."""
    import re
    return re.sub(r"[HD]?\d+$", "", code) or code


def seed_snake(names, strengths, n_heats=3):
    """Snake-seed by strength with two constraints:
       (1) Heats stay balanced (e.g. 13 teams → 5+4+4, not 5+3+5).
       (2) Two boats from the same club never share a heat.
    If the snake target would violate either, shift to the next valid heat."""
    n = len(names)
    base = n // n_heats
    extra = n - base * n_heats
    caps = [base + (1 if i < extra else 0) for i in range(n_heats)]  # e.g. [5,4,4]
    ordered = sorted(names, key=lambda t: -strengths[t])
    heats = [[] for _ in range(n_heats)]
    pattern, forward = [], True
    while len(pattern) < n:
        seq = range(n_heats) if forward else range(n_heats - 1, -1, -1)
        for i in seq:
            pattern.append(i)
            if len(pattern) >= n:
                break
        forward = not forward
    for t, start in zip(ordered, pattern):
        club = club_of(t)
        target = start
        for _ in range(n_heats):
            full = len(heats[target]) >= caps[target]
            conflict = any(club_of(x) == club for x in heats[target])
            if not full and not conflict:
                break
            target = (target + 1) % n_heats
        else:
            # Both constraints can't be satisfied — drop the club rule, keep size cap.
            target = start
            while len(heats[target]) >= caps[target]:
                target = (target + 1) % n_heats
        heats[target].append(t)
    return heats


# Plackett–Luce temperature. < 1 = mer deterministisk (sterke lag dominerer mer).
# Kalibrert lavt fordi roing er mindre tilfeldig enn ren P-L antar — en topp-båt
# slår nesten alltid en mye svakere båt over 1600m.
PL_TEMP = 0.45


def sample_pl(team_strengths):
    """Gumbel-max trick med temperatur τ: sample en P-L-rangering."""
    draws = []
    for t, s in team_strengths.items():
        u = -math.log(-math.log(random.random())) * PL_TEMP
        draws.append((t, s + u))
    draws.sort(key=lambda x: -x[1])
    return [t for t, _ in draws]


def simulate_bracket(strengths, heats, nsim=NSIM):
    """Monte Carlo: each race samples a P-L ordering. Returns per-team probs."""
    names = list(strengths.keys())
    tallies = {n: {"win": 0, "medal": 0, "a_final": 0, "heat_win": 0, "via_rep": 0}
               for n in names}
    for _ in range(nsim):
        heat_winners, rest = [], []
        for heat in heats:
            order = sample_pl({t: strengths[t] for t in heat})
            heat_winners.append(order[0])
            rest.extend(order[1:])
        for w in heat_winners:
            tallies[w]["heat_win"] += 1
        rep_order = sample_pl({t: strengths[t] for t in rest})
        rep_adv = rep_order[:3]
        for t in rep_adv:
            tallies[t]["via_rep"] += 1
        a_final = heat_winners + rep_adv
        for t in a_final:
            tallies[t]["a_final"] += 1
        fin = sample_pl({t: strengths[t] for t in a_final})
        tallies[fin[0]]["win"] += 1
        for t in fin[:3]:
            tallies[t]["medal"] += 1
    for n in names:
        for k in tallies[n]:
            tallies[n][k] /= nsim
    return tallies


def expected_bracket(strengths, heats):
    """Deterministic most-likely path: order by strength at every stage."""
    heat_results, heat_winners, rest = [], [], []
    for heat in heats:
        ordered = sorted(heat, key=lambda t: -strengths[t])
        heat_results.append(ordered)
        heat_winners.append(ordered[0])
        rest.extend(ordered[1:])
    rep_order = sorted(rest, key=lambda t: -strengths[t])
    rep_adv = rep_order[:3]
    a_final_teams = heat_winners + rep_adv
    final_order = sorted(a_final_teams, key=lambda t: -strengths[t])
    return {
        "heats": heat_results,
        "rep_order": rep_order,
        "rep_adv": rep_adv,
        "final_order": final_order,
    }


def print_report(gender, strengths, probs, alpha):
    label = {"M": "MEN'S 8+", "W": "WOMEN'S 8+"}[gender]
    print(f"\n{'=' * 72}\n{label} — predicted SM 2026 standings\n{'=' * 72}")
    print(f"Scale α = {alpha:5.2f} s per strength unit · model: Plackett–Luce\n")
    rows = sorted(strengths.items(), key=lambda kv: -kv[1])
    print(f"{'Rank':>4}  {'Team':<12}  {'Strength':>8}  {'≈ gap':>8}  "
          f"{'Win%':>6}  {'Medal%':>7}  {'AFinal%':>8}  {'HeatWin%':>8}")
    print("-" * 80)
    best = rows[0][1]
    for i, (t, s) in enumerate(rows, 1):
        p = probs[t]
        print(f"{i:>4}  {t:<12}  {s:>+7.2f}   {(best-s)*alpha:>+6.1f}s  "
              f"{p['win']*100:>5.1f}%  {p['medal']*100:>6.1f}%  "
              f"{p['a_final']*100:>7.1f}%  {p['heat_win']*100:>7.1f}%")


TEAM_LABELS = {
    "OSIH1": "OSI H1", "OSIH2": "OSI H2",
    "BSIH1": "BSI H1", "BSIH2": "BSI H2",
    "NTNUIH1": "NTNUI H1", "NTNUIH2": "NTNUI H2", "NTNUIH3": "NTNUI H3",
    "MRKH1": "MRK H1", "MRKH2": "MRK H2", "MRKJRC": "MRK/JRC",
    "JUSTITIAH1": "Justitia H1", "RNNABC1": "RNNA H1", "RNNABC2": "RNNA H2",
    "CREW": "Crew", "BIA": "BIA", "JRC": "JRC H1",
    "OSID1": "OSI D1", "OSID2": "OSI D2",
    "BSID1": "BSI D1", "BSID2": "BSI D2",
    "NTNUID1": "NTNUI D1", "NTNUID2": "NTNUI D2", "NTNUID3": "NTNUI D3",
    "MRKD1": "MRK D1", "MRKD2": "MRK D2", "JRCD1": "JRC D1",
    "JUSTITIAD1": "Justitia D1", "JUSTITIAD2": "Justitia D2",
    "OKTAGOND1": "Oktagon D1", "OKTAGOND2": "Oktagon D2",
    "SKKRD1": "SKKR D1", "SKKRD2": "SKKR D2", "OMRD1": "OMR D1", "OMRD2": "OMR D2",
}


def pretty(t):
    return TEAM_LABELS.get(t, t)


STAGE_LABEL = {"heat": "Heat", "rep": "Oppsamling", "A": "A-finale", "B": "B-finale"}


def fmt_secs(t):
    if t is None:
        return ""
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:05.2f}".rstrip("0").rstrip(".")


def build_results_html(rows, gender="M"):
    """Render historical results for one gender as a clean nested table."""
    sub = [r for r in rows if r["gender"] == gender]
    by_year = defaultdict(lambda: defaultdict(list))
    for r in sub:
        by_year[r["year"]][r["stage"]].append(r)

    YEAR_NOTES = {
        2025: ("Værforhold ble betydelig dårligere utover dagen, så finalene ble rodd "
               "i tøffere forhold enn heatene. Tidene for finalen er derfor merkbart "
               "dårligere enn tidene fra heatene."),
    }
    RACE_NOTES = {
        (2025, "M", "heat", "1"): ("Tidene fra dette heatet er ikke gyldige (sen start / "
                                   "feil tid). Den interne rekkefølgen er likevel korrekt."),
    }

    gender_label = "herrer" if gender == "M" else "damer"
    out = []
    for year in sorted(by_year.keys(), reverse=True):
        year_rows = by_year[year]
        out.append(f'<details class="src" {"open" if year == max(by_year) else ""}>')
        out.append(f'<summary><b>SM {year}</b> · {gender_label} 8+</summary>')
        if year in YEAR_NOTES:
            out.append(f'<div class="src-note">⚠ {YEAR_NOTES[year]}</div>')
        out.append('<div class="src-body">')
        for stage in ("heat", "rep", "A", "B"):
            if stage not in year_rows:
                continue
            stage_rows = year_rows[stage]
            by_race = defaultdict(list)
            for r in stage_rows:
                by_race[r["race_key"]].append(r)
            for race_key in sorted(by_race.keys()):
                boats = sorted(by_race[race_key], key=lambda b: (b["place"] or 99))
                race_no = race_key[3]
                title = f"{STAGE_LABEL[stage]} {race_no}" if stage in ("heat", "rep") else STAGE_LABEL[stage]
                note = RACE_NOTES.get((race_key[0], race_key[1], race_key[2], race_key[3]))
                out.append(f'<div class="rrace"><div class="rrace-title">{title}</div>')
                if note:
                    out.append(f'<div class="rrace-note">⚠ {note}</div>')
                out.append('<table class="rtbl"><tbody>')
                for b in boats:
                    place = b["place"] if b["place"] else "—"
                    time = fmt_secs(b["time_sec"])
                    label = pretty(b["team"])
                    cls = " winner" if b["place"] == 1 else ""
                    invalid = " invalid-time" if note else ""
                    out.append(
                        f'<tr class="{cls.strip()}"><td class="rp">{place}</td>'
                        f'<td class="rn">{label}</td><td class="rt{invalid}">{time}</td></tr>'
                    )
                out.append('</tbody></table></div>')
        out.append('</div></details>')
    return "\n".join(out)


def write_html(bundles):
    payload = {
        g: {"teams": [], "seeds": bundles[g]["seeds"], "alpha": bundles[g]["alpha"],
            "adj": PRIOR_ADJUSTMENTS.get(g, {})}
        for g in GENDERS
    }
    for g in GENDERS:
        for t, s in bundles[g]["strengths"].items():
            # store UNADJUSTED strength so JS applies adjustments on top
            adj_steps = PRIOR_ADJUSTMENTS.get(g, {}).get(t, 0)
            raw = s - adj_steps * SEC_PER_STEP / bundles[g]["alpha"]
            payload[g]["teams"].append({
                "team": t,
                "label": pretty(t),
                "strength": raw,
            })
        payload[g]["teams"].sort(key=lambda x: -x["strength"])
    data_json = json.dumps(payload)
    rows = load()
    results_m = build_results_html(rows, "M")
    results_w = build_results_html(rows, "W")
    html = (HTML_TEMPLATE
            .replace("__DATA__", data_json)
            .replace("__SEC_PER_STEP__", str(SEC_PER_STEP))
            .replace("__NSIM__", str(8000))
            .replace("__RESULTS_M__", results_m)
            .replace("__RESULTS_W__", results_w))
    (ROOT / "index.html").write_text(html)
    print(f"\n→ wrote index.html ({(ROOT / 'index.html').stat().st_size // 1024} kB)")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SM 2026 8+ prognose · herrer & damer</title>
<style>
  :root{
    --bg:#fafafa; --ink:#1a1a1a; --muted:#666; --rule:#e5e5e5;
    --accent:#0b5394; --accent2:#2b7a78; --hl:#e8f1fb;
    --gold:#f2b544; --silver:#b9b9b9; --bronze:#c78a5a;
    --good:#2c7a3d; --bad:#a93a3a;
  }
  html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Inter,sans-serif;
    font-size:15px;line-height:1.5;}
  .wrap{max-width:1100px;margin:0 auto;padding:32px 24px 80px;}
  .kicker{font-size:13px;color:var(--muted);font-weight:600;}
  h1{font-weight:700;font-size:32px;line-height:1.15;margin:6px 0 8px;letter-spacing:-0.01em;}
  .sub{font-size:15px;color:var(--muted);max-width:680px;margin:0;}
  h2{font-size:22px;font-weight:700;margin:44px 0 8px;letter-spacing:-0.01em;}
  h2 small{font-size:12px;color:var(--muted);font-weight:400;float:right;margin-top:10px;}
  .howto{background:#fff;border-left:3px solid var(--accent);padding:10px 14px;
    margin:20px 0 0;font-size:13px;color:var(--ink);}
  .howto b{color:var(--accent);}
  .ex-wrap{margin-top:16px;background:#fff;border:1px solid var(--rule);border-radius:4px;padding:14px 14px 10px;}
  .ex-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;flex-wrap:wrap;gap:10px;}
  .ex-head h3{margin:0;font-family:Georgia,serif;font-size:17px;}
  .ex-head .ctrl{display:flex;gap:8px;align-items:center;font-size:12px;color:var(--muted);}
  .ex-head button{font:inherit;font-size:12px;padding:4px 10px;border:1px solid var(--rule);background:#fff;border-radius:3px;cursor:pointer;}
  .ex-flow{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:60px;position:relative;margin-top:10px;}
  .ex-arrows{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible;z-index:0;}
  .ex-col{display:flex;flex-direction:column;gap:10px;position:relative;z-index:1;}
  .ex-col-title{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:2px;}
  .ex-heat{border:1px solid var(--rule);border-radius:4px;padding:6px 8px;background:var(--bg);transition:background .1s,border-color .1s;}
  .ex-heat.drop-target{background:var(--hl);border-color:var(--accent);}
  .ex-heat-title{font-size:10px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}
  .ex-pill{display:grid;grid-template-columns:18px 1fr auto;gap:8px;padding:5px 8px;align-items:center;
    font-size:12px;border:1px solid var(--rule);background:#fff;border-radius:3px;margin:3px 0;position:relative;}
  .ex-pill[draggable="true"]{grid-template-columns:14px 18px 1fr auto;cursor:grab;}
  .ex-pill .drag-handle{color:var(--muted);opacity:.55;font-size:13px;letter-spacing:-3px;user-select:none;line-height:1;}
  .ex-pill[draggable="true"]:hover .drag-handle{opacity:1;color:var(--accent2);}
  .ex-pill.picking{box-shadow:0 0 0 2px var(--accent);background:var(--hl);}
  .ex-heat.move-target{background:var(--hl);border-color:var(--accent);cursor:pointer;}
  .ex-pill.dragging{opacity:.4;cursor:grabbing;}
  .ex-pill.winner{border-left:3px solid var(--accent);background:#fff3ea;}
  .ex-pill.advancing{border-left:3px solid var(--accent);background:#fff3ea;}
  .ex-pill.out{opacity:.55;background:var(--bg);}
  .ex-pill.out .ex-name{text-decoration:line-through;}
  .ex-pill.final{padding:8px 10px;font-size:13px;}
  .ex-pill.final.medal-gold{border-left:4px solid var(--gold);background:#fff8e8;}
  .ex-pill.final.medal-silver{border-left:4px solid var(--silver);background:#f6f6f4;}
  .ex-pill.final.medal-bronze{border-left:4px solid var(--bronze);background:#fbf1e8;}
  .ex-pill .ex-pos{color:var(--muted);font-size:11px;text-align:center;}
  .ex-pill .ex-medal{font-size:16px;text-align:center;}
  .ex-pill .ex-name{font-weight:700;}
  .ex-pill .ex-badge{color:var(--muted);font-size:10px;font-variant-numeric:tabular-nums;white-space:nowrap;}
  .ex-pill .ex-badge b{color:var(--accent);font-weight:700;}
  .ex-pill .ex-time{color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums;}
  .ex-legend{display:flex;gap:16px;font-size:11px;color:var(--muted);margin-top:10px;flex-wrap:wrap;align-items:center;}
  .ex-legend .swatch{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:4px;}
  /* Team stats card */
  .team-card{margin-top:14px;display:flex;gap:16px;align-items:stretch;background:#fff;border:1px solid var(--rule);border-radius:4px;padding:14px 16px;flex-wrap:wrap;}
  .team-card .pick{display:flex;flex-direction:column;gap:4px;min-width:190px;}
  .team-card .pick label{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700;}
  .team-card select{font:inherit;font-size:16px;padding:4px 8px;border:1px solid var(--rule);background:#fff;border-radius:3px;font-weight:700;}
  .team-card .stats{display:flex;gap:22px;flex-wrap:wrap;flex:1;}
  .team-card .stat{display:flex;flex-direction:column;}
  .team-card .stat .val{font-family:Georgia,serif;font-size:28px;font-weight:900;line-height:1;font-variant-numeric:tabular-nums;}
  .team-card .stat .val.accent{color:var(--accent);}
  .team-card .stat .lbl{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-top:4px;}
  .team-card .stat .sub{font-size:11px;color:var(--muted);margin-top:2px;}
  .ex-pill.picked{box-shadow:0 0 0 2px var(--accent2);}
  /* Strength-vs-history stepper (inside team card) */
  .strength-stepper{margin-top:10px;display:flex;flex-direction:column;gap:4px;}
  .strength-stepper .lbl{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:700;}
  .strength-stepper .row{display:inline-flex;gap:6px;align-items:center;}
  .strength-stepper .btn{width:22px;height:22px;border:1px solid var(--rule);background:#fff;border-radius:3px;cursor:pointer;font-weight:700;padding:0;line-height:1;font:inherit;}
  .strength-stepper .btn:hover{background:var(--hl);}
  .strength-stepper .dots{display:inline-flex;gap:3px;}
  .strength-stepper .dot{width:11px;height:11px;border:1px solid var(--rule);background:#fff;border-radius:2px;}
  .strength-stepper .dot.zero{background:var(--ink);opacity:.35;width:3px;border-radius:0;margin:0 2px;border-color:var(--ink);}
  .strength-stepper .dot.pos{background:var(--accent);border-color:var(--accent);}
  .strength-stepper .dot.neg{background:var(--accent2);border-color:var(--accent2);}
  .strength-stepper .hint{font-size:10px;color:var(--muted);}
  /* Boost-mark: intuitive green up / red down chip showing crew adjustment. */
  .boost-mark{display:inline-block;font-size:10px;font-weight:700;padding:1px 5px;
    border-radius:8px;margin-left:6px;vertical-align:middle;line-height:1.4;
    font-variant-numeric:tabular-nums;}
  .boost-mark.pos{background:#e1f0e3;color:#2c7a3d;}
  .boost-mark.neg{background:#fbe5e5;color:#a93a3a;}
  /* Historical results — compact, parsed from results.csv (no iframes). */
  .sources{margin-top:14px;display:flex;flex-direction:column;gap:10px;}
  .src{background:#fff;border:1px solid var(--rule);border-radius:4px;overflow:hidden;}
  .src summary{padding:10px 14px;cursor:pointer;font-size:14px;border-bottom:1px solid transparent;}
  .src[open] summary{border-bottom-color:var(--rule);background:var(--bg);}
  .src-body{padding:14px 16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;}
  .rrace{}
  .rrace-title{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:4px;border-bottom:1px solid var(--rule);padding-bottom:2px;}
  .rtbl{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;}
  .rtbl td{padding:3px 4px;border-bottom:1px dotted var(--rule);}
  .rtbl tr.winner td{background:var(--hl);font-weight:700;}
  .rtbl td.rp{color:var(--muted);width:18px;text-align:center;}
  .rtbl td.rt{color:var(--muted);text-align:right;font-size:11px;}
  .rtbl td.rt.invalid-time{text-decoration:line-through;color:#a93a3a;}
  .src-note{padding:8px 16px;background:#fff8e6;border-bottom:1px solid var(--rule);
    color:#7a5800;font-size:12px;font-style:italic;}
  .rrace-note{font-size:10px;color:#a93a3a;font-style:italic;margin-bottom:4px;line-height:1.3;}
  .methodology{margin-top:48px;padding:18px 20px;background:transparent;color:var(--muted);font-size:13px;border-top:1px solid var(--rule);}
  .methodology h3{margin:0 0 8px;font-size:13px;font-weight:700;color:var(--ink);}
  .methodology ol{padding-left:20px;margin:6px 0;}
  .methodology ol li{margin-bottom:4px;}
  .methodology .caveats{font-size:12px;margin-top:10px;}
  .footer{margin-top:24px;font-size:11px;color:var(--muted);text-align:center;}
  /* ----- Mobile (≤700px) ----- */
  @media (max-width: 700px){
    .wrap{padding:18px 12px 60px;}
    h1{font-size:24px;line-height:1.2;}
    .sub{font-size:14px;}
    h2{font-size:18px;margin-top:32px;}
    .ex-wrap{padding:10px;}
    .ex-head h3{font-size:15px;}
    .ex-head .ctrl{flex-wrap:wrap;}
    /* Stack the three flow columns vertically. */
    .ex-flow{grid-template-columns:1fr;gap:14px;}
    /* SVG arrows look weird when columns stack — hide on mobile. */
    .ex-arrows{display:none;}
    .ex-col-title{font-size:12px;}
    .ex-pill{font-size:13px;padding:7px 8px;}
    .ex-pill .ex-badge{font-size:11px;}
    .ex-pill[draggable="true"]{grid-template-columns:14px 18px 1fr auto;}
    /* Team card: stack vertically. */
    .team-card{flex-direction:column;padding:12px;}
    .team-card .pick{min-width:0;width:100%;}
    .team-card select{font-size:15px;}
    .team-card .stats{gap:14px;}
    .team-card .stat .val{font-size:22px;}
    /* Selected-pick highlight thinner on mobile. */
    .ex-pill.picked{box-shadow:0 0 0 1.5px var(--accent2);}
    /* Source results: single column. */
    .src-body{grid-template-columns:1fr;gap:10px;}
    /* Tap-to-move hint when no drag. */
    .ex-heat .heat-tap-hint{display:none;}
  }
</style>
</head><body>
<div class="wrap">
  <h1>Hvem vinner 8+ på SM 2026?</h1>
  <p class="sub">Offisiell trekning er låst inn. Velg ditt lag og se hvor sannsynlig det
    er at dere vinner. Basert på plasseringer fra SM24 og SM25.</p>

  <div id="mens-container"></div>
  <div id="womens-container"></div>

  <h2>Resultater fra tidligere år <small>kildedata bak modellen</small></h2>
  <div class="sources">
    <div style="font-size:12px;color:var(--muted);font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-top:4px">Herrer</div>
    __RESULTS_M__
    <div style="font-size:12px;color:var(--muted);font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-top:10px">Damer</div>
    __RESULTS_W__
  </div>

  <details class="methodology">
    <summary style="cursor:pointer;font-weight:700;color:var(--ink);font-size:13px;">For nerder: hvordan modellen fungerer ▾</summary>
    <div style="margin-top:10px">
      <ol>
        <li><b>Rangeringsbasert.</b> Vi bruker kun <i>plasseringer</i> fra hvert
          SM24/25-løp, aldri absolutte tider. Vind, vann og startlinjefeil påvirker alle
          båtene i et løp likt, så plasseringer er forholds-invariante per konstruksjon.</li>
        <li><b>Tilpasningen.</b> <code>choix.ilsr_rankings</code> estimerer en latent
          "styrke" for hvert lag slik at sjansen for at lag <i>i</i> slår lag <i>j</i> i et
          løp er <code>e<sup>sᵢ</sup> / (e<sup>sᵢ</sup> + e<sup>sⱼ</sup>)</code>. A-finaler
          vektes 2,5×, oppsamling 1,6×, heat 1×; 2025 vektes 2× vs 2024 fordi
          studentmannskap roterer fra år til år.</li>
        <li><b>Bracket-simulering.</b> For hver simulering trekkes en plassering fra
          Plackett-Luce ved hjelp av Gumbel-max-trikset. Heat-vinnere går rett til A-finalen;
          de 3 raskeste fra Oppsamling går videre. 8 000 simuleringer gir sannsynlighetene
          som vises på hvert lag.</li>
        <li><b>Trekning.</b> Heatene er den offisielle trekningen fra arrangør
          (herrer 5+4+4, damer 6+6+6). Du kan dra båter mellom heat for å teste
          hvordan alternative trekninger ville sett ut.</li>
      </ol>
      <div class="caveats">
        <b>Forbehold.</b> Studentmannskap roterer fra år til år, så årsvekter dekker bare
        delvis dette. To regattaer er et lite datagrunnlag. Heat vektes lavt fordi lederne
        ofte sparer kreftene; snake-seeding gjør at de uansett ikke havner i samme heat.
      </div>
    </div>
  </details>

  <div class="footer">Bygget fra SM24 Heatoppsett og SM25 Resultatliste (se ovenfor) · Plackett-Luce via choix.</div>
</div>

<script>
const DATA = __DATA__;
const SEC_PER_STEP = __SEC_PER_STEP__;
const NSIM = __NSIM__;

const GENDERS = ['M', 'W'];

// ADJ: integer steps per team. Positive = stronger in 2026.
const ADJ = {M:{}, W:{}};
Object.assign(ADJ.M, DATA.M.adj || {});
Object.assign(ADJ.W, DATA.W.adj || {});

// Official (locked) draw — used as both the default and the reset target.
const OFFICIAL = {
  M: DATA.M.seeds.map(h=>[...h]),
  W: DATA.W.seeds.map(h=>[...h]),
};
// Live heat assignments (draggable — user can still tinker).
const HEATS = {
  M: DATA.M.seeds.map(h=>[...h]),
  W: DATA.W.seeds.map(h=>[...h]),
};

// User's chosen team per division (for the stats card + pill highlight).
const SELECTED = {M: 'OSIH2', W: null};

// Apply ADJ (steps → strength units) → return {team: strength}.
function teamStrengths(g){
  const out = {};
  for(const t of DATA[g].teams){
    const adjSteps = ADJ[g][t.team] || 0;
    // step in seconds → strength units = seconds / alpha
    out[t.team] = t.strength + adjSteps * SEC_PER_STEP / DATA[g].alpha;
  }
  return out;
}
function labelOf(g, code){
  return (DATA[g].teams.find(t=>t.team===code) || {label:code}).label;
}
function clubOf(code){
  return code.replace(/[HD]?\d+$/, '') || code;
}
// Format a 0–1 probability for display: <1% becomes "—" (effectively zero).
function pct(p, digits=0){
  if(p < 0.005) return '—';     // < 0.5% → vis ingenting
  return (p*100).toFixed(digits) + '%';
}
function pct1(p){ return pct(p, 1); }
function boostMark(g, code){
  const adj = ADJ[g][code] || 0;
  if(adj === 0) return '';
  const cls = adj > 0 ? 'pos' : 'neg';
  const arrow = adj > 0 ? '↑' : '↓';
  const sign = adj > 0 ? '+' : '−';
  return `<span class="boost-mark ${cls}" title="${sign}${Math.abs(adj)} trinn vs form i 2024-25">${arrow} ${sign}${Math.abs(adj)}</span>`;
}
// Place ordered team codes into nHeats respecting a club constraint
// (two boats from same club never share a heat). Uses snake order on the
// supplied ordering (caller decides whether to sort by strength or shuffle).
function placeWithConstraint(ordered, nHeats){
  const n = ordered.length;
  // Balanced size caps: 13 teams → [5,4,4], 14 → [5,5,4], 12 → [4,4,4].
  const base = Math.floor(n / nHeats);
  const extra = n - base * nHeats;
  const caps = Array.from({length: nHeats}, (_,i) => base + (i < extra ? 1 : 0));
  const heats = Array.from({length: nHeats}, () => []);
  const pattern = []; let forward = true;
  while(pattern.length < n){
    const seq = forward ? [...Array(nHeats).keys()] : [...Array(nHeats).keys()].reverse();
    for(const i of seq){ pattern.push(i); if(pattern.length >= n) break; }
    forward = !forward;
  }
  for(let i=0; i<n; i++){
    const t = ordered[i];
    const club = clubOf(t);
    let start = pattern[i], target = start, placed = false;
    for(let k=0; k<nHeats; k++){
      const full = heats[target].length >= caps[target];
      const conflict = heats[target].some(x => clubOf(x) === club);
      if(!full && !conflict){ placed = true; break; }
      target = (target + 1) % nHeats;
    }
    if(!placed){
      // Couldn't satisfy both — drop club rule, keep size cap.
      target = start;
      while(heats[target].length >= caps[target]) target = (target + 1) % nHeats;
    }
    heats[target].push(t);
  }
  return heats;
}

// Gumbel-max P-L sampling med temperatur (mindre støy = mer deterministisk).
const PL_TEMP = 0.45;
function sampleOrder(teamStr){
  const entries = Object.entries(teamStr);
  const draws = entries.map(([t,s]) => [t, s + gumbel()*PL_TEMP]);
  draws.sort((a,b)=>b[1]-a[1]);
  return draws.map(d=>d[0]);
}
function gumbel(){
  let u = Math.random();
  while(u === 0) u = Math.random();
  return -Math.log(-Math.log(u));
}

// Monte Carlo bracket sim → per-team probabilities.
function simulateBracket(strengths, heats, nsim){
  const names = Object.keys(strengths);
  const t = {};
  for(const n of names) t[n] = {win:0, medal:0, afinal:0, heatWin:0, viaRep:0};
  for(let s=0; s<nsim; s++){
    const winners = [], rest = [];
    for(const heat of heats){
      const sub = {}; for(const x of heat) sub[x] = strengths[x];
      const o = sampleOrder(sub);
      winners.push(o[0]);
      for(let i=1;i<o.length;i++) rest.push(o[i]);
    }
    for(const w of winners) t[w].heatWin++;
    const subR = {}; for(const x of rest) subR[x] = strengths[x];
    const repO = sampleOrder(subR);
    const adv = repO.slice(0,3);
    for(const x of adv) t[x].viaRep++;
    const finalTeams = [...winners, ...adv];
    for(const x of finalTeams) t[x].afinal++;
    const subF = {}; for(const x of finalTeams) subF[x] = strengths[x];
    const finO = sampleOrder(subF);
    t[finO[0]].win++;
    for(let i=0;i<3;i++) t[finO[i]].medal++;
  }
  for(const n of names){
    for(const k in t[n]) t[n][k] /= nsim;
  }
  return t;
}

// Deterministic most-likely path.
function expectedBracket(strengths, heats){
  const heatRes = heats.map(h => [...h].sort((a,b)=>strengths[b]-strengths[a]));
  const winners = heatRes.map(r => r[0]);
  const rest = heatRes.flatMap(r => r.slice(1));
  const repOrder = [...rest].sort((a,b)=>strengths[b]-strengths[a]);
  const repAdv = repOrder.slice(0,3);
  const finalTeams = [...winners, ...repAdv];
  const finalOrder = [...finalTeams].sort((a,b)=>strengths[b]-strengths[a]);
  return {heatRes, winners: new Set(winners), repOrder, repAdv: new Set(repAdv), finalOrder};
}

function renderFlow(g, containerId, title){
  const el = document.getElementById(containerId);
  const strengths = teamStrengths(g);
  const exp = expectedBracket(strengths, HEATS[g]);
  const probs = simulateBracket(strengths, HEATS[g], NSIM);

  const teams = Object.keys(strengths);
  const meanS = teams.reduce((a,t)=>a+strengths[t],0)/teams.length;

  const medal = ['🥇','🥈','🥉'];
  const medalCls = ['medal-gold','medal-silver','medal-bronze'];

  const pick = SELECTED[g];
  const pickedCls = code => code === pick ? ' picked' : '';

  // Team stats card.
  const options = DATA[g].teams
    .map(t => t.team).sort((a,b) => labelOf(g,a).localeCompare(labelOf(g,b)))
    .map(t => `<option value="${t}" ${t===pick?'selected':''}>${labelOf(g,t)}</option>`).join('');
  const cardButtons = `
    <div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap">
      <button class="resim" data-g="${g}" style="font:inherit;font-size:11px;padding:4px 8px;border:1px solid var(--accent);background:var(--accent);color:#fff;border-radius:3px;cursor:pointer;font-weight:700">↻ Simuler på nytt</button>
    </div>`;

  function stepperFor(team){
    const adj = ADJ[g][team] || 0;
    const dots = [];
    for(let v=-3; v<=3; v++){
      let cls = 'dot';
      if(v===0) cls += ' zero';
      if(v>0 && adj>=v) cls += ' pos';
      if(v<0 && adj<=v) cls += ' neg';
      dots.push(`<span class="${cls}"></span>`);
    }
    const phrases = {
      '-3': 'mye svakere enn i 2024-25',
      '-2': 'merkbart svakere enn i 2024-25',
      '-1': 'litt svakere enn i 2024-25',
      '0':  'omtrent samme form som i 2024-25',
      '1':  'litt sterkere enn i 2024-25',
      '2':  'merkbart sterkere enn i 2024-25',
      '3':  'mye sterkere enn i 2024-25',
    };
    const hint = phrases[String(adj)] || phrases['0'];
    return `
      <div class="strength-stepper">
        <span class="lbl">Mannskapet vs tidligere år</span>
        <div class="row">
          <button class="btn stepdn" data-g="${g}" data-t="${team}" title="svakere">−</button>
          <span class="dots">${dots.join('')}</span>
          <button class="btn stepup" data-g="${g}" data-t="${team}" title="sterkere">+</button>
        </div>
        <span class="hint">${hint}</span>
      </div>`;
  }
  let cardHtml;
  if(pick && probs[pick]){
    const p = probs[pick];
    const expectedRank = exp.finalOrder.indexOf(pick);
    const placeWord = ['🥇','🥈','🥉','4. plass','5. plass','6. plass'][expectedRank] || ((expectedRank+1)+'. plass');
    const expectedStr = expectedRank >= 0
      ? `mest sannsynlig ${expectedRank===0?'vinner':placeWord} i A-finalen`
      : `går ikke til A-finale i mest sannsynlige utfall · plass ${exp.repOrder.indexOf(pick)+1} i Oppsamling`;
    cardHtml = `
      <div class="pick">
        <label>Ditt lag</label>
        <select class="pickteam" data-g="${g}">
          <option value="">(ingen)</option>${options}
        </select>
        <div style="font-size:11px;color:var(--muted);margin-top:4px">${expectedStr}</div>
        ${stepperFor(pick)}
        ${cardButtons}
      </div>
      <div class="stats">
        <div class="stat"><span class="val accent">${pct1(p.win)}</span><span class="lbl">Seier</span><span class="sub">SM-gull</span></div>
        <div class="stat"><span class="val">${pct1(p.medal)}</span><span class="lbl">Medalje</span><span class="sub">topp 3</span></div>
        <div class="stat"><span class="val">${pct1(p.afinal)}</span><span class="lbl">A-finale</span><span class="sub">topp 6</span></div>
        <div class="stat"><span class="val">${pct1(p.heatWin)}</span><span class="lbl">Heat-seier</span><span class="sub">rett til finale</span></div>
        <div class="stat"><span class="val">${pct1(p.viaRep)}</span><span class="lbl">Via Oppsamling</span><span class="sub">går videre fra Oppsamling</span></div>
      </div>`;
  } else {
    cardHtml = `
      <div class="pick">
        <label>Ditt lag</label>
        <select class="pickteam" data-g="${g}">
          <option value="" selected>(ingen)</option>${options}
        </select>
        <div style="font-size:11px;color:var(--muted);margin-top:4px">Velg et lag for å se sannsynligheter</div>
        ${cardButtons}
      </div>
      <div class="stats" style="color:var(--muted);font-size:12px;align-self:center">
        Seier · Medalje · A-finale · Heat-seier · Oppsamling vises her.
      </div>`;
  }

  const heatsHtml = exp.heatRes.map((heat, hi) => {
    const pills = heat.map((code, i) => {
      const p = probs[code];
      const winner = i === 0;
      const toId = winner ? `ex-${g}-f-${code}` : `ex-${g}-r-${code}`;
      const arr = winner ? 'advance' : 'rep';
      const badge = winner
        ? `<span class="ex-badge">heat-seier <b>${pct(p.heatWin)}</b></span>`
        : `<span class="ex-badge">A-finale <b>${pct(p.afinal)}</b></span>`;
      return `<div class="ex-pill${winner?' winner':''}${pickedCls(code)}" id="ex-${g}-h-${code}"
          data-to="${toId}" data-arrow="${arr}">
        <span class="ex-pos">${i+1}</span>
        <span class="ex-name">${labelOf(g, code)}${boostMark(g, code)}</span>
        ${badge}
      </div>`;
    }).join('');
    return `<div class="ex-heat" data-g="${g}" data-heat="${hi}">
      <div class="ex-heat-title">Heat ${String.fromCharCode(65+hi)} · raskeste → A-finale</div>${pills}
    </div>`;
  }).join('');

  const repHtml = exp.repOrder.map((code, i) => {
    const adv = exp.repAdv.has(code);
    const p = probs[code];
    const toAttr = adv ? `data-to="ex-${g}-f-${code}" data-arrow="advance"` : '';
    const badge = `<span class="ex-badge">A-finale ${adv?'<b>':''}${pct(p.afinal)}${adv?'</b>':''}</span>`;
    return `<div class="ex-pill ${adv?'advancing':'out'}${pickedCls(code)}" id="ex-${g}-r-${code}" ${toAttr}>
      <span class="ex-pos">${i+1}</span>
      <span class="ex-name">${labelOf(g, code)}${boostMark(g, code)}</span>
      ${badge}
    </div>`;
  }).join('');

  const finalHtml = exp.finalOrder.map((code, i) => {
    const p = probs[code];
    const cls = i < 3 ? medalCls[i] : '';
    const badge = i === 0
      ? `<span class="ex-badge">seier <b>${pct(p.win)}</b> · medalje ${pct(p.medal)}</span>`
      : `<span class="ex-badge">medalje <b>${pct(p.medal)}</b> · seier ${pct(p.win)}</span>`;
    return `<div class="ex-pill final ${cls}${pickedCls(code)}" id="ex-${g}-f-${code}">
      <span class="ex-medal">${medal[i] || (i+1)}</span>
      <span class="ex-name">${labelOf(g, code)}${boostMark(g, code)}</span>
      ${badge}
    </div>`;
  }).join('');

  el.innerHTML = `
    <h2>${title} <small>${HEATS[g].flat().length} lag · ${NSIM.toLocaleString()} simuleringer</small></h2>
    <div class="team-card">${cardHtml}</div>
    <div class="ex-wrap">
      <div class="ex-head">
        <h3>Mest sannsynlige SM 2026-bracket</h3>
      </div>
      <div class="ex-flow" id="flow-${g}">
        <svg class="ex-arrows" xmlns="http://www.w3.org/2000/svg"></svg>
        <div class="ex-col">
          <div class="ex-col-title">Offisiell trekning</div>
          ${heatsHtml}
        </div>
        <div class="ex-col">
          <div class="ex-col-title">Oppsamling · 3 raskeste → A-finale</div>
          ${repHtml}
        </div>
        <div class="ex-col">
          <div class="ex-col-title">A-finale</div>
          ${finalHtml}
        </div>
      </div>
      <div class="ex-legend">
        <span><svg width="22" height="6" style="vertical-align:middle"><path d="M 0 3 L 22 3" stroke="#0b5394" stroke-width="2"/></svg> går videre</span>
        <span><svg width="22" height="6" style="vertical-align:middle"><path d="M 0 3 L 22 3" stroke="#a8a296" stroke-width="1.5" opacity="0.5"/></svg> til Oppsamling</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#fff;border:1px solid var(--rule);border-left:3px solid var(--accent);border-radius:2px;vertical-align:middle"></span> går til A-finalen</span>
        <span style="opacity:.55;text-decoration:line-through">slått ut</span> i Oppsamling
        <span class="boost-mark pos">↑ +1</span> = sterkere enn i fjor
        <span class="boost-mark neg">↓ −1</span> = svakere enn i fjor
        <span style="color:var(--muted)">%-tall = sannsynlighet fra ${NSIM.toLocaleString()} simuleringer</span>
      </div>
    </div>`;
  requestAnimationFrame(() => drawArrows(document.getElementById('flow-'+g)));
}

function drawArrows(container){
  if(!container) return;
  const svg = container.querySelector('svg.ex-arrows');
  if(!svg) return;
  const rect = container.getBoundingClientRect();
  svg.setAttribute('width', rect.width);
  svg.setAttribute('height', rect.height);
  svg.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
  const id = container.id;
  const defs = `<defs>
    <marker id="adv-${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#ed713a"/></marker>
    <marker id="rep-${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#a8a296"/></marker>
  </defs>`;
  let paths = '';
  container.querySelectorAll('[data-to]').forEach(fromEl => {
    const toEl = document.getElementById(fromEl.dataset.to);
    if(!toEl) return;
    const f = fromEl.getBoundingClientRect();
    const t = toEl.getBoundingClientRect();
    const x1 = f.right - rect.left;
    const y1 = (f.top + f.bottom) / 2 - rect.top;
    const x2 = t.left - rect.left - 6;
    const y2 = (t.top + t.bottom) / 2 - rect.top;
    const dx = Math.max(30, (x2 - x1) * 0.45);
    const adv = fromEl.dataset.arrow === 'advance';
    const color = adv ? '#ed713a' : '#a8a296';
    const opacity = adv ? 0.9 : 0.35;
    const marker = adv ? `adv-${id}` : `rep-${id}`;
    paths += `<path d="M ${x1} ${y1} C ${x1+dx} ${y1}, ${x2-dx} ${y2}, ${x2} ${y2}" stroke="${color}" stroke-width="${adv?2:1.5}" fill="none" opacity="${opacity}" marker-end="url(#${marker})"/>`;
  });
  svg.innerHTML = defs + paths;
}

function render(){
  renderFlow('M', 'mens-container', "Herrer 8+");
  renderFlow('W', 'womens-container', "Damer 8+");
}

// Events
document.addEventListener('click', e => {
  if(e.target.classList.contains('resim')){
    render();
    return;
  }
  if(e.target.classList.contains('stepup') || e.target.classList.contains('stepdn')){
    const g = e.target.dataset.g, t = e.target.dataset.t;
    const cur = ADJ[g][t] || 0;
    const delta = e.target.classList.contains('stepup') ? 1 : -1;
    const next = Math.max(-3, Math.min(3, cur + delta));
    if(next === 0) delete ADJ[g][t]; else ADJ[g][t] = next;
    render();
    return;
  }
});
document.addEventListener('change', e => {
  if(e.target.classList.contains('pickteam')){
    SELECTED[e.target.dataset.g] = e.target.value || null;
    render();
  }
});

// Redraw arrows on resize.
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    drawArrows(document.getElementById('flow-M'));
    drawArrows(document.getElementById('flow-W'));
  }, 80);
});

// ---------------------------------------------------------------------------
// URL state: encode HEATS, ADJ, and SELECTED as query params so any view is
// shareable. Format (per division, prefix m./w.):
//   m.heats=TEAM1,TEAM2|TEAM3,TEAM4|TEAM5,TEAM6   (| separates heats)
//   m.adj=OSIH1:2,OSIH2:1                          (team:step pairs)
//   m.pick=OSIH2
// ---------------------------------------------------------------------------
function encodeState(){
  const parts = [];
  for(const g of GENDERS){
    const prefix = g.toLowerCase() + '.';
    const adjs = Object.entries(ADJ[g]).filter(([,v]) => v).map(([t,v])=>`${t}:${v}`).join(',');
    if(adjs) parts.push(`${prefix}adj=${adjs}`);
    if(SELECTED[g]) parts.push(`${prefix}pick=${SELECTED[g]}`);
  }
  return parts.length ? '?' + parts.join('&') : window.location.pathname;
}
function decodeState(){
  const q = new URLSearchParams(window.location.search);
  for(const g of GENDERS){
    const prefix = g.toLowerCase() + '.';
    const adj = q.get(prefix+'adj');
    if(adj !== null){
      ADJ[g] = {};
      for(const pair of adj.split(',')){
        const [t, v] = pair.split(':');
        const n = parseInt(v, 10);
        if(t && Number.isFinite(n)) ADJ[g][t] = Math.max(-3, Math.min(3, n));
      }
    }
    const pick = q.get(prefix+'pick');
    if(pick !== null) SELECTED[g] = pick || null;
  }
}
let urlTimer;
function updateUrl(){
  clearTimeout(urlTimer);
  urlTimer = setTimeout(() => {
    const url = window.location.pathname + encodeState() + window.location.hash;
    window.history.replaceState(null, '', url);
  }, 80);
}
// Patch render() to also sync URL.
const _render = render;
render = function(){ _render(); updateUrl(); };

// Copy-link button
document.addEventListener('click', e => {
  if(e.target.classList.contains('copylink')){
    const btn = e.target;
    navigator.clipboard.writeText(window.location.href).then(() => {
      const prev = btn.textContent;
      btn.textContent = '✓ copied';
      setTimeout(() => btn.textContent = prev, 1400);
    });
  }
});

decodeState();
render();
</script>
</body></html>
"""


def main():
    rows = load()
    bundles = {}
    for gender in GENDERS:
        _, params, alpha = fit_plackett_luce(rows, gender)

        roster = TEAMS_2026[gender] if TEAMS_2026[gender] else sorted(params.keys())
        strengths = {t: params.get(t, 0.0) for t in roster}

        # Apply 2026 prior adjustments (step units → strength units via alpha).
        adj = PRIOR_ADJUSTMENTS.get(gender, {}) or {}
        for t, steps in adj.items():
            if t in strengths:
                strengths[t] += steps * SEC_PER_STEP / alpha

        # Offisiell trekning har prioritet; ellers seed på styrke.
        heats = OFFICIAL_HEATS.get(gender) or seed_snake(list(strengths.keys()), strengths, 3)
        probs = simulate_bracket(strengths, heats, nsim=NSIM)
        print_report(gender, strengths, probs, alpha)

        bundles[gender] = {
            "strengths": strengths,
            "probs": probs,
            "alpha": alpha,
            "seeds": heats,
        }
    write_html(bundles)


if __name__ == "__main__":
    main()
