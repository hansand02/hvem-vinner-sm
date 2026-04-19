"""Parse the two Google-Sheets HTML exports into a single results.csv.

Output columns:
  year, gender, stage, race_no, team, lane, time_sec, place
where stage ∈ {heat, rep, A, B}.
"""
import re
import csv
from pathlib import Path

ROOT = Path(__file__).parent
SOURCES = [
    (ROOT / "Resultatliste-SM-2025.xlsx - Google Disk_files" / "sheet.html", 2025),
    (ROOT / "Heatoppsett SM24 - Google Disk_files" / "sheet.html", 2024),
]


def parse_time(s: str) -> float | None:
    """Parse times like '5:21,5', '05:42,9', '04.44.67', '5:17.7' → seconds."""
    if not s:
        return None
    s = s.strip().replace(" ", "")
    # Replace comma decimal with dot
    s = s.replace(",", ".")
    # Formats: M:SS.s / MM:SS.ss / MM.SS.ss
    m = re.match(r"^(\d{1,2})[:.](\d{2})[:.]?(\d{1,3})?$", s)
    if m:
        mins = int(m.group(1))
        secs = int(m.group(2))
        frac = m.group(3) or "0"
        # If frac length >2, treat as hundredths-with-extra-digit (strip)
        if len(frac) > 2:
            frac = frac[:2]
        return mins * 60 + secs + int(frac) / (10 ** len(frac))
    m = re.match(r"^(\d{1,2})[:.](\d{2}[.,]\d+)$", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


def extract_rows(html_path: Path):
    html = html_path.read_text()
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    out = []
    for r in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.DOTALL)
        txt = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip() for c in cells]
        out.append(txt)
    return out


def classify_stage(label: str) -> tuple[str, str] | None:
    """Return (gender, stage) from a race label like 'Herrer forsøk 1'."""
    label = label.lower()
    gender = None
    if "herre" in label:
        gender = "M"
    elif "dame" in label:
        gender = "W"
    if not gender:
        return None
    if "forsøk" in label or "forsok" in label:
        return gender, "heat"
    if "oppsamling" in label:
        return gender, "rep"
    if "a-finale" in label or "a finale" in label:
        return gender, "A"
    if "b-finale" in label or "b finale" in label:
        return gender, "B"
    return None


def canon_team(name: str) -> str:
    """Canonicalize team names across years."""
    n = name.upper().replace(" ", "").replace("-", "")
    # Known aliases
    aliases = {
        "JUSTITA": "JUSTITIA",
        "OKTAGON": "OKTAGON",
        "RNNABCTQ": "RNNATQ",
        "RNNABCH1": "RNNABC1",
        "RNNABCH2": "RNNABC2",
        "RNNABC": "RNNABC",
        "CREWH1": "CREW",
        "MRKJRCH1": "MRKJRC",
        "MRKJRCD1": "MRKJRCD1",
        "BISIH1": "BISI",
    }
    # Normalize Justita spelling
    n = n.replace("JUSTITA", "JUSTITIA")
    # RNNA BC TQ (2025) is the same program as RNNA H2 / RNNA BC H2 (2024).
    if n in {"RNNATQ", "RNNABCTQ"}:
        n = "RNNABC2"
    if n in {"RNNABCH1", "RNNABC"}:
        n = "RNNABC1"
    # BISI (2024 BISI H1) == BIA (2025 BIA) — same program, different label.
    if n in {"BISI", "BIA", "BISIH1"}:
        n = "BIA"
    # Women's: "OKTAGON" (2025 rep only) == "OKTAGON D1" (their single D1 boat).
    if n in {"OKTAGON", "OKTAGOND1"}:
        n = "OKTAGOND1"
    # Women's: "SKKR" (2024) == "SKKR D1" (2025) — single boat program, just relabelled.
    if n in {"SKKR", "SKKRD1"}:
        n = "SKKRD1"
    # 2024 MRK-JRC crews were really just JRC (MRK label was administrative).
    if n == "MRKJRCH1":
        n = "JRC"
    if n in {"MRKJRCD1", "MRKJRCD1"}:
        n = "JRCD1"
    return aliases.get(n, n)


def parse_file(path: Path, year: int):
    rows = extract_rows(path)
    current = None  # (gender, stage, race_no)
    results = []
    for cells in rows:
        # Row format roughly: [row_num, '', time, race_no, label, lane, start#, team, place, time, forsoktime, ...]
        if len(cells) < 11:
            continue
        _, _, _, race_no_cell, label_cell, lane, startnum, team, place, result_cell, time_cell, *_ = cells[:12]
        # In SM25 raw `result_cell` sometimes holds a note like 'b' — prefer time_cell,
        # but fall back to result_cell if time_cell is empty or unparseable.
        if label_cell:
            c = classify_stage(label_cell)
            if c:
                rno = race_no_cell or str(len(results))
                current = (c[0], c[1], rno)
            continue
        if current is None:
            continue
        if not team:
            continue
        t = parse_time(time_cell) or parse_time(result_cell)
        # NOTE: SM25 Men Heat 1 was timed with a common offset (late start);
        # relative gaps within the race are valid, so we keep these times —
        # our model uses within-race gaps and naturally cancels any constant shift.
        try:
            pl = int(place) if place else None
        except ValueError:
            pl = None
        if t is None and pl is None:
            continue
        results.append(
            {
                "year": year,
                "gender": current[0],
                "stage": current[1],
                "race_no": current[2],
                "team_raw": team,
                "team": canon_team(team),
                "lane": lane,
                "time_sec": t,
                "place": pl,
            }
        )
    return results


def main():
    all_results = []
    for path, year in SOURCES:
        all_results.extend(parse_file(path, year))
    out = ROOT / "results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["year", "gender", "stage", "race_no", "team_raw", "team", "lane", "time_sec", "place"],
        )
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"wrote {len(all_results)} rows → {out}")


if __name__ == "__main__":
    main()
