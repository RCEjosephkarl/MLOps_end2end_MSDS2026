"""
Synthesize ~2000 additional rows for Computer_Durability_Plus.csv.

Distribution matches the original 999-row dataset but introduces mild drift
so Evidently can detect it:
  - Hours Used Per Day shifted +2h (heavier usage in new cohort)
  - Cost shifted -$3000 (cheaper machines in new cohort)
  - Class rate ~7% positive (up from 5%), still realistic

The synthesized rows are appended to the original data and saved as
Computer_Durability_Plus.csv at data/raw/.
"""
from __future__ import annotations

import csv
import random
import math
from pathlib import Path

SEED = 42
N_SYNTH = 2000
ROOT = Path(__file__).parent.parent

# Original distribution parameters (derived from EDA)
ORIG = {
    "hours_mean": 12.648, "hours_std": 6.558, "hours_min": 1.0, "hours_max": 24.0,
    "cost_mean": 33789.0, "cost_std": 9647.0, "cost_min": 5000.0, "cost_max": 50000.0,
    "age_mean": 36.637, "age_std": 16.560, "age_min": 8.0, "age_max": 65.0,
    "comp_age_mean": 29.654, "comp_age_std": 16.862, "comp_age_min": 1.0, "comp_age_max": 60.0,
}
# Mild drift: heavier usage, cheaper machines
DRIFT = {
    "hours_mean": 14.8,    # +2.15h drift
    "hours_std": 6.2,
    "cost_mean": 30500.0,  # -$3289 drift
    "cost_std": 9200.0,
    "age_mean": 36.637,    # unchanged
    "age_std": 16.560,
    "comp_age_mean": 29.654,
    "comp_age_std": 16.862,
}

HEADER = [
    "Hours Used Per Day", "Cost", "User Age",
    "Needs Replacement", "Primary Usage", "Brand", "Computer Age (Months)"
]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def box_muller(mean: float, std: float, rng: random.Random) -> float:
    u1, u2 = rng.random(), rng.random()
    z = math.sqrt(-2 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
    return mean + std * z


def replacement_probability(hours: float, cost: float, user_age: float) -> float:
    """
    Signal calibrated to match observed replacement rates:
      <12h/day → ~0%,  12-16h → ~8%,  >16h → ~11%
    Cost modifier: cheaper machine → higher risk.
    Age modifier: older users slightly more at risk.
    Overall positive rate: ~5-7% (original ~5%, drift cohort ~7%).
    """
    if hours < 12.0:
        base = 0.002
    elif hours < 16.0:
        base = 0.07
    else:
        base = 0.11
    # Cost: scale from 1.8 (cheapest, $5k) to 0.2 (most expensive, $50k)
    cost_factor = 2.0 - 1.8 * (cost - 5000.0) / 45000.0
    # User age: mild uplift for older users
    age_factor = 1.0 + 0.01 * max(0.0, user_age - 40.0)
    return min(base * cost_factor * age_factor, 0.85)


def synthesize(n: int, rng: random.Random) -> list[list]:
    rows = []
    for _ in range(n):
        hours = clamp(box_muller(DRIFT["hours_mean"], DRIFT["hours_std"], rng), 1.0, 24.0)
        cost = clamp(box_muller(DRIFT["cost_mean"], DRIFT["cost_std"], rng), 5000.0, 50000.0)
        user_age = clamp(box_muller(DRIFT["age_mean"], DRIFT["age_std"], rng), 8.0, 65.0)
        comp_age = clamp(box_muller(DRIFT["comp_age_mean"], DRIFT["comp_age_std"], rng), 1.0, 60.0)
        primary_usage = rng.randint(1, 4)
        brand = rng.randint(1, 5)
        p_replace = replacement_probability(hours, cost, user_age)
        needs_replacement = 1 if rng.random() < p_replace else 0
        rows.append([
            round(hours, 8), round(cost, 5), round(user_age, 8),
            needs_replacement, primary_usage, brand, round(comp_age, 7)
        ])
    return rows


def main() -> None:
    rng = random.Random(SEED)

    data_raw = ROOT / "data" / "raw"
    src = data_raw / "Computer_Durability.csv"
    dst = data_raw / "Computer_Durability_Plus.csv"

    if dst.exists():
        print(f"Already exists, skipping synthesis → {dst}")
        return

    orig_rows = []
    with src.open(encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for r in reader:
            orig_rows.append([
                float(r[0]), float(r[1]), float(r[2]),
                int(r[3]), int(r[4]), int(r[5]), float(r[6])
            ])

    synth_rows = synthesize(N_SYNTH, rng)
    all_rows = orig_rows + synth_rows

    pos = sum(1 for r in all_rows if r[3] == 1)
    print(f"Original rows : {len(orig_rows)}")
    print(f"Synthesized   : {len(synth_rows)}")
    print(f"Total rows    : {len(all_rows)}")
    print(f"Positive class: {pos} ({100*pos/len(all_rows):.1f}%)")

    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(all_rows)

    print(f"Saved → {dst}")

    # Also copy raw originals into data/raw/
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(src, raw_dir / src.name)
    shutil.copy(dst, raw_dir / dst.name)
    print(f"Copied raw files → {raw_dir}")


if __name__ == "__main__":
    main()
