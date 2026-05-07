#!/usr/bin/env python3

import re
from pathlib import Path

# ===============================
# CONFIG
# ===============================
DATA_DIR = Path(".")

UL_PREFIXES = {
    "DUSA":   "ul_dusa",
    "Trad":   "ul_trad",
    "Static": "ul_sem5qi-static",
    "Rand":   "ul_dynamic-random",
}

# ===============================
# PARSERS
# ===============================
def extract_success_rate(filepath):
    text = filepath.read_text(errors="replace")
    m = re.search(r"Overall avg\s*:\s*([0-9.eE+\-]+)", text)
    if m is None:
        raise ValueError(f"Missing success rate in {filepath}")
    return float(m.group(1))


def extract_harq_rate(filepath):
    text = filepath.read_text(errors="replace")
    m = re.search(r"Overall Avg HARQ Infeasibility Rate\s*:\s*([0-9.eE+\-]+)", text)
    if m is None:
        raise ValueError(f"Missing HARQ rate in {filepath}")
    return float(m.group(1))


# ===============================
# METRICS
# ===============================
def percent_increase(dusa, other):
    return (dusa - other) / other * 100.0


def percent_decrease(dusa, other):
    return (other - dusa) / other * 100.0


# ===============================
# CORE LOGIC
# ===============================
def compute_case(label_suffix, file_suffix):
    success = {}
    harq = {}

    for label, prefix in UL_PREFIXES.items():
        f = DATA_DIR / f"{prefix}_{file_suffix}.txt"
        success[label] = extract_success_rate(f)
        harq[label] = extract_harq_rate(f)

    print(f"\n==============================")
    print(f" UL RESULTS: {label_suffix}")
    print(f"==============================")

    print("\nTask Success Rate:")
    for label in ["Trad", "Static", "Rand"]:
        pct = percent_increase(success["DUSA"], success[label])
        print(f"  DUSA vs {label}: {pct:.1f}% GREATER")

    print("\nHARQ Infeasibility Rate:")
    for label in ["Trad", "Static", "Rand"]:
        pct = percent_decrease(harq["DUSA"], harq[label])
        print(f"  DUSA vs {label}: {pct:.1f}% LOWER")


# ===============================
# MAIN
# ===============================
def main():
    # 40 UE case → 10ue files
    compute_case("40 UE", "10ue")

    # 1 CB case → 1cb files
    compute_case("1 CB", "1cb")


if __name__ == "__main__":
    main()
