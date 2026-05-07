#!/usr/bin/env python3
# ======================================================
# Parse path task counts from all experiment log files
# ======================================================

import os
import re
import glob

# ======================================================
# Config: map filename label -> human-readable name
# ======================================================

EXPERIMENT_LABELS = {
    # Experiment 1: Compute Budget Sweep (n_ue=12)
    "1cb": "CB1 (du=2.0, ru=0.5, n_ue=12)",
    "2cb": "CB2 (du=3.5, ru=1.0, n_ue=12)",
    "3cb": "CB3 (du=5.0, ru=2.0, n_ue=12)",
    "4cb": "CB4 (du=7.0, ru=3.5, n_ue=12)",
    # Experiment 2: UE Count Sweep (CB3 fixed)
    "1ue": "UE1 (du=5.0, ru=2.0, n_ue=4)",
    "3ue": "UE3 (du=5.0, ru=2.0, n_ue=12)",
    "5ue": "UE5 (du=5.0, ru=2.0, n_ue=20)",
   "10ue": "UE10 (du=5.0, ru=2.0, n_ue=40)",
}

TECHNIQUES = {
    "dusa":           "DUSA",
    "trad":           "TRAD",
    "sem5qi-static":  "STATIC",
    "dynamic-random": "RAND",
}

# ======================================================
# File pattern: ul_{technique}_{label}.txt
# ======================================================

def parse_path_counts(filepath):
    """Extract per-path task counts from a log file."""
    counts = {}
    in_section = False
    with open(filepath, "r") as f:
        for line in f:
            if "Total Tasks Served Per Path" in line:
                in_section = True
                continue
            if in_section:
                m = re.match(r"\s*Path\s+(\d+)\s*:\s*(\d+)", line)
                if m:
                    counts[int(m.group(1))] = int(m.group(2))
                else:
                    # Stop if we hit a non-matching line after the section starts
                    if counts:
                        break
    return counts if len(counts) == 4 else None


def find_files(directory="."):
    """Find all ul_*_*.txt files in the given directory."""
    return glob.glob(os.path.join(directory, "ul_*.txt"))


def parse_filename(filename):
    """
    Extract technique key and config label from filename.
    Pattern: ul_{technique}_{label}.txt
    """
    basename = os.path.basename(filename)
    m = re.match(r"ul_(.+?)_([^_]+)\.txt$", basename)
    if not m:
        return None, None
    technique_raw = m.group(1)
    label         = m.group(2)
    return technique_raw, label


# ======================================================
# Main
# ======================================================

def main():
    files = find_files(".")
    if not files:
        print("No ul_*.txt files found in current directory.")
        return

    # Organize: data[label][technique] = {0: N, 1: N, 2: N, 3: N}
    data = {}

    missing = []
    unparseable = []

    for f in sorted(files):
        tech_raw, label = parse_filename(f)
        if tech_raw is None or label is None:
            continue

        # Match technique key
        technique = None
        for key in TECHNIQUES:
            if tech_raw == key:
                technique = key
                break
        if technique is None:
            continue

        counts = parse_path_counts(f)
        if counts is None:
            unparseable.append(f)
            continue

        if label not in data:
            data[label] = {}
        data[label][technique] = counts

    # ======================================================
    # Print results
    # ======================================================

    print("=" * 70)
    print("  PATH TASK COUNT SUMMARY")
    print("=" * 70)

    # Sort configs: CB sweep first, then UE sweep
    cb_labels = [l for l in ["1cb","2cb","3cb","4cb"] if l in data]
    ue_labels = [l for l in ["1ue","3ue","5ue","10ue"] if l in data]
    other_labels = [l for l in sorted(data.keys()) if l not in cb_labels + ue_labels]
    ordered_labels = cb_labels + ue_labels + other_labels

    for label in ordered_labels:
        config_name = EXPERIMENT_LABELS.get(label, label)
        print(f"\n  CONFIG: {config_name}")
        print(f"  {'-'*60}")
        print(f"  {'Technique':<12}  {'Path 0':>10}  {'Path 1':>10}  {'Path 2':>10}  {'Path 3':>10}  {'Total':>10}")
        print(f"  {'-'*60}")

        for tech_key, tech_name in TECHNIQUES.items():
            if tech_key not in data[label]:
                print(f"  {tech_name:<12}  {'MISSING':>10}")
                missing.append(f"{tech_name} / {label}")
                continue
            c = data[label][tech_key]
            total = sum(c.values())
            print(f"  {tech_name:<12}  {c[0]:>10}  {c[1]:>10}  {c[2]:>10}  {c[3]:>10}  {total:>10}")

    print(f"\n{'='*70}")

    if missing:
        print(f"\n  WARNING — missing results for:")
        for m in missing:
            print(f"    {m}")

    if unparseable:
        print(f"\n  WARNING — could not parse path counts from:")
        for u in unparseable:
            print(f"    {u}")

    print()


if __name__ == "__main__":
    main()
