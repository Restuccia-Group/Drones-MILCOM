import os
import numpy as np
import random
import glob
from scipy.io import loadmat

np.random.seed(42)
random.seed(42)

# ======================================================
# Dataset loaders (identical to your sim scripts)
# ======================================================

def load_cfr_dataset(folder):
    files = sorted(glob.glob(folder + "/CFR_nFFT1024_tx*.mat"))
    usable = np.concatenate([
        np.arange(1024//2 - 350, 1024//2 - 3),
        np.arange(1024//2 + 3, 1024//2 + 350)
    ])
    channels = []
    for f in files:
        mat = loadmat(f)
        CFR = mat["CFR"]
        CFR = CFR[usable, :]
        for rx in range(8):
            H = CFR[:, rx]
            H = H / np.sqrt(np.mean(np.abs(H)**2))
            channels.append(H)
    return channels

def load_sub6_dataset(folder):
    files = sorted(glob.glob(folder + "/*.mat"))
    usable = np.concatenate([
        np.arange(2048//2 - 600, 2048//2 - 3),
        np.arange(2048//2 + 3, 2048//2 + 600)
    ])
    channels = []
    for f in files:
        mat = loadmat(f)
        cores_raw = mat["cores"]
        for pkt_idx in range(cores_raw.shape[1]):
            c0 = cores_raw[0, pkt_idx]
            for rx in range(4):
                c00       = c0[0, rx]
                nss       = c00["nss"][0, 0]
                nss_inner = nss[0, 0]
                data      = nss_inner["data"][0, 0]
                H         = data[0, usable]
                H = H / np.sqrt(np.mean(np.abs(H)**2))
                channels.append(H)
    return channels

sub6_channels   = load_sub6_dataset("data_dec12_sub6/data_dec12_sub6/input_files/processed_files")
mmwave_channels = load_cfr_dataset("2026-01-04_12_15_15")

# ======================================================
# Sem5QI Table
# ======================================================

SEM5QI_TABLE = {
    150: dict(type="bit", max_ber=0.16, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=5.00),
    151: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.60, A_clean=0.70, deadline=5.00),
    152: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.68, A_clean=0.80, deadline=5.00),
    153: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.77, A_clean=0.90, deadline=5.00),
    154: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.85, A_clean=1.00, deadline=5.00),
    155: dict(type="bit", max_ber=0.14, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=6.00),
    156: dict(type="ai",  max_ber=0.14, min_throughput=0.00, A_min=0.61, A_clean=0.70, deadline=6.00),
    157: dict(type="ai",  max_ber=0.14, min_throughput=0.00, A_min=0.70, A_clean=0.80, deadline=6.00),
    158: dict(type="ai",  max_ber=0.14, min_throughput=0.00, A_min=0.78, A_clean=0.90, deadline=6.00),
    159: dict(type="ai",  max_ber=0.14, min_throughput=0.00, A_min=0.87, A_clean=1.00, deadline=6.00),
    160: dict(type="bit", max_ber=0.12, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=7.00),
    161: dict(type="ai",  max_ber=0.12, min_throughput=0.00, A_min=0.62, A_clean=0.70, deadline=7.00),
    162: dict(type="ai",  max_ber=0.12, min_throughput=0.00, A_min=0.71, A_clean=0.80, deadline=7.00),
    163: dict(type="ai",  max_ber=0.12, min_throughput=0.00, A_min=0.80, A_clean=0.90, deadline=7.00),
    164: dict(type="ai",  max_ber=0.12, min_throughput=0.00, A_min=0.89, A_clean=1.00, deadline=7.00),
    165: dict(type="bit", max_ber=0.17, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=4.00),
    166: dict(type="ai",  max_ber=0.17, min_throughput=0.00, A_min=0.59, A_clean=0.70, deadline=4.00),
    167: dict(type="ai",  max_ber=0.17, min_throughput=0.00, A_min=0.68, A_clean=0.80, deadline=4.00),
    168: dict(type="ai",  max_ber=0.17, min_throughput=0.00, A_min=0.76, A_clean=0.90, deadline=4.00),
    169: dict(type="ai",  max_ber=0.17, min_throughput=0.00, A_min=0.84, A_clean=1.00, deadline=4.00),
    170: dict(type="bit", max_ber=0.15, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=5.00),
    171: dict(type="ai",  max_ber=0.15, min_throughput=0.00, A_min=0.60, A_clean=0.70, deadline=5.00),
    172: dict(type="ai",  max_ber=0.15, min_throughput=0.00, A_min=0.69, A_clean=0.80, deadline=5.00),
    173: dict(type="ai",  max_ber=0.15, min_throughput=0.00, A_min=0.77, A_clean=0.90, deadline=5.00),
    174: dict(type="ai",  max_ber=0.15, min_throughput=0.00, A_min=0.86, A_clean=1.00, deadline=5.00),
    175: dict(type="bit", max_ber=0.13, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=6.00),
    176: dict(type="ai",  max_ber=0.13, min_throughput=0.00, A_min=0.61, A_clean=0.70, deadline=6.00),
    177: dict(type="ai",  max_ber=0.13, min_throughput=0.00, A_min=0.70, A_clean=0.80, deadline=6.00),
    178: dict(type="ai",  max_ber=0.13, min_throughput=0.00, A_min=0.79, A_clean=0.90, deadline=6.00),
    179: dict(type="ai",  max_ber=0.13, min_throughput=0.00, A_min=0.88, A_clean=1.00, deadline=6.00),
    180: dict(type="bit", max_ber=0.20, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=30.00),
    181: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.57, A_clean=0.70, deadline=30.00),
    182: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.66, A_clean=0.80, deadline=30.00),
    183: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.74, A_clean=0.90, deadline=30.00),
    184: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.82, A_clean=1.00, deadline=30.00),
    185: dict(type="bit", max_ber=0.18, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=32.00),
    186: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.58, A_clean=0.70, deadline=32.00),
    187: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.67, A_clean=0.80, deadline=32.00),
    188: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.75, A_clean=0.90, deadline=32.00),
    189: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.84, A_clean=1.00, deadline=32.00),
    190: dict(type="bit", max_ber=0.16, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=34.00),
    191: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.60, A_clean=0.70, deadline=34.00),
    192: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.68, A_clean=0.80, deadline=34.00),
    193: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.77, A_clean=0.90, deadline=34.00),
    194: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.85, A_clean=1.00, deadline=34.00),
    195: dict(type="bit", max_ber=0.22, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=0.50),
    196: dict(type="ai",  max_ber=0.22, min_throughput=0.00, A_min=0.56, A_clean=0.70, deadline=0.50),
    197: dict(type="ai",  max_ber=0.22, min_throughput=0.00, A_min=0.64, A_clean=0.80, deadline=0.50),
    198: dict(type="ai",  max_ber=0.22, min_throughput=0.00, A_min=0.72, A_clean=0.90, deadline=0.50),
    199: dict(type="ai",  max_ber=0.22, min_throughput=0.00, A_min=0.80, A_clean=1.00, deadline=0.50),
    200: dict(type="bit", max_ber=0.20, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=2.00),
    201: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.57, A_clean=0.70, deadline=2.00),
    202: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.66, A_clean=0.80, deadline=2.00),
    203: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.74, A_clean=0.90, deadline=2.00),
    204: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.82, A_clean=1.00, deadline=2.00),
    205: dict(type="bit", max_ber=0.18, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=5.00),
    206: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.58, A_clean=0.70, deadline=5.00),
    207: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.67, A_clean=0.80, deadline=5.00),
    208: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.75, A_clean=0.90, deadline=5.00),
    209: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.84, A_clean=1.00, deadline=5.00),
}

# ======================================================
# Shared input generation
# ======================================================

N_ITERATIONS = 70
N_RU_PER_DU  = 3
N_UE_MAX     = 40  # largest UE count across all experiments

sem5qi_keys  = list(SEM5QI_TABLE.keys())
ue_names_max = [f"UE{i+1}" for i in range(N_UE_MAX)]

# Task assignments — one per iteration, for max UE count
ITERATION_TASKS = []
for _ in range(N_ITERATIONS):
    assignment = {ue: random.choice(sem5qi_keys) for ue in ue_names_max}
    ITERATION_TASKS.append(assignment)

# Sub-6 channels — one per iteration, per (UE, RU index)
all_ue_ru_sub6_channels = [
    {
        (ue, ru_idx): random.choice(sub6_channels)
        for ue in ue_names_max
        for ru_idx in range(N_RU_PER_DU)
    }
    for _ in range(N_ITERATIONS)
]

# mmWave channels — one per iteration, per (path, RU index)
all_ru_mm_channels = [
    {
        (path, ru_idx): random.choice(mmwave_channels)
        for path in range(4)
        for ru_idx in range(N_RU_PER_DU)
    }
    for _ in range(N_ITERATIONS)
]

np.save("shared_iteration_tasks.npy",   ITERATION_TASKS,           allow_pickle=True)
np.save("shared_ue_ru_sub6.npy",        all_ue_ru_sub6_channels,   allow_pickle=True)
np.save("shared_ru_mm.npy",             all_ru_mm_channels,         allow_pickle=True)

print("Saved:")
print(f"  shared_iteration_tasks.npy  ({N_ITERATIONS} iterations x {N_UE_MAX} UEs)")
print(f"  shared_ue_ru_sub6.npy       ({N_ITERATIONS} iterations x {N_UE_MAX * N_RU_PER_DU} UE/RU pairs)")
print(f"  shared_ru_mm.npy            ({N_ITERATIONS} iterations x {4 * N_RU_PER_DU} RU/DU links)")
