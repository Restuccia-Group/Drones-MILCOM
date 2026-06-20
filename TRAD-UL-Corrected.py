# ======================================================
# Semantic Aerial Open Radio Access Networks Simulation
# ======================================================

# ======================================================
# GPU-SAFE PREAMBLE
# ======================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import random
from scipy.io import loadmat
import glob

from sionna.phy.mapping import Constellation, Mapper, Demapper
from sionna.phy.utils import ebnodb2no
from sklearn.cluster import KMeans

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--du_max", type=float, default=5.0)
_parser.add_argument("--ru_max", type=float, default=2.0)
_parser.add_argument("--n_ue",   type=int,   default=12)
_args = _parser.parse_args()

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ======================================================
# Global Parameters
# ======================================================

BATCH_SIZE = 1

N_BITS       = 696 * 10
BPS          = 2
EBNO_DB_SUB6 = 6.0 #12.0
EBNO_DB_MM   = 2.0 #5.0
CODERATE     = 1.0

MAX_HARQ = 4
MAX_RLC  = 8
N_TRIALS = 70
N_ITERATIONS = 70

N_RB = 150

sub6_tx_time = 0.125
mm_tx_time   = 0.0625
harq_time    = 1.0
rlc_time     = 3.0

du_max_compute = _args.du_max #2.0 #3.5 #5.0 #7.0
ru_max_compute = _args.ru_max #0.5 #1.0 #2.0 #3.5

N_RU_PER_DU = 3
FAIL_THRESHOLD = 3

MM_CSI_NOISE_VAR   = 0.01
SUB6_CSI_NOISE_VAR = 0.001

HARQ_RTT_BUDGET = 0.5

# ======================================================
# Compute Cost Model
# ======================================================

C_RU_RX_SUB6 = 1.0
C_RU_TX_MM   = 0.5

C_DU_DEMAP   = 1.0
C_DU_HARQ    = 0.5
C_DU_RLC     = 1.5
C_DU_TX_MM   = 0.5

# ======================================================
# Modulation
# ======================================================

constellation = Constellation("qam", BPS)
mapper   = Mapper(constellation=constellation)
demapper = Demapper("app", constellation=constellation)

# ======================================================
# Utilities
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
        CFR = mat["CFR"]   # shape (1024, 8)
        CFR = CFR[usable, :]
        for rx in range(8):
            H = CFR[:, rx]
            H = H / np.sqrt(np.mean(np.abs(H)**2))  # normalize to unit power
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
        cores_raw = mat["cores"]  # (1, N_packets)
        for pkt_idx in range(cores_raw.shape[1]):
            c0 = cores_raw[0, pkt_idx]  # (1, 4)
            for rx in range(4):
                c00      = c0[0, rx]
                nss      = c00["nss"][0, 0]
                nss_inner = nss[0, 0]
                data     = nss_inner["data"][0, 0]  # (1, 2048)
                H        = data[0, usable]           # (1194,) complex
                H = H / np.sqrt(np.mean(np.abs(H)**2))  # normalize to unit power
                channels.append(H)
    return channels

sub6_channels  = load_sub6_dataset("data_dec12_sub6/data_dec12_sub6/input_files/processed_files")
mmwave_channels = load_cfr_dataset("2026-01-04_12_15_15")

#sub6_channels  = load_sub6_dataset("/workspace/envs/simple-wireless-coms/data_dec12_sub6/data_dec12_sub6/input_files/processed_files")
#mmwave_channels = load_cfr_dataset("/workspace/envs/simple-wireless-coms/2026-01-04_12_15_15")

def freq_selective_channel(x, H):
    n_sc  = len(H)
    H_tf  = tf.constant(H, dtype=tf.complex64)
    reps  = tf.cast(tf.math.ceil(tf.shape(x)[1] / n_sc), tf.int32)
    H_rep = tf.tile(H_tf, [reps])
    H_rep = H_rep[:tf.shape(x)[1]]
    H_rep = tf.reshape(H_rep, [1, -1])
    y     = x * H_rep
    return y, H_rep

def awgn(x, no):
    noise = tf.complex(
        tf.random.normal(tf.shape(x), stddev=tf.sqrt(no/2)),
        tf.random.normal(tf.shape(x), stddev=tf.sqrt(no/2))
    )
    return x + noise

def compute_ber(b1, b2):
    return tf.reduce_mean(tf.cast(b1 != b2, tf.float32)).numpy()

def rician_channel(batch, K=10.0):
    scale  = tf.sqrt(1.0 / (2.0 * (K + 1.0)))
    h_nlos = tf.complex(
        tf.random.normal([batch, 1]) * scale,
        tf.random.normal([batch, 1]) * scale
    )
    h_los  = tf.complex(tf.sqrt(K / (K + 1.0)), 0.0)
    return h_nlos + h_los

# ======================================================
# Semantic Accuracy Model
# ======================================================

def semantic_accuracy_confidence(ber, A_clean, k=1.0):
    return A_clean * np.exp(-k * ber)

# ======================================================
# UE → RU
# ======================================================

def ue_to_ru(bits, H=None):
    x = mapper(bits)
    H    = H if H is not None else random.choice(sub6_channels)
    y, H_tf = freq_selective_channel(x, H)
    y    = awgn(y, ebnodb2no(EBNO_DB_SUB6, BPS, CODERATE))
    H_hat = H_tf + tf.complex( # Imperfect sub-6 CSI estimate
        tf.random.normal(tf.shape(H_tf), stddev=np.sqrt(SUB6_CSI_NOISE_VAR/2)),
        tf.random.normal(tf.shape(H_tf), stddev=np.sqrt(SUB6_CSI_NOISE_VAR/2))
    )
    return y, H_hat

# ======================================================
# RU → DU
# ======================================================

def ru_to_du(iq_ru, h_sub6_hat, H_mm=None):
    if H_mm is None:
        H_mm = random.choice(mmwave_channels) # Real mmWave channel
    y, H_tf = freq_selective_channel(iq_ru, H_mm)
    y       = awgn(y, ebnodb2no(EBNO_DB_MM, BPS, CODERATE))
    H_hat   = H_tf + tf.complex( # Imperfect mmWave CSI estimate
        tf.random.normal(tf.shape(H_tf), stddev=np.sqrt(MM_CSI_NOISE_VAR/2)),
        tf.random.normal(tf.shape(H_tf), stddev=np.sqrt(MM_CSI_NOISE_VAR/2))
    )
    y_eq    = y / (H_hat * h_sub6_hat)
    return y_eq, ebnodb2no(EBNO_DB_MM, BPS, CODERATE)

# ======================================================
# DU High-PHY
# ======================================================

def du_high_phy(bits_ref, H_mm_fixed, H_sub6_fixed, harq, rlc, task_ber_max, du_compute_shared, du_max_compute):

    success          = False
    total_rlc_rounds = 0
    harq_rounds_per_rlc = []
    du_compute_local = 0.0
    task_time        = 0.0
    harq_rtt_violations = 0

    task_ber_max *= 1.0

    for _ in range(MAX_RLC if rlc else 1):
        total_rlc_rounds += 1
        if rlc:
            du_compute_local += C_DU_RLC
            du_compute_shared[0] += C_DU_RLC

        iq_ru, h_sub6 = ue_to_ru(bits_ref, H_sub6_fixed)
        H_mm_this_rlc = H_mm_fixed
        llr_acc = tf.zeros_like(bits_ref, tf.float32)
        h_rounds = 0

        for _ in range(MAX_HARQ if harq else 1):
            h_rounds += 1
            y_eq, no = ru_to_du(iq_ru, h_sub6, H_mm_this_rlc)
            llr      = demapper(y_eq, no)

            du_compute_local += C_DU_DEMAP
            du_compute_shared[0] += C_DU_DEMAP

            # always accumulate LLR and compute bits_hat/BER first
            llr_acc  += llr
            bits_hat  = tf.cast(llr_acc > 0, tf.int32)
            ber       = tf.reduce_mean(tf.cast(bits_hat != bits_ref, tf.float32)).numpy()

            # check RTT first, regardless of BER outcome
            if harq:
                du_delay_now = max(0.0, (du_compute_shared[0] - du_max_compute) / du_max_compute)
                fronthaul_rtt = 2 * mm_tx_time + du_delay_now
                if fronthaul_rtt > HARQ_RTT_BUDGET:
                    harq_rtt_violations += 1

            if ber <= task_ber_max:
                success = True
                break

            # only break on violation if BER wasn't already good enough
            if harq and fronthaul_rtt > HARQ_RTT_BUDGET:
                break

        harq_rounds_per_rlc.append(h_rounds)

        if success:
            break

    if rlc:
        task_time += (total_rlc_rounds - 1) * rlc_time
    if harq:
        task_time += (harq_rounds_per_rlc[-1] - 1) * harq_time

    avg_harq_per_rlc = float(np.mean(harq_rounds_per_rlc))
    succ_tensor      = tf.constant([success])

    return bits_hat, succ_tensor, avg_harq_per_rlc, total_rlc_rounds, task_time, du_compute_local, harq_rtt_violations

# ======================================================
# DU → CU
# ======================================================

def du_to_cu(bits_du):
    x       = mapper(bits_du)
    H       = random.choice(mmwave_channels)
    y, H_tf = freq_selective_channel(x, H) # Real mmWave channel
    y       = awgn(y, ebnodb2no(EBNO_DB_MM, BPS, CODERATE))
    H_hat   = H_tf + tf.complex( # Imperfect mmWave CSI estimate
        tf.random.normal(tf.shape(H_tf), stddev=np.sqrt(MM_CSI_NOISE_VAR/2)),
        tf.random.normal(tf.shape(H_tf), stddev=np.sqrt(MM_CSI_NOISE_VAR/2))
    )
    llr = demapper(y / H_hat, ebnodb2no(EBNO_DB_MM, BPS, CODERATE))
    return tf.cast(llr > 0, tf.int32)

# ======================================================
# Stateful DUSA
# ======================================================

PATH_CFG = {
    0: (True,  True),
    1: (True,  True),
    2: (True, False),
    3: (True, False),
}

def DUSA_stateful(ue_state):
    return {
        ue: dict(path=st["path"], harq=PATH_CFG[st["path"]][0], rlc=PATH_CFG[st["path"]][1])
        for ue, st in ue_state.items()
    }

# ======================================================
# Sem5QI Table
# ======================================================

SEM5QI_TABLE = {
    150: dict(type="bit", max_ber=0.16, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=5.00), # Path 0
    151: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.60, A_clean=0.70, deadline=5.00),
    152: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.68, A_clean=0.80, deadline=5.00),
    153: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.77, A_clean=0.90, deadline=5.00),
    154: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.85, A_clean=1.00, deadline=5.00),
    #################################################################################################
    155: dict(type="bit", max_ber=0.14, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=6.00),
    156: dict(type="ai",  max_ber=0.14, min_throughput=0.00, A_min=0.61, A_clean=0.70, deadline=6.00),
    157: dict(type="ai",  max_ber=0.14, min_throughput=0.00, A_min=0.70, A_clean=0.80, deadline=6.00),
    158: dict(type="ai",  max_ber=0.14, min_throughput=0.00, A_min=0.78, A_clean=0.90, deadline=6.00),
    159: dict(type="ai",  max_ber=0.14, min_throughput=0.00, A_min=0.87, A_clean=1.00, deadline=6.00),
    #################################################################################################
    160: dict(type="bit", max_ber=0.12, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=7.00),
    161: dict(type="ai",  max_ber=0.12, min_throughput=0.00, A_min=0.62, A_clean=0.70, deadline=7.00),
    162: dict(type="ai",  max_ber=0.12, min_throughput=0.00, A_min=0.71, A_clean=0.80, deadline=7.00),
    163: dict(type="ai",  max_ber=0.12, min_throughput=0.00, A_min=0.80, A_clean=0.90, deadline=7.00),
    164: dict(type="ai",  max_ber=0.12, min_throughput=0.00, A_min=0.89, A_clean=1.00, deadline=7.00),

    165: dict(type="bit", max_ber=0.17, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=4.00), # Path 1
    166: dict(type="ai",  max_ber=0.17, min_throughput=0.00, A_min=0.59, A_clean=0.70, deadline=4.00),
    167: dict(type="ai",  max_ber=0.17, min_throughput=0.00, A_min=0.68, A_clean=0.80, deadline=4.00),
    168: dict(type="ai",  max_ber=0.17, min_throughput=0.00, A_min=0.76, A_clean=0.90, deadline=4.00),
    169: dict(type="ai",  max_ber=0.17, min_throughput=0.00, A_min=0.84, A_clean=1.00, deadline=4.00),
    #################################################################################################
    170: dict(type="bit", max_ber=0.15, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=5.00),
    171: dict(type="ai",  max_ber=0.15, min_throughput=0.00, A_min=0.60, A_clean=0.70, deadline=5.00),
    172: dict(type="ai",  max_ber=0.15, min_throughput=0.00, A_min=0.69, A_clean=0.80, deadline=5.00),
    173: dict(type="ai",  max_ber=0.15, min_throughput=0.00, A_min=0.77, A_clean=0.90, deadline=5.00),
    174: dict(type="ai",  max_ber=0.15, min_throughput=0.00, A_min=0.86, A_clean=1.00, deadline=5.00),
    #################################################################################################
    175: dict(type="bit", max_ber=0.13, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=6.00),
    176: dict(type="ai",  max_ber=0.13, min_throughput=0.00, A_min=0.61, A_clean=0.70, deadline=6.00),
    177: dict(type="ai",  max_ber=0.13, min_throughput=0.00, A_min=0.70, A_clean=0.80, deadline=6.00),
    178: dict(type="ai",  max_ber=0.13, min_throughput=0.00, A_min=0.79, A_clean=0.90, deadline=6.00),
    179: dict(type="ai",  max_ber=0.13, min_throughput=0.00, A_min=0.88, A_clean=1.00, deadline=6.00),

    180: dict(type="bit", max_ber=0.20, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=30.00), # Path 2
    181: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.57, A_clean=0.70, deadline=30.00),
    182: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.66, A_clean=0.80, deadline=30.00),
    183: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.74, A_clean=0.90, deadline=30.00),
    184: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.82, A_clean=1.00, deadline=30.00),
    #################################################################################################
    185: dict(type="bit", max_ber=0.18, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=32.00),
    186: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.58, A_clean=0.70, deadline=32.00),
    187: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.67, A_clean=0.80, deadline=32.00),
    188: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.75, A_clean=0.90, deadline=32.00),
    189: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.84, A_clean=1.00, deadline=32.00),
    #################################################################################################
    190: dict(type="bit", max_ber=0.16, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=34.00),
    191: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.60, A_clean=0.70, deadline=34.00),
    192: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.68, A_clean=0.80, deadline=34.00),
    193: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.77, A_clean=0.90, deadline=34.00),
    194: dict(type="ai",  max_ber=0.16, min_throughput=0.00, A_min=0.85, A_clean=1.00, deadline=34.00),

    195: dict(type="bit", max_ber=0.22, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=0.50), # Path 3
    196: dict(type="ai",  max_ber=0.22, min_throughput=0.00, A_min=0.56, A_clean=0.70, deadline=0.50),
    197: dict(type="ai",  max_ber=0.22, min_throughput=0.00, A_min=0.64, A_clean=0.80, deadline=0.50),
    198: dict(type="ai",  max_ber=0.22, min_throughput=0.00, A_min=0.72, A_clean=0.90, deadline=0.50),
    199: dict(type="ai",  max_ber=0.22, min_throughput=0.00, A_min=0.80, A_clean=1.00, deadline=0.50),
    #################################################################################################
    200: dict(type="bit", max_ber=0.20, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=2.00),
    201: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.57, A_clean=0.70, deadline=2.00),
    202: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.66, A_clean=0.80, deadline=2.00),
    203: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.74, A_clean=0.90, deadline=2.00),
    204: dict(type="ai",  max_ber=0.20, min_throughput=0.00, A_min=0.82, A_clean=1.00, deadline=2.00),
    #################################################################################################
    205: dict(type="bit", max_ber=0.18, min_throughput=0.00, A_min=0.00, A_clean=0.00, deadline=5.00),
    206: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.58, A_clean=0.70, deadline=5.00),
    207: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.67, A_clean=0.80, deadline=5.00),
    208: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.75, A_clean=0.90, deadline=5.00),
    209: dict(type="ai",  max_ber=0.18, min_throughput=0.00, A_min=0.84, A_clean=1.00, deadline=5.00),
}

sem5qi_keys = list(SEM5QI_TABLE.keys())
#ue_names    = [f"UE{i+1}" for i in range(4)] ########################################### THIS IS FOR 1 UES PER PATH ###################################
#ue_names    = [f"UE{i+1}" for i in range(12)] ########################################### THIS IS FOR 3 UES PER PATH ###################################
#ue_names    = [f"UE{i+1}" for i in range(20)] ########################################### THIS IS FOR 5 UES PER PATH ###################################
#ue_names    = [f"UE{i+1}" for i in range(40)] ########################################### THIS IS FOR 10 UES PER PATH ###################################
#ue_names    = [f"UE{i+1}" for i in range(60)] ############################################################## THIS IS FOR 15 UES PER PATH ##################################
ue_names = [f"UE{i+1}" for i in range(_args.n_ue)]

_all_tasks = np.load("shared_iteration_tasks.npy", allow_pickle=True).tolist()
ITERATION_TASKS = [{ue: it[ue] for ue in ue_names} for it in _all_tasks]

#np.save('iteration_tasks.npy', ITERATION_TASKS)
#print("Saved ITERATION_TASKS to iteration_tasks.npy")

print("Pre-generated Sem5QI assignments:")
for i, assignment in enumerate(ITERATION_TASKS):
    print(f"  Iteration {i+1}: {assignment}")

# ======================================================
# Sem5QI �~F~R Initial Path
# ======================================================

def assign_paths_balanced(tasks):
    """
    UEs with sem5qi <= 179 → evenly split between path 0 and path 1
    UEs with sem5qi > 179  → evenly split between path 2 and path 3
    """
    group_low  = [ue for ue, cfg in tasks.items() if cfg["sem5qi"] <= 179]
    group_high = [ue for ue, cfg in tasks.items() if cfg["sem5qi"] > 179]

    path_assignments = {}

    for i, ue in enumerate(group_low):
        path_assignments[ue] = 0 if i % 2 == 0 else 1

    for i, ue in enumerate(group_high):
        path_assignments[ue] = 2 if i % 2 == 0 else 3

    return path_assignments

# ======================================================
# Monte Carlo Simulation
# ======================================================

print("\n")

all_iteration_avg_success = []
all_iteration_avg_harq    = []
all_iteration_avg_rlc     = []
all_iteration_harq_infeasible = []
all_path_task_counts = {0: 0, 1: 0, 2: 0, 3: 0}

_all_ue_ru_sub6 = np.load("shared_ue_ru_sub6.npy", allow_pickle=True).tolist()
_all_ru_mm      = np.load("shared_ru_mm.npy",      allow_pickle=True).tolist()

for iteration in range(N_ITERATIONS):
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration+1}/{N_ITERATIONS}")
    print(f"{'='*60}")

    TASKS = {}
    for ue in ue_names:
        sem5qi = ITERATION_TASKS[iteration][ue]
        TASKS[ue] = dict(sem5qi=sem5qi)
        TASKS[ue].update(SEM5QI_TABLE[sem5qi])

    print("Sem5QI assignments:", {ue: TASKS[ue]['sem5qi'] for ue in TASKS})

    path_assignments = assign_paths_balanced(TASKS)

    ue_state = {
        u: dict(
            path=path_assignments[u],
            fail_streak=[0, 0, 0]
        )
        for i, u in enumerate(TASKS)
    }

    metrics = {
        u: dict(success=[], ber=[], throughput=[], time=[], harq=[], rlc=[], A_v=[], harq_infeasible=[])
        for u in TASKS
    }
    path_task_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    ue_ru_sub6_channels = _all_ue_ru_sub6[iteration]
    ru_mm_channels      = _all_ru_mm[iteration]

    for _ in range(N_TRIALS):

        dusa_map = DUSA_stateful(ue_state)
        #print("UE state:", {ue: ue_state[ue] for ue in TASKS})

        ru_compute = {p: [0.0]*N_RU_PER_DU for p in range(4)}
        du_compute = {p: 0.0 for p in range(4)}
        ru_rr_ptr  = {p: 0 for p in range(4)}

        results = {}

        for ue, cfg in TASKS.items():

            p  = dusa_map[ue]["path"]
            ru = ru_rr_ptr[p]
            ru_rr_ptr[p] = (ru + 1) % N_RU_PER_DU

            bits = tf.random.uniform([BATCH_SIZE, N_BITS], 0, 2, tf.int32)
            H_sub6_fixed = ue_ru_sub6_channels[(ue, ru)]
            H_mm_fixed   = ru_mm_channels[(p, ru)]

            ru_compute[p][ru] += C_RU_RX_SUB6 + C_RU_TX_MM

            # ← pass du_compute[p] as a mutable single-element list so du_high_phy can update it
            du_compute_shared = [du_compute[p]]

            bits_du, succ_phy, avg_harq_per_rlc, rlc_r, t_phy, du_c, harq_rtt_violations = du_high_phy(
                bits, H_mm_fixed, H_sub6_fixed,
                dusa_map[ue]["harq"],
                dusa_map[ue]["rlc"],
                cfg["max_ber"],
                du_compute_shared,   # ← pass shared state
                du_max_compute       # ← pass budget
            )

            # ← sync back the updated compute state and add TX cost
            du_compute[p] = du_compute_shared[0] + C_DU_TX_MM
            du_delay_this_ue = max(0.0, (du_compute[p] - du_max_compute) / du_max_compute)

            bits_cu = bits_du #du_to_cu(bits_du)
            ber     = compute_ber(bits, bits_cu)

            A_v = semantic_accuracy_confidence(ber, cfg["A_clean"]) if cfg["type"] == "ai" else None

            ru_delay_task = max(
                0.0,
                (ru_compute[p][ru] - ru_max_compute) / ru_max_compute
            )

            results[ue] = dict(
                ber=ber,
                A_v=A_v,
                time=t_phy,             
                ru_delay=ru_delay_task,
                du_delay=du_delay_this_ue,
                path=p,
                avg_harq_per_rlc=avg_harq_per_rlc, 
                rlc_r=rlc_r,              
                success_phy=bool(tf.reduce_all(succ_phy)),
                harq_rtt_violations=harq_rtt_violations  # ← per-round violation count
            )   

            path_task_counts[p] += 1

        #du_delay = {
        #    p: max(0.0, (du_compute[p] - du_max_compute) / du_max_compute)
        #    for p in du_compute
        #}

        # ← HARQ infeasibility now based on per-round violations inside du_high_phy
        for ue in TASKS:
            r = results[ue]
            if dusa_map[ue]["harq"]:
                results[ue]["harq_infeasible"] = r["harq_rtt_violations"] > 0
            else:
                results[ue]["harq_infeasible"] = False

        for ue, cfg in TASKS.items():
            r = results[ue]

            total_time = (
                r["time"]
                + sub6_tx_time
                + mm_tx_time
                + r["ru_delay"]
                + r["du_delay"]
            )

            throughput = N_RB * (1 - r["ber"]) * N_BITS / (total_time * 1e6)

            # Bit-wise Tasks
            if cfg["type"] == "bit":
                success = (
                    r["ber"] <= cfg["max_ber"]
                    and total_time <= cfg["deadline"]
                    #and not r["harq_infeasible"]    
                )

                metrics[ue]["success"].append(success)
                metrics[ue]["harq_infeasible"].append(r["harq_infeasible"])
                metrics[ue]["harq"].append(r["avg_harq_per_rlc"])
                metrics[ue]["rlc"].append(r["rlc_r"])

                if success:
                    ue_state[ue]["fail_streak"] = [0, 0, 0]
                    metrics[ue]["ber"].append(r["ber"])
                    metrics[ue]["throughput"].append(throughput)
                    metrics[ue]["time"].append(total_time)
                    #metrics[ue]["harq"].append(r["avg_harq_per_rlc"])
                    #metrics[ue]["rlc"].append(r["rlc_r"])
                else:
                    if r["ber"] > cfg["max_ber"]:
                        ue_state[ue]["fail_streak"][0] += 1

                    if total_time > cfg["deadline"]:
                        ue_state[ue]["fail_streak"][2] += 1

            # AI Tasks
            else:
                success = (
                    r["A_v"] >= cfg["A_min"]
                    and total_time <= cfg["deadline"]
                    #and not r["harq_infeasible"]
                )

                metrics[ue]["success"].append(success)
                metrics[ue]["harq_infeasible"].append(r["harq_infeasible"])
                metrics[ue]["harq"].append(r["avg_harq_per_rlc"])
                metrics[ue]["rlc"].append(r["rlc_r"])

                if success:
                    ue_state[ue]["fail_streak"][1] = 0
                    ue_state[ue]["fail_streak"][2] = 0
                    metrics[ue]["A_v"].append(r["A_v"])
                    metrics[ue]["time"].append(total_time)
                    #metrics[ue]["harq"].append(r["avg_harq_per_rlc"])
                    #metrics[ue]["rlc"].append(r["rlc_r"])
                else:
                    if r["A_v"] < cfg["A_min"]:
                        ue_state[ue]["fail_streak"][1] += 1

                    if total_time > cfg["deadline"]:
                        ue_state[ue]["fail_streak"][2] += 1

    # ======================================================
    # Results
    # ======================================================
    for ue, cfg in TASKS.items():
        if cfg["type"] == "bit":
            print(f"{ue}")
            print(f"  Success Rate                                : {np.mean(metrics[ue]['success']):.3f}")
            print(f"  Avg BER (successful tasks)                  : {np.mean(metrics[ue]['ber']):.3e}")
            print(f"  Avg throughput (successful tasks)           : {np.mean(metrics[ue]['throughput']):.3e}")
            print(f"  Avg total task time (successful tasks)      : {np.mean(metrics[ue]['time']):.3e}")
            print(f"  Avg HARQ rounds per RLC of successful tasks : {np.mean(metrics[ue]['harq']):.2f}")
            print(f"  Avg RLC rounds of successful tasks          : {np.mean(metrics[ue]['rlc']):.2f}\n")
        else:
            print(f"{ue}")
            print(f"  Success Rate                                : {np.mean(metrics[ue]['success']):.3f}")
            print(f"  Avg A_v (successful tasks)                  : {np.mean(metrics[ue]['A_v']):.3e}")
            print(f"  Avg total task time (successful tasks)      : {np.mean(metrics[ue]['time']):.3e}")
            print(f"  Avg HARQ rounds per RLC of successful tasks : {np.mean(metrics[ue]['harq']):.2f}")
            print(f"  Avg RLC rounds of successful tasks          : {np.mean(metrics[ue]['rlc']):.2f}\n")

    iter_avg = np.mean([np.mean(metrics[ue]['success']) for ue in ue_names])
    iter_harq = np.mean([np.mean(metrics[ue]['harq']) if metrics[ue]['harq'] else 0 for ue in ue_names])
    iter_rlc  = np.mean([np.mean(metrics[ue]['rlc'])  if metrics[ue]['rlc']  else 0 for ue in ue_names])
    iter_harq_infeasible = np.mean([np.mean(metrics[ue]['harq_infeasible']) for ue in ue_names])
    print(f"  --> Iteration {iteration+1} avg success rate: {iter_avg:.3f}")
    print(f"  --> Path task counts: { {f'Path {p}': path_task_counts[p] for p in range(4)} }")
    for p in range(4):
        all_path_task_counts[p] += path_task_counts[p]
    all_iteration_avg_success.append(iter_avg)
    all_iteration_avg_harq.append(iter_harq)
    all_iteration_avg_rlc.append(iter_rlc)
    all_iteration_harq_infeasible.append(iter_harq_infeasible)

print(f"\n{'='*60}")
print(f"FINAL SUMMARY ACROSS {N_ITERATIONS} ITERATIONS")
print(f"{'='*60}")
for i, avg in enumerate(all_iteration_avg_success):
    print(f"  Iteration {i+1:2d}: {avg:.3f}")
print(f"\n  Overall avg : {np.mean(all_iteration_avg_success):.3f}")
print(f"  Std dev     : {np.std(all_iteration_avg_success):.3f}")
print(f"  Overall Avg HARQ rounds : {np.mean(all_iteration_avg_harq):.3f}")
print(f"  Std HARQ rounds         : {np.std(all_iteration_avg_harq):.3f}")
print(f"  Overall Avg RLC rounds  : {np.mean(all_iteration_avg_rlc):.3f}")
print(f"  Std RLC rounds          : {np.std(all_iteration_avg_rlc):.3f}")
print(f"  Overall Avg HARQ Infeasibility Rate : {np.mean(all_iteration_harq_infeasible):.3f}")
print(f"  Std HARQ Infeasibility Rate         : {np.std(all_iteration_harq_infeasible):.3f}")
print(f"\n  Total Tasks Served Per Path (all iterations):")
for p in range(4):
    print(f"    Path {p}: {all_path_task_counts[p]}")
