import csv
import itertools
import scipy.stats
import numpy as np

Edis_arr =  [470e3,570e3]
Vdis_arr = [4e-6,1.5e-5]
decoupling_arr = [80e3, 120e3]
vc_arr = [6.5, 8.5]
Tp_arr = [1350.0, 1425.0]
thermal_options = [0, 1]

lbounds = [Edis_arr[0], Vdis_arr[0],  vc_arr[0], Tp_arr[0]]
hbounds = [Edis_arr[1], Vdis_arr[1],  vc_arr[1], Tp_arr[1]]

n=3000
sampler = scipy.stats.qmc.LatinHypercube(d=4)
sample = sampler.random(n=n)
scaled_sample = scipy.stats.qmc.scale(sample, lbounds, hbounds)
random_cases = np.linspace(0, n-1, num=n, dtype=int)

rows = []
case_id = 0

for index_random, opt_th,dc in itertools.product(
    random_cases,
    thermal_options,
    decoupling_arr):
    rows.append({
        "case_id": case_id,
        "Edis": scaled_sample[index_random, 0],
        "Vdis": scaled_sample[index_random, 1],
        "decoupling": dc,
        "vc": scaled_sample[index_random, 2],
        "Tp": scaled_sample[index_random, 3],
        "option_thermal": opt_th,
    })
    case_id += 1

with open("cases_random.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case_id",
            "Edis",
            "Vdis",
            "decoupling",
            "vc",
            "Tp",
            "option_thermal",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} cases to cases.csv")