import csv
import itertools
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

_PLOT_ = False 

def plot(scaled_sample:np.ndarray)->None:

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(scaled_sample[:, 0], scaled_sample[:, 1], alpha=0.5)
    plt.xlabel('Age (Myr)')
    plt.ylabel('Convergence Velocity (cm/yr)')
    plt.title('Latin Hypercube Sampling of Age vs Convergence Velocity')
    plt.grid()
    plt.show()
    plt.savefig('age_vs_vc_latin_hypercube.png')

    thermal_parameter = scaled_sample[:,0] * scaled_sample[:,1] * (1/1e5 * 1e6) * np.sin(np.radians(45))

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(thermal_parameter, scaled_sample[:, 2], alpha=0.5)
    plt.xlabel('Thermal Parameter (Myr * cm/yr)')
    plt.ylabel('Tp (°C)')
    plt.title('Latin Hypercube Sampling of Thermal Parameter vs Tp')
    plt.grid()
    plt.show()
    plt.savefig('thermal_parameter_vs_Tp_latin_hypercube.png')  

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(thermal_parameter, scaled_sample[:, 0], alpha=0.5)
    plt.xlabel('Thermal Parameter (Myr * cm/yr)')
    plt.ylabel('Age (Myr)')
    plt.title('Latin Hypercube Sampling of Thermal Parameter vs Age')
    plt.grid()
    plt.show()
    plt.savefig('thermal_parameter_vs_age_latin_hypercube.png')  


def main(n:int = 10000
         ,vc_arr:list[float]=[4.6, 6.6]
         ,Tp_arr:list[float]=[1350.0, 1450.0]
         ,age_arr:list[float]=[10,15]
         ,phi_arr:list[float]=[0.2,11]
         ,decoupling_arr:list[float]=[55,70])->None:


    lbounds = [age_arr[0],  vc_arr[0], Tp_arr[0],phi_arr[0],decoupling_arr[0]]
    hbounds = [age_arr[1],  vc_arr[1], Tp_arr[1],phi_arr[1],decoupling_arr[1]]

    sampler = scipy.stats.qmc.LatinHypercube(d=5)
    sample = sampler.random(n=n)
    scaled_sample = scipy.stats.qmc.scale(sample, lbounds, hbounds)

    # Non linear cases 

    random_cases = np.linspace(0, (n)-1, num= n, dtype=int)



    rows = []
    case_id = 0

    for index_random, k in itertools.product(random_cases, [0]):

        rows.append({
            "case_index": case_id,
            "age": scaled_sample[index_random, 0],
            "vc": scaled_sample[index_random, 1],
            "Tp": scaled_sample[index_random, 2],
            "phi":scaled_sample[index_random,3],
            "decoupling":scaled_sample[index_random,4],

        })
        case_id += 1



    with open("cases_Sensitivity_mexico.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_index",
                "age",
                "vc",
                "Tp",
                "phi",
                "decoupling"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} cases to cases_Sensitivity.csv")  
    if _PLOT_: 
        plot(scaled_sample=scaled_sample)
    

main()