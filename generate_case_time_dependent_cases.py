import csv
import itertools
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


# Small script to create time dependent cases



def plots(array_exp):
    

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(array_exp[:, 0], array_exp[:, 1], alpha=0.5)
    plt.xlabel('Age (Myr)')
    plt.ylabel('Convergence Velocity (cm/yr)')
    plt.title('Latin Hypercube Sampling of Age vs Convergence Velocity')
    plt.grid()
    plt.show()
    plt.savefig('age_vs_vc_latin_hypercube.png')

    thermal_parameter = array_exp[:,0] * array_exp[:,1] * (1/1e5 * 1e6) * np.sin(np.radians(45))



    fig = plt.figure(figsize=(10, 6))
    plt.scatter(thermal_parameter, array_exp[:, 0], alpha=0.5)
    plt.xlabel('Thermal Parameter (Myr * cm/yr)')
    plt.ylabel('Age (Myr)')
    plt.title('Latin Hypercube Sampling of Thermal Parameter vs Age')
    plt.grid()
    plt.show()
    plt.savefig('thermal_parameter_vs_age_latin_hypercube.png')  



def main(vc_arr:list[float] = [3.0, 15.0], age_arr:list[float] = [30, 100.0])->None: 
    """
    Simple script: Time dependent problem are not suitable for massive generation of data. 
    The only strategy would be using ROM modelling - topic that I am not expert. 
    
    """

    vc = np.linspace(vc_arr[0],vc_arr[1],num=10)
    age = np.linspace(age_arr[0],age_arr[1],num=10)

    array_exp = np.zeros([100,2],dtype=float)

    vc_, age_ = np.meshgrid(vc,age)
    a = vc_.flatten()
    b = age_.flatten()
    len_array = len(age)*len(vc)

    for i in range(len_array):
        array_exp[i,0] = b[i]
        array_exp[i,1] = a[i]


    rows = []
    case_id = 0

    for  i in range(len_array):

        rows.append({
            "case_index": case_id,
            "age": array_exp[i, 0],
            "vc": array_exp[i, 1],
            "Tp": 1350.0,
        })
        case_id += 1



    with open("cases_Sensitivity_td.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_index",
                "age",
                "vc",
                "Tp"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} cases to cases_Sensitivity.csv")
    
# main function -> future introducing arguments
main()