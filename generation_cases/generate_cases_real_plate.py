"""Generate the .csv table of sensitivity-test cases for a given
plate/suite: Latin-hypercube-sample the parameter ranges defined in
real_plate_parametric_config.yml and write one row per case.
"""

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

_PLOT_ = False


def plot(scaled_sample: np.ndarray, path_save: str, keys: list) -> None:
    """Save diagnostic scatter plots of the Latin-hypercube sample: age vs
    convergence velocity, and thermal parameter vs Tp / age.

    Args:
        scaled_sample (np.ndarray): sampled parameter values, shape
                (n_cases, n_params)
        path_save (str): directory to save the figures in
        keys (list): [name, column_index] pairs mapping parameter names to
                columns of `scaled_sample`
    """

    dict_keys = {i[0]: i[1] for i in keys}

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(
        scaled_sample[:, dict_keys["age"]], scaled_sample[:, dict_keys["vc"]], alpha=0.5
    )
    plt.xlabel(f"{dict_keys}")
    plt.ylabel(f"{dict_keys}")
    plt.title("Latin Hypercube Sampling of Age vs Convergence Velocity")
    plt.grid()
    plt.show()
    plt.savefig(f"{path_save}/Age_vc_latin_hypercube.png")

    thermal_parameter = (
        scaled_sample[:, dict_keys["age"]]
        * scaled_sample[:, dict_keys["vc"]]
        * (1 / 1e5 * 1e6)
    )

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(thermal_parameter, scaled_sample[:, dict_keys["Tp"]], alpha=0.5)
    plt.xlabel("Thermal Parameter (Myr * cm/yr)")
    plt.ylabel("Tp (°C)")
    plt.title("Latin Hypercube Sampling of Thermal Parameter vs Tp")
    plt.grid()
    plt.show()
    plt.savefig(f"{path_save}/thermal_parameter_vs_Tp_latin_hypercube.png")

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(thermal_parameter, scaled_sample[:, dict_keys["age"]], alpha=0.5)
    plt.xlabel("Thermal Parameter (Myr * cm/yr)")
    plt.ylabel("Age (Myr)")
    plt.title("Latin Hypercube Sampling of Thermal Parameter vs Age")
    plt.grid()
    plt.show()
    plt.savefig(f"{path_save}/thermal_parameter_vs_age_latin_hypercube.png")


def read_yml_configuration_file(path: str, plate: str) -> dict:
    """Read the parameter ranges for one plate/suite from
    real_plate_parametric_config.yml.

    Args:
        path (Path): directory containing real_plate_parametric_config.yml
        plate (str): top-level key of the suite to read (e.g. "Mexico")

    Returns:
        dict: {param_name: [low, high]} plus "n_cases", as defined in the yml
    """

    import yaml

    path_yaml = path / "real_plate_parametric_config.yml"
    with open(path_yaml, "r") as f:
        dict_configuration_main = yaml.safe_load(f)

    dict_value = dict_configuration_main[plate]

    return dict_value


def extract_information(dict_conf: dict) -> tuple[np.ndarray, np.ndarray, list, int]:
    """Split a suite's configuration dict into the sampling bounds and
    number of cases.

    Args:
        dict_conf (dict): {param_name: [low, high]} plus "n_cases", as
                returned by `read_yml_configuration_file`

    Returns:
        tuple[np.ndarray, np.ndarray, list, int]: (lbounds, hbounds, keys,
                n_cases) - lower/upper bounds per parameter, [name, index]
                pairs, and the number of cases to sample
    """

    len_dictionary = len(dict_conf)
    # remove the n_cases
    hbounds = np.zeros(len_dictionary - 1, dtype=float)
    lbounds = np.zeros(len_dictionary - 1, dtype=float)

    cnt = 0
    keys = []
    for i in dict_conf.keys():
        if i != "n_cases":
            lbounds[cnt] = dict_conf[i][0]
            hbounds[cnt] = dict_conf[i][1]
            keys.append([i, cnt])
            cnt = cnt + 1
        else:
            n_cases = dict_conf[i]

    return lbounds, hbounds, keys, n_cases


def main() -> None:
    """Sample the cases for the suite named by --plate and write them to
    cases_Sensitivity_{plate}.csv.
    """
    # Parse the input argument
    # Find the real folder of the current file
    # Read yml file
    # Produce a dictionary of values (i.e., read yml file for the given plate, extract the target parameter and relative value)
    # Extract the hbounds and lbounds and number of case and the list of the parameter name
    # Produce the random esemble and scale it
    # write .csv in the same folder with the appropriate name.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plate", default="Mexico", type=str, help="name of the plate/suite"
    )

    args = parser.parse_args()
    #
    path_real = Path(__file__).resolve().parents[0]

    dict_value = read_yml_configuration_file(path=path_real, plate=args.plate)

    # Define lbounds and hbounds
    lbounds, hbounds, keys, n = extract_information(dict_value)

    sampler = scipy.stats.qmc.LatinHypercube(d=len(keys))
    sample = sampler.random(n=n)
    scaled_sample = scipy.stats.qmc.scale(sample, lbounds, hbounds)
    random_cases = np.linspace(0, (n) - 1, num=n, dtype=int)
    rows = []
    temporary_dict = {p[0]: None for p in keys}
    temporary_dict["case_index"] = None

    for index_random in itertools.product(random_cases):
        for i in keys:
            temporary_dict[i[0]] = scaled_sample[index_random, i[1]].item()

        temporary_dict["case_index"] = index_random[0].item()

        rows.append(temporary_dict.copy())

    file = path_real / f"cases_Sensitivity_{args.plate}.csv"
    with open(file, "w", newline="") as f:
        fnames = []
        [fnames.append(k) for k in temporary_dict.keys()]

        writer = csv.DictWriter(
            f,
            fieldnames=fnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} cases to cases_Sensitivity.csv")
    if _PLOT_:
        path_save = path_real / f"{args.plate}_random_pic"
        path_save.mkdir(exist_ok=True)
        plot(scaled_sample=scaled_sample, path_save=path_save, keys=keys)


main()
