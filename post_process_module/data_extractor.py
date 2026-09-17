import os
import warnings

import h5py
import numpy as np

r_grid = 800


def get_nested_attr(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


class Test:
    """Class containing the data of the current test

    path_2_test: the current path of the test at hand
    MeshData: class containing the mesh data
    Data_Raw: class containing the field

    methods:
    _interpolate_data(): function that interpolate the raw data (unstructured triangular grid) to a regular quadrangular
    grid. It uses the additional information provided by the boundary to filter artifact

    _interpolate_data_slab(): function that interpolate the raw data of the simulation and select only the slab domain.

    """

    def __init__(self, path_2_test: str, td: bool = False):
        """Initialisation of the class

        Args:
            path_2_test (str): the path to the test
            td (bool, optional): _description_.Tell the class if the test is steady state or not. False: means steady state, True: means timedependent
        """
        self.path_2_test = path_2_test
        self.MeshData = MeshData(self.path_2_test, td)
        self.Data_raw = Data_Raw(
            self.path_2_test, num=len(self.MeshData.X[:, 0]), td=td
        )

    def _interpolate_data(self, Data_field):
        """Interpolate the data on a regular grid for visualisation
        Args:
            Data_field (str): field to be interpolated
        Returns:
            Zi (np.array): interpolated data

            Paraview is not able to visualise unstructured data properly, so we need to interpolate the data on a regular grid for visualisation purposes.
            The nan mask is used to mask the data outside the domain, and giving you the illusion that I was able to generate a curved mesh in python. Ahah.
        """
        import numpy as np
        from scipy.interpolate import griddata

        Xi = self.MeshData.Xi
        Yi = self.MeshData.Yi

        # Extrac the field data
        values = get_nested_attr(self.Data_raw, Data_field)

        Zi = griddata(self.MeshData.X, values, (Xi, Yi), method="linear")

        Zi[self.MeshData.ar == False] = np.nan

        return Zi

    def _interpolate_data_slab(self, Data_field, ts=0, steady_state=True):
        """Interpolate the data on a regular grid for visualisation
        Args:
            Data_field (str): field to be interpolated
        Returns:
            Zi (np.array): interpolated data

            Paraview is not able to visualise unstructured data properly, so we need to interpolate the data on a regular grid for visualisation purposes.
            The nan mask is used to mask the data outside the domain, and giving you the illusion that I was able to generate a curved mesh in python. Ahah.
        """
        import numpy as np
        from scipy.interpolate import griddata

        Xi = self.MeshData.Xi
        Yi = self.MeshData.Yi

        # Extrac the field data
        values = get_nested_attr(self.Data_raw, Data_field)
        X = self.MeshData.X

        if steady_state:

            Zi = griddata(X, values, (Xi, Yi), method="linear")
        else:
            val = values[:, ts].copy()
            Zi = griddata(X, val, (Xi, Yi), method="linear")

        Zi[self.MeshData.ar_s == False] = np.nan

        return Zi


# ---
class Data_Raw:

    def __init__(self, f: str, num: int, td: bool = False):
        """_summary_

        Args:
            f (str): path 2 the file
            num (int): number of nodes of the unstructured grid
            td (bool, optional): _description_. Defaults to False. time dependent flag: True: Time Dependent; False: Steady state
        """

        if not td:
            self.SteadyState = Data_experiment(f, num, ts=False)
        else:
            self.TimeDependent = Data_experiment(f, num, ts=True)


class Data_experiment:
    """
    Class to extract the data from either the steady state or time dependent h5 file.
    the init function requires the path to the test, the number of points and the number of timestep
    The main issue is for the timedependent case, as I need to extract the number of timesteps from the
    the h5 file, but I will create the function later, and will be in the metadata field of the test.
    times     = time array
    time_list = time list
    Temp      = Temperature array [C]
    Pres      = Dynamic Pressure array [GPa]
    LitPres   = Lithostatic Pressure array [GPa]
    vx        = velocity along x [cm/yr]
    vy        = velocity along y [cm/yr]
    qx        = heat flux x component [W/m]
    qy        = heat flux y component [W/m]
    kappa     = Thermal diffusivity [mm^2/s]
    alpha     = Thermal expansivity
    eta       = Viscosity [Pas]
    rho       = Density [kg/m3]
    k         = Conductivity [W/K/m]
    Cp        = Heat capacity [J/kg/K]
    NoTD      = False
    method:
    extract_data() method that read the output files of StonedFenicsx and fill up all the arrays previously preparred.


    """

    def __init__(self, f: str, num: int, ts: bool):
        """Extract the steady state data from the h5 file.
        The function infers if the test is time dependent, extract the number of timestep
        and allocate arrays with dimension [num,TS], where num is the total number of nodes,
        and TS the total amout of timestep. Then it calls the routine to extract the data
        that automatically generate all the array.

        Args:
            f (str)  : path to the test
            num (int): number of points
            ts (int) : time step
        """

        import h5py
        import numpy as np

        self.times = None
        self.time_list = None
        self.Temp = None
        self.Pres = None
        self.LitPres = None
        self.vx = None
        self.vy = None
        self.qx = None
        self.qy = None
        self.kappa = None
        self.alpha = None
        self.eta = None
        self.rho = None
        self.k = None
        self.Cp = None
        self.NoTD = False

        if ts == True:
            # Rather necessary as the h5 file was not created in reasonable way
            # Direct to the time dependent file
            if os.path.exists("%s/time_dependent.h5" % f):
                print("The file exists.")
            else:
                print(
                    "The file does not exist. Either the time dependent simulation was not run or the path is incorrect."
                )
                self.NoTD = True
                return None

            fl = h5py.File("%s/time_dependent.h5" % f, "r")
            field = "Function/Temperature  [degC]"
            times = list(fl[field].keys())
            time_list = [float(s.replace("_", ".")) for s in times]
            time_sort = np.argsort(time_list)
            time_list = [time_list[i] for i in time_sort]
            times = [times[i] for i in time_sort]
            TS = len(times)
            self.times = times
            self.time_list = time_list
            fl.close()
        else:
            TS = 0

        self.Temp = np.zeros([num, TS], dtype=float)
        self.Pres = np.zeros([num, TS], dtype=float)
        self.LitPres = np.zeros([num, TS], dtype=float)
        self.vx = np.zeros([num, TS], dtype=float)
        self.vy = np.zeros([num, TS], dtype=float)
        self.qx = np.zeros([num, TS], dtype=float)
        self.qy = np.zeros([num, TS], dtype=float)
        self.kappa = np.zeros([num, TS], dtype=float)
        self.alpha = np.zeros([num, TS], dtype=float)
        self.eta = np.zeros([num, TS], dtype=float)
        self.rho = np.zeros([num, TS], dtype=float)
        self.k = np.zeros([num, TS], dtype=float)
        self.Cp = np.zeros([num, TS], dtype=float)

        self.extract_data(f, ts)

    def extract_data(self, f: str, ts: bool):

        import h5py
        import numpy as np

        if ts == True:
            # Direct to the time dependent file
            with h5py.File("%s/time_dependent.h5" % f, "r") as fl:
                for it, time in enumerate(self.times):
                    field_temp = "/Function/Temperature  [degC]/%s" % time
                    field_pres = "/Function/Pressure  [GPa]/%s" % time
                    field_litpres = "/Function/Lit Pres  [GPa]/%s" % time
                    field_v = "/Function/Velocity  [cm/yr]/%s" % time

                    self.Temp[:, it] = np.array(fl[field_temp]).flatten()
                    self.Pres[:, it] = np.array(fl[field_pres]).flatten()
                    self.LitPres[:, it] = np.array(fl[field_litpres]).flatten()

                    v = np.array(fl[field_v])
                    self.vx[:, it] = v[:, 0]
                    self.vy[:, it] = v[:, 1]

        else:
            f1 = ""
            try:
                f1 = "%s/Steady_state.h5" % f
                fl = h5py.File(f1, "r")
                fl.close()
                proceed = True
            except:
                string = f"Steady state solution does not exist in {f}. Either has not be run or there is a problem with the path"
                warnings.warn(string)
                proceed = False

            if proceed:
                with h5py.File(f1, "r") as fl:
                    self.Temp = np.array(
                        fl["/Function/Temperature  [degC]/0"]
                    ).flatten()

                    self.Pres = np.array(fl["/Function/Pressure  [GPa]/0"]).flatten()

                    self.LitPres = np.array(fl["/Function/Lit Pres  [GPa]/0"]).flatten()

                    v = np.array(fl["Function/Velocity  [cm/yr]/0"])

                    self.vx = v[:, 0]

                    self.vy = v[:, 1]

                    qS = np.array(fl["Function/Heat flux [W/m2]/0"])

                    self.qx = qS[:, 0]

                    self.qy = qS[:, 1]

                    self.Cp = np.array(fl["/Function/Cp  [J/kg]/0"]).flatten()

                    self.k = np.array(fl["/Function/k  [W/m/K]/0"]).flatten()

                    self.rho = np.array(fl["/Function/Density  [kg/m3]/0"]).flatten()

                    self.eta = np.array(fl["/Function/Viscosity  [Pa.s]/0"]).flatten()

                    self.alpha = np.array(fl["/Function/alpha  [1/K]/0"]).flatten()

                    self.kappa = np.array(fl["/Function/kappa  [m2/s]/0"]).flatten()


# ---
class MeshData:
    def __init__(self, f: str, td: bool = False):
        """Extract the mesh data from the h5 file. Produce the visualisation grid and the polygon

        Args:
            f (str): path to the test
        """
        if td:
            f1 = "%s/time_dependent.h5" % f
        else:
            f1 = "%s/Steady_state.h5" % f

        with h5py.File(f1, "r") as fl:
            X = np.array(fl["/Mesh/mesh/geometry"])

            ar_point = np.array(fl["Function/MeshTAG/0"])

        ind = np.where(ar_point != 0.0)

        ind = ind[0]

        self.mesh_tag = ar_point.flatten()

        self.X = X

        self.xi = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), r_grid)

        self.yi = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), r_grid)

        self.Xi, self.Yi = np.meshgrid(self.xi, self.yi)

        self.polygon, self.ar = self.create_polygon()

        self.polygon_S, self.ar_s = self.create_polygon_slab()

        self.ind_topSlab = (self.mesh_tag == 8.0) | (self.mesh_tag == 9.0)

        self.ind_Oceanic = self.mesh_tag == 10.0

    # ---
    def create_polygon_slab(self):

        def extract_coordinates(mesh_tag: list) -> np.ndarray:

            if len(mesh_tag) == 2:
                condition = (self.mesh_tag == mesh_tag[0]) | (
                    self.mesh_tag == mesh_tag[1]
                )
            else:
                condition = self.mesh_tag == mesh_tag[0]

            xbuf = self.X[condition, 0]

            ybuf = self.X[condition, 1]

            sort = np.argsort(xbuf)

            X = np.array([xbuf[sort], ybuf[sort]])

            X = np.transpose(X)

            return X

        x = self.X[:, 0]

        y = self.X[:, 1]

        # Left inlet:
        X_L = extract_coordinates([7])
        # Bottom slab:
        X_BS = extract_coordinates([6])
        #
        X_BT = extract_coordinates([5])
        # Slab
        X_Sl = extract_coordinates([9, 8])

        a = X_Sl[:, 0]
        b = X_Sl[:, 1]
        a = a[::-1]
        b = b[::-1]
        X_Sl[:, 0] = a
        X_Sl[:, 1] = b

        from shapely import contains_xy as scontains_xy
        from shapely.geometry import Polygon as sPolygon

        polygon = sPolygon(
            np.vstack((np.array(X_L), np.array(X_BS), np.array(X_BT), np.array(X_Sl)))
        )

        ar = scontains_xy(polygon, self.Xi, self.Yi)

        return polygon, ar

    # ---
    def create_polygon(self):

        x_min = np.min(self.X[:, 0])

        x_max = np.max(self.X[:, 0])

        y_min = np.min(self.X[:, 1])

        y_max = np.max(self.X[:, 1])

        x = self.X[:, 0]

        y = self.X[:, 1]

        top = np.array([x[self.X[:, 1] == y_max], y[self.X[:, 1] == y_max]])

        bottom = np.array([x[self.X[:, 1] == y_min], y[self.X[:, 1] == y_min]])

        left = np.array([x[self.X[:, 0] == x_min], y[self.X[:, 0] == x_min]])

        right = np.array([x[self.X[:, 0] == x_max], y[self.X[:, 0] == x_max]])

        l_min = np.min(left[1, :])

        p0 = np.array([x_min, l_min])

        p_list = []

        p_list.append(p0)

        xbt = self.X[(self.mesh_tag == 6.0), 0]
        ybt = self.X[(self.mesh_tag == 6.0), 1]
        sort = np.argsort(xbt)

        p_list = np.array([xbt[sort], ybt[sort]])

        p_list = p_list.transpose()

        bottom = bottom.transpose()

        right = right.transpose()

        top = top.transpose()

        left = left.transpose()
        # order bottom boundary

        x_bottom = bottom[:, 0]

        ind_arg = np.argsort(x_bottom)

        bottom = bottom[ind_arg, :]
        # order right boundary

        y_right = right[:, 1]

        ind_arg = np.argsort(y_right)

        right = right[ind_arg, :]
        # order top boundary

        x_top = top[:, 0]

        ind_arg = np.argsort(x_top)

        top = top[ind_arg[::-1], :]

        from shapely import contains_xy as scontains_xy
        from shapely.geometry import Polygon as sPolygon

        polygon = sPolygon(
            np.vstack(
                (
                    np.array(p_list),
                    np.array(bottom),
                    np.array(right),
                    np.array(top),
                    np.array(left),
                )
            )
        )

        ar = scontains_xy(polygon, self.Xi, self.Yi)

        return polygon, ar
