"""
plot_thermal_properties.py  --  rebuild of the lost "thermal properties" figure.

Panels (bottom -> top): [a] Cp, [b] rho, [c] k, [d] kappa  vs T (degC), P = 0.
Curves : This study (NL), McKenzie et al. 2005, Xu et al. 2004, Richards et al. (reconstruction, see below).
Band   : This study with pressure dependence (NLP), P from 0 to PMAX.
Dotted : values of the linear models (Cp = 1250, rho = 3300, k = 3.1 and 5, kappa = k/(rho Cp)).

"This study" coefficients are copied from stonedfenicsx
  src/stonedfenicsx/config/material_properties_data_base/thermal_properties_data_base_raw.yml
and the formulas from src/stonedfenicsx/config/phase_db.py (density, heat_capacity, heat_conductivity),
commit 3d592e1. Mantle Cp = 0.9 Fo + 0.1 Fa, Berman & Aranovich (1996).

Usage:  python plot_thermal_properties.py   -> thermal_properties.png / .pdf + summary table on stdout
"""
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- settings
T_MIN, T_MAX = 50.0, 1300.0          # degC, plotted range
PMAX = 12e9                          # Pa, upper edge of the NLP band (12 GPa reproduces the rho band of the old figure; set to your domain)
GRAIN = 0.5                          # cm, grain size for k_rad (Grose & Afonso 2013, App. A)
T_REF = 298.15                       # K, reference temperature (pdb.temp_ref)
RHO0 = 3300.0                        # kg/m3
K_LIN = (3.1, 5.0)                   # linear models
CP_LIN, RHO_LIN = 1250.0, 3300.0

TC = np.linspace(T_MIN, T_MAX, 1000)
TK = TC + 273.15

# ----------------------------------------------------------------------------- this study (StonedFeniCSx)
# heat capacity, J/kg/K, T in K  (Bermann_Aranovich_Fosterite / _Fayalite in the yml)
CP_BA_FO = (1657.3910200368182, -12805.367791827477, 0.0, -1904457285.8249636, 0.0, 0.0)
CP_BA_FA = (1236.6823542113452, -9882.171653473753, 0.0, -305195538.1285856, 0.0, 0.0)
# Berman (1988), same layout (Bermann_Fosterite / _Fayalite), used for the Richards reconstruction
CP_B88_FO = (1696.1994726030803, -14224.790498326116, 0.0, -826207788.7000589, 0.0, 0.0)
CP_B88_FA = (1221.616422356469, -9441.480878044473, 0.0, -682629029.6460243, 0.0, 0.0)


def cp_law(c, T):
    return c[0] + c[1] * T**-0.5 + c[2] * T**-2 + c[3] * T**-3 + c[4] * T + c[5] * T**2


def cp_mix(cfo, cfa, xfo, T):
    return xfo * cp_law(cfo, T) + (1 - xfo) * cp_law(cfa, T)


# thermal expansivity / density (Mantle in the yml)
A0, A1, A2 = 2.832e-5, 3.79e-8, 3.63e-11      # 1/K, 1/K^2, 1/Pa
KB = (2 * 100e9 * (1 + 0.25)) / (3 * (1 - 0.25 * 2))   # bulk modulus as in phase_db.py (~166.7 GPa)


def rho_study(T, P):
    """phase_db.density, option_rho = 2 (P dependent); at P = 0 this is the NL density."""
    r = RHO0 * (1 - np.exp(-P * A2) * (A0 * (T - T_REF) + A1 / 2 * (T**2 - T_REF**2)))
    return r * np.exp(P / KB)


# lattice diffusivity (Fosterite_Faylite90 in the yml)
KA, KB_, KC, KD, KE, KF = 0.565e-6, 0.67e-6, 590.0, 1.4e-6, 135.0, 0.05e-9


def kappa_lat(T, P):
    return (KA + KB_ * np.exp(-(T - T_REF) / KC) + KD * np.exp(-(T - T_REF) / KE)) * np.exp(KF * P)


def k_rad(T, d=GRAIN):
    """Grose & Afonso (2013) Eq. A1-A2, d in cm, T in K (as in PhaseDataBase.__init__)."""
    a = 1.8 * (1 - np.exp(-(d**1.3) / 0.15)) - (1 - np.exp(-(d**0.5) / 5.0))
    b = 11.7 * np.exp(-d / 0.159) + 6.0 * np.exp(-(d**3) / 10.0)
    ta = 490.0 + 1850.0 * np.exp(-(d**0.315) / 0.825) + 875.0 * np.exp(-d / 0.18)
    tb = 2700.0 + 9000.0 * np.exp(-(d**0.5) / 0.205)
    xa = 167.5 + 505.0 * np.exp(-(d**0.5) / 0.85)
    xb = 465.0 + 1700.0 * np.exp(-(d**0.94) / 0.175)
    return a * np.exp(-((T - ta) ** 2) / (2 * xa**2)) + b * np.exp(-((T - tb) ** 2) / (2 * xb**2))


def props_study(T, P):
    cp = cp_mix(CP_BA_FO, CP_BA_FA, 0.9, T)
    rho = rho_study(T, P)
    k = kappa_lat(T, P) * rho * cp + k_rad(T)
    return cp, rho, k, k / (rho * cp)


# ----------------------------------------------------------------------------- literature laws (k only)
def k_mckenzie(T):
    """McKenzie et al. (2005) Eq. 4: b/(1 + c T[degC]) + sum d_m (T[degC]+273)^m."""
    d = (1.753e-2, -1.0365e-4, 2.2451e-7, -3.4071e-11)
    return 5.3 / (1 + 0.0015 * (T - 273.15)) + sum(dm * T**m for m, dm in enumerate(d))


def k_xu(T):
    """Xu et al. (2004) as written in McKenzie et al. (2005) Eq. 6, P = 0."""
    return 4.08 * (298.0 / T) ** 0.406


def props_richards(T):
    """RECONSTRUCTION, not Richards' own code: same lattice diffusivity and k_rad as this study,
    Cp from Berman (1988) with 11% fayalite (Richards 2018 Fig. 7e), and rho from integrating
    alpha = a0 + a1 T exponentially (EoS form at P = 0). Replace with Richards' SI parameterisation if needed."""
    cp = cp_mix(CP_B88_FO, CP_B88_FA, 0.89, T)
    rho = RHO0 * np.exp(-(A0 * (T - T_REF) + A1 / 2 * (T**2 - T_REF**2)))
    k = kappa_lat(T, 0.0) * rho * cp + k_rad(T)
    return cp, rho, k, k / (rho * cp)


# ----------------------------------------------------------------------------- compute
cp0, rho0, k0, kap0 = props_study(TK, 0.0)                 # NL (P = 0)
_, rhoP, kP, kapP = props_study(TK, PMAX)                  # upper edge of the NLP band
cpR, rhoR, kR, kapR = props_richards(TK)
kM, kX = k_mckenzie(TK), k_xu(TK)
kapM, kapX = kM / (rho0 * cp0), kX / (rho0 * cp0)          # same rho*Cp as this study (as in the old figure)

# ----------------------------------------------------------------------------- plot
C = dict(study="#a31f1f", mck="#6a1b9a", xu="#f39c12", ric="#2e7d32", band="#a31f1f")
fig, ax = plt.subplots(4, 1, figsize=(6.5, 10), sharex=True)
a_cp, a_rho, a_k, a_kap = ax[3], ax[2], ax[1], ax[0]

a_cp.plot(TC, cp0, color=C["study"], lw=1.5)
a_cp.axhline(CP_LIN, ls=":", color="k", lw=1)
a_cp.set_ylabel(r"$C_p$ [J/kg/K]")

a_rho.fill_between(TC, rho0, rhoP, color=C["band"], alpha=0.12, lw=0)
a_rho.plot(TC, rho0, color=C["study"], lw=1.5)
a_rho.axhline(RHO_LIN, ls=":", color="k", lw=1)
a_rho.set_ylabel(r"$\rho$ [kg/m$^3$]")

a_k.fill_between(TC, k0, kP, color=C["band"], alpha=0.12, lw=0)
for y, lab, c in [(k0, "This study", C["study"]), (kM, "McKenzie et al. 2005", C["mck"]),
                  (kX, "Xu et al. 2004", C["xu"]), (kR, "Richards et al. (reconstr.)", C["ric"])]:
    a_k.plot(TC, y, color=c, lw=1.5, label=lab)
for kl in K_LIN:
    a_k.axhline(kl, ls=":", color="k", lw=1)
a_k.set_ylabel(r"$k$ [W/m/K]")

a_kap.fill_between(TC, kap0 * 1e6, kapP * 1e6, color=C["band"], alpha=0.12, lw=0)
for y, c in [(kap0, C["study"]), (kapM, C["mck"]), (kapX, C["xu"]), (kapR, C["ric"])]:
    a_kap.plot(TC, y * 1e6, color=c, lw=1.5)
for kl in K_LIN:
    a_kap.axhline(kl / (RHO_LIN * CP_LIN) * 1e6, ls=":", color="k", lw=1)
a_kap.set_ylabel(r"$\kappa$ [mm$^2$/s]")
a_k.legend(fontsize=8, loc="upper right", frameon=True)

for a, lab in zip([a_cp, a_rho, a_k, a_kap], ["[a]", "[b]", "[c]", "[d]"]):
    a.text(0.02, 0.92, lab, transform=a.transAxes, fontsize=9, va="top",
           bbox=dict(boxstyle="round,pad=0.2", fc="#444", ec="none"), color="w")
    a.grid(alpha=0.25)
a_cp.set_xlabel(r"Temperature [$^\circ$C]")
a_cp.set_xlim(T_MIN, T_MAX)
fig.tight_layout()
fig.savefig("thermal_properties.png", dpi=300)
fig.savefig("thermal_properties.pdf")

# ----------------------------------------------------------------------------- summary table (kappa, mm2/s)
win = [("150-450", 150, 450), ("0-600", 0, 600), ("600-1000", 600, 1000),
       ("1000-1400", 1000, 1400), ("200-1200", 200, 1200)]
TCw = np.linspace(0, 1400, 2801); TKw = TCw + 273.15
c0w, r0w, k0w, kap0w = props_study(TKw, 0.0)
kw = {"This study (NL)": kap0w,
      "McKenzie 2005": k_mckenzie(TKw) / (r0w * c0w),
      "Xu 2004": k_xu(TKw) / (r0w * c0w),
      "Richards (reconstr.)": props_richards(TKw)[3]}
print(f"{'kappa [mm2/s]':22s}" + "".join(f"{w[0]:>20s}" for w in win))
for n, a in kw.items():
    row = ""
    for _, lo, hi in win:
        s = a[(TCw >= lo) & (TCw <= hi)] * 1e6
        row += f"{s.mean():8.2f} ±{100 * s.std() / s.mean():5.1f}%     "
    print(f"{n:22s}" + row)
print("linear: k3 -> %.3f, k5 -> %.3f mm2/s" % tuple(k / (RHO_LIN * CP_LIN) * 1e6 for k in K_LIN))
for z in (50, 100, 200, 300):
    P = RHO0 * 9.81 * z * 1e3
    print(f"NLP factor on k_lat at {z:3d} km (P = {P / 1e9:4.1f} GPa, T = 800 C): "
          f"{(kappa_lat(1073.15, P) * rho_study(1073.15, P)) / (kappa_lat(1073.15, 0) * rho_study(1073.15, 0)):.2f}")