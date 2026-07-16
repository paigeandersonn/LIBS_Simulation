# -*- coding: utf-8 -*-
"""
Onion-layer Aluminum LIBS model (Spyder-ready, single file)

- 10 concentric layers (back -> front along line of sight)
- T, ne, n_plasma follow exponential falloff from a hot/dense core to ambient
- Saha (I <-> II) per layer
- Voigt line shapes (Doppler ⊕ instrument ⊕ Stark)
- Physics-based pointwise self-absorption with layer-by-layer radiative transfer:
    I_out = I_in*exp(-tau) + J_thin * (1 - exp(-tau))/tau
- Diagnostic plot: T, ne, n_plasma vs line-of-sight position

Runs out of the box with a tiny built-in Al line list; you can later swap in a CSV.
"""

# ==========================
# CONFIG
# ==========================
CONFIG = {
    # Wavelength grid (nm)
    "lambda_min_nm": 370.0,
    "lambda_max_nm": 410.0,
    "delta_lambda_nm": 0.002,

    # Geometry / layering
    "num_layers": 20,                 # onion layers
    "total_path_length_cm": 0.05,     # total LOS thickness (cm)

    # Core/ambient states (exponential falloff with layer index i=0..N-1)
    "core_T_K": 9000.0,
    "core_ne_cm3": 5e16,
    "core_nplasma_cm3": 1e17,

    "ambient_T_K": 500.0,
    "ambient_ne_cm3": 2e15,
    "ambient_nplasma_cm3": 1e15,

    # Composition (number fraction)
    "composition": {"Al": 1.0},

    # Ionization: Saha (I <-> II) per layer
    "use_saha": True,
    "delta_chi_eV": 0.0,              # ionization potential lowering (optional)
    "partition_functions": {},        # defaults to 1.0 if absent (keys: 'Al I', 'Al II')

    # Broadening / instrument
    "instrument_fwhm_nm": 0.04,       # Gaussian instrument LSF FWHM (nm)
    "include_doppler": True,
    "include_stark": True,
    "estimate_missing_stark": False,   # estimate per-line width at 1e16 cm^-3 if missing

    # Self-absorption & RT
    "enable_self_absorption": True,

    # Output/plots
    "normalize_to_max": True,
    "plot_result": True,
    "save_plot_png": "onion_aluminum_spectrum.png",
    "save_profiles_png": "onion_aluminum_profiles.png",
    "output_csv": "onion_aluminum_spectrum.csv",
    "profile_type_T":  "supergauss",  # "parabola" | "gauss" | "supergauss" | "raisedcos" | "logistic"
    "profile_type_ne": "raisedcos",
    "profile_type_np": "raisedcos",
    # shape params
    "parabola_power": 2.0,           # for "parabola"
    "gauss_sigma_frac": 0.35,        # sigma/L for "gauss"
    "super_m": 3.0,                  # m for super-Gaussian
    "raisedcore_frac": 0.1,          # L0/L for raised-cosine flat core fraction
    "raisededge_frac": 0.3,          # Δ/L for raised-cosine edge thickness
    "logistic_xhalf_frac": 0.3,      # x_half/L for logistic midpoint
    "logistic_width_frac": 0.08      # w/L for logistic transition width
}

# ==========================
# Imports & constants
# ==========================
import numpy as np
import pandas as pd
import math
from scipy.special import wofz
import matplotlib.pyplot as plt

# SI constants
K_B = 1.380649e-23
H   = 6.62607015e-34
C   = 2.99792458e8
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
AMU = 1.66053906660e-27
M_E = 9.1093837015e-31
PI  = math.pi

# Aluminum data (extend as needed)
IONIZATION_POTENTIAL_EV = {"Al": 5.98577}
MASS_AMU = {"Al": 26.98}

# Tiny demo Al line list (replace with CSV later if desired)
# columns: species, wl_nm, Aki, Ek_eV, Ei_eV, gk, gi, mass, stark_w_nm_at_1e16
AL_LINES = pd.DataFrame([
    ["Al I",  394.401, 4.9e7, 3.14, 0.00, 2, 2, MASS_AMU["Al"], 0.015],
    ["Al I",  396.152, 9.9e7, 3.14, 0.00, 4, 2, MASS_AMU["Al"], 0.020],
    ["Al II", 358.657, 3.0e7, 6.37, 0.00, 4, 2, MASS_AMU["Al"], 0.010],
], columns=["species","wavelength_nm","Aki_s","Ek_eV","Ei_eV","gk","gi","mass_amu","stark_w_nm_at_1e16"])

# ==========================
# Utilities
# ==========================
def parse_species(s):
    s = str(s).strip()
    if " " in s:
        el, st = s.split()
        return el.strip(), st.strip().upper()
    return s, "I"

def get_partition_function(species, T, Ucfg):
    U = Ucfg.get(species, 1.0)
    return float(U(T)) if callable(U) else float(U)

def saha_first_ion_fractions(element, T_K, ne_cm3, U_I=1.0, U_II=1.0, delta_chi_eV=0.0):
    """Return (f_I, f_II) for first ionization balance (cm^-3 inputs)."""
    chi_eV = IONIZATION_POTENTIAL_EV.get(element, 7.0)
    kT_eV  = (K_B*T_K)/E_CHARGE
    S      = ((2.0*PI*M_E*K_B*T_K)/(H*H))**1.5 / 1e6   # m^-3 -> cm^-3
    rhs    = S * 2.0 * (U_II/max(U_I,1e-30)) * math.exp(-(chi_eV - delta_chi_eV)/max(kT_eV,1e-30))
    R      = rhs / max(ne_cm3,1e-30)
    fII    = R/(1.0+R)
    fI     = 1.0 - fII
    return max(0.0,min(1.0,fI)), max(0.0,min(1.0,fII))

def doppler_fwhm_nm(lambda_nm, T_K, mass_amu):
    lam = float(lambda_nm)
    M   = max(mass_amu, 1e-6)
    return 7.16e-7 * lam * math.sqrt(T_K / M)

def effective_gaussian_fwhm_nm(doppler_nm, instrument_nm):
    return math.sqrt(max(doppler_nm,0.0)**2 + max(instrument_nm,0.0)**2)

def stark_fwhm_nm(ne_cm3, w_nm_at_1e16):
    if (w_nm_at_1e16 is None) or (w_nm_at_1e16 <= 0): return 0.0
    return w_nm_at_1e16 * (ne_cm3/1.0e16)

def voigt_profile_nm(lam_nm_grid, lam0_nm, fwhm_g_nm, fwhm_l_nm):
    """Area-normalized Voigt on wavelength grid (nm)."""
    x = lam_nm_grid - lam0_nm
    sigma = fwhm_g_nm/(2.0*math.sqrt(2.0*math.log(2.0))) if fwhm_g_nm>0 else 0.0
    gamma = 0.5*fwhm_l_nm
    if sigma<=0 and gamma<=0:
        prof = np.zeros_like(lam_nm_grid)
        prof[np.argmin(np.abs(x))] = 1.0/max(abs(lam_nm_grid[1]-lam_nm_grid[0]),1e-12)
        return prof
    if sigma<=0:  # Lorentz
        prof = gamma/PI/(x**2 + gamma**2)
        area = np.trapz(prof, lam_nm_grid); 
        if area>0: prof/=area
        return prof
    z = (x + 1j*gamma)/(sigma*math.sqrt(2.0))
    V = np.real(wofz(z))/(sigma*math.sqrt(2.0*PI))
    area = np.trapz(V, lam_nm_grid); 
    if area>0: V/=area
    return V

def voigt_center_freq(fwhm_g_Hz, fwhm_l_Hz):
    """Area-normalized Voigt at line center in frequency domain (1/Hz)."""
    sigma = fwhm_g_Hz/(2.0*math.sqrt(2.0*math.log(2.0))) if fwhm_g_Hz>0 else 0.0
    gamma = 0.5*fwhm_l_Hz
    if sigma<=0 and gamma<=0: return 0.0
    if sigma<=0: return 1.0/(PI*gamma) if gamma>0 else 0.0
    z0 = 1j*gamma/(sigma*math.sqrt(2.0))
    return float(np.real(wofz(z0))/(sigma*math.sqrt(2.0*PI)))

def f_from_Aki_lambda_gi_gk(Aki, lambda_nm, gi, gk):
    lam_m = lambda_nm*1e-9
    if gi<=0 or gk<=0 or Aki<=0 or lam_m<=0: return np.nan
    return (M_E*C*EPS0*(gk/gi)*Aki*(lam_m**2))/(PI*(E_CHARGE**2))

# ==========================
# Layer profiles
# ==========================
def shape_profile(x, L, Xcore, Xamb, kind, cfg):
    """
    Return X(x) on 0..L for the requested 'kind'.
    All shapes are 'anchored' so X(0)=Xcore and X(L)=Xamb exactly.
    """
    kind = (kind or "parabola").lower()
    if L <= 0.0:
        return float(Xcore)
    # normalized position 0..1 at the i-th layer
    s = x / L
    s = 0.0 if s < 0 else (1.0 if s > 1.0 else s)

    if kind == "supergauss":
        # Anchored super-Gaussian:
        # raw: w = exp(-(s/σ)^m), tail at s=1 is w1 = exp(-(1/σ)^m)
        # anchored: wA = (w - w1)/(1 - w1) ⇒ wA(0)=1, wA(1)=0
        m      = float(cfg.get("super_m", 4.0))
        sigmaf = float(cfg.get("super_sigma_frac", 0.35))
        sigma  = max(sigmaf, 1e-12)
        w  = math.exp(- (s / sigma)**m)
        w1 = math.exp(- (1.0 / sigma)**m)
        denom = max(1.0 - w1, 1e-12)
        wA = (w - w1) / denom
        return float(Xamb + (Xcore - Xamb) * wA)

    elif kind == "gauss":
        # Anchored Gaussian (same anchoring idea)
        sigmaf = float(cfg.get("gauss_sigma_frac", 0.35))
        sigma  = max(sigmaf, 1e-12)
        w  = math.exp(- (s / sigma)**2)
        w1 = math.exp(- (1.0 / sigma)**2)
        denom = max(1.0 - w1, 1e-12)
        wA = (w - w1) / denom
        return float(Xamb + (Xcore - Xamb) * wA)

    elif kind == "parabola":
        p = float(cfg.get("parabola_power", 2.0))
        return float(Xamb + (Xcore - Xamb) * max(0.0, 1.0 - s**p))

    elif kind == "raisedcos":
        L0 = float(cfg.get("raisedcore_frac", 0.5)) * L   # flat core length
        d  = float(cfg.get("raisededge_frac", 0.2)) * L   # edge thickness
        if s*L <= L0:
            w = 1.0
        elif s*L >= L0 + d:
            w = 0.0
        else:
            xi = (s*L - L0) / max(d, 1e-12)
            w  = 0.5 * (1.0 + math.cos(math.pi * xi))
        return float(Xamb + (Xcore - Xamb) * w)

    elif kind == "logistic":
        xh = float(cfg.get("logistic_xhalf_frac", 0.7)) * L
        w  = float(cfg.get("logistic_width_frac", 0.08)) * L
        wA = 1.0 / (1.0 + math.exp((s*L - xh) / max(w, 1e-12)))
        # Anchor: logistic tends to 0,1 asymptotically; this form is already ~anchored on 0..L.
        return float(Xamb + (Xcore - Xamb) * wA)

    # Fallback: anchored parabola
    p = float(cfg.get("parabola_power", 2.0))
    return float(Xamb + (Xcore - Xamb) * max(0.0, 1.0 - s**p))

def build_layers_with_profiles(cfg):
    """
    Build N layers of thickness dz = Ltot/(N), with anchored profiles for T, ne, n_plasma.
    Ensures last layer equals ambient by construction.
    """
    N    = int(cfg.get("num_layers", 10))
    Ltot = float(cfg["total_path_length_cm"])           # LOS distance in cm
    dz   = Ltot / max(N, 1)
    # Which shapes to use for each field
    kT   = cfg.get("profile_type_T",  "parabola")
    kne  = cfg.get("profile_type_ne", "parabola")
    knp  = cfg.get("profile_type_np", "parabola")

    layers = []
    for i in range(N):
        x = i * dz
        T_i  = shape_profile(x, Ltot, cfg["core_T_K"],         cfg["ambient_T_K"],         kT,  cfg)
        ne_i = shape_profile(x, Ltot, cfg["core_ne_cm3"],      cfg["ambient_ne_cm3"],      kne, cfg)
        np_i = shape_profile(x, Ltot, cfg["core_nplasma_cm3"], cfg["ambient_nplasma_cm3"], knp, cfg)
        layers.append({"T_K": T_i, "ne_cm3": ne_i, "n_plasma_cm3": np_i, "thickness_cm": dz})
    return layers

# ==========================
# Main compute
# ==========================
def main(cfg):
    # wavelength grid
    lam = np.arange(cfg["lambda_min_nm"], cfg["lambda_max_nm"] + 0.5*cfg["delta_lambda_nm"], cfg["delta_lambda_nm"])

    # build layers (deepest first => observed through remaining layers)
    layers = build_layers_with_profiles(cfg)

    # --- Diagnostic plot: layer profiles along line of sight ---
    if cfg.get("plot_result", True):
        pos_cm = np.linspace(0, cfg["total_path_length_cm"], len(layers))
        T_list  = [L["T_K"] for L in layers]
        ne_list = [L["ne_cm3"] for L in layers]
        np_list = [L["n_plasma_cm3"] for L in layers]

        fig, ax1 = plt.subplots()
        color1, color2, color3 = "tab:red", "tab:blue", "tab:green"
        ax1.set_xlabel("Line-of-sight position (cm)")
        ax1.set_ylabel("Temperature (K)", color=color1)
        ax1.plot(pos_cm, T_list, "o-", color=color1, label="T")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_title("Onion-layer plasma profiles (Aluminum, inverted parabola)")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Density (cm$^{-3}$)", color=color2)
        ax2.semilogy(pos_cm, ne_list, "s--", color=color2, label="nₑ")
        ax2.semilogy(pos_cm, np_list, "d-.", color=color3, label="n_plasma")
        ax2.tick_params(axis="y", labelcolor=color2)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        if cfg.get("save_profiles_png"):
            plt.savefig(cfg["save_profiles_png"], dpi=150)
        plt.show()

    # prep line list
    lines = AL_LINES.copy()

    # initialize emergent spectrum
    I = np.zeros_like(lam, dtype=float)

    # loop layers from back (deepest) to front (ambient), accumulating RT
    for idx, layer in enumerate(layers):
        T_K  = layer["T_K"]
        ne   = layer["ne_cm3"]
        npl  = layer["n_plasma_cm3"]
        L_cm = layer["thickness_cm"]
        L_m  = L_cm * 1e-2
        Ucfg = cfg.get("partition_functions", {})

        # species number densities (Al I / Al II) for this layer
        comp = cfg["composition"]
        species_ntot = {}
        for el, frac in comp.items():
            ntot_el = frac * npl
            if cfg.get("use_saha", True):
                U_I  = get_partition_function(f"{el} I",  T_K, Ucfg)
                U_II = get_partition_function(f"{el} II", T_K, Ucfg)
                fI, fII = saha_first_ion_fractions(el, T_K, ne, U_I=U_I, U_II=U_II, delta_chi_eV=cfg.get("delta_chi_eV",0.0))
                species_ntot[f"{el} I"]  = ntot_el * fI
                species_ntot[f"{el} II"] = ntot_el * fII
            else:
                species_ntot[f"{el} I"] = ntot_el  # neutral-only fallback

        # layer emission and SA -> emergent from this layer
        J_layer = np.zeros_like(lam, dtype=float)

        for _, row in lines.iterrows():
            species = str(row["species"]).strip()
            if species_ntot.get(species,0) <= 0:
                continue

            lam0   = float(row["wavelength_nm"])
            if lam0 < cfg["lambda_min_nm"] or lam0 > cfg["lambda_max_nm"]:
                continue
            lam0_m = lam0*1e-9

            Aki    = float(row["Aki_s"])
            Ek_eV  = float(row["Ek_eV"])
            Ei_eV  = float(row["Ei_eV"])
            gk     = float(row["gk"]) if row["gk"]>0 else 1.0
            gi     = float(row["gi"]) if row["gi"]>0 else 1.0
            mass   = float(row["mass_amu"]) if row["mass_amu"]>0 else MASS_AMU["Al"]
            w16    = float(row["stark_w_nm_at_1e16"]) if not np.isnan(row["stark_w_nm_at_1e16"]) else 0.0
            if (w16<=0.0) and cfg.get("estimate_missing_stark", True):
                w16 = (0.015 if parse_species(species)[1]=="I" else 0.010) * (lam0/400.0)**2

            # LTE populations (upper/lower)
            U_T = get_partition_function(species, T_K, Ucfg)
            kT  = (K_B*T_K)/E_CHARGE
            n_species   = species_ntot[species]   # cm^-3
            boltz_upper = (gk/max(U_T,1e-30)) * math.exp(-Ek_eV/max(kT,1e-30))
            boltz_lower = (gi/max(U_T,1e-30)) * math.exp(-Ei_eV/max(kT,1e-30))

            # thin emissivity * shape (arbitrary units)
            S_em  = n_species * boltz_upper * Aki
            dop_nm = doppler_fwhm_nm(lam0, T_K, mass) if cfg.get("include_doppler", True) else 0.0
            inst_nm= float(cfg.get("instrument_fwhm_nm", 0.0))
            fwhm_g = effective_gaussian_fwhm_nm(dop_nm, inst_nm)  # instrument included for displayed profile
            fwhm_l = stark_fwhm_nm(ne, w16) if cfg.get("include_stark", True) else 0.0

            prof = voigt_profile_nm(lam, lam0, fwhm_g, fwhm_l)  # area=1 over λ
            J_thin = S_em * prof

            # --- Self-absorption (pointwise) ---
            if cfg.get("enable_self_absorption", True) and L_m>0:
                # oscillator strength (fallback route)
                f_ik = f_from_Aki_lambda_gi_gk(Aki, lam0, gi, gk)

                # For SA cross-section, use PHYSICAL Gaussian (Doppler) without instrument
                fwhm_g_phys = dop_nm
                nm_to_Hz    = C/(lam0_m**2) * 1e-9
                phi0        = voigt_center_freq(fwhm_g_phys*nm_to_Hz, fwhm_l*nm_to_Hz)  # 1/Hz

                # center optical depth τ0
                tau0 = np.nan
                if (not np.isnan(f_ik)) and f_ik>0 and phi0>0:
                    n_i_cm3 = n_species * boltz_lower
                    if n_i_cm3 > 0:
                        sigma0 = (PI*(E_CHARGE**2)/(EPS0*M_E*C)) * f_ik * phi0   # m^2
                        tau0   = (n_i_cm3*1e6) * sigma0 * L_m                    # dimensionless

                if (not np.isnan(tau0)) and tau0>0:
                    peak = np.max(prof) if np.max(prof)>0 else 1.0
                    tau_lambda = tau0 * (prof/peak)
                    R = np.ones_like(tau_lambda)
                    m = tau_lambda > 1e-9
                    R[m] = (1.0 - np.exp(-tau_lambda[m]))/tau_lambda[m]
                    J_layer += J_thin * R
                else:
                    J_layer += J_thin
            else:
                J_layer += J_thin

        # Radiative transfer update through this layer
        I = I + J_layer

    # normalize & output
    if CONFIG.get("normalize_to_max", True):
        m = np.max(I) if I.size else 1.0
        if m>0: I = I/m

    pd.DataFrame({"wavelength_nm": lam, "intensity_au": I}).to_csv(CONFIG["output_csv"], index=False)

    if CONFIG.get("plot_result", True):
        plt.figure()
        plt.plot(lam, I, lw=1.0)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (a.u.)")
        plt.title("Onion-layer Aluminum LIBS (inverted-parabola T/ne)")
        plt.tight_layout()
        if CONFIG.get("save_plot_png"):
            plt.savefig(cfg["save_plot_png"], dpi=150)
        plt.show()

if __name__ == "__main__":
    cfg = CONFIG
    main(cfg)