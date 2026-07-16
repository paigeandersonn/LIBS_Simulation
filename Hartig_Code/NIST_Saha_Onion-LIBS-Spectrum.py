# -*- coding: utf-8 -*-
"""
Onion-layer Aluminum LIBS (NIST ASD + Saha + Voigt + pointwise SA)

- Reads NIST ASD CSV (Observed wavelength, nm)
- 10–20 "onion" layers with anchored profile shapes for T, ne, n_plasma
- Saha (I <-> II) ion split per layer (cm^-3 convention for densities)
- Voigt line shapes: Doppler ⊕ instrument (Gaussian, in quadrature) ⊕ Stark (Lorentz)
- Self-absorption: physics-based, pointwise; τ0 from f_ik, gi, Ei, and Voigt center in frequency
- Layer-by-layer radiative transfer (emergent from each slab added to output)
"""

# ==============================
# CONFIG
# ==============================
CONFIG = {
    # Wavelength grid (nm)
    "lambda_min_nm": 300.0,
    "lambda_max_nm": 410.0,
    "delta_lambda_nm": 0.002,

    # Layers
    "num_layers": 20,
    "total_path_length_cm": 0.05,     # LOS thickness [cm]

    # Core/ambient state (densities in cm^-3)
    "core_T_K": 9000.0,
    "core_ne_cm3": 5.0e16,
    "core_nplasma_cm3": 1.0e17,
    "ambient_T_K": 500.0,
    "ambient_ne_cm3": 2.0e15,
    "ambient_nplasma_cm3": 1.0e15,

    # Composition (by number)
    "composition": {"Al": 1.0},

    # Saha (I <-> II)
    "use_saha": True,
    "delta_chi_eV": 0.0,              # IP lowering (optional)
    "partition_functions": {},        # e.g., {"Al I": 2.0, "Al II": 1.0}
    "ion_fractions": {("Al","I"): 0.6, ("Al","II"): 0.4},  # used only if use_saha=False

    # Broadening
    "instrument_fwhm_nm": 0.04,       # instrument Gaussian FWHM [nm]
    "include_doppler": True,
    "include_stark": True,
    "estimate_missing_stark": False,  # if True, estimate w16 when CSV lacks it

    # Self-absorption (slab)
    "enable_self_absorption": True,
    "optical_path_length_cm": 0.05,   # [cm] used in τ0 = n_i σ_ν0 L (converted to m)

    # Profile shapes (anchored)
    "profile_type_T":  "supergauss",  # "parabola"|"gauss"|"supergauss"|"raisedcos"|"logistic"
    "profile_type_ne": "raisedcos",
    "profile_type_np": "raisedcos",
    "parabola_power": 2.0,
    "gauss_sigma_frac": 0.35,
    "super_m": 3.0,
    "super_sigma_frac": 0.35,
    "raisedcore_frac": 0.10,
    "raisededge_frac": 0.30,
    "logistic_xhalf_frac": 0.30,
    "logistic_width_frac": 0.08,

    # NIST CSV (Observed wavelength)
    "use_nist_file": True,
    "nist_file_path": r"nist_al_cache.csv",  # <-- set to your NIST ASD CSV path
    "write_example_lines_csv": False,

    # Outputs
    "normalize_to_max": True,
    "plot_result": True,
    "save_plot_png": "onion_aluminum_spectrum.png",
    "save_profiles_png": "onion_aluminum_profiles.png",
    "output_csv": "onion_aluminum_spectrum.csv",
    "print_diagnostics": True
}

# ==============================
# Imports & constants
# ==============================
import numpy as np
import pandas as pd
import math, re
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

# Aluminum data
IONIZATION_POTENTIAL_EV = {"Al": 5.98577}   # Al -> Al+
MASS_AMU = {"Al": 26.98}

# ==============================
# Helpers
# ==============================
def parse_species(s):
    s = str(s).strip()
    if " " in s:
        el, st = s.split()
        return el.strip(), st.strip().upper()
    return s, "I"

def get_partition_function(species, T_K, Ucfg):
    U = Ucfg.get(species, 1.0)
    return float(U(T_K)) if callable(U) else float(U)

def saha_first_ion_fractions(element, T_K, ne_cm3, U_I=1.0, U_II=1.0, delta_chi_eV=0.0):
    """Return (f_I, f_II) using Saha for first ionization; all densities in cm^-3."""
    chi_eV = IONIZATION_POTENTIAL_EV.get(element, 7.0)
    kT_eV  = (K_B*T_K)/E_CHARGE
    # S in m^-3 → convert to cm^-3 (divide by 1e6) to stay consistent with ne_cm3
    S = ((2.0*PI*M_E*K_B*T_K)/(H*H))**1.5 / 1e6
    rhs = S * 2.0 * (U_II/max(U_I,1e-30)) * math.exp(-(chi_eV - delta_chi_eV)/max(kT_eV,1e-30))
    R   = rhs / max(ne_cm3,1e-30)   # n(II)/n(I)
    fII = R/(1.0+R); fI = 1.0 - fII
    return max(0.0,min(1.0,fI)), max(0.0,min(1.0,fII))

def doppler_fwhm_nm(lambda_nm, T_K, mass_amu):
    lam = float(lambda_nm)
    M   = max(mass_amu, 1e-6)
    return 7.16e-7 * lam * math.sqrt(T_K / M)

def effective_gaussian_fwhm_nm(dop_nm, inst_nm):
    return math.sqrt(max(dop_nm,0.0)**2 + max(inst_nm,0.0)**2)

def stark_fwhm_nm(ne_cm3, w_nm_at_1e16):
    if (w_nm_at_1e16 is None) or (w_nm_at_1e16 <= 0):
        return 0.0
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
    if sigma<=0:
        prof = gamma/PI/(x**2 + gamma**2)
        area = np.trapz(prof, lam_nm_grid)
        if area>0: prof/=area
        return prof
    V = np.real(wofz((x + 1j*gamma)/(sigma*math.sqrt(2.0))))/(sigma*math.sqrt(2.0*PI))
    area = np.trapz(V, lam_nm_grid)
    if area>0: V/=area
    return V

def voigt_center_freq(fwhm_g_Hz, fwhm_l_Hz):
    """Area-normalized Voigt at line center in frequency (1/Hz)."""
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

# ==============================
# Anchored layer profiles
# ==============================
def shape_profile(x, L, Xcore, Xamb, kind, cfg):
    """Anchored shapes: X(0)=Xcore, X(L)=Xamb."""
    kind = (kind or "parabola").lower()
    if L <= 0.0: return float(Xcore)
    s = max(0.0, min(1.0, x/L))  # 0..1

    if kind == "supergauss":
        m      = float(cfg.get("super_m", 4.0))
        sigmaf = float(cfg.get("super_sigma_frac", 0.35))
        sigma  = max(sigmaf, 1e-12)
        w  = math.exp(-(s/sigma)**m)
        w1 = math.exp(-(1.0/sigma)**m)
        wA = (w - w1)/max(1.0 - w1, 1e-12)
        return Xamb + (Xcore - Xamb)*wA

    if kind == "gauss":
        sigmaf = float(cfg.get("gauss_sigma_frac", 0.35))
        sigma  = max(sigmaf, 1e-12)
        w  = math.exp(-(s/sigma)**2)
        w1 = math.exp(-(1.0/sigma)**2)
        wA = (w - w1)/max(1.0 - w1, 1e-12)
        return Xamb + (Xcore - Xamb)*wA

    if kind == "raisedcos":
        L0 = float(cfg.get("raisedcore_frac", 0.5))*L
        d  = float(cfg.get("raisededge_frac", 0.2))*L
        if s*L <= L0: w = 1.0
        elif s*L >= L0+d: w = 0.0
        else:
            xi = (s*L - L0)/max(d,1e-12)
            w  = 0.5*(1.0 + math.cos(math.pi*xi))
        return Xamb + (Xcore - Xamb)*w

    if kind == "logistic":
        xh = float(cfg.get("logistic_xhalf_frac", 0.7))*L
        w  = float(cfg.get("logistic_width_frac", 0.08))*L
        y  = 1.0/(1.0 + math.exp((s*L - xh)/max(w,1e-12)))
        return Xamb + (Xcore - Xamb)*y

    # default parabola
    p = float(cfg.get("parabola_power", 2.0))
    return Xamb + (Xcore - Xamb)*max(0.0, 1.0 - s**p)

def build_layers_with_profiles(cfg):
    """Build N layers, thickness dz=Ltot/N, anchored T/ne/n_plasma."""
    N    = int(cfg.get("num_layers", 10))
    Ltot = float(cfg["total_path_length_cm"])
    dz   = Ltot/max(N,1)
    kT, kne, knp = cfg.get("profile_type_T","parabola"), cfg.get("profile_type_ne","parabola"), cfg.get("profile_type_np","parabola")
    layers = []
    for i in range(N):
        x = i*dz
        T_i  = shape_profile(x, Ltot, cfg["core_T_K"],         cfg["ambient_T_K"],         kT,  cfg)
        ne_i = shape_profile(x, Ltot, cfg["core_ne_cm3"],      cfg["ambient_ne_cm3"],      kne, cfg)
        np_i = shape_profile(x, Ltot, cfg["core_nplasma_cm3"], cfg["ambient_nplasma_cm3"], knp, cfg)
        layers.append({"T_K": T_i, "ne_cm3": ne_i, "n_plasma_cm3": np_i, "thickness_cm": dz})
    if cfg.get("print_diagnostics", False):
        print(f"[layers] T_core={layers[0]['T_K']:.0f} K, T_edge={layers[-1]['T_K']:.0f} K (amb={cfg['ambient_T_K']:.0f})")
        print(f"[layers] ne_core={layers[0]['ne_cm3']:.3e}, ne_edge={layers[-1]['ne_cm3']:.3e} (amb={cfg['ambient_ne_cm3']:.3e})")
        print(f"[layers] np_core={layers[0]['n_plasma_cm3']:.3e}, np_edge={layers[-1]['n_plasma_cm3']:.3e} (amb={cfg['ambient_nplasma_cm3']:.3e})")
    return layers

# ==============================
# NIST CSV (Observed wavelength in nm)
# ==============================
def _num_cell(x):
    if x is None: return np.nan
    if isinstance(x,(int,float,np.floating)): return float(x)
    s = str(x).strip()
    if s.startswith('="') and s.endswith('"'): s = s[2:-1]
    s = s.replace(',', '').replace('×','x')
    try: return float(s)
    except: pass
    if 'x10' in s:
        try:
            base, exp = s.split('x10'); base = float(base); exp = float(exp.replace('^',''))
            return base*(10.0**exp)
        except: pass
    m = re.search(r'[-+]?\d+(\.\d+)?([eE][-+]?\d+)?', s)
    return float(m.group(0)) if m else np.nan

def load_nist_csv_observed(path, cfg):
    """
    Read a NIST ASD CSV and map it to the columns your model expects:
      species, wavelength_nm, Aki_s, Ek_eV, Ei_eV, gk, gi, mass_amu, stark_w_nm_at_1e16, log_gf
    Priorities for g-factors:
      1) use gk/gi columns if present,
      2) else compute from J_upper/J_lower: g = 2J+1,
      3) else parse level/term strings to extract J,
      4) else fall back to 1.0.
    Uses Observed wavelength in nm (converts Å→nm if needed).
    """
    import re
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = list(df.columns)

    def find(*cands):
        for c in cands:
            for cn in cols:
                if c.lower() in cn.lower():
                    return cn
        return None

    def _num_cell(x):
        if x is None: return np.nan
        if isinstance(x,(int,float,np.floating)): return float(x)
        s = str(x).strip()
        if s.startswith('="') and s.endswith('"'): s = s[2:-1]
        s = s.replace(',', '').replace('×','x')
        try: return float(s)
        except: pass
        if 'x10' in s:
            try:
                base, exp = s.split('x10'); base=float(base); exp=float(exp.replace('^',''))
                return base*(10.0**exp)
            except: pass
        m = re.search(r'[-+]?\d+(\.\d+)?([eE][-+]?\d+)?', s)
        return float(m.group(0)) if m else np.nan

    # --- Species (Spectrum or element+stage) ---
    col_spec = find('spectrum','species')
    col_el   = find('element')
    col_spn  = find('sp_num','spectrum number','stage','ion')
    if col_spec:
        species = df[col_spec].astype(str).str.strip()
    else:
        if not col_el: raise ValueError("NIST CSV missing 'Spectrum/Species' and 'element'.")
        el = df[col_el].astype(str).str.strip().str.capitalize()
        if col_spn:
            spn = df[col_spn].apply(_num_cell).astype(float)
            roman = {1:'I',2:'II',3:'III',4:'IV',5:'V'}
            st = spn.apply(lambda v: roman.get(int(v),'I') if not math.isnan(v) else 'I')
        else:
            st = 'I'
        species = el + ' ' + st

    # --- Observed wavelength (nm) ---
    col_wl = find('obs_wl','observed','wavelength')
    if not col_wl: raise ValueError("NIST CSV missing Observed Wavelength column.")
    wl = df[col_wl].apply(_num_cell).astype(float)
    wl_nm = wl/10.0 if np.nanmedian(wl) > 1000 else wl

    # --- Einstein A ---
    col_A = find('A (s-1)','aki','aki(s^-1)')
    Aki = df[col_A].apply(_num_cell).astype(float) if col_A else pd.Series(np.nan, index=df.index)

    # --- Energies (cm^-1 -> eV if present) ---
    def cm1_to_eV(series):
        vals = series.apply(_num_cell).astype(float)
        return (H*C*vals*100.0)/E_CHARGE
    col_Ek = find('Ek','E upper','upper'); Ek_eV = cm1_to_eV(df[col_Ek]) if col_Ek else pd.Series(np.nan,index=df.index)
    col_Ei = find('Ei','E lower','lower'); Ei_eV = cm1_to_eV(df[col_Ei]) if col_Ei else pd.Series(np.nan,index=df.index)

    # --- Degeneracies: prefer direct gk/gi; else from Ju/Jl; else from term strings; else 1.0 ---
    col_gk = find('g_k','gu','upper g','g_upper')
    col_gi = find('g_i','gl','lower g')
    gk = df[col_gk].apply(_num_cell).astype(float) if col_gk else pd.Series(np.nan,index=df.index)
    gi = df[col_gi].apply(_num_cell).astype(float) if col_gi else pd.Series(np.nan,index=df.index)

    # If missing, try J columns
    def _j_from(cname):
        return df[cname].apply(_num_cell).astype(float)
    if gk.isna().any():
        col_Ju = find('J upper','J_upper','J up','J_u','Ju')
        if col_Ju:
            Ju = _j_from(col_Ju)
            gk = gk.where(gk.notna(), 2.0*Ju + 1.0)
    if gi.isna().any():
        col_Jl = find('J lower','J_lower','J low','J_l','Jl')
        if col_Jl:
            Jl = _j_from(col_Jl)
            gi = gi.where(gi.notna(), 2.0*Jl + 1.0)

    # If still missing, parse term/level strings for trailing J or explicit "J=..."
    def _g_from_term(col_term):
        if not col_term: return pd.Series(np.nan, index=df.index)
        s = df[col_term].astype(str)
        # 1) J=...
        mJ = s.str.extract(r'J\s*=\s*([0-9]+(?:\.[0-9]+)?)', expand=False)
        J  = pd.to_numeric(mJ, errors='coerce')
        # 2) fallback: last token numeric (e.g., D_2 or D 2)
        m2 = s.str.extract(r'(?:_|[\s])([0-9]+(?:\.[0-9]+)?)\s*$', expand=False)
        J2 = pd.to_numeric(m2, errors='coerce')
        J  = J.where(J.notna(), J2)
        return 2.0*J + 1.0
    if gk.isna().any() or gi.isna().any():
        col_term_up = find('Upper level','Upper term','Term upper','Upper')
        col_term_lo = find('Lower level','Lower term','Term lower','Lower')
        if gk.isna().any() and col_term_up:
            gk = gk.where(gk.notna(), _g_from_term(col_term_up))
        if gi.isna().any() and col_term_lo:
            gi = gi.where(gi.notna(), _g_from_term(col_term_lo))

    # Final fallback
    gk = gk.fillna(1.0); gi = gi.fillna(1.0)

    # --- Optional fields ---
    col_lg = find('log(gf)','log gf','log_gf'); log_gf = pd.to_numeric(df[col_lg], errors='coerce') if col_lg else pd.Series(np.nan,index=df.index)
    col_stark = find('stark_w_nm_at_1e16','stark width','stark')
    w16 = df[col_stark].apply(_num_cell).astype(float) if col_stark else pd.Series(np.nan,index=df.index)

    # Assemble
    out = pd.DataFrame({
        "species": species,
        "wavelength_nm": wl_nm,
        "Aki_s": Aki,
        "Ek_eV": Ek_eV,
        "Ei_eV": Ei_eV,
        "gk": gk,
        "gi": gi,
        "mass_amu": species.apply(lambda s: MASS_AMU.get(parse_species(s)[0], MASS_AMU["Al"])),
        "stark_w_nm_at_1e16": w16,
        "log_gf": log_gf
    })

    # Filter window; Al I/II only; positive Aki
    lam_min, lam_max = cfg["lambda_min_nm"], cfg["lambda_max_nm"]
    out = out[(out["wavelength_nm"]>=lam_min) & (out["wavelength_nm"]<=lam_max)]
    out = out[pd.to_numeric(out["Aki_s"], errors='coerce') > 0]
    out = out[out["species"].astype(str).str.contains(r"\bAl\s+(?:I|II)\b", case=False, regex=True)]  # non-capturing
    out = out.reset_index(drop=True)

    if cfg.get("print_diagnostics", False) and len(out):
        print(f"[NIST] Lines in {lam_min:.1f}-{lam_max:.1f} nm: {len(out)}")
        print(out[["species","wavelength_nm","Aki_s","Ek_eV","Ei_eV","gk","gi","stark_w_nm_at_1e16"]].to_string(index=False))

    return out

# ==============================
# Main
# ==============================
def main(cfg):
    lam = np.arange(cfg["lambda_min_nm"], cfg["lambda_max_nm"] + 0.5*cfg["delta_lambda_nm"], cfg["delta_lambda_nm"])
    layers = build_layers_with_profiles(cfg)

    # Plot profiles (T, ne, n_plasma)
    if cfg.get("plot_result", True):
        pos_cm = np.linspace(0, cfg["total_path_length_cm"], len(layers))
        T_list  = [L["T_K"] for L in layers]
        ne_list = [L["ne_cm3"] for L in layers]
        np_list = [L["n_plasma_cm3"] for L in layers]
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Line-of-sight position (cm)")
        ax1.set_ylabel("Temperature (K)", color="tab:red")
        ax1.plot(pos_cm, T_list, "o-", color="tab:red", label="T")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Density (cm$^{-3}$)", color="tab:blue")
        ax2.semilogy(pos_cm, ne_list, "s--", color="tab:blue", label="nₑ")
        ax2.semilogy(pos_cm, np_list, "d-.", color="tab:green", label="n_plasma")
        ax1.set_title("Onion-layer plasma profiles (Al)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
        plt.tight_layout()
        if cfg.get("save_profiles_png"): plt.savefig(cfg["save_profiles_png"], dpi=150)
        plt.show()

    # Load lines (NIST or fallback)
    if cfg.get("use_nist_file", True):
        try:
            lines = load_nist_csv_observed(cfg["nist_file_path"], cfg)
        except Exception as e:
            print(f"[WARN] NIST CSV load failed: {e}\nUsing demo Al lines.")
            lines = None
    else:
        lines = None

    if lines is None or len(lines) == 0:
        lines = pd.DataFrame({
            "species": ["Al I","Al I","Al II"],
            "wavelength_nm": [394.401, 396.152, 358.657],
            "Aki_s": [4.9e7, 9.9e7, 3.0e7],
            "Ek_eV": [3.14, 3.14, 6.37],
            "Ei_eV": [0.0, 0.0, 0.0],
            "gk": [2,4,4],
            "gi": [2,2,2],
            "mass_amu": [MASS_AMU["Al"]]*3,
            "stark_w_nm_at_1e16": [0.015, 0.020, 0.010],
            "log_gf": [np.nan, np.nan, np.nan]
        })

    # Build emergent spectrum
    I = np.zeros_like(lam, dtype=float)
    for idx, layer in enumerate(layers):
        T_K  = float(layer["T_K"])
        ne   = float(layer["ne_cm3"])       # cm^-3
        npl  = float(layer["n_plasma_cm3"]) # cm^-3
        L_cm = float(layer["thickness_cm"]); L_m = L_cm*1e-2
        Ucfg = cfg.get("partition_functions", {})

        # Saha ion split
        species_ntot = {}
        for el, frac in cfg["composition"].items():
            ntot_el = frac*npl
            if cfg.get("use_saha", True):
                U_I  = get_partition_function(f"{el} I",  T_K, Ucfg)
                U_II = get_partition_function(f"{el} II", T_K, Ucfg)
                fI, fII = saha_first_ion_fractions(el, T_K, ne, U_I=U_I, U_II=U_II, delta_chi_eV=cfg.get("delta_chi_eV",0.0))
                species_ntot[f"{el} I"]  = ntot_el*fI
                species_ntot[f"{el} II"] = ntot_el*fII
            else:
                species_ntot[f"{el} I"]  = ntot_el*cfg["ion_fractions"].get((el,"I"),0.0)
                species_ntot[f"{el} II"] = ntot_el*cfg["ion_fractions"].get((el,"II"),0.0)

        J_layer = np.zeros_like(lam, dtype=float)

        for _, row in lines.iterrows():
            sp = str(row["species"]).strip()
            if species_ntot.get(sp,0.0) <= 0.0: continue

            lam0_nm = float(row["wavelength_nm"])
            if not (cfg["lambda_min_nm"] <= lam0_nm <= cfg["lambda_max_nm"]): continue
            lam0_m  = lam0_nm*1e-9

            Aki   = float(row["Aki_s"])
            Ek_eV = float(row.get("Ek_eV", np.nan)) if not pd.isna(row.get("Ek_eV", np.nan)) else 0.0
            Ei_eV = float(row.get("Ei_eV", np.nan)) if not pd.isna(row.get("Ei_eV", np.nan)) else 0.0
            gk    = float(row.get("gk", np.nan)) if not pd.isna(row.get("gk", np.nan)) and row["gk"]>0 else 1.0
            gi    = float(row.get("gi", np.nan)) if not pd.isna(row.get("gi", np.nan)) and row["gi"]>0 else 1.0
            mass  = float(row.get("mass_amu", np.nan)) if not pd.isna(row.get("mass_amu", np.nan)) else MASS_AMU.get(parse_species(sp)[0], 26.98)
            w16   = float(row.get("stark_w_nm_at_1e16", np.nan)) if not pd.isna(row.get("stark_w_nm_at_1e16", np.nan)) else 0.0
            if (w16<=0.0) and cfg.get("estimate_missing_stark", False):
                st = parse_species(sp)[1]
                w16 = (0.015 if st=="I" else 0.010) * (lam0_nm/400.0)**2

            # LTE upper population → line strength
            U = get_partition_function(sp, T_K, Ucfg)
            kT_eV = (K_B*T_K)/E_CHARGE
            n_upper = species_ntot[sp]*(gk/max(U,1e-30))*math.exp(-max(Ek_eV,0.0)/max(kT_eV,1.0e-30))
            S_em = n_upper*Aki

            # Broadening (display profile includes instrument)
            dop_nm = doppler_fwhm_nm(lam0_nm, T_K, mass) if CONFIG.get("include_doppler", True) else 0.0
            inst_nm= float(CONFIG.get("instrument_fwhm_nm", 0.0))
            fwhm_g = effective_gaussian_fwhm_nm(dop_nm, inst_nm)
            fwhm_l = stark_fwhm_nm(ne, w16) if CONFIG.get("include_stark", True) else 0.0
            prof = voigt_profile_nm(lam, lam0_nm, fwhm_g, fwhm_l)
            J_thin = S_em * prof

            # Self-absorption (use physical widths for φν(ν0))
            R_lambda = 1.0
            if CONFIG.get("enable_self_absorption", True) and cfg.get("optical_path_length_cm", 0.0) > 0:
                # f_ik from log(gf) if available; else from Aki
                log_gf = row.get("log_gf", np.nan)
                if not pd.isna(log_gf) and gi>0:
                    f_ik = (10.0**float(log_gf))/gi
                else:
                    f_ik = f_from_Aki_lambda_gi_gk(Aki, lam0_nm, gi, gk)

                dop_nm_phys = dop_nm                 # Doppler only
                nm_to_Hz    = C/(lam0_m**2) * 1e-9
                phi0 = voigt_center_freq(dop_nm_phys*nm_to_Hz, fwhm_l*nm_to_Hz)

                tau0 = np.nan
                if f_ik and f_ik>0 and phi0>0:
                    n_i_cm3 = species_ntot[sp]*(gi/max(U,1.0e-30))*math.exp(-max(Ei_eV,0.0)/max(kT_eV,1.0e-30))
                    if n_i_cm3>0:
                        sigma0 = (PI*(E_CHARGE**2)/(EPS0*M_E*C))*f_ik*phi0  # m^2
                        n_i_m3 = n_i_cm3*1.0e6
                        L_m = cfg["optical_path_length_cm"]*1.0e-2
                        tau0 = n_i_m3*sigma0*L_m

                if tau0 and tau0>0:
                    peak = np.max(prof) if np.max(prof)>0 else 1.0
                    tl = tau0*(prof/peak)
                    R_lambda = np.ones_like(tl)
                    msk = tl>1e-9
                    R_lambda[msk] = (1.0 - np.exp(-tl[msk]))/tl[msk]
                else:
                    R_lambda = 1.0

            J_layer += J_thin*R_lambda

        I += J_layer

    # Normalize & save
    if CONFIG.get("normalize_to_max", True):
        m = np.max(I) if I.size else 1.0
        if m>0: I = I/m

    pd.DataFrame({"wavelength_nm": lam, "intensity_au": I}).to_csv(CONFIG["output_csv"], index=False)

    if cfg.get("plot_result", True):
        plt.figure()
        plt.plot(lam, I, lw=1.0)
        plt.xlabel("Wavelength (nm)"); plt.ylabel("Intensity (a.u.)")
        plt.title("Onion-layer Aluminum LIBS (NIST ASD + Saha + SA)")
        plt.tight_layout()
        if cfg.get("save_plot_png"): plt.savefig(cfg["save_plot_png"], dpi=150)
        plt.show()

# Entry
if __name__ == "__main__":
    main(CONFIG)