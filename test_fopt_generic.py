
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
start_time = time.time()

# Find the absolute path to the src directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')

# Add this path to the start of sys.path if it's not already there
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from temperatures import find_T_min, find_T_max, refine_Tmin, R_sepH
from espinosa import Vt_vec
from utils import interpolation_narrow
from temperatures import compute_logP_f, N_bubblesH, R_sepH, R0, compute_Gamma_f, R_meanH
from GWparams import cs2, alpha_th_bar, beta, GW_SuperCooled
from dof_interpolation import g_rho_spline
from model import model_no_quantum

from pbh import FKSCollapse
from external_constants import *

# Helper functions
def is_increasing(arr):
    return np.all(arr[:-1] <= arr[1:])

def rho_rad(T):
    g_rad = g_rho_spline(T) # + dark_DoF?
    return np.pi**2 / 30 * g_rad * T**4

def H_rad(T):
    M_PL = 1.221e19
    return np.sqrt(8*np.pi/3/M_PL**2 * rho_rad(T)) # using G = 1/M_Pl^2


# Default params
# FKS benchmark
params1 = {
    'a': 0.0626,
    'lam': 0.275,
    'c': 0.1052,
    'd': 2.725,
    'vev': 1.0,
}

# Random scan point
params2 = {
    'a': 0.02389641372958019,
    'lam': 0.45560461320509543,
    'c': 1.4477345640951926e-05,
    'd': 0.710259204962238,
    'vev': 0.1,
}
#{"vev": 0.01, "a": 0.07096676838168296, "d": 0.8625353201731807, "c": 6.782313922284097e-05, "lambda": 0.07052424336983912, "Tstar": 0.0018374099208954332, "Tc": 0.0018921358977234673, "SeByTstar": 162.863735833008, "alpha": 0.15362157752019864, "betaByHstar": 9450.163272049394, "v_wall": 0.31074543656842735, "f_peak": 1.1088512675095657e-05, "h2Omega": 9.324412623498865e-16, "NormedCollapseTime": 1.3566654836248906e-22, "MPBH": 1.9517426608599386e+47, "fPBH": 0.0},

params3 = {
    'a': 0.07096676838168296,
    'lam': 0.07052424336983912,
    'c': 6.782313922284097e-05,
    'd': 0.8625353201731807,
    'vev': 0.01
}

point = params3

T0sq = (point['lam'] * point['vev']**2 - 3*point['c']*point['vev'])/(2*point['d'])
Tc = (point['c']*point['a'] + np.sqrt(point['lam']*point['d']*(point['c']**2 + (point['lam']*point['d'] - point['a']**2)*T0sq)))/(point['lam']*point['d'] - point['a']**2)
phi_critical = Tc * (2*(point['a'] + point['c']/Tc)/point['lam'])
wall_tension = np.power(phi_critical, 3) * np.power(point['lam']/2, 0.5) / 6

##################################
# Initialize Effective Potential #
##################################

# write to log file
with open('fopt_generic.log', 'w') as f:
    f.write("--------------------------------------\n")
    f.write("Starting FOPT calculation\n")
    f.write("-------------------------------------\n")

    vev, units = point['vev'], 'GeV'
    f.write(f"vev: {vev}, units: {units}\n")
    f.write(f"phi_critical: {phi_critical}\n")
    f.write(f"wall_tension: {wall_tension}\n")
    f.write(f"T0sq: {T0sq}\n")
    f.write(f"Tc: {Tc}\n")


    model = model_no_quantum(a=point['a'], lam=point['lam'], c=point['c'], d=point['d'], vev=point['vev'])

    V = model.DVtot # This is the scalar potential shifted such that the false vacuum is located at ϕ = 0 for each value of the temperature
    dV = model.gradV # This is the gradient of the scalar potential

    # Computation of P_f requires a domain wall speed.
    point['v_w'] = 1.0
    v_w = point['v_w']

    #########################
    # Bounding Temperatures #
    #########################
    T_max, vevs_max, max_min_vals, _ = find_T_max(V, dV, precision= 1e-5, Phimax = 3*vev, step_phi = vev * 1e-3, tmax=2.5 * vev)
    T_min, vevs_min, _ = find_T_min(V, dV, tmax=T_max, precision = 1e-5, Phimax = 3*vev, step_phi = vev * 1e-3, max_min_vals = max_min_vals)
    
    if T_max is not None and T_min is not None:
        maxvev = np.max(np.concatenate((vevs_max, vevs_min)))
    elif T_max is not None:
        maxvev = np.max(vevs_max)
    elif T_min is not None:
        maxvev = np.max(vevs_min)
    else:
        maxvev = None
    
    T_min = refine_Tmin(T_min, V, dV, maxvev, log_10_precision = 7) if T_min is not None else None
    if T_min == 0.0:
        T_min = 1e-7

    point['T_max'] = T_max
    point['T_min'] = T_min

    f.write(f"T_max: {T_max}, T_min: {T_min}\n")
    f.write(f"vevs_max: {maxvev}\n")

    true_vev = {}
    S3overT = {}
    V_min_value = {}
    phi0_min = {}
    false_vev = {}
    def action_over_T(T, c_step_phi = 1e-3, precision = 1e-3):
        instance = Vt_vec(T, V, dV, step_phi = c_step_phi, precision = precision, vev0 = maxvev, ratio_vev_step0=50)
        if instance.barrier:
            true_vev[T] = instance.true_min
            false_vev[T] = instance.phi_original_false_vev
            S3overT[T] = instance.action_over_T
            V_min_value[T] = instance.min_V
            phi0_min[T] = instance.phi0_min
            return instance.action_over_T
        else:
            return None
    
    n_points = 100
    temperatures = np.linspace(T_min, T_max, n_points)
    action_vec = np.vectorize(action_over_T)


    action_vec(temperatures)

    temperatures = np.array([T for T in temperatures if T in S3overT])

    plt.plot(temperatures, action_vec(temperatures))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Temperature')
    plt.ylabel('S3/T')
    plt.show()

    # Nucleation, Percolation, Completion
    is_physical = True
    counter = 0
    while counter <= 1:
        logP_f, Temps, ratio_V, Gamma, H = compute_logP_f(model, V_min_value, S3overT, v_w, units = units, cum_method= 'None')
        Gamma_f_list, Temps_G, ratio_V, Gamma_list, H = compute_Gamma_f(model, V_min_value, S3overT, v_w, logP_f, units='GeV', cum_method='cumulative_simpson')

        logP_f = np.nan_to_num(logP_f)
        Gamma_f_list = np.nan_to_num(Gamma_f_list)
        plt.plot(Temps, -logP_f)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Temperature')
        plt.ylabel('-logP_f')
        plt.show()

        plt.plot(Temps, Gamma_f_list)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Temperature')
        plt.ylabel('Gamma_f')
        plt.show()

        RH, R = R_sepH(Temps, Gamma, logP_f, H, ratio_V)
        nH = N_bubblesH(Temps, Gamma, logP_f, H, ratio_V)
        logP_t = np.log(1 - np.exp(logP_f))
        Nf_list = N_bubblesH(Temps, Gamma_f_list, logP_t, H, ratio_V)
        
        mask_nH = ~np.isnan(nH)
        T_nuc = interpolation_narrow(np.log(nH[mask_nH]), Temps[mask_nH], 0)
        mask_Pf = ~np.isnan(logP_f)
        T_perc = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf], np.log(0.71))
        T_perc_false = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf], np.log(0.29))
        T_completion = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf], np.log(0.01))
        idx_compl = np.max([np.argmin(np.abs(Temps - T_completion)), 1])
        test_completion = np.array([logP_f[idx_compl - 1], logP_f[idx_compl], logP_f[idx_compl + 1]])
        test_completion = test_completion[~np.isnan(test_completion)]
        if not is_increasing(test_completion):
            T_completion = np.nan
        if counter == 1:
            d_dT_logP_f = np.gradient(logP_f, Temps)
            log_at_T_perc = interpolation_narrow(Temps, d_dT_logP_f, T_perc)
            ratio_V_at_T_perc = interpolation_narrow(Temps, ratio_V, T_perc)
            log_at_T_completion = interpolation_narrow(Temps, d_dT_logP_f, T_completion)
            if ratio_V_at_T_perc > log_at_T_perc:
                is_physical = False
                print("\n *** The physical volume at percolation is not decreasing. The production of GW is questionable ***")
        counter += 1

    milestones = [T_max, T_nuc, T_perc, T_completion, T_min]
    milestones = [milestone for milestone in milestones if milestone is not None and not np.isnan(milestone)]
    action_vec(milestones)


    # update point
    point['T_nuc'] = T_nuc
    point['T_perc'] = T_perc
    point['T_perc_false'] = T_perc_false
    point['T_completion'] = T_completion
    f.write(f"T_nuc: {T_nuc}, T_perc: {T_perc}, T_perc_false: {T_perc_false}, T_completion: {T_completion}\n")

    #########################
    # False-Vacuum Remnants #
    #########################
    mask_Gf = ~np.isnan(Gamma_f_list)
    Gammaf_perc = interpolation_narrow(Temps[mask_Gf], Gamma_f_list[mask_Gf], T_perc)
    print("GAMMAF_PERC", Gammaf_perc)
    Nf_perc = interpolation_narrow(Temps[mask_Gf], Nf_list[mask_Gf], T_perc)
    H_perc = interpolation_narrow(Temps[mask_Gf], H[mask_Gf], T_perc)
    Hubble_vol_perc = 4*np.pi / 3 * H_perc**(-3)
    nf_perc = Nf_perc / Hubble_vol_perc # number density of false-vacuum bubbles

    # Same quantities, but evaluated when P_f(T_perc_false) = 0.29
    Gammaf_perc_false = interpolation_narrow(Temps[mask_Gf], Gamma_f_list[mask_Gf], T_perc_false)
    Nf_perc_false = interpolation_narrow(Temps[mask_Gf], Nf_list[mask_Gf], T_perc_false)
    H_perc_false = interpolation_narrow(Temps[mask_Gf], H[mask_Gf], T_perc_false)
    Hubble_vol_perc_false = 4*np.pi / 3 * H_perc_false**(-3)
    nf_perc_false = Nf_perc_false / Hubble_vol_perc_false # number density of false-vacuum bubbles

    # update point
    point['Gammaf_perc'] = Gammaf_perc
    point['Nf_perc'] = Nf_perc
    point['H_perc'] = H_perc
    point['nf_perc'] = nf_perc

    point['Gammaf_perc_false'] = Gammaf_perc_false
    point['Nf_perc_false'] = Nf_perc_false
    point['H_perc_false'] = H_perc_false
    point['nf_perc_false'] = nf_perc_false

    f.write(f"Gammaf_perc: {Gammaf_perc}, Nf_perc: {Nf_perc}, H_perc: {H_perc}, nf_perc: {nf_perc}\n")
    f.write(f"Gammaf_perc_false: {Gammaf_perc_false}, Nf_perc_false: {Nf_perc_false}, H_perc_false: {H_perc_false}, nf_perc_false: {nf_perc_false}\n")

    # FKS https://arxiv.org/pdf/2402.13341 eq. (61)
    R_perc = interpolation_narrow(Temps, R, T_perc)
    logP_f_R, Temps_, _, _, _ = compute_logP_f(model, V_min_value, S3overT, v_w = point['v_w'], units = units, cum_method= 'None', R_0=R_perc)
    mask_PfR = ~np.isnan(logP_f_R)
    logPf_Rperc = interpolation_narrow(Temps_[mask_PfR], logP_f_R[mask_PfR], T_perc)
    point['P_surv'] = np.exp(logPf_Rperc)

    f.write(f"P_surv: {point['P_surv']}\n")
    f.write(f"R_perc: {R_perc}\n")


    # Compute abundance of PBHs
    f.write(f"DeltaV: {V_min_value[T_perc]}\n")
    fks = FKSCollapse(deltaV=abs(V_min_value[T_perc]), sigma=wall_tension, vw=v_w)
    pbh_forms = fks.does_pbh_form(R_perc)
    f.write(f"pbh_forms: {pbh_forms}\n")

    # PBH mass
    m_pbh = fks.M0(R_perc)
    f.write(f"m_pbh: {m_pbh}\n")

    # PBH abundance
    normalization = np.power(HUBBLE0, 3) / (4 * np.pi * RHO_CRIT_GEV4 * OMEGA_DM / 3)
    abundance = normalization * m_pbh * point['P_surv'] * Nf_perc
    f.write(f"abundance: {abundance}\n")