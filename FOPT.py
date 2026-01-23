# Calculates relevant data for a given FOPT.

import numpy as np
import sys
import os
import time
start_time = time.time()

# Find the absolute path to the ELENA/src directory
# This assumes we're in the 'ELENA' folder
src_path = os.path.abspath('ELENA/src')

# Add this path to the start of sys.path if it's not already there
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from temperatures import find_T_min, find_T_max, refine_Tmin, R_sepH
from espinosa import Vt_vec
from utils import interpolation_narrow
from temperatures import compute_logP_f, N_bubblesH, R_sepH, R0, compute_Gamma_f, R_meanH
from GWparams import cs2, alpha_th_bar, beta, GW_SuperCooled
from BL_model import BL_model
from dof_interpolation import g_rho_spline

##################################
# Initialize Effective Potential #
##################################
point = {
    "alpha_BL0": 0.025136634430275393,
    "alpha_Y0": 0.028781817487679596,
    "mu": 0.001
}

vev, units = point['mu'], 'GeV'

params = {
    'alpha_BL0': point['alpha_BL0'],
    'alpha_Y0': point['alpha_Y0'],
    'mu': vev
}

model = BL_model(**params)

V = model.DVtot # This is the scalar potential shifted such that the false vacuum is located at ϕ = 0 for each value of the temperature
dV = model.gradV # This is the gradient of the scalar potential

# Computation of P_f requires a domain wall speed.
point['v_w'] = 1
v_w = point['v_w']

#########################
# Bounding Temperatures #
#########################
T_max, vevs_max, max_min_vals, false_min_tmax = find_T_max(V, dV, precision= 1e-2, Phimax = 2*vev, step_phi = vev * 1e-2, tmax=2.5 * vev)
T_min, vevs_min, false_min_tmin = find_T_min(V, dV, tmax=T_max, precision = 1e-2, Phimax = 2*vev, step_phi = vev * 1e-2, max_min_vals = max_min_vals)

if T_max is not None and T_min is not None:
    maxvev = np.max(np.concatenate((vevs_max, vevs_min)))
elif T_max is not None:
    maxvev = np.max(vevs_max)
elif T_min is not None:
    maxvev = np.max(vevs_min)
else:
    maxvev = None

T_min = refine_Tmin(T_min, V, dV, maxvev, log_10_precision = 6) if T_min is not None else None

# update point
point['T_max'] = T_max
point['T_min']  = T_min # potential barrier vanishes

##########################
# Milestone Temperatures #
##########################
# Euclidean Action
true_vev = {}
S3overT = {}
V_min_value = {}
phi0_min = {}
V_exit = {}
false_vev = {}

def action_over_T(T, c_step_phi = 1e-3, precision = 1e-3):
    instance = Vt_vec(T, V, dV, step_phi = c_step_phi, precision = precision, vev0 = maxvev, ratio_vev_step0=50)
    if instance.barrier:
        true_vev[T] = instance.true_min
        false_vev[T] = instance.phi_original_false_vev
        S3overT[T] = instance.action_over_T
        V_min_value[T] = instance.min_V
        phi0_min[T] = instance.phi0_min
        V_exit[T] = instance.V_exit
        return instance.action_over_T
    else:
        return None


n_points = 100
temperatures = np.linspace(T_min, T_max, n_points)
action_vec = np.vectorize(action_over_T)

action_vec(temperatures)

temperatures = np.array([T for T in temperatures if T in S3overT])

# Nucleation, Percolation, Completion
is_physical = True

def is_increasing(arr):
    return np.all(arr[:-1] <= arr[1:])

counter = 0
while counter <= 1:
    if counter == 1:
        temperatures = np.linspace(np.nanmax([T_min, 0.95 * T_completion]), np.nanmin([T_max, 1.05 * T_nuc]), n_points, endpoint = True)
        action_vec(temperatures)
    logP_f, Temps, ratio_V, Gamma, H = compute_logP_f(model, V_min_value, S3overT, v_w, units = units, cum_method= 'None')
    Gamma_f_list, Temps_G, ratio_V, Gamma_list, H = compute_Gamma_f(model, V_min_value, S3overT, v_w, logP_f, units='GeV', cum_method='cumulative_simpson')

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
    print(counter, T_completion, test_completion)
    print(is_increasing(test_completion))
    if not is_increasing(test_completion):
        T_completion = np.nan
    if counter == 1:
        d_dT_logP_f = np.gradient(logP_f, Temps)
        log_at_T_perc = interpolation_narrow(Temps, d_dT_logP_f, T_perc)
        ratio_V_at_T_perc = interpolation_narrow(Temps, ratio_V, T_perc)
        log_at_T_completion = interpolation_narrow(Temps, d_dT_logP_f, T_completion)
        ratio_V_at_T_completion = interpolation_narrow(Temps, ratio_V, T_completion)
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


#########################
# False-Vacuum Remnants #
#########################
mask_Gf = ~np.isnan(Gamma_f_list)
Gammaf_perc = interpolation_narrow(Temps[mask_Gf], Gamma_f_list[mask_Gf], T_perc)
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

# FKS https://arxiv.org/pdf/2402.13341 eq. (61)
R_perc = interpolation_narrow(Temps, R, T_perc)
logP_f_R, Temps_, _, _, _ = compute_logP_f(model, V_min_value, S3overT, v_w = point['v_w'], units = units, cum_method= 'None', R_0=R_perc)
mask_PfR = ~np.isnan(logP_f_R)
logPf_Rperc = interpolation_narrow(Temps_[mask_PfR], logP_f_R[mask_PfR], T_perc)
point['P_surv'] = np.exp(logPf_Rperc)


#################################
# Gravitational Wave Parameters #
#################################
if T_perc is not None:
    action_over_T(T_perc)
    c_s2 = cs2(T_perc, model, true_vev, units = units)[0]

def c_alpha_inf(T, units):
    '''
    "Efficiency parameter" representing the minimal α required to overcome leading-order
    friction in plasma due to changes in field masses.
    '''
    v_true = true_vev[T]
    v_false = false_vev[T]
    Dm2_Zprime = 3 * 4*(model.g(v_true, T)**2*v_true**2 - model.g(v_false, T)**2*v_false**2)
    Dm2_Phi = 3*(model.lam(v_true, T)*v_true**2 - model.lam(v_false, T)*v_false**2)
    Dm2_RHN = 0.5 * 2*0.5*(model.y(v_true, T)**2*v_true**2 - model.y(v_false, T)**2*v_false**2)
    # ignoring Goldstone boson?
    numerator = (Dm2_Zprime + Dm2_Phi + Dm2_RHN) * T**2 / 24
    rho_tot = - T * 3 * (model.dVdT(v_false, T, include_radiation=True, include_SM = True, units = units) ) / 4
    rho_DS = - T * 3 * (model.dVdT(v_false, T, include_radiation=True, include_SM = False, units = units) ) / 4
    return numerator/ rho_tot, numerator / rho_DS

def c_alpha_eq(T, units):
    '''
    If domain walls reach relativistic speeds, next-to-leading-order friction terms
    (proportional to the Lorentz factor) become relevant. This α_eq is used in calculating
    the terminal speed of walls.
    '''
    v_true = true_vev[T]
    v_false = false_vev[T]
    numerator = (3 * (2*(model.g(v_true,T)**3*v_true - model.g(v_false,T)**3*v_false)) * T**3)
    rho_tot = - T * 3 * (model.dVdT(v_false, T, include_radiation=True, include_SM = True, units = units) ) / 4
    rho_DS = - T * 3 * (model.dVdT(v_false, T, include_radiation=True, include_SM = False, units = units) ) / 4
    return numerator / rho_tot, numerator / rho_DS

# Now compute the alphas at percolation. If `alpha < alpha_inf` the bubble nucleation
# is not in runaway regime, implying that the wall speed v_w does not converge asymptotically
# to 1. GWs are still possible in this regime, however the computation of the transition temperatures
# and thermal parameters above is not reliable, since it assumes v_w --> 1. Thus the
# final GW spectrum should not be trust in this regime.
alpha, alpha_DS = alpha_th_bar(T_perc, model, V_min_value, false_vev, true_vev, units = units)
alpha_inf, alpha_inf_DS = c_alpha_inf(T_perc, units)
alpha_eq, alpha_eq_DS = c_alpha_eq(T_perc, units)

alpha, alpha_inf, alpha_eq = alpha[0], alpha_inf[0], alpha_eq[0]

gamma_eq = (alpha - alpha_inf) / alpha_eq

if alpha < alpha_inf:
    is_physical = False
    print("\n*** Warning, the bubble expansion is not in runaway regime! The results of the computation are not reliable. ***")

v_min = 0.99
if gamma_eq < 1 / np.sqrt(1 - v_min**2):
    is_physical = False
    print(f"\n*** Warning, the NLO pressure could prevent the walls to reach relativistic velocities (gamma_eq = {gamma_eq:.2e}). \
          The results of the computation are not reliable as v_w = {v_w} was used. ***")

# Mean Bubble Separation
RH, R = R_sepH(Temps, Gamma, logP_f, H, ratio_V)
RH_interp = interpolation_narrow(Temps, RH, T_perc)
H_star = interpolation_narrow(Temps, H, T_perc)
R_star = RH_interp / H_star
gamma_star = 2 * R_star / (3 * R0(T_perc, S3overT, V_exit))

# Mean False-Vacuum Bubble Separation
RH_false, R_false = R_sepH(Temps, Gamma_f_list, logP_t, H, ratio_V)
RH_interp_false = interpolation_narrow(Temps, RH_false, T_perc_false)
H_star_false = interpolation_narrow(Temps, H, T_perc_false)
R_star_false = RH_interp_false / H_star_false

# Mean False-Vacuum Bubble Radii
R_meanH_false, R_mean_false = R_meanH(Temps, Gamma_f_list, logP_t, H, ratio_V, RH_false, R_0=0.)
R_meanH_interp_false = interpolation_narrow(Temps, R_meanH_false, T_perc_false)
R_mean_false = R_meanH_interp_false / H_star_false


# GW spectrum peak
dark_dof = 2*1 + 1*3 + 1*1 + 1*1 # https://arxiv.org/pdf/1609.04979.pdf
GW = GW_SuperCooled(T_perc, alpha, alpha_inf, alpha_eq, R_star, gamma_star, H_star, np.sqrt(c_s2), v_w, units, dark_dof)
GW_peaks = GW.find_peak(verbose=True)
kappa_col = GW.kappa_col
tau_sw = GW.tau_sw
T_reh = GW.T_reh

# β and γ
logP_f, Temps, ratio_V, Gamma, H = compute_logP_f(model, V_min_value, S3overT, v_w, units = units, cum_method= 'None')
beta_Hn, gamma_Hn, times, Gamma_t, Temps_t, H_t = beta(Temps, ratio_V, Gamma, H, T_nuc, T_perc, verbose = True)




# update point
point['is_physical'] = is_physical
point['c_s2'] = c_s2
point['alpha'] = alpha
point['alpha_inf'] = alpha_inf
point['alpha_eq'] = alpha_eq

point['R_sepH_perc'] = RH_interp
point['H_perc'] = H_star
point['gamma_star'] = gamma_star

point['R_sep_falseH_perc_false'] = RH_interp_false
point['R_mean_falseH_perc_false'] = R_meanH_interp_false
point['H_perc_false'] = H_star_false

point['GW_peak'] = GW_peaks[0]
point['GW_peak_col'] = GW_peaks[1]
point['GW_peak_sw'] = GW_peaks[2]
point['GW_peak_turb'] = GW_peaks[3]
point['kappa_col'] = kappa_col
point['tau_sw'] = tau_sw
point['T_reh'] = T_reh

point['beta_over_H'] = beta_Hn
point['gamma_over_H'] = gamma_Hn


#####################################
# Check for Local Vacuum Domination #
#####################################
# R * H_rad << 1 should hold for PBH formation mechanism
def rho_rad(T):
    g_rad = g_rho_spline(T) # + dark_DoF?
    return np.pi**2 / 30 * g_rad * T**4
def H_rad(T):
    M_PL = 1.221e19
    return np.sqrt(8*np.pi/3/M_PL**2 * rho_rad(T)) # using G = 1/M_Pl^2

# point['R_sepH_rad'] = R_star * H_rad(T_perc)
point['R_mean_falseH_rad'] = R_mean_false * H_rad(T_perc_false)




print(point)
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")