
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




class FOPTGeneric:
    """
    Class that performs a generic FOPT calculation.
    USAGE:
    fopt = FOPTGeneric(point)
    fopt.calc_temperature_bounds()
    fopt.calc_action_over_T()
    fopt.calc_nucleation_percolation_completion()
    fopt.calc_npatches()
    fopt.calc_mean_bubble_size()
    fopt.calc_pbh_abundance()
    """
    def __init__(self, point, verbose = False):
        self.verbose = verbose

        # init Veff params
        self.point = point
        self.a = point['a']
        self.lam = point['lam']
        self.c = point['c']
        self.d = point['d']
        self.vev = point['vev']

        # Derive Veff quantities
        self.T0sq = (self.lam * self.vev**2 - 3*self.c*self.vev)/(2*self.d)
        self.Tc = self.get_Tc()
        self.phi_crit = self.phi_critical()
        self.sigma = self.wall_tension()

        # Init ELENA quantities
        self.units = 'GeV'
        self.model = model_no_quantum(a=point['a'], lam=point['lam'], c=point['c'], d=point['d'], vev=point['vev'])
        self.V = self.model.DVtot # This is the scalar potential shifted such that the false vacuum is located at ϕ = 0 for each value of the temperature
        self.dV = self.model.gradV # This is the gradient of the scalar potential
        self.v_w = 1.0
        self.n_temp_points = 100
        self.T_max = self.Tc
        self.T_min = None
        self.maxvev = None

        if self.verbose:
            print("a: ", self.a)
            print("lam: ", self.lam)
            print("c: ", self.c)
            print("d: ", self.d)
            print("vev: ", self.vev)
            print("T_max (Tc): ", self.Tc)
            print("T_min (T0): ", np.sqrt(self.T0sq))
            print("Tc: ", self.Tc)
            print("phi_critical: ", self.phi_crit)
            print("wall_tension: ", self.sigma)
            print("T0sq: ", self.T0sq)

        # Arrays over temperature grid
        self.Temps = None
        self.Gamma_f_list = None
        self.Nf_list = None
        self.Hubble = None
        self.action_vec = None
        self.S3overT = None
        self.logP_f = None

        self.V_min_value = None
        self.phi0_min = None
        self.pbh_forms = False

        # Varibles for false-vacuum remnants
        self.Nf_perc = None
        self.H_perc = None
        self.nf_perc = None

        self.Nf_perc_false = None
        self.H_perc_false = None
        self.nf_perc_false = None

        # Variables for PBH formation
        self.P_surv_pbh = None
        self.R_perc = None
        self.m_pbh = None
        self.f_pbh = None
    
    def get_Tc(self) -> float:
        # from FKS
        return (self.c*self.a + np.sqrt(self.lam*self.d*(self.c**2 + (self.lam*self.d - self.a**2)*self.T0sq)))/(self.lam*self.d - self.a**2)

    def phi_critical(self) -> float:
        return self.Tc * (2*(self.a + self.c/self.Tc)/self.lam)
    
    def wall_tension(self) -> float:
        # from thin wall approx
        return np.power(self.phi_critical(), 3) * np.power(self.lam/2, 0.5) / 6

    def calc_temperature_bounds(self, use_elena: bool = True):
        if use_elena:
            T_max, vevs_max, max_min_vals, false_min_tmax = find_T_max(self.V, self.dV, precision= 1e-3, Phimax = 2*self.vev, step_phi = self.vev * 1e-3, tmax=2.5 * self.vev)
            T_min, vevs_min, false_min_tmin = find_T_min(self.V, self.dV, tmax=T_max, precision = 1e-3, Phimax = 2*self.vev, step_phi = self.vev * 1e-3, max_min_vals = max_min_vals)
            
            if self.verbose:
                print(f"T_max: {T_max}, T_min: {T_min}")
                print(f"vevs_max: {vevs_max}, vevs_min: {vevs_min}")
                print("Refining...")
            if T_max is not None and T_min is not None:
                maxvev = np.max(np.concatenate((vevs_max, vevs_min)))
            elif T_max is not None:
                maxvev = np.max(vevs_max)
            elif T_min is not None:
                maxvev = np.max(vevs_min)
            else:
                maxvev = None
            T_min = refine_Tmin(T_min, self.V, self.dV, maxvev, log_10_precision = 7) if T_min is not None else None
            if T_min == 0.0:
                T_min = 1e-7
        else:
            T_max = self.Tc
            T_min = np.sqrt(self.T0sq)
            maxvev = self.phi_critical()
        
        self.T_max = T_max
        self.T_min = T_min
        self.maxvev = maxvev
    
    def calc_action_over_T(self):
        true_vev = {}
        S3overT = {}
        V_min_value = {}
        phi0_min = {}
        false_vev = {}
        def action_over_T(T, c_step_phi = 1e-3, precision = 1e-3):
            instance = Vt_vec(T,
                              self.V,
                              self.dV,
                              step_phi = c_step_phi,
                              precision = precision,
                              vev0 = self.maxvev,
                              ratio_vev_step0=50)

            if instance.barrier:
                true_vev[T] = instance.true_min
                false_vev[T] = instance.phi_original_false_vev
                S3overT[T] = instance.action_over_T
                V_min_value[T] = instance.min_V
                phi0_min[T] = instance.phi0_min
                return instance.action_over_T
            else:
                return None
        
        temperatures = np.linspace(self.T_min, self.T_max, self.n_temp_points)
        action_vec = np.vectorize(action_over_T)

        action_vec(temperatures)
        #self.temperatures = np.array([T for T in temperatures if T in S3overT])
        self.action_vec = action_vec
        self.S3overT = S3overT
        self.V_min_value = V_min_value
        self.phi0_min = phi0_min
        self.false_vev = false_vev
        self.true_vev = true_vev

    def calc_nucleation_percolation_completion(self):
        # In this routine, we will populate the arrays of logP_f and the associated temperature
        # grid, determine T_nuc and T_perc, and T_completion, and check that the physical volume is decreasing

        counter = 0
        while counter <= 1:
            logP_f, Temps, ratio_V, Gamma, H = compute_logP_f(self.model,
                                                              self.V_min_value,
                                                              self.S3overT,
                                                              self.v_w,
                                                              units = self.units,
                                                              cum_method= 'None')
            Gamma_f_list, _, ratio_V, _, H = compute_Gamma_f(self.model,
                                                             self.V_min_value,
                                                             self.S3overT,
                                                             self.v_w,
                                                             logP_f,
                                                             units=self.units,
                                                             cum_method='cumulative_simpson')

            logP_f = np.nan_to_num(logP_f)
            Gamma_f_list = np.nan_to_num(Gamma_f_list)

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
                if ratio_V_at_T_perc > log_at_T_perc:
                    raise ValueError("\n *** The physical volume at percolation is not decreasing. The production of GW is questionable ***")
            counter += 1

        #milestones = [self.T_max, T_nuc, T_perc, T_completion, self.T_min]
        #milestones = [milestone for milestone in milestones if milestone is not None and not np.isnan(milestone)]
        #self.action_vec(milestones)

        self.T_nuc = T_nuc
        self.T_perc = T_perc
        self.T_perc_false = T_perc_false
        self.T_completion = T_completion
        self.Temps = Temps
        self.Gamma_f_list = Gamma_f_list
        self.logP_f = logP_f
        self.Nf_list = Nf_list
        self.Hubble = H
        self.R = R
        self.RH = RH

        if self.verbose:
            print("T_nuc: ", self.T_nuc)
            print("T_perc: ", self.T_perc)
            print("T_perc_false: ", self.T_perc_false)
            print("T_completion: ", self.T_completion)

    def calc_npatches(self):
        mask_Gf = ~np.isnan(self.Gamma_f_list)
        #Gammaf_perc = interpolation_narrow(self.Temps[mask_Gf], self.Gamma_f_list[mask_Gf], self.T_perc)
        Nf_perc = interpolation_narrow(self.Temps[mask_Gf], self.Nf_list[mask_Gf], self.T_perc)
        H_perc = interpolation_narrow(self.Temps[mask_Gf], self.Hubble[mask_Gf], self.T_perc)
        Hubble_vol_perc = 4*np.pi / 3 * H_perc**(-3)
        nf_perc = Nf_perc / Hubble_vol_perc # number density of false-vacuum bubbles

        # Same quantities, but evaluated when P_f(T_perc_false) = 0.29
        #Gammaf_perc_false = interpolation_narrow(self.Temps[mask_Gf], self.Gamma_f_list[mask_Gf], self.T_perc_false)
        Nf_perc_false = interpolation_narrow(self.Temps[mask_Gf], self.Nf_list[mask_Gf], self.T_perc_false)
        H_perc_false = interpolation_narrow(self.Temps[mask_Gf], self.Hubble[mask_Gf], self.T_perc_false)
        Hubble_vol_perc_false = 4*np.pi / 3 * H_perc_false**(-3)
        nf_perc_false = Nf_perc_false / Hubble_vol_perc_false # number density of false-vacuum bubbles

        self.Nf_perc = Nf_perc
        self.H_perc = H_perc
        self.nf_perc = nf_perc

        self.Nf_perc_false = Nf_perc_false
        self.H_perc_false = H_perc_false
        self.nf_perc_false = nf_perc_false
        if self.verbose:
            print("Nf_perc: ", self.Nf_perc)
            print("H_perc: ", self.H_perc)
            print("nf_perc: ", self.nf_perc)
            print("Nf_perc_false: ", self.Nf_perc_false)
            print("H_perc_false: ", self.H_perc_false)
            print("nf_perc_false: ", self.nf_perc_false)

    def calc_mean_bubble_size(self):
        R_perc = interpolation_narrow(self.Temps, self.R, self.T_perc)
        logP_f_R, Temps_, _, _, _ = compute_logP_f(self.model,
                                                   self.V_min_value,
                                                   self.S3overT,
                                                   v_w = self.v_w,
                                                   units = self.units,
                                                   cum_method= 'None',
                                                   R_0=R_perc)
        mask_PfR = ~np.isnan(logP_f_R)
        logPf_Rperc = interpolation_narrow(Temps_[mask_PfR], logP_f_R[mask_PfR], self.T_perc)
        self.P_surv_pbh = np.exp(logPf_Rperc)
        self.R_perc = R_perc
        
        if self.verbose:
            print("R_perc: ", self.R_perc)
            print("P_surv_pbh: ", self.P_surv_pbh)
        
        # Interpolate self.V_min_value to get the value at T_perc, but remember that self.V_min_value is a dictionary
        V_min_Temps = list(self.V_min_value.keys())
        V_min_values = list(self.V_min_value.values())

        V_min_value_at_T_perc = -np.interp(self.T_perc, V_min_Temps, V_min_values)

        fks = FKSCollapse(deltaV=abs(V_min_value_at_T_perc), sigma=self.sigma, vw=self.v_w)
        self.pbh_forms = fks.does_pbh_form(self.R_perc)
        self.m_pbh = fks.M0(self.R_perc)

        if self.verbose:
            print("pbh_forms: ", self.pbh_forms)
            print("m_pbh (g): ", self.m_pbh / GEV_PER_G)

    def calc_pbh_abundance(self, verbose: bool = False):
        normalization = np.power(HUBBLE0, 3) / (4 * np.pi * RHO_CRIT_GEV4 * OMEGA_DM / 3)
        abundance = normalization * self.m_pbh * self.P_surv_pbh * self.Nf_perc_false
        self.f_pbh = abundance
        
        if verbose:
            print("normalization: ", normalization)
            print("m_pbh (GeV): ", self.m_pbh)
            print("P_surv_pbh: ", self.P_surv_pbh)
            print("Nf_perc: ", self.Nf_perc)
            print("Nf_perc_false: ", self.Nf_perc_false)
            print("f_pbh: ", self.f_pbh)
        
        return abundance












