
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import sys
import os
from time import time
from scipy.integrate import cumulative_trapezoid


# Find the absolute path to the src directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')

# Add this path to the start of sys.path if it's not already there
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from espinosa import Vt_vec
from utils import (interpolation_narrow, interpolation_narrow_log,
                   interp_positive_log, s_SM, log_trapz as _log_trapz,
                   log_cumulative_trapz, stable_logP_t, _log_trapz_cell,
                   convert_units, drho_SM_spline, d2rho_SM_spline)
from temperatures import (compute_logP_f,
                          N_bubblesH,
                          R_sepH,
                          compute_Gamma_f,
                          Gamma_f_vals_at,
                          logP_f_at_T,
                          _logP_f_single_slice,
                          find_T_min,
                          find_T_max,
                          refine_Tmin,
                          M_pl as M_pl_BARRIER)
from dof_interpolation import g_rho_spline, g_rho
from model import model_no_quantum, model_bl

from pbh import FKSCollapse
from external_constants import *

# #region agent log
_AGENT_DEBUG_LOG = os.path.normpath(
    os.path.join(script_dir, '..', '..', 'debug-272c5d.log'))


def _agent_debug_log(location, message, data, hypothesis_id, run_id='cliff-enrich-v2'):
    try:
        import json
        payload = {
            'sessionId': '272c5d',
            'runId': run_id,
            'hypothesisId': hypothesis_id,
            'location': location,
            'message': message,
            'data': data,
            'timestamp': int(time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\n')
    except Exception:
        pass
# #endregion

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
    def __init__(self, point, verbose = False, model="polynomial"):
        self.verbose = verbose

        # init Veff params
        self.point = point
        if model == "polynomial":
            self.a = point['a']
            self.lam = point['lam']
            self.c = point['c']
            self.d = point['d']
            self.vev = point['vev']
            self.model = model_no_quantum(a=point['a'], lam=point['lam'], c=point['c'], d=point['d'], vev=point['vev'])
            self.T0sq = (self.lam * self.vev**2 - 3*self.c*self.vev)/(2*self.d)
        elif model == "BL":
            self.gBL = point['gBL']
            self.vev = point['vev']
            self.model = model_bl(gBL=point['gBL'], vev=point['vev'])

        # Derive Veff quantities
        self.Tc = self.get_Tc()
        self.phi_crit = self.phi_critical()
        self.sigma = self.wall_tension()

        # Init ELENA quantities
        self.units = 'GeV'
        self.V = self.model.DVtot # This is the scalar potential shifted such that the false vacuum is located at ϕ = 0 for each value of the temperature
        self.dV = self.model.gradV # This is the gradient of the scalar potential
        self.v_wall = 1.0
        self.n_temp_points = 400
        self.n_cliff_points = 48
        self.n_cliff_focus_points = 64
        self.cliff_max_dT_frac = 5e-8
        self.T_max = self.Tc
        self.T_min = None
        self.maxvev = None

        if self.verbose:
            print(point)
            print("T_max (Tc): ", self.Tc)
            print("T_min (T0): ", np.sqrt(self.T0sq))
            print("Tc: ", self.Tc)
            print("phi_critical: ", self.phi_crit)
            print("wall_tension: ", self.sigma)

        # Arrays over temperature grid
        self.Temps = None
        self.Gamma_f_list = None
        self.Gamma = None
        self.Nf_list = None
        self.Hubble = None
        self.action_vec = None
        self.S3overT = None
        self.logP_f = None
        self.ratio_V = None

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
        self.nf_perc_false_approx = None

        self.alpha = None
        self.beta = None
        self.beta_by_Hn = None

        # Variables for PBH formation
        self.P_surv_pbh = None
        self.R_perc = None
        self.m_pbh = None
        self.f_pbh = None
        self.f_pbh_approx = None
    
    def run_pipeline(self, time_verbose: bool = False,
                           log_temps: bool = False,
                           n_refine: int = 0):
        try:
            start_time = time()
            self.calc_temperature_bounds()
            end_time = time()
            if time_verbose:
                print(f"Temperature bounds calculated in {end_time - start_time} seconds")
            #print("Temperature bounds calculated")
        except Exception as e:
            print(f"Error: {e}")

        try:
            start_time = time()
            self.calc_action_over_T(log_temps=log_temps)
            end_time = time()
            if time_verbose:
                print(f"Action over T calculated in {end_time - start_time} seconds")
            #print("Action over T calculated")
        except Exception as e:
            print(f"Error: {e}")
        
        try:
            start_time = time()
            self.calc_nucleation_percolation_completion()
            end_time = time()
            if time_verbose:
                print(f"Nucleation percolation completion calculated in {end_time - start_time} seconds")
            #print("Nucleation percolation completion calculated")
            if np.isnan(self.T_completion):
                print("T_completion is nan")
            if np.isnan(self.T_nuc):
                print("T_nuc is nan")
            if np.isnan(self.T_perc):
                print("T_perc is nan")
        except Exception as e:
            print(f"Error: {e}")

        try:
            start_time = time()
            self.calc_npatches(n_refine=n_refine)
            end_time = time()
            if time_verbose:
                print(f"Number of patches calculated in {end_time - start_time} seconds")
            #print("Number of patches calculated")
        except Exception as e:
            print(f"Error: {e}")

        try:
            start_time = time()
            self.calc_mean_bubble_size()
            end_time = time()
            if time_verbose:
                print(f"Mean bubble size calculated in {end_time - start_time} seconds")
            #print("Mean bubble size calculated")
        except Exception as e:
            print(f"Error: {e}")

        try:
            start_time = time()
            self.calc_pbh_abundance(verbose=False)
            end_time = time()
            if time_verbose:
                print(f"PBH abundance calculated in {end_time - start_time} seconds")
            #print("PBH abundance calculated")
        except Exception as e:
            print(f"Error: {e}")


    def get_Tc(self) -> float:
        # from FKS
        return (self.c*self.a + np.sqrt(self.lam*self.d*(self.c**2 + (self.lam*self.d - self.a**2)*self.T0sq)))/(self.lam*self.d - self.a**2)

    def phi_critical(self) -> float:
        return self.Tc * (2*(self.a + self.c/self.Tc)/self.lam)
    
    def wall_tension(self) -> float:
        # from thin wall approx
        return np.power(self.phi_critical(), 3) * np.power(self.lam/2, 0.5) / 6

    def calc_temperature_bounds(self):
        # Analytic solution
        T_max = self.Tc
        T_min = np.sqrt(self.T0sq)
        maxvev = self.phi_critical()
    
        self.T_max = T_max
        self.T_min = T_min
        self.maxvev = maxvev
    
    def calc_action_over_T(self, log_temps = True):
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
        
        # Do not sample at T_max = Tc: Vt_vec's barrier check intermittently
        # fails on the topmost grid point
        if log_temps:
            temperatures = np.logspace(np.log10(self.T_min), np.log10(self.T_max), self.n_temp_points)[:-1]
        else:
            temperatures = np.linspace(self.T_min, self.T_max, self.n_temp_points)[:-1]
        action_vec = np.vectorize(action_over_T)
        action_vec(temperatures)

        # The lowest-T point where a barrier first appears has S3/T ~ 0.15
        # (log evidence: S3overT_min while physical values are > 20 one cell
        # up). It poisons the logP_f cliff and shifts as lambda moves.
        _sorted_T = sorted(S3overT.keys())
        while _sorted_T and S3overT[_sorted_T[0]] < 1.0 and len(_sorted_T) > 10:
            t0 = _sorted_T[0]
            for _d in (S3overT, V_min_value, phi0_min, true_vev, false_vev):
                _d.pop(t0, None)
            _sorted_T = sorted(S3overT.keys())

        self.action_vec = action_vec
        self.S3overT = S3overT
        self.V_min_value = V_min_value
        self.phi0_min = phi0_min
        self.false_vev = false_vev
        self.true_vev = true_vev

    def cs2(self, T, true_vev):
        speed2 = self.model.dVdT([true_vev], T, units = self.units) / (T * self.model.d2VdT2([true_vev], T, units = self.units))
        return min(1/3, speed2.flatten())
    
    def calc_vw(self) -> float:
        # Determines v_wall once the history is computed and alpha is known
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            return 1.0
        
        vJ = (np.sqrt(2*self.alpha/3 + self.alpha**2) + np.sqrt(1/3))/(1+self.alpha)
        rho_r = np.pi**2 * GSTAR_SM * self.T_perc**4 / 30

        V_min_Temps = list(self.V_min_value.keys())
        V_min_values = list(self.V_min_value.values())

        V_min_value_at_T_perc = -np.interp(self.T_perc, V_min_Temps, V_min_values)
        
        denom = self.alpha * rho_r
        if denom <= 0 or V_min_value_at_T_perc < 0:
            return 1.0
        
        v_candidate = np.sqrt(V_min_value_at_T_perc / denom)
        self.v_wall = v_candidate if v_candidate < vJ else (1.0 + vJ)/2
        return self.v_wall

    def calc_alpha(self):
        # Computes alpha at T_perc once the history is computed
        V_min_Temps = list(self.V_min_value.keys())
        V_min_values = list(self.V_min_value.values())
        false_vev_values = list(self.false_vev.values())
        true_vev_values = list(self.true_vev.values())

        V_min_value = -np.interp(self.T_perc, V_min_Temps, V_min_values)
        false_vev = -np.interp(self.T_perc, V_min_Temps, false_vev_values)
        true_vev = -np.interp(self.T_perc, V_min_Temps, true_vev_values)

        delta_rho = - V_min_value \
            -  self.T_perc * (self.model.dVdT([false_vev], self.T_perc, include_SM = True, units = self.units) \
            - self.model.dVdT([true_vev], self.T_perc, include_SM = True, units = self.units))
        delta_p = V_min_value / self.cs2(self.T_perc, true_vev)
        wf = - self.T_perc * self.model.dVdT([false_vev], self.T_perc, units = self.units, include_SM = True)

        return (delta_rho - delta_p) / (3 * wf)
    
    def c_alpha_inf(self, T):
        v_true = self.high_vev[T]
        v_false = self.false_vev[T]
        Dm2_photon = 3 * self.g**2 * (v_true**2 - v_false**2)
        Dm2_scalar = 3 * self.lambda_ * (v_true**2 - v_false**2) 
        numerator = (Dm2_photon + Dm2_scalar) * T**2 / 24
        rho_tot = - T * 3 * (self.dp.dVdT(v_false, T, include_radiation=True, include_SM = True, units = self.units) ) / 4
        rho_DS = - T * 3 * (self.dp.dVdT(v_false, T, include_radiation=True, include_SM = False, units = self.units) ) / 4
        return numerator/ rho_tot, numerator / rho_DS

    def c_alpha_eq(self, T):
        v_true = self.high_vev[T]
        v_false = self.false_vev[T]
        numerator = (self.g**2 * 3 * (self.g * (v_true - v_false)) * T**3)
        rho_tot = - T * 3 * (self.dp.dVdT(v_false, T, include_radiation=True, include_SM = True, units = self.units) ) / 4
        rho_DS = - T * 3 * (self.dp.dVdT(v_false, T, include_radiation=True, include_SM = False, units = self.units) ) / 4
        return numerator / rho_tot, numerator / rho_DS
    
    def calc_beta(self):
        # Computes beta at T_nuc once the history is computed
        idx_nuc = np.argmin(np.abs(self.Temps - self.T_nuc))
        idx_perc = np.argmin(np.abs(self.Temps - self.T_perc))

        Gamma_n = self.Gamma_f_list[idx_nuc]
        H_n = self.Hubble[idx_nuc]

        times = cumulative_trapezoid(-np.flip((self.ratio_V[idx_perc:idx_nuc+1] / (3 * self.Hubble[idx_perc:idx_nuc+1]))),
                                      np.flip(self.Temps[idx_perc:idx_nuc+1]), initial=0)
        times = np.flip(times)

        t = np.flip(H_n*times)
        ft = np.flip(self.Gamma_f_list[idx_perc:idx_nuc+1]/Gamma_n)
        ln_ft = np.log(ft)

        # Create the design matrix
        X = np.vstack((t, t**2)).T  # Stack t and t^2 as columns to create a design matrix

        # Fit
        coefs, _, _, _ = lstsq(X, ln_ft, rcond=None)  # Fit 'ln_ft' against 't' and 't^2'

        # Extract coefficients
        a_fit = coefs[0]  # Coefficient for the linear term (t)
        beta_Hn = a_fit

        self.beta = beta_Hn
        self.beta_by_Hn = beta_Hn / H_n

        return beta_Hn

    def _integrate_barrier_profiles(self, Pf_true_choice, Pf_false_choice,
                                    check_volume=True):
        """Build logP_f / Gamma_f / Nf_list and milestone temperatures on current keys."""
        logP_f, Temps, ratio_V, Gamma, H = compute_logP_f(
            self.model, self.V_min_value, self.S3overT, self.v_wall,
            units=self.units, cum_method='log_trapz')
        Gamma_f_list, _, ratio_V, _, H = compute_Gamma_f(
            self.model, self.V_min_value, self.S3overT, self.v_wall, logP_f,
            units=self.units, integrator='log_trapz')

        logP_f = np.nan_to_num(logP_f)
        Gamma_f_list = np.nan_to_num(Gamma_f_list)

        RH, R = R_sepH(Temps, Gamma, logP_f, H, ratio_V)
        nH = N_bubblesH(Temps, Gamma, logP_f, H, ratio_V)
        logP_t = np.vectorize(stable_logP_t)(logP_f)
        Nf_list = N_bubblesH(Temps, Gamma_f_list, logP_t, H, ratio_V,
                               integrator='log_trapz')

        mask_nH = ~np.isnan(nH) & (nH > 0)
        T_nuc = interpolation_narrow(np.log(nH[mask_nH]), Temps[mask_nH], 0)
        mask_Pf = ~np.isnan(logP_f)
        T_perc = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf],
                                        np.log(Pf_true_choice))
        T_perc_false = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf],
                                            np.log(Pf_false_choice))
        T_completion = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf],
                                            np.log(0.01))

        idx_compl = np.max([np.argmin(np.abs(Temps - T_completion)), 1])
        test_completion = np.array([logP_f[idx_compl - 1], logP_f[idx_compl],
                                    logP_f[idx_compl + 1]])
        test_completion = test_completion[~np.isnan(test_completion)]
        if not is_increasing(test_completion):
            T_completion = np.nan

        d_dT_logP_f = np.gradient(logP_f, Temps)
        log_at_T_perc = interpolation_narrow(Temps, d_dT_logP_f, T_perc)
        ratio_V_at_T_perc = interpolation_narrow(Temps, ratio_V, T_perc)
        if check_volume and ratio_V_at_T_perc > log_at_T_perc:
            raise ValueError("\n *** The physical volume at percolation is not "
                             "decreasing. The production of GW is questionable ***")

        return {
            'T_nuc': T_nuc,
            'T_perc': T_perc,
            'T_perc_false': T_perc_false,
            'T_completion': T_completion,
            'Temps': Temps,
            'ratio_V': ratio_V,
            'Gamma': Gamma,
            'Gamma_f_list': Gamma_f_list,
            'logP_f': logP_f,
            'Nf_list': Nf_list,
            'Hubble': H,
            'R': R,
            'RH': RH,
        }

    def _cliff_keys_in_window(self, T_lo, T_hi):
        return np.array(sorted(k for k in self.S3overT
                               if float(T_lo) <= float(k) <= float(T_hi)),
                        dtype=float)

    def _max_cliff_gap(self, T_lo, T_hi):
        keys = self._cliff_keys_in_window(T_lo, T_hi)
        if keys.size < 2:
            return float('inf'), keys
        gaps = np.diff(keys)
        return float(np.max(gaps)), keys

    def _try_add_barrier_key(self, T, existing, T_hi, tol):
        T = float(T)
        if existing.size and np.min(np.abs(existing - T)) < tol:
            return False, existing
        self.action_vec(T)
        return True, np.append(existing, T)

    def _enrich_barrier_cliff_window(self, T_perc_false, T_perc, T_nuc):
        """
        Insert bounce solves in the percolation cliff window so ``Gamma_f`` /
        ``Nf`` integrators do not sit on sparse ``S3/T`` gaps.

        Uses (i) a baseline log mesh, (ii) extra density between
        ``T_perc_false`` and ``T_perc``, and (iii) adaptive gap-filling until
        the largest cell in the cliff window falls below ``cliff_max_dT_frac``.
        """
        if not hasattr(self, 'action_vec') or self.action_vec is None:
            return 0, float('inf')
        if T_perc_false is None or not np.isfinite(T_perc_false):
            return 0, float('inf')
        if not self.S3overT:
            return 0, float('inf')

        n_pts = int(getattr(self, 'n_cliff_points', 48))
        n_focus = int(getattr(self, 'n_cliff_focus_points', 64))
        dT_frac = float(getattr(self, 'cliff_max_dT_frac', 5e-8))
        T_hi = float(max(self.S3overT.keys()))
        T_lo_key = float(min(self.S3overT.keys()))
        width = max(T_hi - float(T_perc_false), 1e-30)
        T_lo = max(float(T_perc_false) - 0.05 * width, T_lo_key)
        T_focus_hi = max(float(T_perc_false),
                         float(T_perc) if np.isfinite(T_perc) else float(T_perc_false),
                         float(T_nuc) if T_nuc is not None and np.isfinite(T_nuc)
                         else float(T_perc_false))
        T_focus_hi = min(T_focus_hi + 0.02 * width, T_hi)

        candidates = list(np.geomspace(T_lo, T_hi, n_pts, endpoint=False))
        if T_focus_hi > float(T_perc_false):
            candidates.extend(np.geomspace(float(T_perc_false), T_focus_hi,
                                           n_focus, endpoint=False))
        for Tk in (T_perc_false, T_perc, T_nuc):
            if Tk is not None and np.isfinite(Tk):
                candidates.append(float(Tk))

        existing = np.array(sorted(self.S3overT.keys()), dtype=float)
        tol = max(dT_frac * T_hi, 1e-18)
        added = 0
        for T in candidates:
            ok, existing = self._try_add_barrier_key(T, existing, T_hi, tol)
            if ok:
                added += 1

        dT_target = max(dT_frac * T_hi, 1e-18)
        for _ in range(int(getattr(self, 'n_cliff_gap_iters', 48))):
            max_gap, keys = self._max_cliff_gap(T_lo, T_hi)
            if not np.isfinite(max_gap) or max_gap <= dT_target or keys.size < 2:
                break
            idx = int(np.argmax(np.diff(keys)))
            T_mid = 0.5 * (keys[idx] + keys[idx + 1])
            ok, existing = self._try_add_barrier_key(T_mid, existing, T_hi, tol)
            if ok:
                added += 1

        max_gap, _ = self._max_cliff_gap(T_lo, T_hi)
        return added, max_gap

    def calc_nucleation_percolation_completion(self, Pf_true_choice = 0.71, Pf_false_choice = 0.29):
        # Populate barrier profiles, determine milestones, then enrich the cliff
        # window with extra bounce keys before the final profile used downstream.
        self.gamma_f_cum_f = log_cumulative_trapz

        prof_coarse = self._integrate_barrier_profiles(Pf_true_choice, Pf_false_choice,
                                                       check_volume=False)
        n_cliff_added, max_cliff_gap = self._enrich_barrier_cliff_window(
            prof_coarse['T_perc_false'], prof_coarse['T_perc'], prof_coarse['T_nuc'])
        if n_cliff_added > 0:
            prof = self._integrate_barrier_profiles(Pf_true_choice, Pf_false_choice,
                                                    check_volume=False)
        else:
            prof = self._integrate_barrier_profiles(Pf_true_choice, Pf_false_choice,
                                                    check_volume=True)

        self.T_nuc = prof['T_nuc']
        self.T_perc = prof['T_perc']
        self.T_perc_false = prof['T_perc_false']
        self.T_completion = prof['T_completion']
        self.Temps = prof['Temps']
        self.ratio_V = prof['ratio_V']
        self.Gamma = prof['Gamma']
        self.Gamma_f_list = prof['Gamma_f_list']
        self.logP_f = prof['logP_f']
        self.Nf_list = prof['Nf_list']
        self.Hubble = prof['Hubble']
        self.R = prof['R']
        self.RH = prof['RH']
        self.alpha = self.calc_alpha()
        self.v_wall_gamma_f = self.v_wall
        self.v_wall = self.calc_vw()
        self.beta = self.calc_beta()

        if self.verbose:
            print("T_nuc: ", self.T_nuc)
            print("T_perc: ", self.T_perc)
            print("T_perc_false: ", self.T_perc_false)
            print("T_completion: ", self.T_completion)

        # #region agent log
        idx_pf = int(np.argmin(np.abs(self.Temps - self.T_perc_false)))
        dT_pf = float(np.min(np.abs(self.Temps - self.T_perc_false)))
        mask_nf = ~np.isnan(self.Nf_list)
        Nf_lin_at_pf = float(np.interp(self.T_perc_false,
                                       self.Temps[mask_nf], self.Nf_list[mask_nf]))
        Nf_log_at_pf = float(interp_positive_log(
            self.T_perc_false, self.Temps[mask_nf], self.Nf_list[mask_nf]))
        _agent_debug_log(
            'fopt_generic.py:calc_nucleation_percolation_completion',
            'stage1 milestones and Nf_list interp comparison',
            {
                'n_temp_points': int(getattr(self, 'n_temp_points', -1)),
                'n_cliff_added': int(n_cliff_added),
                'max_cliff_gap': float(max_cliff_gap),
                'cliff_dT_target': float(getattr(self, 'cliff_max_dT_frac', 5e-8)
                                         * float(max(self.S3overT.keys()))),
                'n_barrier_keys': int(len(self.S3overT)),
                'n_barrier_grid': int(len(self.Temps)),
                'T_perc_false': float(self.T_perc_false),
                'T_perc': float(self.T_perc),
                'dT_to_nearest_grid': dT_pf,
                'idx_bracket': idx_pf,
                'Nf_list_lin_interp': Nf_lin_at_pf,
                'Nf_list_log_interp': Nf_log_at_pf,
                'Nf_list_log_lin_ratio': Nf_log_at_pf / max(Nf_lin_at_pf, 1e-300),
            },
            'H6',
            run_id='cliff-enrich-v2',
        )
        # #endregion

    def _s3overT_at(self, T):
        """
        Interpolate ``S3/T`` from bounce keys onto ``T`` with log-linear
        rule in ``S3/T`` (see ``FOPTGenericRefineGrid`` docstring).
        """
        T = np.asarray(T, dtype=float)
        S3_Temps = np.array(sorted(self.S3overT.keys()), dtype=float)
        S3ot = np.array([self.S3overT[t] for t in S3_Temps], dtype=float)
        if S3_Temps.size == 0:
            return np.full_like(T, np.nan, dtype=float)
        if S3_Temps.size == 1:
            return np.full_like(T, S3ot[0], dtype=float)
        pos = S3ot > 0.0
        if not np.all(pos):
            return np.interp(T, S3_Temps, S3ot)
        log_s3ot = np.interp(T, S3_Temps, np.log(S3ot))
        return np.exp(log_s3ot)

    def _gamma_at_T(self, T):
        """Decay width at T from log-interpolated S3/T."""
        s3ot = float(self._s3overT_at(float(T)))
        return float(T) ** 4 * (s3ot / (2.0 * np.pi)) ** (3.0 / 2.0) * np.exp(-s3ot)

    def _gamma_from_S3(self, T):
        """Vectorised Gamma(T) from log-interpolated S3/T."""
        T = np.asarray(T, dtype=float)
        s3ot = self._s3overT_at(T)
        return (T ** 4 * (s3ot / (2.0 * np.pi)) ** (3.0 / 2.0) * np.exp(-s3ot))

    def _thermo_from_barrier(self, T):
        """
        Evaluate H(T) and ratio_V(T) from interpolated ``V_min_value`` and
        model thermal derivatives (same physics as ``_barrier_thermo_fields``).
        """
        T = np.asarray(T, dtype=float)
        keys = np.array(sorted(self.V_min_value.keys()), dtype=float)
        if keys.size < 2:
            nan = np.full_like(T, np.nan, dtype=float)
            return nan, nan

        vmin_vals = np.array([self.V_min_value[float(k)] for k in keys],
                             dtype=float)
        V_min = np.interp(T, keys, vmin_vals)

        T_step = max(float(keys[-1] - keys[0]) * 1e-3, 1e-30)
        V = self.model.Vtot
        cu = convert_units[self.units]
        phi0 = np.array([0.0])

        H_out = np.empty(T.size, dtype=float)
        rv_out = np.empty(T.size, dtype=float)
        for i, t in enumerate(T.ravel()):
            tp = t + T_step
            tm = t - T_step
            dvdT_mid = (V(phi0, tp) - V(phi0, tm)) / (2.0 * T_step)
            dVdT0 = float(dvdT_mid - drho_SM_spline(t) / 3.0)
            dvdT_tp = (V(phi0, tp + T_step) - V(phi0, tp - T_step)) / (2.0 * T_step)
            dvdT_tm = (V(phi0, tm + T_step) - V(phi0, tm - T_step)) / (2.0 * T_step)
            d2VdT2_0 = float((dvdT_tp - dvdT_tm) / (2.0 * T_step)
                             - d2rho_SM_spline(t) / 3.0)

            e_vac = -float(V_min.ravel()[i]) - t * dVdT0
            e_rad = np.pi ** 2 * g_rho(t / cu) * t ** 4 / 30.0
            H_out.ravel()[i] = (np.sqrt(max((e_vac + e_rad) / 3.0, 0.0))
                                / (M_pl_BARRIER * cu))
            rv_out.ravel()[i] = d2VdT2_0 / dVdT0 if dVdT0 != 0.0 else np.nan

        return H_out.reshape(T.shape), rv_out.reshape(T.shape)

    def _augment_grid_at(self, T_target, *fields):
        """
        Build the ascending temperature grid that starts exactly at T_target
        and includes every existing grid point strictly above it, along with
        the field arrays linearly-interpolated at T_target. This lets us
        compute integrals from T_target to T_max without the lower limit
        having to land on a grid point.

        Returns (T_arr, *augmented_field_arrays).
        """
        Temps = self.Temps
        if T_target >= Temps[-1]:
            return (None,) * (1 + len(fields))
        mask = Temps > T_target
        T_above = Temps[mask]
        T_arr = np.concatenate([[T_target], T_above])
        out = [T_arr]
        for arr in fields:
            val_at = float(np.interp(T_target, Temps, arr))
            out.append(np.concatenate([[val_at], arr[mask]]))
        return tuple(out)

    def _refine_grid_above(self, T_target, T_arr, n_refine, *field_arrays):
        """
        Insert log-spaced sub-points in the first cell above T_target to
        reduce grid-locking when T_target slides within a coarse cell.
        """
        if n_refine <= 0 or len(T_arr) < 2:
            return (T_arr,) + field_arrays

        T_hi = T_arr[1]
        if T_hi <= T_target:
            return (T_arr,) + field_arrays

        T_lo = T_target + (T_hi - T_target) * 1e-6
        T_sub = np.geomspace(T_lo, T_hi * (1.0 - 1e-9), n_refine, endpoint=False)
        T_new = np.concatenate([[T_target], T_sub, T_arr[1:]])

        out = [T_new]
        for arr in field_arrays:
            sub_vals = np.interp(T_sub, T_arr, arr)
            out.append(np.concatenate([[arr[0]], sub_vals, arr[1:]]))
        return tuple(out)

    def _R_sepH_at(self, T_target):
        """
        Compute the mean bubble separation R = n^(-1/3) AT a specific
        temperature T_target by integrating directly from T_target to T_max
        with T_target inserted as the lower integration limit.

        The integrand `Gamma * exp(logP_f) * ratio_V/(3H) * exp(-cum_rv)`
        decays approximately exponentially in T just above T_perc (Gamma
        drops with S3 growing roughly linearly). Plain trapezoidal
        over-estimates exponential decay by ~ (k*dx)/(1 - exp(-k*dx))
        per cell, and that factor swings with where T_target lands
        relative to the grid -- which manifests as a smooth-looking
        oscillation in R(lam). We use a log-linear-per-cell rule
        (`_log_trapz`) which is exact for exponential integrands.
        """
        if not np.isfinite(T_target):
            return np.nan
        aug = self._augment_grid_at(T_target, self.Gamma, self.ratio_V,
                                    self.logP_f, self.Hubble)
        if aug[0] is None:
            return np.nan
        T_arr, G_arr, rv_arr, lp_arr, H_arr = aug
        cum_rv = cumulative_trapezoid(rv_arr, x=T_arr, initial=0)
        f_ext = G_arr * rv_arr * np.exp(lp_arr) / (3.0 * H_arr)
        f1 = f_ext * np.exp(-cum_rv)
        # Log-linear per-cell quadrature (exact for exponential integrand)
        n_at = _log_trapz(f1, T_arr)
        if not (n_at > 0) or not np.isfinite(n_at):
            return np.nan
        return n_at ** (-1.0 / 3.0)

    def _Nf_list_at(self, T_target, n_refine=32, *,
                    temps=None, hubble=None, ratio_v=None,
                    gamma=None, logp_f=None):
        """
        Nf patches integral from ``T_target`` to ``T_max`` with ``T_target``
        inserted as the lower limit.  Recomputes ``Gamma_f_vals_at`` at each slice
        (matching ``compute_Gamma_f`` + ``N_bubblesH(..., logP_t)``) rather
        than interpolating the tabulated ``Nf_list`` profile at milestones.

        Optional ``temps``/``hubble``/... override the barrier arrays (stage-1
        snapshots on ``FOPTGenericRefineGrid``).
        """
        if not np.isfinite(T_target):
            return np.nan

        Temps = np.asarray(self.Temps if temps is None else temps, dtype=float)
        Hubble = np.asarray(self.Hubble if hubble is None else hubble, dtype=float)
        ratio_V = np.asarray(self.ratio_V if ratio_v is None else ratio_v, dtype=float)
        Gamma = np.asarray(self.Gamma if gamma is None else gamma, dtype=float)
        logP_f = np.asarray(self.logP_f if logp_f is None else logp_f, dtype=float)

        if T_target >= Temps[-1]:
            return np.nan

        v_w = getattr(self, "v_wall_gamma_f", self.v_wall)
        logP_f_exact = logP_f_at_T(
            self.model, self.V_min_value, self.S3overT, v_w, T_target,
            units=self.units)
        if not np.isfinite(logP_f_exact):
            return np.nan

        mask = Temps > T_target
        T_arr = np.concatenate([[T_target], Temps[mask]])
        H_tab = float(np.interp(T_target, Temps, Hubble))
        H_log = float(interp_positive_log(T_target, Temps, Hubble))
        rv_tab = float(np.interp(T_target, Temps, ratio_V))
        S3_Temps = np.array(sorted(self.S3overT.keys()))
        S3_vals = np.array([self.S3overT[t] for t in S3_Temps]) if S3_Temps.size else np.array([])
        S3_lin = float(np.interp(T_target, S3_Temps, S3_vals)) if S3_Temps.size else float('nan')
        Gamma_lin = ((float(T_target) ** 4 * (S3_lin / (2.0 * np.pi)) ** (3.0 / 2.0)
                      * np.exp(-S3_lin)) if np.isfinite(S3_lin) else float('nan'))
        Gamma_log = self._gamma_at_T(T_target)
        H_arr, rv_arr = self._thermo_from_barrier(T_arr)
        H_bar = float(H_arr[0])
        rv_bar = float(rv_arr[0])
        Gamma_arr = self._gamma_from_S3(T_arr)
        logP_f_arr = np.concatenate([[logP_f_exact], logP_f[mask]])

        on_grid = (np.min(np.abs(Temps - T_target))
                   < 1e-10 * max(abs(T_target), 1e-30))
        dT_first = float(T_arr[1] - T_target) if len(T_arr) >= 2 else 0.0
        refine_ok = (dT_first > max(1e-9 * abs(T_target), 1e-18))
        if not on_grid and n_refine > 0 and refine_ok:
            T_arr, rv_arr, H_arr, Gamma_arr, logP_f_arr = self._refine_grid_above(
                T_target, T_arr, n_refine, rv_arr, H_arr, Gamma_arr, logP_f_arr)
            logP_f_arr[0] = logP_f_exact
            H_arr, rv_arr = self._thermo_from_barrier(T_arr)
            Gamma_arr = self._gamma_from_S3(T_arr)

        logP_t_arr = np.array([stable_logP_t(lp) for lp in logP_f_arr])

        cum_f = getattr(self, 'gamma_f_cum_f', log_cumulative_trapz)
        gamma_f_vals_arr = np.zeros_like(T_arr)
        for k in range(len(T_arr) - 1):
            gamma_f_vals_arr[k] = Gamma_f_vals_at(
                T_arr[k:], H_arr[k:], rv_arr[k:], Gamma_arr[k:],
                logP_f_arr[k], v_w, cum_f=cum_f)

        integrand = gamma_f_vals_arr * np.exp(logP_t_arr) * rv_arr / H_arr ** 4
        val = _log_trapz(integrand, T_arr)
        Nf = 4.0 * np.pi / 9.0 * val
        first_cell = (float(_log_trapz(integrand[:2], T_arr[:2]))
                      if len(T_arr) >= 2 else float('nan'))
        # #region agent log
        _agent_debug_log(
            'fopt_generic.py:_Nf_list_at',
            'direct Nf integration diagnostics',
            {
                'n_temp_points': int(getattr(self, 'n_temp_points', -1)),
                'T_target': float(T_target),
                'on_grid': bool(on_grid),
                'dT_first': dT_first,
                'refine_ok': bool(refine_ok),
                'n_refine_used': int(n_refine if (not on_grid and refine_ok) else 0),
                'H_lin_vs_log_ratio': H_log / max(H_tab, 1e-300),
                'H_tab_vs_barrier_ratio': H_bar / max(H_tab, 1e-300),
                'rv_tab_vs_barrier_ratio': rv_bar / max(abs(rv_tab), 1e-300),
                'Gamma_lin_vs_log_ratio': Gamma_log / max(Gamma_lin, 1e-300),
                'integrand_0': float(integrand[0]),
                'first_cell_frac': first_cell / val if val > 0 else float('nan'),
                'Nf_direct': float(Nf),
            },
            'H2',
            run_id='cliff-enrich-v2',
        )
        # #endregion
        return Nf

    def calc_npatches(self, n_refine=32):
        # Direct integration at T_perc / T_perc_false with recomputed Gamma_f
        # on an augmented barrier grid (avoids interpolating Nf_list at cliffs).
        Nf_perc       = self._Nf_list_at(self.T_perc, n_refine=n_refine)
        Nf_perc_false = self._Nf_list_at(self.T_perc_false, n_refine=n_refine)
        # H is smooth in T, plain linear interpolation is fine.
        H_perc        = float(np.interp(self.T_perc,       self.Temps, self.Hubble))
        H_perc_false  = float(np.interp(self.T_perc_false, self.Temps, self.Hubble))

        Hubble_vol_perc = 4*np.pi / 3 * H_perc**(-3)
        nf_perc = Nf_perc / Hubble_vol_perc # number density of false-vacuum bubbles
        Hubble_vol_perc_false = 4*np.pi / 3 * H_perc_false**(-3)
        nf_perc_false = Nf_perc_false / Hubble_vol_perc_false # number density of false-vacuum bubbles

        self.Nf_perc = Nf_perc
        self.H_perc = H_perc
        self.nf_perc = nf_perc

        self.Nf_perc_false = Nf_perc_false
        self.H_perc_false = H_perc_false
        self.nf_perc_false = nf_perc_false
        # #region agent log
        _agent_debug_log(
            'fopt_generic.py:calc_npatches',
            'npatches milestone outputs',
            {
                'n_temp_points': int(getattr(self, 'n_temp_points', -1)),
                'Nf_perc_false': float(Nf_perc_false),
                'nf_perc_false': float(nf_perc_false),
                'T_perc_false': float(self.T_perc_false),
            },
            'H3',
            run_id='cliff-enrich-v2',
        )
        # #endregion
        if self.verbose:
            print("Nf_perc: ", self.Nf_perc)
            print("H_perc: ", self.H_perc)
            print("nf_perc: ", self.nf_perc)
            print("Nf_perc_false: ", self.Nf_perc_false)
            print("H_perc_false: ", self.H_perc_false)
            print("nf_perc_false: ", self.nf_perc_false)

    def calc_mean_bubble_size(self):
        # Compute R_perc by directly integrating from T_perc to T_max, with
        # T_perc inserted as the lower limit. The per-grid-point self.R has
        # near-discontinuous growth across the percolation cliff (R[i+1]/R[i]
        # ~ 10-15 within a couple of grid cells of T_perc), so any 2-point
        # interpolation -- linear OR log -- snaps as T_perc crosses grid
        # points. Direct integration is smooth in T_perc.
        R_perc = self._R_sepH_at(self.T_perc)
        logP_f_R, Temps_, _, _, _ = compute_logP_f(self.model,
                                                   self.V_min_value,
                                                   self.S3overT,
                                                   v_w = self.v_wall,
                                                   units = self.units,
                                                   cum_method= 'None',
                                                   R_0=R_perc)
        mask_PfR = ~np.isnan(logP_f_R)
        logPf_Rperc = interpolation_narrow(Temps_[mask_PfR], logP_f_R[mask_PfR], self.T_perc)
        self.P_surv_pbh = 1 - np.exp(logPf_Rperc)
        self.R_perc = R_perc
        
        if self.verbose:
            print("R_perc: ", self.R_perc)
            print("P_surv_pbh: ", self.P_surv_pbh)
        
        # Interpolate self.V_min_value to get the value at T_perc, but remember that self.V_min_value is a dictionary
        V_min_Temps = list(self.V_min_value.keys())
        V_min_values = list(self.V_min_value.values())

        V_min_value_at_T_perc = -np.interp(self.T_perc, V_min_Temps, V_min_values)

        fks = FKSCollapse(deltaV=abs(V_min_value_at_T_perc), sigma=self.sigma, vw=self.v_wall)
        self.pbh_forms = fks.does_pbh_form(self.R_perc)
        collapse_time = fks.get_collapse_time(self.R_perc) * np.sqrt(fks.get_HV2())
        if collapse_time > 1.0:
            self.pbh_forms = False

        self.m_pbh = fks.M0(self.R_perc)

        if self.verbose:
            print("pbh_forms: ", self.pbh_forms)
            print("m_pbh (g): ", self.m_pbh / GEV_PER_G)

    def calc_pbh_abundance(self, verbose: bool = False):
        if self.pbh_forms is False:
            self.f_pbh = 0.0
            return 0.0
        prefactor = (8*np.pi / 3 / M_PL**2) / OMEGA_DM / HUBBLE0**2
        abundance = prefactor * self.m_pbh * self.nf_perc_false
        # Get reheat temperature using ELENA's method
        T_reh = (1 + self.alpha)**(1/4) * self.T_perc

        self.f_pbh = abundance * S0_SM / s_SM(T_reh)

        if verbose:
            print("m_pbh (GeV): ", self.m_pbh)
            print("P_surv_pbh: ", self.P_surv_pbh)
            print("Nf_perc: ", self.Nf_perc)
            print("Nf_perc_false: ", self.Nf_perc_false)
            print("f_pbh: ", self.f_pbh)
        
        return abundance
    
    def calc_pbh_abundance_approx(self):
        prefactor = (8 * np.pi / 3 / M_PL**2) / OMEGA_DM / HUBBLE0**2
        T_reh = (1 + self.alpha) ** (1 / 4) * self.T_perc
        s_dilution = S0_SM / s_SM(T_reh)
        beta = self.beta * self.H_perc
        #nf_approx = 0.29 * np.power(1.238*beta, 4) / (192 * self.v_wall**3) / beta
        nf_approx = max(self.Gamma_f_list) / (self.beta * self.H_perc)

        self.f_pbh_approx = prefactor * self.m_pbh * nf_approx * s_dilution
        
        return prefactor * self.m_pbh * nf_approx * s_dilution
    



class FOPTGenericRefineGrid(FOPTGeneric):
    """
    Two-stage variant of :class:`FOPTGeneric`.

    Stage 1 (coarse, expensive action solve done ONCE):
        ``calc_temperature_bounds`` -> ``calc_action_over_T`` ->
        ``calc_nucleation_percolation_completion`` on an initial grid of
        ``n_temp_points`` points. This fixes the milestone temperatures
        (T_nuc, T_perc, T_perc_false, T_completion), v_wall, alpha, beta and
        the coarse profiles of Gamma, Hubble, ratio_V and logP_f.

    Stage 2 (adaptive, cheap):
        Build a second temperature grid for ``R_perc`` via ``_R_sepH_at``.
        ``Nf`` via direct ``_Nf_list_at`` on the stage-1 barrier grid;
        ``P_surv_pbh`` from barrier ``compute_logP_f(R_0=R_perc)``.
        The adaptive mesh still feeds integrand diagnostics/plots.
    """

    def __init__(self, point, verbose=False, model="polynomial",
                 n_temp_points=200, n_adaptive=80, adaptive_floor=0.05,
                 n_sub=32, sub_span_frac=0.3,
                 nf_cum_f=log_cumulative_trapz):
        super().__init__(point, verbose, model)
        self.n_temp_points = n_temp_points
        self.n_adaptive = n_adaptive
        self.adaptive_floor = adaptive_floor
        # Lower-limit sub-refinement: the npatches integrand is sharply peaked
        # at the integration lower limit (T_perc / T_perc_false), so the integral
        # is dominated by the first cell there. Insert n_sub log-spaced points
        # within the first sub_span_frac of [T_lower, Tc] above each lower limit
        # to resolve that spike and make the integral converge.
        self.n_sub = n_sub
        self.sub_span_frac = sub_span_frac
        # Cumulative integrator handed to ``Gamma_f_vals_at``. Defaults to the
        # log-linear rule (``log_cumulative_trapz``), paired with stage-1
        # ``compute_Gamma_f(..., integrator='log_trapz')``.
        # Pass ``nf_cum_f=None`` to fall back to ``Gamma_f_vals_at``'s own default.
        self.nf_cum_f = nf_cum_f

        # Stage-1 (coarse) profiles, retained for the stage-2 reconstruction.
        self.Temps_init = None
        self.Gamma_init = None
        self.Hubble_init = None
        self.ratio_V_init = None
        self.logP_f_init = None
        self.Nf_list_init = None
        self.Gamma_f_list_init = None
        self.timings = {}
        # Populated by run_pipeline: milestone locations vs barrier/adaptive grids.
        self.milestone_drift = None
        self.npatches_diag = None

    # ------------------------------------------------------------------
    # Milestone drift diagnostics (stage-1 limit vs grid placement)
    # ------------------------------------------------------------------
    @staticmethod
    def _grid_snap(T_grid, T_target):
        """Nearest grid point to T_target and bracketing cell widths."""
        out = {
            "T": float("nan"),
            "idx": -1,
            "err_abs": float("nan"),
            "err_rel": float("nan"),
            "dT_lo": float("nan"),
            "dT_hi": float("nan"),
            "on_grid": False,
        }
        if T_target is None or not np.isfinite(T_target):
            return out
        T_grid = np.asarray(T_grid, dtype=float)
        if T_grid.size == 0:
            out["T"] = float(T_target)
            return out
        idx = int(np.argmin(np.abs(T_grid - T_target)))
        err = float(abs(T_grid[idx] - T_target))
        scale = max(abs(T_target), 1e-30)
        out.update({
            "T": float(T_target),
            "idx": idx,
            "err_abs": err,
            "err_rel": err / scale,
            "dT_lo": float(T_grid[idx] - T_grid[idx - 1]) if idx > 0 else float("nan"),
            "dT_hi": float(T_grid[idx + 1] - T_grid[idx]) if idx < len(T_grid) - 1 else float("nan"),
            "on_grid": err <= 1e-12 * scale,
        })
        return out

    @staticmethod
    def _logpf_bracket_residual(T_grid, logP_f, T_milestone, logP_target):
        """How well logP_f(T_milestone) matches the Pf milestone target."""
        if (T_milestone is None or not np.isfinite(T_milestone)
                or logP_target is None or not np.isfinite(logP_target)):
            return float("nan")
        T_grid = np.asarray(T_grid, dtype=float)
        logP_f = np.asarray(logP_f, dtype=float)
        mask = np.isfinite(logP_f)
        if mask.sum() < 2:
            return float("nan")
        logP_at = float(np.interp(T_milestone, T_grid[mask], logP_f[mask]))
        return logP_at - float(logP_target)

    def _collect_milestone_drift(self):
        """
        Snapshot milestone temperatures and their placement on the stage-1
        barrier grid (Temps_init) and the stage-2 adaptive grid (Temps).

        Used to diagnose n_temp sweeps: if T_perc_false jumps between runs,
        integration limits move discontinuously even when f_pbh looks stable.
        """
        T_barrier = self.Temps_init if self.Temps_init is not None else np.array([])
        T_adaptive = np.asarray(self.Temps, dtype=float)
        logPf_bar = self.logP_f_init if self.logP_f_init is not None else np.array([])

        logPf_true = np.log(0.71)
        logPf_false = np.log(0.29)

        milestone_specs = (
            ("T_nuc", self.T_nuc, None),
            ("T_perc", self.T_perc, logPf_true),
            ("T_perc_false", self.T_perc_false, logPf_false),
            ("T_completion", self.T_completion, np.log(0.01)),
        )
        milestones = {}
        for name, T_m, logP_target in milestone_specs:
            entry = {
                "barrier": self._grid_snap(T_barrier, T_m),
                "adaptive": self._grid_snap(T_adaptive, T_m),
            }
            if logP_target is not None and len(T_barrier) >= 2 and len(logPf_bar) >= 2:
                entry["logP_f_residual"] = self._logpf_bracket_residual(
                    T_barrier, logPf_bar, T_m, logP_target)
            milestones[name] = entry

        s3_keys = sorted(self.S3overT.keys()) if self.S3overT else []
        if s3_keys:
            T_s3_lo = float(s3_keys[0])
            s3_lo = float(self.S3overT[s3_keys[0]])
        else:
            T_s3_lo, s3_lo = float("nan"), float("nan")

        pf_entry = milestones.get("T_perc_false", {})
        if self.npatches_diag and self.npatches_diag.get("T_perc_false"):
            pf_entry["npatches"] = dict(self.npatches_diag["T_perc_false"])
            milestones["T_perc_false"] = pf_entry

        self.milestone_drift = {
            "n_temp_points": int(self.n_temp_points),
            "n_adaptive": int(self.n_adaptive),
            "n_sub": int(self.n_sub),
            "n_s3": len(self.S3overT),
            "n_barrier": int(len(T_barrier)),
            "n_adaptive_grid": int(len(T_adaptive)),
            "Tc": float(self.Tc),
            "S3_lo_T": T_s3_lo,
            "S3_lo_val": s3_lo,
            "milestones": milestones,
        }
        return self.milestone_drift

    @staticmethod
    def compare_milestone_drift(prev, curr):
        """
        Step-to-step deltas between two ``milestone_drift`` snapshots
        (e.g. consecutive n_temp_points in a sweep).
        """
        if prev is None or curr is None:
            return {}
        deltas = {
            "n_barrier": curr["n_barrier"] - prev["n_barrier"],
            "n_s3": curr["n_s3"] - prev["n_s3"],
        }
        for name in ("T_nuc", "T_perc", "T_perc_false", "T_completion"):
            if name not in prev["milestones"] or name not in curr["milestones"]:
                continue
            T_prev = prev["milestones"][name]["barrier"]["T"]
            T_curr = curr["milestones"][name]["barrier"]["T"]
            if np.isfinite(T_prev) and np.isfinite(T_curr):
                deltas[f"d{name}"] = T_curr - T_prev
                deltas[f"d{name}_rel"] = (T_curr - T_prev) / max(abs(T_prev), 1e-30)
            b_prev = prev["milestones"][name]["barrier"]
            b_curr = curr["milestones"][name]["barrier"]
            if np.isfinite(b_prev.get("err_rel", np.nan)):
                deltas[f"{name}_bar_err_rel"] = b_curr["err_rel"]
        pf = curr["milestones"].get("T_perc_false", {})
        if "logP_f_residual" in pf:
            deltas["T_perc_false_logPf_res"] = pf["logP_f_residual"]
        return deltas

    @staticmethod
    def format_milestone_drift(drift, deltas=None):
        """One-line summary for logging."""
        if drift is None:
            return "(no milestone drift snapshot)"
        pf = drift["milestones"]["T_perc_false"]
        b = pf["barrier"]
        a = pf["adaptive"]
        parts = [
            f"n_s3={drift['n_s3']}",
            f"n_bar={drift['n_barrier']}",
            f"T_pf={b['T']:.6g}",
            f"bar_err={b['err_rel']:.2e}",
            f"adp_err={a['err_rel']:.2e}",
            f"adp_pin={a['on_grid']}",
        ]
        if "logP_f_residual" in pf:
            parts.append(f"logPf_res={pf['logP_f_residual']:.2e}")
        npd = pf.get("npatches") or {}
        if npd:
            dT1 = npd.get("dT_first", float("nan"))
            frac = npd.get("first_cell_frac", float("nan"))
            Nf = npd.get("Nf", float("nan"))
            if np.isfinite(Nf):
                parts.append(f"Nf={Nf:.3e}")
        if deltas:
            dT = deltas.get("dT_perc_false", float("nan"))
            if np.isfinite(dT):
                parts.append(f"dT_pf={dT:+.3e}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def run_pipeline(self, time_verbose: bool = False,
                     log_temps: bool = True,
                     n_adaptive: int = None):
        if n_adaptive is not None:
            self.n_adaptive = n_adaptive

        # ---- Stage 1: coarse grid, run once -------------------------------
        t0 = time()
        self.calc_temperature_bounds()
        self.calc_action_over_T(log_temps=log_temps)
        self.calc_nucleation_percolation_completion()
        self.timings['stage1'] = time() - t0
        if time_verbose:
            print(f"[stage1] coarse pipeline: {self.timings['stage1']:.3f} s "
                  f"(n_temp_points={self.n_temp_points}, grid={len(self.Temps)})")

        # Snapshot the coarse profiles before overwriting with the adaptive grid.
        self.Temps_init = np.asarray(self.Temps, dtype=float).copy()
        self.Gamma_init = np.asarray(self.Gamma, dtype=float).copy()
        self.Hubble_init = np.asarray(self.Hubble, dtype=float).copy()
        self.ratio_V_init = np.asarray(self.ratio_V, dtype=float).copy()
        self.logP_f_init = np.asarray(self.logP_f, dtype=float).copy()
        self.Nf_list_init = np.asarray(self.Nf_list, dtype=float).copy()
        self.Gamma_f_list_init = np.asarray(self.Gamma_f_list, dtype=float).copy()

        # ---- Stage 2: adaptive grid recompute -----------------------------
        t0 = time()
        T_adaptive = self._build_adaptive_grid(self.n_adaptive)
        self._recompute_on_grid(T_adaptive)
        self.timings['stage2_recompute'] = time() - t0

        t0 = time()
        self.calc_npatches()
        self.timings['calc_npatches'] = time() - t0

        t0 = time()
        self.calc_mean_bubble_size()
        self.timings['calc_mean_bubble_size'] = time() - t0

        t0 = time()
        self.calc_pbh_abundance(verbose=False)
        self.timings['calc_pbh_abundance'] = time() - t0
        self.timings['total'] = sum(self.timings[k] for k in
                                     ('stage1', 'stage2_recompute',
                                      'calc_npatches', 'calc_mean_bubble_size',
                                      'calc_pbh_abundance'))
        self._collect_milestone_drift()
        if time_verbose:
            print(f"[stage2] adaptive grid={len(self.Temps)} | "
                  f"recompute={self.timings['stage2_recompute']:.3f}s "
                  f"npatches={self.timings['calc_npatches']:.3f}s "
                  f"total={self.timings['total']:.3f}s")
            print(f"[milestones] {self.format_milestone_drift(self.milestone_drift)}")

    # ------------------------------------------------------------------
    # Stage-2 helpers
    # ------------------------------------------------------------------
    def _build_adaptive_grid(self, n_adaptive):
        """
        Place ``n_adaptive`` points over the coarse grid range with a density
        that follows the profile of Gamma. The weight is the normalised
        |d ln(Gamma)/dT| plus a uniform floor so the whole range stays
        sampled. Points are equidistributed in the cumulative-weight
        coordinate, then the milestone temperatures are inserted exactly so
        the integration limits land on grid points, bounce-solve temperatures
        in the cliff window are pinned, and finally a cluster of log-spaced
        sub-points is added just above each integration lower limit
        (T_perc_false, T_perc) to resolve the integrand spike there.
        """
        T0 = self.Temps_init
        G0 = self.Gamma_init

        pos = G0 > 0
        lnG = np.empty_like(T0)
        if pos.any():
            lnG[pos] = np.log(G0[pos])
            lnG[~pos] = np.min(lnG[pos])
        else:
            lnG[:] = 0.0

        dlnG = np.abs(np.gradient(lnG, T0))
        peak = np.max(dlnG)
        weight = (dlnG / peak if peak > 0 else np.zeros_like(dlnG))
        weight = weight + self.adaptive_floor

        cumw = cumulative_trapezoid(weight, T0, initial=0.0)
        if cumw[-1] <= 0:
            T_adaptive = np.linspace(T0[0], T0[-1], n_adaptive)
        else:
            cumw /= cumw[-1]
            targets = np.linspace(0.0, 1.0, n_adaptive)
            T_adaptive = np.interp(targets, cumw, T0)

        # Pin the milestone temperatures so idx-snapping in _Nf_list_at /
        # _R_sepH_at lands exactly on them.
        for Tk in (self.T_perc_false, self.T_perc, self.T_nuc):
            if Tk is not None and np.isfinite(Tk) and T0[0] <= Tk <= T0[-1]:
                T_adaptive = np.append(T_adaptive, float(Tk))

        # Pin bounce-solve temperatures in the cliff / npatches window so
        # Gamma and logP_f re-integration never sit on large S3 gaps.
        T_adaptive = self._pin_s3_bounce_keys(T_adaptive)

        # Resolve the integrand spike at each integration lower limit.
        for Tk in (self.T_perc_false, self.T_perc):
            T_adaptive = self._refine_lower_limit(T_adaptive, Tk)

        T_adaptive = np.unique(T_adaptive)  # sorted ascending, deduplicated
        return T_adaptive

    def _pin_s3_bounce_keys(self, T_grid):
        """
        Append every bounce-solve temperature from ``S3overT`` that lies in
        the percolation cliff / ``Nf`` integration window.

        Stage-2 ``Gamma`` is reconstructed from interpolated ``S3/T`` between
        keys; when ``n_temp_points`` is small the adaptive mesh can land
        entirely in a gap, amplifying ``nf`` and ``f_pbh``.  Inserting the
        actual solve temperatures removes those gaps without re-running the
        bounce action.
        """
        if not self.S3overT:
            return T_grid
        T_hi = float(self.Temps_init[-1])
        T_lo = float(self.Temps_init[0])
        anchors = [T_lo, T_hi]
        for Tk in (self.T_perc_false, self.T_perc, self.T_nuc):
            if Tk is not None and np.isfinite(Tk):
                anchors.append(float(Tk))
        T_win_lo = min(anchors)
        if self.T_perc_false is not None and np.isfinite(self.T_perc_false):
            width = max(self.Tc - self.T_perc_false, 1e-30)
            T_win_lo = min(T_win_lo, float(self.T_perc_false) - 0.05 * width)
        T_win_lo = max(T_win_lo, T_lo)

        s3_T = np.array(sorted(self.S3overT.keys()), dtype=float)
        in_window = s3_T[(s3_T >= T_win_lo) & (s3_T <= T_hi)]
        if in_window.size == 0:
            return T_grid
        return np.concatenate([T_grid, in_window])

    def _refine_lower_limit(self, T_grid, T_lower, n_sub=None, span_frac=None):
        """
        Insert ``n_sub`` log-spaced points just above ``T_lower`` (an integration
        lower limit) to resolve the npatches integrand, which is sharply peaked
        there. Points are spaced geometrically in the distance above T_lower, so
        they cluster tightly at the limit and thin out toward
        ``T_lower + span_frac*(Tc - T_lower)``.
        """
        n_sub = self.n_sub if n_sub is None else n_sub
        span_frac = self.sub_span_frac if span_frac is None else span_frac
        if n_sub <= 0 or T_lower is None or not np.isfinite(T_lower):
            return T_grid
        width = self.Tc - T_lower
        if width <= 0:
            return T_grid
        d_lo = width * 1e-4
        d_hi = width * span_frac
        sub = float(T_lower) + np.geomspace(d_lo, d_hi, n_sub)
        # Keep sub-points strictly inside the coarse grid range used for interp.
        sub = sub[sub <= self.Temps_init[-1]]
        return np.concatenate([T_grid, sub])

    def _augment_grid_at(self, T_target, *fields):
        """
        Like the parent, but H and ratio_V at ``T_target`` use
        ``_thermo_from_barrier``; Gamma uses ``_gamma_from_S3``.
        """
        Temps = self.Temps
        if T_target >= Temps[-1]:
            return (None,) * (1 + len(fields))
        mask = Temps > T_target
        T_above = Temps[mask]
        T_arr = np.concatenate([[T_target], T_above])
        out = [T_arr]

        field_names = ("Gamma", "ratio_V", "logP_f", "Hubble")
        for j, arr in enumerate(fields):
            name = field_names[j] if j < len(field_names) else None
            if name == "Gamma":
                val_at = float(self._gamma_from_S3(np.array([T_target]))[0])
            elif name == "ratio_V":
                _, val_rv = self._thermo_from_barrier(np.array([T_target]))
                val_at = float(val_rv[0])
            elif name == "Hubble":
                val_h, _ = self._thermo_from_barrier(np.array([T_target]))
                val_at = float(val_h[0])
            else:
                val_at = float(np.interp(T_target, Temps, arr))
            out.append(np.concatenate([[val_at], arr[mask]]))
        return tuple(out)

    def _recompute_on_grid(self, T_adaptive):
        """
        Reconstruct the quantities feeding calc_npatches on ``T_adaptive``:
          - Gamma  : ``_gamma_from_S3`` with log-interpolated S3/T cliff
          - Hubble : ``_thermo_from_barrier`` (V_min + model derivatives)
          - ratio_V: ``_thermo_from_barrier``
          - logP_f : re-integrated per slice via _logP_f_single_slice
        """
        T_adaptive = np.asarray(T_adaptive, dtype=float)

        Gamma_ad = self._gamma_from_S3(T_adaptive)
        H_ad, rV_ad = self._thermo_from_barrier(T_adaptive)
        f_ext_ad = rV_ad * Gamma_ad / H_ad

        v_w = getattr(self, "v_wall_gamma_f", self.v_wall)
        N = len(T_adaptive)
        logP_f_ad = np.zeros(N, dtype=float)
        for i in range(N - 1):
            logP_f_ad[i] = _logP_f_single_slice(
                T_adaptive[i:], rV_ad[i:], H_ad[i:], f_ext_ad[i:], v_w,
                integrator='log_trapz')
        logP_f_ad = np.nan_to_num(logP_f_ad)

        self.Temps = T_adaptive
        self.Gamma = Gamma_ad
        self.Hubble = H_ad
        self.ratio_V = rV_ad
        self.logP_f = logP_f_ad

    # ------------------------------------------------------------------
    # calc_npatches: same physics as the parent, but the O(N^3) false-vacuum
    # integral nf(T) is built ONCE and reused for both T_perc and T_perc_false
    # (the parent recomputes it twice).
    # ------------------------------------------------------------------
    def _compute_gamma_f_array(self):
        Temps, H = self.Temps, self.Hubble
        rV, G, lp = self.ratio_V, self.Gamma, self.logP_f
        v_w = self.v_wall_gamma_f
        cum_f = self.nf_cum_f if self.nf_cum_f is not None else log_cumulative_trapz
        gamma_f_vals = np.zeros_like(Temps)
        for i in range(len(Temps) - 1):
            gamma_f_vals[i] = Gamma_f_vals_at(
                Temps[i:], H[i:], rV[i:], G[i:], lp[i], v_w, cum_f=cum_f)
        return gamma_f_vals

    def _nf_integrand_on_grid(self, T_arr, G_arr, H_arr, rv_arr, lp_arr):
        """Build ``Gamma_f`` slice values and the npatches integrand."""
        v_w = self.v_wall_gamma_f
        cum_f = self.nf_cum_f if self.nf_cum_f is not None else log_cumulative_trapz
        logP_t = np.array([stable_logP_t(lp) for lp in lp_arr])
        gamma_f_vals = np.zeros_like(T_arr)
        for k in range(len(T_arr) - 1):
            gamma_f_vals[k] = Gamma_f_vals_at(
                T_arr[k:], H_arr[k:], rv_arr[k:], G_arr[k:],
                lp_arr[k], v_w, cum_f=cum_f)
        integrand = gamma_f_vals * np.exp(logP_t) * rv_arr / H_arr ** 4
        return gamma_f_vals, integrand

    def _logP_f_on_grid(self, T_arr, G_arr, H_arr, rv_arr):
        """Re-integrate ``logP_f`` on an ascending temperature slice."""
        v_w = getattr(self, "v_wall_gamma_f", self.v_wall)
        f_ext = rv_arr * G_arr / H_arr
        lp = np.zeros(len(T_arr), dtype=float)
        for i in range(len(T_arr) - 1):
            lp[i] = _logP_f_single_slice(
                T_arr[i:], rv_arr[i:], H_arr[i:], f_ext[i:], v_w,
                integrator='log_trapz')
        return np.nan_to_num(lp)

    def _Nf_at(self, T_target, n_refine=None):
        """
        Nf patches integral from ``T_target`` to ``Tc`` with ``T_target``
        inserted as the lower limit (``_augment_grid_at``), matching the
        grid-locking fix used in ``_R_sepH_at``.
        """
        if not np.isfinite(T_target):
            return np.nan, None
        n_refine = self.n_sub if n_refine is None else n_refine
        aug = self._augment_grid_at(T_target, self.Gamma, self.ratio_V,
                                    self.logP_f, self.Hubble)
        if aug[0] is None:
            return np.nan, None
        T_arr, G_arr, rv_arr, lp_arr, H_arr = aug

        if n_refine > 0 and len(T_arr) >= 2:
            T_arr, rv_arr, H_arr, G_arr, lp_arr = self._refine_grid_above(
                T_target, T_arr, n_refine, rv_arr, H_arr, G_arr, lp_arr)

        G_arr = self._gamma_from_S3(T_arr)
        H_arr, rv_arr = self._thermo_from_barrier(T_arr)
        lp_arr = self._logP_f_on_grid(T_arr, G_arr, H_arr, rv_arr)

        idx_max = int(np.argmin(np.abs(T_arr - self.Tc))) - 1
        if idx_max < 1:
            return np.nan, None

        T_slice = T_arr[:idx_max + 1]
        gamma_f_vals, integrand = self._nf_integrand_on_grid(
            T_slice, G_arr[:idx_max + 1], H_arr[:idx_max + 1],
            rv_arr[:idx_max + 1], lp_arr[:idx_max + 1])

        val = _log_trapz(integrand, T_slice)
        if not np.isfinite(val) or val <= 0:
            return np.nan, None

        Nf = 4.0 * np.pi / 9.0 * val
        dT_first = float(T_slice[1] - T_slice[0])
        first_cell = float(_log_trapz(integrand[:2], T_slice[:2]))
        diag = {
            "T_lower": float(T_target),
            "dT_first": dT_first,
            "integrand_0": float(integrand[0]),
            "gamma_f_0": float(gamma_f_vals[0]),
            "first_cell_contrib": first_cell,
            "first_cell_frac": first_cell / val if val > 0 else float("nan"),
            "Nf": float(Nf),
        }
        return Nf, diag

    def calc_npatches(self, n_refine=0):
        """
        ``Nf`` via direct ``_Nf_list_at`` on the stage-1 barrier grid
        (same path as ``FOPTGeneric.calc_npatches``, using ``Temps_init``).
        """
        n_ref = self.n_sub if n_refine == 0 else n_refine
        barrier = dict(
            temps=self.Temps_init,
            hubble=self.Hubble_init,
            ratio_v=self.ratio_V_init,
            gamma=self.Gamma_init,
            logp_f=self.logP_f_init,
        )
        Tb = self.Temps_init
        H_b = self.Hubble_init

        Nf_perc = self._Nf_list_at(self.T_perc, n_refine=n_ref, **barrier)
        Nf_perc_false = self._Nf_list_at(self.T_perc_false, n_refine=n_ref,
                                         **barrier)
        H_perc = float(np.interp(self.T_perc, Tb, H_b))
        H_perc_false = float(np.interp(self.T_perc_false, Tb, H_b))

        self.npatches_diag = {
            "T_perc": {"Nf": float(Nf_perc), "source": "direct_Nf_list_at"},
            "T_perc_false": {
                "Nf": float(Nf_perc_false),
                "source": "direct_Nf_list_at",
            },
        }

        Hubble_vol_perc = 4 * np.pi / 3 * H_perc ** (-3)
        nf_perc = Nf_perc / Hubble_vol_perc
        Hubble_vol_perc_false = 4 * np.pi / 3 * H_perc_false ** (-3)
        nf_perc_false = Nf_perc_false / Hubble_vol_perc_false

        self.Nf_perc = Nf_perc
        self.H_perc = H_perc
        self.nf_perc = nf_perc
        self.Nf_perc_false = Nf_perc_false
        self.H_perc_false = H_perc_false
        self.nf_perc_false = nf_perc_false

        if self.verbose:
            print("Nf_perc: ", self.Nf_perc)
            print("Nf_perc_false: ", self.Nf_perc_false)
            print("nf_perc_false: ", self.nf_perc_false)

    # ------------------------------------------------------------------
    # Bubble size + abundance: same physics as the parent, but the abundance
    # folds in the survival probability P_surv_pbh (candidate A).
    # ------------------------------------------------------------------
    def calc_mean_bubble_size(self):
        """
        Mean bubble size on the stage-2 adaptive grid; survival probability
        on the stage-1 barrier grid.

        ``R_perc`` is integrated on the active adaptive arrays via
        ``_R_sepH_at``.  ``P_surv_pbh`` uses ``logP_f(T, R_0=R_perc)``
        from ``compute_logP_f`` on ``V_min_value`` / ``S3overT`` (the
        parent ELENA path), not the stage-2 re-integrated ``logP_f``
        profile — the latter was returning ``P_surv ≈ 1`` when stage-1
        was sparse even though ``Nf`` on the adaptive grid was wrong for
        a different reason.
        """
        R_perc = self._R_sepH_at(self.T_perc)
        self.R_perc = R_perc

        logP_f_R, Temps_, _, _, _ = compute_logP_f(
            self.model,
            self.V_min_value,
            self.S3overT,
            v_w=self.v_wall,
            units=self.units,
            cum_method='None',
            R_0=R_perc,
            integrator='log_trapz')
        mask_PfR = ~np.isnan(logP_f_R)
        logPf_Rperc = interpolation_narrow(Temps_[mask_PfR], logP_f_R[mask_PfR],
                                           self.T_perc)
        self.P_surv_pbh = 1 - np.exp(logPf_Rperc)

        if self.verbose:
            print("R_perc: ", self.R_perc)
            print("P_surv_pbh: ", self.P_surv_pbh)

        V_min_Temps = list(self.V_min_value.keys())
        V_min_values = list(self.V_min_value.values())
        V_min_value_at_T_perc = -np.interp(self.T_perc, V_min_Temps, V_min_values)

        fks = FKSCollapse(deltaV=abs(V_min_value_at_T_perc), sigma=self.sigma,
                          vw=self.v_wall)
        self.pbh_forms = fks.does_pbh_form(self.R_perc)
        collapse_time = fks.get_collapse_time(self.R_perc) * np.sqrt(fks.get_HV2())
        if collapse_time > 1.0:
            self.pbh_forms = False

        self.m_pbh = fks.M0(self.R_perc)

        if self.verbose:
            print("pbh_forms: ", self.pbh_forms)
            print("m_pbh (g): ", self.m_pbh / GEV_PER_G)

    def calc_pbh_abundance(self, verbose: bool = False):
        """
        Candidate A: fold the survival probability ``P_surv_pbh`` into f_pbh.

        The parent computes ``P_surv_pbh = 1 - exp(logP_f at R_perc)`` in
        ``calc_mean_bubble_size`` but then discards it, leaving
            f_pbh = prefactor * m_pbh * nf_perc_false * s_dilution.
        ``P_surv_pbh`` is the (strongly point-dependent) probability that a
        Hubble patch is still in the false vacuum at the collapse scale
        ``R_perc``; omitting it both inflates f_pbh (the refine-grid method
        produced unphysical f_pbh > 1) and washes out the point-to-point
        spread. Multiplying it back in restores the missing suppression.
        """
        if self.pbh_forms is False:
            self.f_pbh = 0.0
            return 0.0
        prefactor = (8 * np.pi / 3 / M_PL**2) / OMEGA_DM / HUBBLE0**2
        abundance = (prefactor * self.m_pbh * self.nf_perc_false
                     * self.P_surv_pbh)
        # Get reheat temperature using ELENA's method
        T_reh = (1 + self.alpha)**(1/4) * self.T_perc

        self.f_pbh = abundance * S0_SM / s_SM(T_reh)

        if verbose:
            print("m_pbh (GeV): ", self.m_pbh)
            print("P_surv_pbh: ", self.P_surv_pbh)
            print("nf_perc_false: ", self.nf_perc_false)
            print("f_pbh: ", self.f_pbh)

        return abundance





"""
PBH = FKSCollapse(DeltaV, sigma, point['v_w'])
H_form, RH_form = point['H_perc_false'], point['R_mean_falseH_perc_false']
R_star = RH_form / H_form
t_coll = PBH.get_collapse_time(R_star)
M_star = PBH.M0(R_star)
# M_tp = PBH.get_M_at_zTP()
# Calculate PBH abundance
nf_form = point['nf_perc_false']
nPBH_today = nf_form * S0_SM / s_SM(point['T_reh']) # entropy injection from reheating?
OmegaCDMh2 = 0.120
H0_h = 100 * 3.24078e-20 * (1 / 6.582119569e-25)**(-1)
f_PBH = (8*np.pi / 3 / M_PL**2) / OmegaCDMh2 / H0_h**2 * M_star * nPBH_today
"""









