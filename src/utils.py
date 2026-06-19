# Here we store useful functions and constants

import numpy as np
from dof_interpolation import g_rho, g_s, g_rho_spline, dg_rho_spline, d2g_rho_spline

M_pl = 2.4353234600842885e+18

def is_increasing(arr):
    return np.all(arr[:-1] <= arr[1:])


def stable_logP_t(logP_f):
    """log(1 - exp(logP_f)) without underflow when logP_f << 0."""
    logP_f = float(logP_f)
    if logP_f >= -1e-12:
        return np.log(max(-logP_f, 1e-300))
    return float(np.log1p(-np.exp(logP_f)))


def _log_trapz_cell(yi, yj, dx):
    """Single-cell contribution assuming log-linear y(x) between endpoints."""
    if yi <= 0.0 or yj <= 0.0 or not np.isfinite(yi) or not np.isfinite(yj):
        return 0.5 * (yi + yj) * dx
    if yi == yj:
        return yi * dx
    log_ratio = np.log(yj / yi)
    if abs(log_ratio) < 1e-12:
        return yi * dx
    return (yj - yi) * dx / log_ratio


def log_trapz(y, x):
    """
    Integrate y(x) with log-linear interpolation per cell (exact for
    exponential integrands). Falls back to trapezoidal on bad cells.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    total = 0.0
    for i in range(len(x) - 1):
        total += _log_trapz_cell(y[i], y[i + 1], x[i + 1] - x[i])
    return total


def log_cumulative_trapz(y, x, initial=0.0):
    """Cumulative version of log_trapz, matching cumulative_trapezoid layout."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    out[0] = initial
    total = float(initial)
    for i in range(n - 1):
        total += _log_trapz_cell(y[i], y[i + 1], x[i + 1] - x[i])
        out[i + 1] = total
    return out


def interp_positive_log(xq, x, y):
    """
    Interpolate positive ``y(x)`` at ``xq`` with log-linear rule per cell
    (exact for exponential profiles). Falls back to ``np.interp`` on bad data.
    """
    xq = float(xq)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float(y[0]) if len(y) else float("nan")
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[-1]:
        return float(y[-1])
    i = int(np.searchsorted(x, xq) - 1)
    x0, x1 = float(x[i]), float(x[i + 1])
    y0, y1 = float(y[i]), float(y[i + 1])
    if (not np.isfinite(y0)) or (not np.isfinite(y1)) or y0 <= 0.0 or y1 <= 0.0:
        return float(np.interp(xq, x, y))
    if y0 == y1:
        return y0
    log_ratio = np.log(y1 / y0)
    if abs(log_ratio) < 1e-12:
        return y0
    t = (xq - x0) / (x1 - x0)
    return float(y0 * (y1 / y0) ** t)


def interpolation_narrow_log(y, x, target):
    """
    Bracket ``target`` in ``y`` (same logic as ``interpolation_narrow``) but
    interpolate ``x`` via log-linear rule in ``y`` when both bracket values
    are positive; otherwise fall back to linear ``np.interp``.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    greater_than_target = y[y > target]
    less_than_target = y[y < target]
    if len(greater_than_target) == 0 or len(less_than_target) == 0:
        return np.nan
    mask = np.zeros(y.shape, dtype=bool)
    closest_greater = greater_than_target[np.abs(greater_than_target - target).argmin()]
    mask[y == closest_greater] = True
    closest_less = less_than_target[np.abs(less_than_target - target).argmin()]
    mask[y == closest_less] = True
    yb = y[mask]
    xb = x[mask]
    if yb[-1] < yb[0]:
        yb = np.flip(yb)
        xb = np.flip(xb)
    if yb[0] > 0.0 and yb[1] > 0.0 and target > 0.0:
        t = (np.log(target) - np.log(yb[0])) / (np.log(yb[1]) - np.log(yb[0]))
        return float(xb[0] + t * (xb[1] - xb[0]))
    return float(np.interp(target, yb, xb))

# Assume the standad unit is GeV
convert_units = {
    'MeV': 1e3,
    'GeV': 1,
    'TeV': 1e-3
}

def s_SM(T, units = 'GeV'):
    return (2 * np.pi**2 / 45) * g_s(T / convert_units[units]) * T**3

def rho_SM(T, units = 'GeV'):
    return (np.pi**2 / 30) * g_rho(T / convert_units[units]) * T**4

def rho_SM_spline(T, units = 'GeV'):
    T_units = T / convert_units[units]
    return (np.pi**2 / 30) * g_rho_spline(T_units) * T**4
def drho_SM_spline(T, units = 'GeV'):
    T_units = T / convert_units[units]
    return (np.pi**2 / 30) * (
        dg_rho_spline(T_units) * T**4 \
        + 4*g_rho_spline(T_units) * T**3
    )
def d2rho_SM_spline(T, units = 'GeV'):
    T_units = T / convert_units[units]
    return (np.pi**2 / 30) * (
        0*d2g_rho_spline(T_units) * T**4 # !!! the second derivative of the data is simply too sporatic
        + 4*dg_rho_spline(T_units) * T**3
        + 4*dg_rho_spline(T_units) * T**3 + 12*g_rho_spline(T_units) * T**2
    )


def interpolation_narrow(y, x, target):
    # Identify elements greater than the target
    greater_than_target = y[y > target]
    # Identify elements less than the target
    less_than_target = y[y < target]
    
    if len(greater_than_target) > 0 and len(less_than_target) > 0:
        # Initialize an empty mask
        mask = np.zeros(y.shape, dtype=bool)

        # Find closest element greater than target
        closest_greater = greater_than_target[np.abs(greater_than_target - target).argmin()]
        mask[y == closest_greater] = True

        # Find closest element less than target
        closest_less = less_than_target[np.abs(less_than_target - target).argmin()]
        mask[y == closest_less] = True

        if y[mask][-1] >= y[mask][0]:
            return np.interp(target, y[mask], x[mask])
        else:
            return np.interp(target, np.flip(y[mask]), np.flip(x[mask]))
    else:
        return np.nan
    

def g_crit(lambda_):
        return ( ((32 * np.pi**2 * lambda_)/3) * (1 - lambda_ * (18 + 4*np.log(2) )/(32 * np.pi**2) ) )**(1/4)

def g_roll(lambda_):
    return ( ((16 * np.pi**2 * lambda_)/3) * (1 - lambda_ * (5 + 2*np.log(2) )/(8 * np.pi**2) ) )**(1/4)


def good_point(instance, which_T_volume = 'percolation_and_completion'):
    if instance.T_completion is None:
        return False
    if which_T_volume == 'percolation_and_completion':
        return instance.Vf_contracting_at_T_perc and instance.Vf_contracting_at_T_completion
    if which_T_volume == 'percolation':
        return instance.Vf_contracting_at_T_perc
    if which_T_volume == 'completion':
        return instance.Vf_contracting_at_T_completion
    if which_T_volume == 'any':
        return instance.Vf_contracting_somewhere


def find_g_max(lambda_, physical_vev, precision = 1e-2, T_step = 1e-2, refine_Tmin_precision = 2, which_T_volume = 'percolation_and_completion'):
    from temperatures import Temperatures
    delta = 1e-1
    g0 = g_roll(lambda_)
    if np.isnan(g0) or g0==0:
        g0 = g_crit(lambda_)
    g = g0
    instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision)

    if good_point(instance, which_T_volume):
        while delta >= precision:
            first = True
            while good_point(instance, which_T_volume):
                g_old = g
                instance_old = instance
                g = (1 + delta) * g_old
                if g == g_old:
                    return g_old, [instance_old.T_max, instance_old.T_nuc, instance_old.T_perc, instance_old.T_completion, instance_old.T_min], [instance_old.Vf_contracting_at_T_perc, instance_old.Vf_contracting_at_T_completion, instance_old.Vf_contracting_somewhere]
                instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision)
                while first and not good_point(instance, which_T_volume):
                    delta = delta / 10
                    g = (1 + delta) * g_old
                    if g == g_old:
                        return g_old, [instance_old.T_max, instance_old.T_nuc, instance_old.T_perc, instance_old.T_completion, instance_old.T_min], [instance_old.Vf_contracting_at_T_perc, instance_old.Vf_contracting_at_T_completion, instance_old.Vf_contracting_somewhere]
                    instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision)
                first = False  
            instance_return = instance_old
            instance = instance_old
            g_return = g_old
            g = g_old
            delta = delta / 10
    else:
        while delta >= precision:
            first = True
            while not good_point(instance, which_T_volume):
                g_old = g
                instance_old = instance
                g = (1 - delta) * g_old
                if g == g_old:
                    return g_old, [instance_old.T_max, instance_old.T_nuc, instance_old.T_perc, instance_old.T_completion, instance_old.T_min], [instance_old.Vf_contracting_at_T_perc, instance_old.Vf_contracting_at_T_completion, instance_old.Vf_contracting_somewhere]
                instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision)
                while first and good_point(instance, which_T_volume):
                    delta = delta / 10
                    g = (1 - delta) * g_old
                    if g == g_old:
                        return g_old, [instance_old.T_max, instance_old.T_nuc, instance_old.T_perc, instance_old.T_completion, instance_old.T_min], [instance_old.Vf_contracting_at_T_perc, instance_old.Vf_contracting_at_T_completion, instance_old.Vf_contracting_somewhere]
                    instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision)
                first = False  
                if g < g0 / 10:
                    if delta > 1e-3:
                        g = g0   
                        delta = delta / 10
                    else:
                        g = None
                        return g, [instance.T_max, instance.T_nuc, instance.T_perc, instance.T_completion, instance.T_min], [instance.Vf_contracting_at_T_perc, instance.Vf_contracting_at_T_completion, instance.Vf_contracting_somewhere]       
            instance_return = instance
            instance = instance_old
            g_return = g
            g = g_old
            delta = delta / 10
    
    return g_return, [instance_return.T_max, instance_return.T_nuc, instance_return.T_perc, instance_return.T_completion, instance_return.T_min], [instance_return.Vf_contracting_at_T_perc, instance_return.Vf_contracting_at_T_completion, instance_return.Vf_contracting_somewhere]

    

def find_g_min(lambda_, physical_vev, precision = 1e-2, g0 = None, T_step = 1e-2, refine_Tmin_precision = 2, which_T_volume = 'percolation_and_completion'):
    from temperatures import Temperatures
    delta = 1e-1
    if g0 is None:
        g0 = g_roll(lambda_) / 2
        if np.isnan(g0) or g0==0:
            g0 = g_crit(lambda_) / 2
    g = g0
    instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision)

    if good_point(instance, which_T_volume):
        while delta >= precision:
            first = True
            while good_point(instance, which_T_volume):
                g_old = g
                instance_old = instance
                g = (1 - delta) * g_old
                if g == g_old:
                    return g_old, [instance_old.T_max, instance_old.T_nuc, instance_old.T_perc, instance_old.T_completion, instance_old.T_min], [instance_old.Vf_contracting_at_T_perc, instance_old.Vf_contracting_at_T_completion, instance_old.Vf_contracting_somewhere]
                instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision)
                while first and not good_point(instance, which_T_volume):
                    delta = delta / 10
                    g = (1 - delta) * g_old
                    if g == g_old:
                        return g_old, [instance_old.T_max, instance_old.T_nuc, instance_old.T_perc, instance_old.T_completion, instance_old.T_min], [instance_old.Vf_contracting_at_T_perc, instance_old.Vf_contracting_at_T_completion, instance_old.Vf_contracting_somewhere]
                    instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision) 
                first = False                  
            instance_return = instance_old
            instance = instance_old
            g_return = g_old
            g = g_old
            delta = delta / 10
    else:
        while delta >= precision:
            first = True
            while not good_point(instance, which_T_volume):
                g_old = g
                instance_old = instance
                g = (1 + delta) * g_old
                if g == g_old:
                    return g_old, [instance_old.T_max, instance_old.T_nuc, instance_old.T_perc, instance_old.T_completion, instance_old.T_min], [instance_old.Vf_contracting_at_T_perc, instance_old.Vf_contracting_at_T_completion, instance_old.Vf_contracting_somewhere]
                instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision)
                while first and good_point(instance, which_T_volume):
                    delta = delta / 10
                    g = (1 + delta) * g_old
                    if g == g_old:
                        return g_old, [instance_old.T_max, instance_old.T_nuc, instance_old.T_perc, instance_old.T_completion, instance_old.T_min], [instance_old.Vf_contracting_at_T_perc, instance_old.Vf_contracting_at_T_completion, instance_old.Vf_contracting_somewhere]
                    instance = Temperatures(lambda_, g, physical_vev, T_step = T_step, refine_Tmin_precision = refine_Tmin_precision)
                first = False  
                if g > 2 * g_crit(lambda_):
                    if delta > 1e-3:
                        g = g0   
                        delta = delta / 10
                    else:
                        g = None
                        return g, [instance.T_max, instance.T_nuc, instance.T_perc, instance.T_completion, instance.T_min], [instance.Vf_contracting_at_T_perc, instance.Vf_contracting_at_T_completion, instance.Vf_contracting_somewhere]
            instance_return = instance
            instance = instance_old
            g_return = g
            g = g_old
            delta = delta / 10
    
    return g_return, [instance_return.T_max, instance_return.T_nuc, instance_return.T_perc, instance_return.T_completion, instance_return.T_min], [instance_return.Vf_contracting_at_T_perc, instance_return.Vf_contracting_at_T_completion, instance_return.Vf_contracting_somewhere]