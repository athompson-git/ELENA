from fopt_generic import *

import json

test_params = {
    'a': 0.07096676838168296,
    'lam': 0.07052424336983912,
    'c': 6.782313922284097e-05,
    'd': 0.8625353201731807,
    'vev': 0.01
}


def test_point(point, verbose: bool = False):
    try:
        fopt = FOPTGeneric(point, verbose)
        print("FOPTGeneric initialized")
        if fopt.T0sq <= 0:
            print("T0sq is negative")
            return None, None, None
        if np.isnan(fopt.Tc):
            print("Tc is nan")
            return None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    try:
        fopt.calc_temperature_bounds(use_elena=False)
        print("Temperature bounds calculated")
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    try:
        fopt.calc_action_over_T()
        print("Action over T calculated")
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    try:
        fopt.calc_nucleation_percolation_completion()
        print("Nucleation percolation completion calculated")
        if np.isnan(fopt.T_completion):
            print("T_completion is nan")
            return None, None, None
        if np.isnan(fopt.T_nuc):
            print("T_nuc is nan")
            return None, None, None
        if np.isnan(fopt.T_perc):
            print("T_perc is nan")
            return None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    try:
        fopt.calc_npatches()
        print("Number of patches calculated")
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    try:
        fopt.calc_mean_bubble_size()
        print("Mean bubble size calculated")
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    try:
        fopt.calc_pbh_abundance(verbose=True)
        print("PBH abundance calculated")
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    print(f"Point: {point} passed")
    print("m_pbh (g): ", fopt.m_pbh / GEV_PER_G)
    print("f_pbh: ", fopt.f_pbh)
    return fopt.T_perc, fopt.m_pbh / GEV_PER_G, fopt.f_pbh


def scan_params(vev: float, n_points: int, out_file_name: str = "scans/pbh_scan_1MeV.json", verbose: bool = False):
    
    with open(out_file_name, "w") as outfile:
        outfile.write("[")
    
        # constraints:
        # self.lam*self.d - self.a**2 > 0 : keeps Tc real
        # c < vev * lam / 3 : keeps T0^2 positive
    
        for i in range(n_points):
            d = np.random.uniform(0.1, 1.0)
            lam = 10**np.random.uniform(-4, -1)
            a_max = np.sqrt(lam * d)
            a = 10**np.random.uniform(np.log10(a_max) - 2, np.log10(a_max))
            log10_cmax = np.log10(vev * lam / 3)
            c = 10**np.random.uniform(log10_cmax - 2, log10_cmax)
            
            point = {
                'a': a,
                'lam': lam,
                'c': c,
                'd': d,
                'vev': vev
            }
            print("################################################################################")
            print(f"On point i={i}")
            T_perc, m_pbh, f_pbh = test_point(point, verbose)
            if T_perc is None or m_pbh is None or f_pbh is None:
                continue
            
            param_dict = {
                'a': a,
                'lam': lam,
                'c': c,
                'd': d,
                'vev': vev,
                'T_perc': T_perc,
                'm_pbh': m_pbh,
                'f_pbh': f_pbh
            }
            json_object = json.dumps(param_dict)
            outfile.write(json_object)
            if i < n_points - 1:
                outfile.write(",\n")
        outfile.write("]")


if __name__ == "__main__":
    
    #test_point(test_params, verbose=True)
    
    scan_params(vev=0.001, n_points=1000, out_file_name="scans/pbh_scan_1MeV_debug.json", verbose=True)