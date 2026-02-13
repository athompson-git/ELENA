# Default units: cm, GeV, s
import numpy as np

KILOMETERS = 1e5

C_LIGHT  = 2.99792458e10  # Speed of light
HBAR = 6.58212e-25 # Reduced Planck's Constant (GeV * s)
HBARC = 1.97236e-14 # hbar * c (GeV * cm)
M_PL = 2.4353234600842885e+18
M_PL_REDUCED = M_PL * np.power(8*np.pi, -0.5)

HUBBLE0 = 67.4 * HBAR * KILOMETERS / 3.086e24 # km / (s Mpc)  -> GeV
OMEGA_DM = 0.12
RHO_CRIT_GEV_CM3 = 1.053672e-5 # GeV / cm^3
RHO_CRIT_GEV4 = RHO_CRIT_GEV_CM3 * (HBARC**3)
S0_SM = 2891.2  * HBARC**3 # cm^-3 --> GeV^3 entropy density today

# unit conversions
GEV_PER_G = 5.6095887e23  # GeV/g