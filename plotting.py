# plotting macros for visualizing parameter scans
# 
# Copyright (c) 2025 Adrian Thompson via MIT License

import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import numpy as np

from external_constants import *
from gw import GravitationalWave


"""
Make a 6-plot 'cornerplot' over the parameters for a quartic potential as scatterplots
color-coded by a parameter string of choice.
"""
def corner_plots_quartic_potential(json_filepath, parameter_to_colorcode="f_pbh"):

    fig = plt.figure(constrained_layout=True, figsize=[10.0, 10.0])
    spec = fig.add_gridspec(3, 3)


    with open(json_filepath, "r") as file:
        param_json = json.load(file)

    a_list = []
    d_list = []
    c_list = []
    lam_list = []
    color_param_list = []

    for i in range(len(param_json)):
        p = param_json[i]
        d_param = p["d"]
        c_param = p["c"]
        a_param = p["a"]
        lam_param = p["lam"]

        T_perc = p["T_perc"]
        f_pbh = p["f_pbh"]
        mpbh = p["m_pbh"]
        
        if T_perc is None:
            continue

        if f_pbh is None:
            continue

        if f_pbh <= 1e-20:
            continue

        if mpbh is None:
            continue


        color_param = p[parameter_to_colorcode]

        a_list.append(a_param)
        d_list.append(d_param)
        c_list.append(c_param)
        lam_list.append(lam_param)
        color_param_list.append(color_param)

    param_choice = np.array(color_param_list)
    param_min = (min(param_choice))
    param_max = (max(param_choice))

    def get_color_log(alpha):
        ln_alpha = np.nan_to_num(np.log10(alpha))
        return (ln_alpha - np.log10(param_min))/(np.log10(param_max) - np.log10(param_min))
    
    def get_color(alpha):
        return (alpha - param_min)/(param_max - param_min)
    
    color_ids = get_color_log(param_choice)
    colors = plt.cm.viridis(color_ids)

    ax1 = fig.add_subplot(spec[0, 0])
    ax1.scatter(a_list, d_list, marker=".", c=colors, alpha=0.8)
    ax1.set_ylabel(r"$D$", fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax2 = fig.add_subplot(spec[1, 0])
    ax2.scatter(a_list, c_list, marker=".", c=colors, alpha=0.8)
    ax2.set_ylabel(r"$C/\langle \phi \rangle$", fontsize=14)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    ax3 = fig.add_subplot(spec[1, 1])
    ax3.scatter(d_list, c_list, marker=".", c=colors, alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_xscale('log')

    ax4 = fig.add_subplot(spec[2, 0])
    ax4.scatter(a_list, lam_list, marker=".", c=colors, alpha=0.8)
    ax4.set_ylabel(r"$\lambda$", fontsize=14)
    ax4.set_xlabel(r"$A$", fontsize=14)
    ax4.set_yscale('log')
    ax4.set_xscale('log')

    ax5 = fig.add_subplot(spec[2, 1])
    ax5.scatter(d_list, lam_list, marker=".", c=colors, alpha=0.8)
    ax5.set_xlabel(r"$D$", fontsize=14)
    ax5.set_yscale('log')
    ax5.set_xscale('log')

    ax6 = fig.add_subplot(spec[2, 2])
    ax6.scatter(c_list, lam_list, marker=".", c=colors, alpha=0.8)
    ax6.set_xlabel(r"$C/\langle \phi \rangle$", fontsize=14)
    ax6.set_yscale('log')
    ax6.set_xscale('log')

    cbar_ax = fig.add_axes([0.8, 0.4, 0.03, 0.6])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_clim(vmin=np.log10(param_min), vmax=np.log10(param_max))

    # Make a log-scale colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\log_{10}(" + parameter_to_colorcode + ")$")

    plt.show()




"""
Make a 6-plot 'cornerplot' over the parameters for a quartic potential as scatterplots
color-coded by a parameter string of choice.
"""
def corner_plots_4param(json_filepath,
                        parameter_to_colorcode="f_pbh",
                        color_log=False,
                        fpbh_cut_lower=False,
                        fpbh_cut_upper=False,
                        color_label="",
                        save_name=None):
    

    fig = plt.figure(constrained_layout=True, figsize=[12.0, 8.0])
    spec = fig.add_gridspec(3, 3)


    with open(json_filepath, "r") as file:
        param_json = json.load(file)

    param1_list = []
    param2_list = []
    param3_list = []
    param4_list = []
    color_param_list = []

    for i in range(len(param_json)):
        p = param_json[i]
        param1_param = p["m_pbh"]
        param2_param = p["f_pbh"]

        if fpbh_cut_lower:
            if param2_param is None or param2_param <= 1e-80:
                continue
        if fpbh_cut_upper:
            if param2_param is None or param2_param >= 1.0:
                continue

        alpha = p["alpha"]
        betaByHstar = p["beta_by_Hn"]
        if betaByHstar is None or betaByHstar == 0.0:
            continue
        vw = 1.0 # FIXME: should be vw
        Tstar = p["T_perc"]
        gw = GravitationalWave()
        gw.alpha = alpha
        gw.betaByHstar = betaByHstar
        gw.vw = vw
        gw.Tstar = Tstar

        f_peak = gw.f_peak()
        h2Omega = gw.omega(f_peak)
        param3_param = f_peak
        param4_param = h2Omega

        color_param = p[parameter_to_colorcode]

        param1_list.append(param1_param)
        param2_list.append(param2_param)
        param3_list.append(param3_param)
        param4_list.append(param4_param)
        color_param_list.append(color_param)

    param_choice = np.array(color_param_list)
    param_min = (min(param_choice))
    param_max = (max(param_choice))

    if color_log:
        norm = mcolors.LogNorm(vmin=param_min, vmax=param_max)
    else:
        norm = mcolors.Normalize(vmin=param_min, vmax=param_max)
    
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.scatter(param1_list, param2_list, marker=".", c=param_choice, norm=norm, alpha=0.8)
    ax1.set_ylabel(r"$f_{\rm PBH}$", fontsize=18)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax2 = fig.add_subplot(spec[1, 0])
    ax2.scatter(param1_list, param3_list, marker=".", c=param_choice, norm=norm, alpha=0.8)
    ax2.set_ylabel(r"$f_{\rm GW}$ [Hz]", fontsize=18)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    ax3 = fig.add_subplot(spec[1, 1])
    ax3.scatter(param2_list, param3_list, marker=".", c=param_choice, norm=norm, alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_xscale('log')

    ax4 = fig.add_subplot(spec[2, 0])
    ax4.scatter(param1_list, param4_list, marker=".", c=param_choice, norm=norm, alpha=0.8)
    ax4.set_ylabel(r"$h^2 \Omega_{\rm GW}$", fontsize=18)
    ax4.set_xlabel(r"$m_{\rm PBH}$ [g]", fontsize=18)
    ax4.set_yscale('log')
    ax4.set_xscale('log')

    ax5 = fig.add_subplot(spec[2, 1])
    ax5.scatter(param2_list, param4_list, marker=".", c=param_choice, norm=norm, alpha=0.8)
    ax5.set_xlabel(r"$f_{\rm PBH}$", fontsize=18)
    ax5.set_yscale('log')
    ax5.set_xscale('log')

    ax6 = fig.add_subplot(spec[2, 2])
    ax6.scatter(param3_list, param4_list, marker=".", c=param_choice, norm=norm, alpha=0.8)
    ax6.set_xlabel(r"$f_{\rm GW}$ [Hz]", fontsize=18)
    ax6.set_yscale('log')
    ax6.set_xscale('log')

    cbar_ax = fig.add_axes([0.8, 0.4, 0.03, 0.6])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)

    # Make a colorbar (log or linear according to norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    # set tick label fontsize to 18
    cbar.ax.tick_params(labelsize=18)
    if color_label is not None:
        cbar.set_label(color_label, fontsize=18)
    else:
        if color_log:
            cbar.set_label(r"$\log_{10}(" + parameter_to_colorcode + ")$", fontsize=18)
        else:
            cbar.set_label(parameter_to_colorcode, fontsize=18)

    if save_name is not None:
        plt.savefig(save_name)
    else:
        plt.show()




"""
Plot a 1D histogram from a json using a parameter string.
"""
def plot_pbh_hist1d(json_filepath, varstr="MPBH", label=r"$M_{PBH}$ [g]"):

    with open(json_filepath, "r") as file:
        param_json = json.load(file)

    var_list = []


    for i in range(len(param_json)):
        p = param_json[i]

        var = p[varstr]
        if var is None:
            continue
        if var <= 0.0:
            continue
        
        var_list.append(var)


    min_mass = min(var_list)
    max_mass = max(var_list)
    mass_bins = np.logspace(np.log10(min_mass), np.log10(max_mass), 50)
    plt.hist(var_list, bins=mass_bins, histtype='step')
    plt.xscale('log')
    plt.ylabel(r"Density of Model Points", fontsize=16)
    plt.xlabel(label, fontsize=16)
    plt.tight_layout()
    plt.show()




"""
Pass two known parameter strings from the json of interest and plot a 2D scatterplot
color coded by a third parameter string choice color_param.

If passing MPBH, automatically rescales to grams.
"""
def plot_2d(json_filepath, varstr1="m_pbh", varstr2 = "f_pbh",
            xlabel=r"$M_{PBH}$ [g]", ylabel=r"$f_{PBH}$",
            ylim=None, xlim=None, color_param="v_wall", color_label="$v_w$",
            color_log=False, cuts=None,
            single_point_idx=None):

    with open(json_filepath, "r") as file:
        param_json = json.load(file)

    var1_list = []
    var2_list = []
    colvar_list = []

    if single_point_idx is not None:
        p = param_json[single_point_idx]
        var1 = p[varstr1]
        var2 = p[varstr2]
        a = p["a"]
        lam = p["lam"]
        d = p["d"]
        vev = p["vev"]
        c = p["c"]
        plt.scatter(var1, var2, marker='x', color='red', s=100, zorder=10)
        print(f"Single point at index {single_point_idx}: {var1}, {var2}")
        print(f"a: {a}, lam: {lam}, d: {d}, vev: {vev}, c: {c}")
    
    for i in range(len(param_json)):
        p = param_json[i]

        var1 = p[varstr1]
        var2 = p[varstr2]

        a = p["a"]
        lam = p["lam"]
        d = p["d"]
        vev = p["vev"]
        c = p["c"]
        #if a > 0.9*np.sqrt(lam * d):
        #    continue
        #if c > 0.9 * vev * lam / 3:
        #    continue
        T0sq = (lam * vev**2 - 3*c*vev)/(2*d)
        Tc = (c*a + np.sqrt(lam*d*(c**2 + (lam*d - a**2)*T0sq)))/(lam*d - a**2)
        
        if color_param == "t_ratio":
            colvar = np.sqrt(T0sq)/Tc
        else:
            colvar = p[color_param] # np.sqrt(T0sq)/Tc #


        if cuts is not None:
            skip = False
            for cut in cuts:
                cutvar = p[cut[0]]
                if cutvar < cut[1]:
                    skip = True
                if cutvar > cut[2]:
                    skip = True
            if skip:
                continue

        if var1 is None:
            continue
        if var2 is None:
            continue
        if var1 <= 0.0:
            continue
        if var2 <= 0.0:
            continue
        if colvar is None or colvar <= 0.0:
            continue
        
        var1_list.append(var1)
        var2_list.append(var2)
        colvar_list.append(colvar)

    param_choice = np.array(colvar_list)
    if min(param_choice) <= 0.0:
        color_log = False

    param_min = (min(param_choice))
    param_max = (max(param_choice))

    if color_log:
        norm = mcolors.LogNorm(vmin=param_min, vmax=param_max)
    else:
        norm = mcolors.Normalize(vmin=param_min, vmax=param_max)

    sc = plt.scatter(var1_list, var2_list, c=param_choice, norm=norm)

    cbar = plt.colorbar(sc)
    cbar.set_label(color_label)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()






# import constraints
import json
with open('../../limits/f_PBH_bounds/AMEGO.json', 'r') as f:
    amego = json.load(f)

amego_M = np.array([float(item['x']) for item in amego])
amego_f = np.array([float(item['y']) for item in amego])

with open('../../limits/f_PBH_bounds/BBN.json', 'r') as f:
    bbn = json.load(f)

bbn_M = np.array([float(item['x']) for item in bbn])
bbn_f = np.array([float(item['y']) for item in bbn])

with open('../../limits/f_PBH_bounds/CMB.json', 'r') as f:
    cmb = json.load(f)

cmb_M = np.array([float(item['x']) for item in cmb])
cmb_f = np.array([float(item['y']) for item in cmb])

with open('../../limits/f_PBH_bounds/extragalactic_gamma_rays.json', 'r') as f:
    egalactic = json.load(f)

ex_galactic_M = np.array([float(item['x']) for item in egalactic])
ex_galactic_f = np.array([float(item['y']) for item in egalactic])

with open('../../limits/f_PBH_bounds/galactic_gamma_rays.json', 'r') as f:
    galactic = json.load(f)

galactic_M = np.array([float(item['x']) for item in galactic])
galactic_f = np.array([float(item['y']) for item in galactic])

with open('../../limits/f_PBH_bounds/HSC.json', 'r') as f:
    hsc = json.load(f)

hsc_M = np.array([float(item['x']) for item in hsc])
hsc_f = np.array([float(item['y']) for item in hsc])

with open('../../limits/f_PBH_bounds/MeV_gamma_rays.json', 'r') as f:
    mev_gamma = json.load(f)

mev_gamma_M = np.array([float(item['x']) for item in mev_gamma])
mev_gamma_f = np.array([float(item['y']) for item in mev_gamma])

with open('../../limits/f_PBH_bounds/Roman.json', 'r') as f:
    roman = json.load(f)

roman_M = np.array([float(item['x']) for item in roman])
roman_f = np.array([float(item['y']) for item in roman])

with open('../../limits/f_PBH_bounds/Voyager_1.json', 'r') as f:
    voyager = json.load(f)

voyager_M = np.array([float(item['x']) for item in voyager])
voyager_f = np.array([float(item['y']) for item in voyager])


def plot_pbh(json_filepaths, varstr1="m_pbh", varstr2 = "f_pbh",
            xlabel=r"$M_{PBH}$ [g]", ylabel=r"$f_{PBH}$",
            ylim=None, xlim=None,
            legends=[],
            save_name=None):
    
    fig, ax = plt.subplots(1,1,figsize=(8,6))

    for j, json_filepath in enumerate(json_filepaths):
        with open(json_filepath, "r") as file:
            param_json = json.load(file)

        var1_list = []
        var2_list = []

        for i in range(len(param_json)):
            p = param_json[i]

            var1 = p[varstr1]
            var2 = p[varstr2]
            if var1 is None:
                continue
            if var2 is None:
                continue
            if var1 <= 0.0:
                continue

            var1_list.append(var1)
            var2_list.append(var2)

        ax.scatter(var1_list, var2_list, alpha=0.8, marker='.', zorder=6, label=legends[j])

    # existing gamma ray
    gamma_color = 'silver'
    ax.fill_between(bbn_M, bbn_f, y2=1.0, color=gamma_color)
    ax.fill_between(ex_galactic_M, ex_galactic_f, y2=1.0, color=gamma_color)
    ax.fill_between(galactic_M, galactic_f, y2=1.0, color=gamma_color)
    ax.fill_between(voyager_M, voyager_f, y2=1.0, color=gamma_color)
    ax.fill_between(cmb_M, cmb_f, y2=1.0, color=gamma_color)
    ax.fill_between(mev_gamma_M, mev_gamma_f, y2=1.0, color=gamma_color)
    
    # existing weak lensing bounds
    ax.fill_between(hsc_M, hsc_f, y2=1.0, color='khaki')

    # Future limits
    ax.plot(roman_M, roman_f, ls='dashed', color='sienna')
    ax.plot(amego_M, amego_f, ls='dashed', color='teal')

    # set text for existing and future limits
    label_fs = 14.0
    ax.text(2e16, 1e-6, "AMEGO", color='teal', rotation=65.0, fontsize=label_fs)
    ax.text(1e24, 3e-4, "ROMAN", color='sienna', rotation=20.0, fontsize=label_fs)
    ax.text(5e14, 1e-6, r"$\gamma$-ray sky", rotation=70.0, fontsize=label_fs)
    ax.text(0.8e25, 1.5e-2, "Subaru-HSC", rotation=25.0, fontsize=label_fs)
    
    plt.legend(loc="lower right", fontsize=18, framealpha=0.0)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name)
    plt.show()






# import external limits
nanograv = np.genfromtxt("../limits/gws/nanograv.txt")
lisa = np.genfromtxt("../limits/gws/lisa.txt")
theia = np.genfromtxt("../limits/gws/theia.txt")
muares = np.genfromtxt("../limits/gws/muares.txt")
bbo = np.genfromtxt("../limits/gws/bbo_1709-02434.txt")
aLIGO = np.genfromtxt("../limits/gws/aLIGO_design_1709-02434.txt")

# import JSON constraints
with open('../limits/gws/GW_sensitivity_curves/THEIA.json', 'r') as f:
    theia2 = json.load(f)

theia_f = np.array([float(item['x']) for item in theia2])
theia_h2Omega = np.array([float(item['y']) for item in theia2])

with open('../limits/gws/GW_sensitivity_curves/A+.json', 'r') as f:
    aplus = json.load(f)

aplus_f = np.array([float(item['x']) for item in aplus])
aplus_h2Omega = np.array([float(item['y']) for item in aplus])

with open('../limits/gws/GW_sensitivity_curves/ALIA.json', 'r') as f:
    alia = json.load(f)

alia_f = np.array([float(item['x']) for item in alia])
alia_h2Omega = np.array([float(item['y']) for item in alia])

with open('../limits/gws/GW_sensitivity_curves/CE.json', 'r') as f:
    ce = json.load(f)

ce_f = np.array([float(item['x']) for item in ce])
ce_h2Omega = np.array([float(item['y']) for item in ce])

with open('../limits/gws/GW_sensitivity_curves/DECIGO.json', 'r') as f:
    decigo = json.load(f)

decigo_f = np.array([float(item['x']) for item in decigo])
decigo_h2Omega = np.array([float(item['y']) for item in decigo])

with open('../limits/gws/GW_sensitivity_curves/EPTA.json', 'r') as f:
    epta = json.load(f)

epta_f = np.array([float(item['x']) for item in epta])
epta_h2Omega = np.array([float(item['y']) for item in epta])

with open('../limits/gws/GW_sensitivity_curves/NANOGrav.json', 'r') as f:
    nanograv2 = json.load(f)

nanograv_f = np.array([float(item['x']) for item in nanograv2])
nanograv_h2Omega = np.array([float(item['y']) for item in nanograv2])

with open('../limits/gws/GW_sensitivity_curves/TianQin.json', 'r') as f:
    TianQin = json.load(f)

tianqin_f = np.array([float(item['x']) for item in TianQin])
tianqin_h2Omega = np.array([float(item['y']) for item in TianQin])

with open('../limits/gws/GW_sensitivity_curves/Taiji.json', 'r') as f:
    taiji = json.load(f)

taiji_f = np.array([float(item['x']) for item in taiji])
taiji_h2Omega = np.array([float(item['y']) for item in taiji])

with open('../limits/gws/GW_sensitivity_curves/Gaia.json', 'r') as f:
    gaia = json.load(f)

gaia_f = np.array([float(item['x']) for item in gaia])
gaia_h2Omega = np.array([float(item['y']) for item in gaia])

with open('../limits/gws/GW_sensitivity_curves/BBO.json', 'r') as f:
    bbo2 = json.load(f)

bbo_f = np.array([float(item['x']) for item in bbo2])
bbo_h2Omega = np.array([float(item['y']) for item in bbo2])




def plot_gw_spectra_from_json(json_filepaths, curves=True, labels=[], save_name=None):

    cmap = plt.get_cmap('tab20')

    fig, ax = plt.subplots(1, 1, figsize=(10,6))

    for j, json_filepath in enumerate(json_filepaths):

        with open(json_filepath, "r") as file:
            param_json = json.load(file)

        gw = GravitationalWave()

        fs = np.logspace(-10, 14, 1000)

        f_peaks = []
        h2Omegas = []

        n_curves = 0
        for i in range(len(param_json)):
            p = param_json[i]
            Tc = p["Tc"]
            Tstar = p["T_perc"]
            alpha = p["alpha"]
            betaByHstar = p["beta_by_Hn"]
            vw = 1.0 # FIXME: should be vw

            if betaByHstar is None or betaByHstar == 0.0:
                continue

            gw.alpha = alpha
            gw.betaByHstar = betaByHstar
            gw.vw = vw
            gw.Tstar = Tstar

            f_peak = gw.f_peak()
            h2Omega = gw.omega(f_peak)

            if curves:
                if gw.f_peak() < 1e-10:
                    continue
                if gw.omega(gw.f_peak()) < 1e-30:
                    continue
                if gw.omega(gw.f_peak()) > 1e-7:
                    continue

                n_curves += 1
                if n_curves > 5:
                    continue

                print("Plotting for alpha={}, beta={}, vw={}, T*={}, f={}".format(alpha, betaByHstar, vw, Tc, gw.f_peak()))

                ax.plot(fs, h2Omega, linewidth=1.0)
            else:
                f_peaks.append(f_peak)
                h2Omegas.append(h2Omega)

        ax.scatter(f_peaks, h2Omegas, color=cmap(2*j+1), marker='.',
                    alpha=1.0, edgecolors=cmap(2*j), linewidth=1.0, s=70)

    # add legends
    handles, ls = plt.gca().get_legend_handles_labels()
    lines = []
    for i in range(len(labels)):
        line1 = Line2D([0], [0], color=cmap(2*i), marker='.', label=labels[i], linestyle='none')
        lines.append(line1)

    handles.extend(lines)

    # plot external limits
    ax.fill_between(nanograv[:,0], nanograv[:,1], y2=1.0, color='silver')
    ax.fill_between(aLIGO[:,0], aLIGO[:,1], y2=1.0, color='silver')

    # plot projections
    text_fontsize = 12
    ax.plot(lisa[:,0], lisa[:,1], linewidth=2.0)
    ax.plot(muares[:,0], muares[:,1], linewidth=2.0)
    #plt.plot(theia[:,0], theia[:,1])
    ax.plot(theia_f, theia_h2Omega, linewidth=2.0)
    #plt.plot(bbo[:,0], bbo[:,1])
    ax.plot(bbo_f, bbo_h2Omega, linewidth=2.0)

    
    ax.plot(aplus_f, aplus_h2Omega, linewidth=2.0)
    ax.plot(decigo_f, decigo_h2Omega, linewidth=2.0)
    ax.plot(alia_f, alia_h2Omega, linewidth=2.0)
    ax.plot(ce_f, ce_h2Omega, linewidth=2.0)
    #plt.plot(nanograv_f, nanograv_h2Omega, color='k')
    ax.plot(gaia_f, gaia_h2Omega, linewidth=2.0)
    #plt.plot(epta_f, epta_h2Omega, ls='dashed')
    ax.plot(taiji_f, taiji_h2Omega, ls='dashed', linewidth=2.0)
    ax.plot(tianqin_f, tianqin_h2Omega, ls='dashed', linewidth=2.0)
    

    ax.text(4.0e-9, 2.0e-15, "THEIA", rotation=45.0, fontsize=text_fontsize)
    ax.text(5.0e-6, 5.0e-15, r"$\mu$Ares", rotation=-50.0, fontsize=text_fontsize)
    ax.text(1.0e-5, 1.0e-10, "LISA", rotation=-60.0, fontsize=text_fontsize)
    ax.text(0.004, 6e-16, "BBO", rotation=-40.0, fontsize=text_fontsize)
    ax.text(0.002, 5e-17, "ALIA", rotation=-50.0, fontsize=text_fontsize)
    ax.text(10, 1e-13, "DECIGO", rotation=70.0, fontsize=text_fontsize)
    ax.text(20.0, 5.0e-9, "aLIGO", rotation=90.0, fontsize=text_fontsize)
    ax.text(4e-9, 2.0e-9, "NANOGrav", rotation=90.0, fontsize=text_fontsize)
    ax.text(500.0, 2e-12, "CE", rotation=40.0, fontsize=text_fontsize)
    ax.text(3.3e-8, 4e-9, "Gaia", rotation=50.0, fontsize=text_fontsize)
    ax.text(0.15, 4e-9, "Taiji", rotation=65.0, fontsize=text_fontsize)
    ax.text(0.6, 1e-9, "TianQin", rotation=65.0, fontsize=text_fontsize)
    ax.text(73, 4e-11, "A+", rotation=0.0, fontsize=text_fontsize)

    fig.set_size_inches(10.0, 6.0)
    plt.legend(handles=handles, loc="lower right", framealpha=1.0, fontsize=14)

    ax.set_xlabel(r"$f$ [Hz]", fontsize=16)
    ax.set_ylabel(r"max[$h^2 \Omega(f)$]", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    plt.ylim((1e-21, 1e-7))
    plt.xlim((1e-10, 1e4))
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


