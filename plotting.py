# plotting macros for visualizing parameter scans
# 
# Copyright (c) 2025 Adrian Thompson via MIT License

import json
import matplotlib.pyplot as plt
import numpy as np
from external_constants import *

import matplotlib.colors as mcolors

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

        if f_pbh <= 0.0:
            continue

        if f_pbh > 1.0:
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
            color_log=False, cuts=None):

    with open(json_filepath, "r") as file:
        param_json = json.load(file)

    var1_list = []
    var2_list = []
    colvar_list = []

    for i in range(len(param_json)):
        p = param_json[i]

        var1 = p[varstr1]
        var2 = p[varstr2]

        colvar = p[color_param]

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
            if var2 <= 1.0e-7:
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



