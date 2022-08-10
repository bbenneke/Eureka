#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:32:43 2022

@author: caroline
Plots for Eureka! Stage 1
"""

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..lib import util
from ..lib.plots import figure_filetype

def saturation_mask(sat_mask, meta, log, step=""):
    '''Plot the saturation mask (Plot 1000)

    Parameters
    ----------
    sat_mask : Numpy array
        The array of saturated pixels
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    step : string
        additional label
    Returns
    -------
    None
    '''
    log.writelog('  Plotting the saturation mask...',
                 mute=(not meta.verbose))

    ngroups = sat_mask.shape[0]
    fig, ax = plt.subplots(ngroups, 1, num=1000)
    
    for i, ax_i in enumerate(ax):
        ax_i.imshow(sat_mask[i,:,:])
        ax_i.set_title("Group "+str(i+1))
    
    fname = (f'figs{os.sep}fig1000_' + step + '_SatMask'+figure_filetype)
    fig.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)
