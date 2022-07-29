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
from .source_pos import gauss
from ..lib import util
from ..lib.plots import figure_filetype

def image_and_background(data, meta, log, m):
    '''Make image+background plot. (Figs 3301)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    m : int
        The file number.

    Returns
    -------
    None
    '''
    log.writelog('  Creating figures for background subtraction...',
                 mute=(not meta.verbose))

    intstart = data.attrs['intstart']
    subdata = np.ma.masked_where(~data.mask.values, data.flux.values)
    subbg = np.ma.masked_where(~data.mask.values, data.bg.values)

    xmin, xmax = data.flux.x.min().values, data.flux.x.max().values
    ymin, ymax = data.flux.y.min().values, data.flux.y.max().values

    iterfn = range(meta.n_int)
    if meta.verbose:
        iterfn = tqdm(iterfn)
    for n in iterfn:
        plt.figure(3301, figsize=(8, 8))
        plt.clf()
        plt.suptitle(f'Integration {intstart + n}')
        plt.subplot(211)
        plt.title('Background-Subtracted Flux')
        max = np.ma.max(subdata[n])
        plt.imshow(subdata[n], origin='lower', aspect='auto',
                   vmin=0, vmax=max/10, extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.ylabel('Detector Pixel Position')
        plt.subplot(212)
        plt.title('Subtracted Background')
        median = np.ma.median(subbg[n])
        std = np.ma.std(subbg[n])
        plt.imshow(subbg[n], origin='lower', aspect='auto', vmin=median-3*std,
                   vmax=median+3*std, extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.ylabel('Detector Pixel Position')
        plt.xlabel('Detector Pixel Position')
        plt.tight_layout()
        file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))
                                       + 1))
        int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
        fname = (f'figs{os.sep}fig3301_file{file_number}_int{int_number}' +
                 '_ImageAndBackground'+figure_filetype)
        plt.savefig(meta.outputdir+fname, dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)
