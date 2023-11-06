#!/usr/bin/env python
# coding: utf-8

# In[199]:

import os
import numpy as np
from collections import OrderedDict as od
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
from matplotlib.cm import ScalarMappable
import matplotlib
import csv
from matplotlib import path

try:
    from astropy.convolution import Gaussian2DKernel, convolve
    astro_smooth = True
except ImportError as IE:
    astro_smooth = False

from astropy.io import fits

from astropy.coordinates import match_coordinates_sky as coords
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles

#from astropy.units import cds
#cds.enable()

from astropy.cosmology import LambdaCDM, FlatLambdaCDM

from astropy.io import ascii
from astropy.io.ascii import masked
from astropy.table import Table

from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from numpy import polyfit
from astropy.time import Time

import uncertainties as unc  
import uncertainties.unumpy as unp
import time

import kcorrect.response


# In[200]:


#ASSEF_CUT = 0.8 #already in vega mags so remember to change vega mag to AB mag conversion
ASSEF_CUT = -0.18
SAVE_FILEPATH = os.path.join(os.path.curdir, 'WISE variations/Final Plots/')
IMPORT_FILEPATH = os.path.join(os.path.curdir, 'WISE variations/')

# ## file imports

# In[201]:


def import_files():
    manga_file = fits.open(IMPORT_FILEPATH + "mnsa-0.3.0.fits")
    manga_hdu = manga_file[2]
    manga_data = manga_hdu.data


    wise_file = fits.open(IMPORT_FILEPATH + "WISE_statistics_2arcsec_sigradec.fits")
    wise_hdu = wise_file[1]
    hdu = wise_hdu.data
    wise_hdu.header

    jiyan_file = fits.open(IMPORT_FILEPATH + "jiyan-agn-0.3.2.fits")
    jiyan_hdu = jiyan_file[1]
    jhdu = jiyan_hdu.data
    return manga_hdu, wise_hdu, jiyan_hdu


# ## data processing for each plot

# In[202]:


def contour(x,y):
    H, xedges, yedges = np.histogram2d((x),(y), bins=(100,100))
    xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])

    # Smooth the contours (if astropy is installed)
    if astro_smooth:
        kernel = Gaussian2DKernel(x_stddev = 1.)
        H=convolve(H,kernel)
    return xmesh, ymesh, H.T


# In[203]:


def process_wise_data_variability_plot(hdu):
    #for x1 and y1
    x1 = hdu['expected var all epochs (mags)']
    y1 = hdu['observed var']
    
    plateifu = hdu['plateifu']
    
    w1 = np.array([])
    w2 = np.array([])
    
    for pifu in plateifu:
        
        w1temp = hdu[hdu['plateifu'] == pifu]['mean W1 per epoch']
        w2temp = hdu[hdu['plateifu'] == pifu]['mean W2 per epoch']
        
        mask_w1 = ~(w1temp == -9999.)
        mask_w2 = ~(w2temp == -9999.)
        
        w1temp, w2temp = w1temp[mask_w1], w2temp[mask_w2]
        
        w1 = np.append(w1, np.nanmean(w1temp))
        w2 = np.append(w2, np.nanmean(w2temp))
    
    #masks
    mask = ~((x1 == -9999.) | (y1 == -9999.) | (w1 == -9999.) | (w2 == -9999.))
    x1, y1, w1, w2, plateifu = x1[mask], y1[mask], w1[mask], w2[mask], plateifu[mask]

    mask1 = ~((np.isnan(x1)) | (np.isnan(y1)) | (np.isnan(w1)) | (np.isnan(w2)))
    x1, y1,w1, w2, plateifu = x1[mask1], y1[mask1], w1[mask1], w2[mask1], plateifu[mask1]
    mask2 = ~((x1 == 0.) | (y1 == 0.))
    x1,y1, w1, w2, plateifu = x1[mask2], y1[mask2], w1[mask2], w2[mask2], plateifu[mask2]

    mask3 = ~((w2 > 100) | (w2 < -100))
    x1, y1, w1, w2, plateifu = x1[mask3], y1[mask3], w1[mask3], w2[mask3], plateifu[mask3]
    
    return x1, y1, w1, w2, plateifu

def w12_cut(cut_ab, x1, y1, w1, w2, plateifu):
    
    w12 = w1-w2
    #W1-W2 colour cut
    rd = kcorrect.response.ResponseDict()
    rd.load_response('wise_w2')
    w2_vega = rd['wise_w2'].vega2ab
    rd.load_response('wise_w1')
    w1_vega = rd['wise_w1'].vega2ab

    #w1_vega, w2_vega = 2.680448504503577, 3.3161984371892177
    #Assef et. al. (2012) cut = -0.18 AB mags. Converting to Vega mags with m_ab = m_vega + m_delta
    cut = cut_ab - (w1_vega - w2_vega)  #(w1_vega - w2_vega) = m_delta

    red_mask = ( w12 > cut)
    X1_red, Y1_red, W1_red, W2_red, red_plateifu = x1[red_mask], y1[red_mask], w1[red_mask], w2[red_mask], plateifu[red_mask]

    X1_blue, Y1_blue, W1_blue, W2_blue, blue_plateifu = x1[~red_mask], y1[~red_mask], w1[~red_mask], w2[~red_mask], plateifu[~red_mask]
    
    return  X1_red, Y1_red, W1_red, W2_red, red_plateifu, X1_blue, Y1_blue, W1_blue, W2_blue, blue_plateifu


# In[204]:


def process_wise_data_per_epoch_var_plot(hdu):
    #for x2 and y2
    
    y2 = hdu['expected var (errors)']
    x2 = hdu['expected var (mags)']

    x2 = x2.reshape(-1)
    y2 = y2.reshape(-1)

    mask = ~((np.isnan(x2)) | (np.isnan(y2)))
    x2, y2 = x2[mask], y2[mask]
    mask1 = ~((x2 == -9999.) | (y2 == -9999.))
    x2, y2 = x2[mask1], y2[mask1]
    mask2 = ~((x2 == 0.) | (y2 == 0.))
    x2,y2 = x2[mask2], y2[mask2]
    
    return x2, y2    


# In[205]:


def process_data_jiyan_cut(jhdu, x1, y1, w1, w2, plateifu):
    #x1r, y1r, w1r, w2r, pifu_r, x1b, y1b, w1b, w2b, pifu_b = w12_cut(ASSEF_CUT, x1, y1, w1, w2, plateifu)
    
    p1 = np.array([])
    p3 = np.array([])

    for i in plateifu:
        p1 = np.append(p1, jhdu['p1'][jhdu['plateifu'] == i])
        p3 = np.append(p3, jhdu['p3'][jhdu['plateifu'] == i])
        

    #masking out -9999. from p1 and p3
    mask = ~((p1 == -9999.) | (p3 == -9999.))
    
    p1, p3, w1, w2 = p1[mask], p3[mask], w1[mask], w2[mask]
    
    return p1, p3, w1, w2


# In[206]:


def spur_objects(x1, y1, w1, w2, plateifu):
    x1r, y1r, w1r, w2r, pifu_r, x1b, y1b, w1b, w2b, pifu_b = w12_cut(ASSEF_CUT, x1, y1, w1, w2, plateifu)
    
    spur_mask = ((np.log10(x1) < -2.0) & (np.log10(x1) > -2.5)) & ((np.log10(y1) < -1.0) & (np.log10(y1) > -2.2))
    
    x1_spur, y1_spur, w1_spur, w2_spur, pifu_spur = x1[spur_mask], y1[spur_mask], w1[spur_mask], w2[spur_mask], plateifu[spur_mask]
    
    return x1_spur, y1_spur, w1_spur, w2_spur, pifu_spur

def spur_w12cut(x1, y1, w1, w2, plateifu):
    x1r, y1r, w1r, w2r, pifu_r, x1b, y1b, w1b, w2b, pifu_b = w12_cut(ASSEF_CUT, x1, y1, w1, w2, plateifu)
    
    x1rs, y1rs, w1rs, w2rs, pifu_rs = spur_objects(x1r, y1r, w1r, w2r, pifu_r)
    x1bs, y1bs, w1bs, w2bs, pifu_bs = spur_objects(x1b, y1b, w1b, w2b, pifu_b)
    
    return x1rs, y1rs, w1rs, w2rs, pifu_rs, x1bs, y1bs, w1bs, w2bs, pifu_bs


# In[207]:


def w2_cut(x1, y1, w1, w2, plateifu):
    w2cut = ((np.log10(x1) < -2.5) & (w2 < 14))
    x1, y1, w1, w2, plateifu = x1[w2cut], y1[w2cut], w1[w2cut], w2[w2cut], plateifu[w2cut]
    
    return x1, y1, w1, w2, plateifu
    
def w2_w12cut(x1, y1, w1, w2, plateifu):
    x1r, y1r, w1r, w2r, pifu_r, x1b, y1b, w1b, w2b, pifu_b = w12_cut(ASSEF_CUT, x1, y1, w1, w2, plateifu)
    
    return x1r, y1r, w1r, w2r, pifu_r, x1b, y1b, w1b, w2b, pifu_b


# ## functions for each plot

# %%
def mnsa_vs_wise(w1, w2, plateifu, whdu, mhdu):

    w1_mnsa = np.array([])
    w2_mnsa = np.array([])

    for i in plateifu:
        w1_mnsa = np.append(w1_mnsa, mhdu[mhdu['plateifu'] == i]['maggies'][:,5])
        w2_mnsa = np.append(w2_mnsa, mhdu[mhdu['plateifu'] == i]['maggies'][:,6])

    #masks
    mask = ~((w1_mnsa == -9999.) | (w2_mnsa == -9999.))
    w1, w2, w1_mnsa, w2_mnsa = w1[mask], w2[mask], w1_mnsa[mask], w2_mnsa[mask]

    mask1 = ~((w1 == 0.) | (w2 == 0.) | (w1_mnsa == 0.) | (w2_mnsa == 0.))
    w1, w2, w1_mnsa, w2_mnsa = w1[mask1], w2[mask1], w1_mnsa[mask1], w2_mnsa[mask1]


    return w1, w2, w1_mnsa, w2_mnsa
# In[208]:


def epoch2epoch_and_perepoch_var(x1r, y1r, w1r, w2r, pifu_r, x1b, y1b, w1b, w2b, pifu_b, fn, save = False):
    #plot 1: epoch-to-epoch variability and per epoch variability
    x2, y2 = process_wise_data_per_epoch_var_plot(wise_hdu.data)
    xc2, yc2, z2 = contour(np.log10(x2), np.log10(y2))
    
    w12b = w1b - w2b
    w12r = w1r - w2r
    #w12 = w1 - w2
    
    fig, ax = plt.subplots(figsize = (20,6), ncols = 2,  gridspec_kw={'width_ratios': [1.35, 1]})
    axa, axb = ax

    #colorbar decimal place format
    fmt = tkr.FormatStrFormatter("%.2f")

    #plot 1
    plot1_blue = axa.scatter(np.log10(x1b), np.log10(y1b), c = w12b, cmap = 'YlGnBu', alpha = 0.5, s = 4)
    plot1_red = axa.scatter(np.log10(x1r), np.log10(y1r), c = w12r, edgecolors = 'black', lw = 0.4, cmap = 'plasma_r', alpha = 0.75, s = 14)
    axa.axline((0, 0), slope= 1, color = 'black', linestyle = 'dashed', label = 'y = x')
    ca_blue = fig.colorbar(plot1_blue, ax=axa, orientation='vertical', pad=0.01, ticks=np.linspace(w12b.min(), w12b.max(), 5), format = fmt)
    ca_red = fig.colorbar(plot1_red, ax=axa, orientation='vertical', pad=0.01,fraction=0.1, ticks=np.linspace(w12r.min(), w12r.max(), 5), format = fmt)
    ca_blue.set_label(label = r'$\mu(\overline{W1}-\overline{W2})$',size=14)
    #plot 2
    axb.scatter(np.log10(x2), np.log10(y2), color = 'darkorange', alpha = 0.2, s = 1)
    axb.axline((0, 0), slope = 1, color = 'black', linestyle = 'dashed', label = 'y = x')
    clevels = axb.contour(xc2, yc2, z2, linewidths=1.3, cmap='winter')
    sm = ScalarMappable(norm=plt.Normalize(z2.min(), z2.max()), cmap='winter')
    fig.colorbar(sm, ax=axb, orientation='vertical')


    axa.legend()

    axa.set_xlim(-4.5, -0.75)
    axa.set_ylim(-4.25, 0)

    axa.set_xlabel(r'expected $\log{Var(W2)}$',  fontsize=14)
    axa.set_ylabel(r'observed $\log{Var(W2)}$',  fontsize=14)

    axb.legend()

    axb.set_xlim(-4, 0)
    axb.set_ylim(-4, -0.25)

    axb.set_xlabel(r'expected $\log{\sigma^2_{W2}}$ per epoch',  fontsize=14)
    axb.set_ylabel(r'expected $\log{Var(W2)}$ per epoch',  fontsize=14)

    if save == True:
        plt.savefig(SAVE_FILEPATH + fn, bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)


# In[209]:


def jiyan_cut(X1r, Y1r, W1r, W2r, pifu_r, X1b, Y1b, W1b, W2b, pifu_b, fn, save = False):
    p1r, p3r, w1r, w2r = process_data_jiyan_cut(jiyan_hdu.data, X1r, Y1r, W1r, W2r, pifu_r)
    p1b, p3b, w1b, w2b = process_data_jiyan_cut(jiyan_hdu.data, X1b, Y1b, W1b, W2b, pifu_b)
    w12b = w1b - w2b
    w12r = w1r - w2r
    #w12 = w1 - w2
    
    
    fig, ax = plt.subplots(ncols = 2, figsize = (16,5))
    axa, axb = ax

    fmt = tkr.FormatStrFormatter("%.2f")

    axa.fill((-0.3, -0.3, 1.3, 1.3),(1.9,0.5, 0.5, 1.9), color = 'lightsteelblue', alpha = 0.2, label = 'Ji & Yan cut')
    plot1 = axa.scatter(p1b, p3b, c = w12b, cmap = 'YlGnBu', s = 8, alpha = 0.6)
    axa.axline((0, 0.5), slope= 0, color = 'black', linestyle = (0, (5, 10)))
    axa.axline((-0.3, 0), slope= 10000, color = 'black', ls = (0, (5, 10)))
    ca_blue = fig.colorbar(plot1, ax=axa, orientation='vertical', pad=0.01, ticks=np.linspace(w12b.min(), w12b.max(), 5), format = fmt)
    ca_blue.set_label(label = r'$\mu(\overline{W1}-\overline{W2})$',size=14)

    axa.set_ylim(-0.7,1.9)
    axa.set_xlim(-1.7, 1.2)

    axa.legend()

    axa.set_ylabel('$P3$',  fontsize=14)
    axa.set_xlabel('$P1$',  fontsize=14)

    axb.fill((-0.3, -0.3, 1.3, 1.3),(1.9,0.5, 0.5, 1.9), color = 'lightsteelblue', alpha = 0.2, label = 'Ji & Yan cut')
    plot2 = axb.scatter(p1r, p3r, c = w12r, cmap = 'plasma_r', edgecolors = 'black', lw = 0.4, s = 12, alpha = 0.6)
    axb.axline((0, 0.5), slope= 0, color = 'black', linestyle = (0, (5, 10)))
    axb.axline((-0.3, 0), slope= 10000, color = 'black', ls = (0, (5, 10)))
    ca_red = fig.colorbar(plot2, ax=axb, orientation='vertical', pad=0.01, ticks=np.linspace(w12r.min(), w12r.max(), 5), format = fmt)
    ca_red.set_label(label = r'$\mu(\overline{W1}-\overline{W2})$',size=14)

    axb.set_ylim(-0.7,1.9)
    axb.set_xlim(-1.7, 1.2)

    axb.legend()

    axb.set_ylabel('$P3$',  fontsize=14)
    axb.set_xlabel('$P1$',  fontsize=14)

    if save == True:
        plt.savefig(SAVE_FILEPATH + fn, bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
    
    
def jiyan_spur(X1r, Y1r, W1r, W2r, pifu_r, X1b, Y1b, W1b, W2b, pifu_b, fn, save = False):
    p1r, p3r, w1r, w2r = process_data_jiyan_cut(jiyan_hdu.data, X1r, Y1r, W1r, W2r, pifu_r)
    p1b, p3b, w1b, w2b = process_data_jiyan_cut(jiyan_hdu.data, X1b, Y1b, W1b, W2b, pifu_b)
    w12b = w1b - w2b
    w12r = w1r - w2r
    #w12 = w1 - w2
    
    fig, ax = plt.subplots(ncols = 1)
    axa =  ax
    fmt = tkr.FormatStrFormatter("%.2f")

    axa.fill((-0.3, -0.3, 1.3, 1.3),(1.9,0.5, 0.5, 1.9), color = 'lightsteelblue', alpha = 0.2, label = 'Ji & Yan cut')
    plot1 = axa.scatter(p1b, p3b, c = w12b, cmap = 'YlGnBu', s = 8, alpha = 0.6)
    plot2 = axa.scatter(p1r, p3r, c = w12r, cmap = 'plasma_r', edgecolors = 'black', lw = 0.4, s = 12, alpha = 0.6)
    axa.axline((0, 0.5), slope= 0, color = 'black', linestyle = (0, (5, 10)))
    axa.axline((-0.3, 0), slope= 10000, color = 'black', ls = (0, (5, 10)))
    ca_blue = fig.colorbar(plot1, ax=axa, orientation='vertical', pad=0.01, ticks=np.linspace(w12b.min(), w12b.max(), 5), format = fmt)
    ca_blue.set_label(label = r'$\mu(\overline{W1}-\overline{W2})$',size=14)
    ca_red = fig.colorbar(plot2, ax=axa, orientation='vertical', pad=0.01, ticks=np.linspace(w12r.min(), w12r.max(), 5), format = fmt)

    axa.set_ylim(-0.7,1.9)
    axa.set_xlim(-1.7, 1.2)

    axa.legend()

    axa.set_ylabel('$P3$',  fontsize=14)
    axa.set_xlabel('$P1$',  fontsize=14)
    
    if save == True:
        plt.savefig(SAVE_FILEPATH + fn, bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)


# In[210]:


def w12_distribution(w1, w2, fn, save = False):
    w12 = w1-w2
    
    fig, ax = plt.subplots(ncols = 1)
    axa = ax
    plot = axa.hist(w12, color = 'dodgerblue', bins = 40, label = 'spur objects');

    axa.legend()

    axa.set_ylabel('$count$',  fontsize=14)
    axa.set_xlabel('$W1-W2$ (Vega mags)',  fontsize=14)
    
    if save == True:
        plt.savefig(SAVE_FILEPATH + fn, bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)


# In[211]:


def W2_vs_err_varW2(x1, y1, w1, w2, plateifu, fn, save = False):
    #contour
    xcw2, ycw2, zw2 = contour(np.log10(x1), w2)
    w12 = w1-w2
    fig, ax = plt.subplots(ncols = 1)
    axa = ax
    axa.set_axisbelow(True)
    plt.grid()


    axa.fill((-2.5, -2.5, -1.8, -1.8),(15.5,9.8, 9.8, 15.5), color = 'lightsteelblue', alpha = 0.5, label = 'discarded points')

    axa.fill((-4.5, -4.5, -2.5, -2.5),(15.5,14, 14, 15.5), color = 'lightsteelblue', alpha = 0.5)

    axa.scatter(np.log10(x1), w2, s = 2, c = 'chocolate', alpha = 0.5)
    axa.axline((-2.5, 0), slope= 100000, color = 'black', linestyle = (0, (5, 7)))
    axa.axline((0, 14), slope= 0, color = 'black', linestyle = (0, (5, 7)))
    clevels = axa.contour(xcw2, ycw2, zw2, linewidths=0.75, cmap='winter')
    sm = ScalarMappable(norm=plt.Normalize(zw2.min(), zw2.max()), cmap='winter')
    fig.colorbar(sm, ax=axa, orientation='vertical')

    axa.set_xlim(-4.5, -1.8)
    axa.set_ylim(9.75, 15.5)

    axa.legend()

    
    
    if save == True:
        plt.savefig(SAVE_FILEPATH + fn, bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)


# In[212]:


def var_w_W2_cut(X1, Y1, W1, W2, plateifu, fn, save = False):
    x1, y1, w1, w2, plateifu = w2_cut(X1, Y1, W1, W2, plateifu)
    x1r, y1r, w1r, w2r, pifu_r, x1b, y1b, w1b, w2b, pifu_b = w2_w12cut(x1, y1, w1, w2, plateifu)
    
    w12b = w1b - w2b
    w12r = w1r - w2r
    w12 = w1 - w2
    
    fig, ax = plt.subplots(ncols = 1)
    axa = ax

    #colorbar decimal place format
    fmt = tkr.FormatStrFormatter("%.2f")

    #plot 1
    plot1_blue = axa.scatter(np.log10(x1b), np.log10(y1b), c = w12b, cmap = 'YlGnBu', alpha = 0.5, s = 4)
    plot1_red = axa.scatter(np.log10(x1r), np.log10(y1r), c = w12r, edgecolors = 'black', lw = 0.4, cmap = 'plasma_r', alpha = 0.75, s = 14)
    axa.axline((0, 0), slope= 1, color = 'black', linestyle = 'dashed', label = 'y = x')
    #axa.axline((-0.5, -2), slope= 0, color = 'gray', linestyle = 'dashed', label = 'AGN cutoff')
    #clevels = axa.contour(Xc1, Yc1, Z1, lw=.2, cmap='winter')
    #axa.fill_between(np.linspace(-4.8, 0,100), -2*np.ones(100)+ 0, -2*np.ones(100) +3*np.ones(100), alpha=0.075, label = 'AGN')
    ca_blue = fig.colorbar(plot1_blue, ax=axa, orientation='vertical', pad=0.01, ticks=np.linspace(w12b.min(), w12b.max(), 5), format = fmt)
    ca_red = fig.colorbar(plot1_red, ax=axa, orientation='vertical', pad=0.01, ticks=np.linspace(w12r.min(), w12r.max(), 5), format = fmt)
    ca_blue.set_label(label = r'$\mu(\overline{W1}-\overline{W2})$',size=14)
    
    axa.set_xlabel(r'expected $\log{Var(W2)}$',  fontsize=14)
    axa.set_ylabel(r'$W2$ (Vega mags)',  fontsize=14)
    
    axa.legend()

    axa.set_xlim(-4.9, -2.4)
    axa.set_ylim(-4.25, 0)

    axa.set_xlabel(r'expected $\log{Var(W2)}$',  fontsize=14)
    axa.set_ylabel(r'observed $\log{Var(W2)}$',  fontsize=14)
    
    if save == True:
        plt.savefig(SAVE_FILEPATH + fn, bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
 


# ## main function

# %%
def mnsa_vs_wise_plots(w1, w2, w1_mnsa, w2_mnsa, fn, save = False):
    #converting wise vega mags to AB mags
    rd = kcorrect.response.ResponseDict()
    rd.load_response('wise_w2')
    w2_vega = rd['wise_w2'].vega2ab
    rd.load_response('wise_w1')
    w1_vega = rd['wise_w1'].vega2ab

    #wise mags from vega to ab to apparent
    w1_ab = w1 + w1_vega
    w2_ab = w2 + w2_vega

    #mnsa mags from ab to apparent
    w1_app = -2.5 * np.log10(w1_mnsa) 
    w2_app = -2.5 * np.log10(w2_mnsa) 
   
   #masking out nan from apparent mags
    mask = ~((np.isnan(w1_app)) | (np.isnan(w2_app)))
    w1_app, w2_app, w1_ab, w2_ab = w1_app[mask], w2_app[mask], w1_ab[mask], w2_ab[mask]

    w12_ab = w1_ab - w2_ab
    w12_app = w1_app - w2_app
    

    xw1, yw1, zw1 = contour(w1_ab, w1_app)
    xw2, yw2, zw2 = contour(w2_ab, w2_app)
    xw12, yw12, zw12 = contour(w12_ab, w12_app)

    fig, ax = plt.subplots(ncols = 3, figsize = (20, 5))
    axa, axb, axc = ax

    axa.scatter(w1_ab, w1_app, color = 'tomato', s = 6, alpha = 0.25)
    axa.axline((0,0), slope = 1, color = 'black')
    clevels1 = axa.contour(xw1, yw1, zw1, linewidths=0.75, cmap='winter')
    sm1 = ScalarMappable(norm=plt.Normalize(zw1.min(), zw1.max()), cmap='winter')
    fig.colorbar(sm1, ax=axa, orientation='vertical')

    axb.scatter(w2_ab, w2_app, color = 'darkorange', s = 6, alpha = 0.25)
    axb.axline((0,0), slope = 1, color = 'black')
    clevels2 = axb.contour(xw2, yw2, zw2, linewidths=0.75, cmap='winter')
    sm2 = ScalarMappable(norm=plt.Normalize(zw2.min(), zw2.max()), cmap='winter')
    fig.colorbar(sm2, ax=axb, orientation='vertical')

    axc.scatter(w12_ab, w12_app, color = 'chocolate', s  = 6, alpha = 0.25)
    axc.axline((0,0), slope = 1, color = 'black')
    clevels3 = axc.contour(xw12, yw12, zw12, linewidths=0.7, alpha = 0.9, cmap='winter')
    sm3 = ScalarMappable(norm=plt.Normalize(zw12.min(), zw12.max()), cmap='winter')
    fig.colorbar(sm3, ax=axc, orientation='vertical')

    axa.set_xlabel('$W1$ (WISE)')
    axa.set_ylabel('$W1$ (MNSA)')
    axa.set_xlim(13, 19)
    axa.set_ylim(13, 19)

    axb.set_xlabel('$W2$ (WISE)')
    axb.set_ylabel('$W2$ (MNSA)')
    axb.set_xlim(13, 19)
    axb.set_ylim(13, 20)

    axc.set_xlabel('$W1-W2$ (WISE)')
    axc.set_ylabel('$W1-W2$ (MNSA)')
    axc.set_xlim(-1, 0.5)
    axc.set_ylim(-1, 0)

    plt.show()

    if save == True:
        fig.savefig(SAVE_FILEPATH + fn, bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
# In[213]:


def main():
    t2 = time.time()
    print('this took ' + str(round(t2-t1, 2)) + ' seconds')
    print('**PLOTTING DATA**')
    epoch2epoch_and_perepoch_var(X1r, Y1r, W1r, W2r, pifu_r, X1b, Y1b, W1b, W2b, pifu_b, 'MaNGA_WISE_variations_colorcut_sigradec.jpg')
    jiyan_cut(X1r, Y1r, W1r, W2r, pifu_r, X1b, Y1b, W1b, W2b, pifu_b, 'MaNGA_WISE_JiYan_colorcut_sigradec.jpg')
    jiyan_spur(X1rs, Y1rs, W1rs, W2rs, pifu_rs, X1bs, Y1bs, W1bs, W2bs, pifu_bs, 'MaNGA_WISE_JiYan_spur_sigradec.jpg')
    w12_distribution(W1_spur, W2_spur, 'MaNGA_WISE_spur_W12_distribution.jpg')
    W2_vs_err_varW2(X1, Y1, W1, W2, plateifu, 'MaNGA_WISE_W2_vs_VarW2.jpg')
    var_w_W2_cut(X1, Y1, W1, W2, plateifu, 'MaNGA_WISE_W2cut_variability.jpg')
    mnsa_vs_wise_plots(w1_w, w2_w, w1_m, w2_m, 'mnsa_vs_wise_mags_distmod.jpg')
    t3 = time.time()
    print('the entire process took ' + str(round(t3-t1, 2)) + ' seconds')


# In[214]:


if __name__ == "__main__":
    t1 = time.time()
    print('**FLITERING DATA**')

    manga_hdu, wise_hdu, jiyan_hdu = import_files()
    X1, Y1, W1, W2, plateifu = process_wise_data_variability_plot(wise_hdu.data)
    X1r, Y1r, W1r, W2r, pifu_r, X1b, Y1b, W1b, W2b, pifu_b = w12_cut(ASSEF_CUT, X1, Y1, W1, W2, plateifu)
    X1_spur, Y1_spur, W1_spur, W2_spur, pifu_spur = spur_objects(X1, Y1, W1, W2, plateifu)
    X1rs, Y1rs, W1rs, W2rs, pifu_rs, X1bs, Y1bs, W1bs, W2bs, pifu_bs = spur_w12cut(X1_spur, Y1_spur, W1_spur, W2_spur, pifu_spur)
    w1_w, w2_w, w1_m, w2_m = mnsa_vs_wise(W1, W2, plateifu, wise_hdu.data, manga_hdu.data)
    
    main()

    
