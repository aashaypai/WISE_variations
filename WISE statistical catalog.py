#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
import astropy.units as u
#from astropy.units import cds
#cds.enable()

from astropy.cosmology import LambdaCDM, FlatLambdaCDM

from PIL import Image

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


# In[14]:


def import_manga():
    manga_file = fits.open(r"C:\Users\paiaa\Downloads\mnsa-0.3.0.fits")
    hdu_manga = manga_file[2]
    manga_data = hdu_manga.data
    return manga_data

def write_query_table():
    RA = manga_file[1].data['objra']
    dec = manga_file[1].data['objdec']
    
    table = Table([RA, dec], names=('ra', 'dec'))
    table.write('objects_all.tbl', format = 'ipac', overwrite = True)
    return table

#spitzer_file = fits.open(r"C:\Users\paiaa\Downloads\asu (2).fit")
#pipe3d_file = fits.open(r"C:\Users\paiaa\Downloads\SDSS17Pipe3D_v3_1_1.fits")


# In[15]:


#hdu_p3d = pipe3d_file[1]
#hdu_spitzer = spitzer_file[1]


#p3d_data = hdu_p3d.data
#spitzer_data = hdu_spitzer.data


# In[16]:


#hdu_manga.header


# In[17]:


#csv with 37 objects
#neowise_data = pd.read_csv(r'C:\Users\paiaa\Blanton Lab\WISE variations\table_irsa_neowise_catalog_search_results.csv')
#allsky_data = pd.read_csv(r'C:\Users\paiaa\Blanton Lab\WISE variations\table_irsa_wise_allsky_catalog_search_results.csv')

#csv with all MaNGA objects
neowise_data = pd.read_csv(r'C:\Users\paiaa\Blanton Lab\WISE variations\neowise_2arcsec_catalog_search_results_all.csv')
allsky_data = pd.read_csv(r'C:\Users\paiaa\Blanton Lab\WISE variations\allsky_2arcsec_catalog_search_results_all.csv')
wise_data = pd.concat([allsky_data, neowise_data], axis = 0)
#wise_data2 = pd.concat([allsky_data2, neowise_data2], axis = 0)


# In[18]:


#processed data (using the process_df function) of all MaNGA objects
#processed_wise_data = pd.read_csv('manga_wise_data.csv')
#wise_avg = pd.read_csv('manga_wise_avg_data.csv')
#wise_var = pd.read_csv('manga_wise_var_data.csv')


# In[ ]:





# # functions to process data

# In[19]:


#global variables
ALL_PAD_LENGTH = 26
PER_PAD_LENGTH = 11273
MANGA_DATA = import_manga()


# In[20]:


def sort(df):
    df = df.sort_values(by=['cntr_01', 'mjd'], ascending = [True, True])
    return df

def add_columns(df):
    #datetime column to sort by date
    df['date'] = pd.to_datetime(df['mjd'].to_numpy() + 2400000.5, origin='julian', unit='D')
    
    #propagating uncertainties to W1 and W2 with the uncertainties package
    w1mpro = unp.uarray(df['w1mpro'], df['w1sigmpro'])
    w2mpro = unp.uarray(df['w2mpro'], df['w2sigmpro'])
    w12 = w1mpro-w2mpro
    w12sig = unp.std_devs(w12)
    df['w12sigmpro (error propagated)'] = w12sig
    df['w1mpro-w2mpro (error propagated)'] = unp.nominal_values(w12)
    
    
    df['error squared'] = np.square(df['w2sigmpro'])
    w2sig = np.square(w2mpro)
    df['error squared (error propagated)'] = unp.std_devs(w2sig)
    
    #replacing NaN values with -9999
    #df = df.fillna(-9999)
    return df

def filter_data(df):
    df = df[df['qual_frame'] != 0]
    df = df[df['cc_flags'] == '0000']
    
    return df

def mean_var_data(df, freq):
    avg = pd.DataFrame()
    var = pd.DataFrame()
    epoch_count = pd.DataFrame()
    epoch_count = pd.DataFrame()
    plateifus = np.array([])
    for i in range(df['cntr_01'].max()):
        
        if i % 2500 == 0:
            print('processing object ' + str(i) + ' out of ' + str(df['cntr_01'].max()))
        
        objects = df[df['cntr_01'] == i + 1]
        
        #calculating mean dataframe
        temp1 = objects.groupby(pd.Grouper(key = 'date', freq = freq), dropna = False).mean().reset_index()
        avg = pd.concat([avg, temp1])
        #calculating variance dataframe
        temp2 = objects.groupby(pd.Grouper(key = 'date', freq = freq), dropna = False).var().reset_index()
        var = pd.concat([var, temp2])
        
        temp3 = objects.groupby(pd.Grouper(key = 'date', freq = freq), dropna = False).size().reset_index(name = 'count')
        epoch_count = pd.concat([epoch_count, temp3])
        
        #adding plateifu column
        plateifu = np.repeat(MANGA_DATA['plateifu'][i], objects.shape[0])
        plateifus = np.append(plateifus, plateifu)
        
        #fixing cntr_01 column
        avg['cntr_01'] = avg['cntr_01'].fillna(method = 'bfill')
        var['cntr_01'] = avg['cntr_01']
    avg['epoch count'] = epoch_count['count']
    df['plateifu'] = plateifus   
        
    return df, avg, var

def calculate_epoch(df, freq):
    df['epoch'] = df.groupby([ pd.Grouper(key = 'date', freq = freq)]).ngroup()
    
    return df

def pad_data(stat_data, pad_length):
    pad_length = pad_length
    padded_data = []
    
    for array in stat_data:
        #print(array)
        if array.shape[0] < pad_length:
            array = np.pad(array, pad_width = (0, int(pad_length - array.shape[0])), constant_values = -9999.)
            padded_data.append(array)
        
        else:
            padded_data.append(array)

    return padded_data


# In[21]:


def process_df(df, freq):
    print('**PROCESSING DATAFRAME**')
    t1 = time.time()
    print('\n**dataframe recieved**')
    result = sort(df)
    print('**dataframe sorted**')
    result = add_columns(result)
    print('**relevant columns added**')
    result = filter_data(result)
    print('**data filtered**')
    result, result_mean, result_var = mean_var_data(result, freq)
    print('**mean, variance calculated**')
    result = calculate_epoch(result, freq)
    print('**epochs labelled**')
    t2 = time.time()
    print('this took: ' + str(round(t2-t1, 2)) + ' seconds')
    return result, result_mean, result_var


# In[22]:


#processed_wise_data, wise_avg, wise_var = process_df(wise_data, '181D')


# In[ ]:





# # Statistical calculations

# ##### (d) mean magnitude at each epoch (unweighted mean), (e) expected variance at each epoch (based on catalog sigma), (f) expected variance at each epoch (based on within-epoch magnitudes)

# In[23]:


#(d)
def mean_w2_per_epoch(df_avg):
    mean_mag = []
    #look at mean mag per epoch in mean dataframe
    for i in range(int(df_avg['cntr_01'].max())):
    
        avg = df_avg[df_avg['cntr_01'] == i+1]
        temp = avg['w2mpro'].to_numpy()

        mean_mag.append(temp)
    padded_data = pad_data(mean_mag, ALL_PAD_LENGTH)
    return padded_data, mean_mag

def mean_w1_per_epoch(df_avg):
    mean_mag = []
    #look at mean mag per epoch in mean dataframe
    for i in range(int(df_avg['cntr_01'].max())):
    
        avg = df_avg[df_avg['cntr_01'] == i+1]
        temp = avg['w1mpro'].to_numpy()

        mean_mag.append(temp)
    padded_data = pad_data(mean_mag, ALL_PAD_LENGTH)
    return padded_data, mean_mag

#(e) #want it to be avg of sigma squared, not square of avg sigma
def expected_var_per_epoch_sigma(df_avg):
    expected_var = []
    #look at variance (sigma squared) of the mean error (sigma) at each epoch
    for i in range(int(df_avg['cntr_01'].max())):
    
        avg = df_avg[df_avg['cntr_01'] == i+1]
        temp = avg['error squared'].to_numpy()

        expected_var.append(temp)
    padded_data = pad_data(expected_var, ALL_PAD_LENGTH)
    return padded_data, expected_var

#(f)
def expected_var_per_epoch_mags(df_var):
    expected_var = []
    #look at variance of magnitudes per epoch
    for i in range(int(df_var['cntr_01'].max())):
    
        var = df_var[df_var['cntr_01'] == i+1]
        temp = var['w2mpro'].to_numpy()

        expected_var.append(temp)
    padded_data = pad_data(expected_var, ALL_PAD_LENGTH)
    return padded_data, expected_var


# ##### (a) observed variance across all epochs, (b) expected variance across all epochs (based on catalog sigma),  (c) expected variance across all epochs (based on within-epoch variances)

# In[24]:


#(a)
def observed_var_all(df_avg):
    observed_var = np.array([])
    #isolate each galaxy
    for i in range(int(df_avg['cntr_01'].max())):
        avg = df_avg[df_avg['cntr_01'] == i+1]
        #look at the variance of the mean W2 across all epochs (the mean was taken across all epochs)
        observed_var = np.append(observed_var, avg['w2mpro'].var())
    return observed_var

#(b)
def expected_var_all_sigma(df_avg):
    expected_var = np.array([])
    padded, expected_var_per_epoch = expected_var_per_epoch_sigma(df_avg)

    for array in expected_var_per_epoch:
        
        var = np.nanmean(array)/array[~np.isnan(array)].shape[0]
        expected_var = np.append(expected_var, var)
    return expected_var

#(c)
def expected_var_all_mags(df_var):
    expected_var = np.array([])
    padded, expected_var_per_epoch = expected_var_per_epoch_mags(df_var)
    
    for array in expected_var_per_epoch:
        var = np.nanmean(array)/array[~np.isnan(array)].shape[0]
        expected_var = np.append(expected_var, var)
    
    return expected_var



# ##### (g) Date associated with epoch, (h) Number of good observations at each epoch, (i) Number of epochs

# In[25]:


#(g)
def date_per_epoch(df_avg):
    dates = []
    for i in range(int(df_avg['cntr_01'].max())):
        objects = df_avg[df_avg['cntr_01'] == i + 1]
        
        t = Time(objects['date'])
        t = t.to_value('mjd', 'float')
        dates.append(t)
    padded_data = pad_data(dates, ALL_PAD_LENGTH)
    return padded_data

#(i)
def epochs_per_object(df):
    epoch_count = np.array([])
    for i in range(int(df['cntr_01'].max())):
        objects = df[df['cntr_01'] == i + 1]
        count = objects['epoch'].max()
        epoch_count = np.append(epoch_count, count)
        
    return epoch_count

#(h)
def good_obs_per_epoch(df_avg):
    obs_count = []
    
    for i in range(int(df_avg['cntr_01'].max())):
    
        objects = df_avg[df_avg['cntr_01'] == i+1]
        obs = objects['epoch count'].to_numpy()

        obs_count.append(obs)
    padded_data = pad_data(obs_count, ALL_PAD_LENGTH)
    return padded_data

def plateifu():
    plateifu = MANGA_DATA['plateifu']
    return plateifu


# ### Create FITS file

# In[26]:


def create_fits_data(df, df_avg, df_var):
    t1 = time.time()
    print('\n**PERFORMING STATISTICAL CALCULATIONS**')
    plateifus = plateifu()
    obs_per_epoch = good_obs_per_epoch(df_avg)
    epochs_per_obj = epochs_per_object(df)
    epoch_date = date_per_epoch(df_avg)
    print('**finished calculating distribution of observations**')
    padded_mean_w1, mean_w1 = mean_w1_per_epoch(df_avg)
    padded_mean_w2, mean_w2 = mean_w2_per_epoch(df_avg)
    padded_exp_var_sigma, exp_var_sigma = expected_var_per_epoch_sigma(df_avg)
    padded_exp_var_mags, exp_var_mags = expected_var_per_epoch_mags(df_var)
    print('**finished calculating within epoch statistics**')
    obs_var_all = observed_var_all(df_avg)
    exp_var_sig_all = expected_var_all_sigma(df_avg)
    exp_var_mags_all = expected_var_all_mags(df_var)
    print('**finished calculating epoch to epoch statistics**')
    t2 = time.time()
    print('this took: ' + str(round(t2-t1, 2)) + ' seconds')
    stat_data = [plateifus, obs_per_epoch, epochs_per_obj, epoch_date, padded_mean_w1, padded_mean_w2, padded_exp_var_sigma, padded_exp_var_mags, obs_var_all, exp_var_sig_all, exp_var_mags_all]
    #stat_data = np.nan_to_num(stat_data, nan = -9999.)
    return stat_data



# In[27]:


def create_fits_file(stat_data):
    print('\n**CREATING FITS HDU**')
    names = np.array(['plateIFU', 'obs per epoch', 'epochs per obj', 'epoch date', 'mean W1 per epoch', 'mean W2 per epoch', 'expected var (errors)', 'expected var (mags)', 'observed var', 'expected var all epochs (errors)', 'expected var all epochs (mags)'])
    formats = np.array(['12A', '26K', 'D', '26D', '26D', '26D', '26D', '26D', 'D', 'D', 'D'])
    hdu = fits.PrimaryHDU()
    cols = np.array([])
    for i, name in enumerate(names):
        
        column = fits.Column(name = name, array =  stat_data[i], format = formats[i])
        cols = np.append(cols, column)
    hdu = fits.BinTableHDU.from_columns(cols)
    #print(hdu.header)
    return hdu


# In[42]:


def save_data(hdu, processed_wise_data, wise_avg, wise_var):
    print('\n**SAVING DATA**')
    hdu.writeto('WISE variations/wise_statistics_2arcsec_sigradec.fits', overwrite=True)
    processed_wise_data.to_csv('WISE variations/processed_manga_data_2arcsec_sigradec.csv')
    wise_avg.to_csv('WISE variations/manga_avg_data_2arcsec_sigradec.csv')
    wise_var.to_csv('WISE variations/manga_var_data_2arcsec_sigradec.csv')


# ## Main function

# In[29]:


def main(freq, save = True):
    t1 = time.time()
    processed_wise_data, wise_avg, wise_var = process_df(wise_data, freq)
    stat_data = create_fits_data(processed_wise_data, wise_avg, wise_var)
    hdu = create_fits_file(stat_data)
    
    if save == True:
        save_data(hdu, processed_wise_data, wise_avg, wise_var)
    
    t2 = time.time()
    print('the entire process took: ' + str(round(t2-t1, 2)) + ' seconds')
    
    return hdu, processed_wise_data, wise_avg, wise_var


# In[30]:


if __name__ == "__main__":
    hdu, processed_wise_data, wise_avg, wise_var = main('181D', save = False)


# ## Testing bad code

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




