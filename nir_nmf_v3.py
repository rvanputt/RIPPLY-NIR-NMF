# -*- coding: utf-8 -*-
"""
nir_nmf_v3.py

RIPPLY-NIR-NMF enables convenient pre-processing and multivariate analysis (semi-NMF and PCA) of (timeseries of) near-infrared spectra. Spectra in Bruker OPUS file format can be used directly.

Code accompanies published work: R. van Putten, K. De Smet, L. Lefort, "RIPPLY: Real-tIme Parallel Progress anaLYsis of organic reactions using near-infrared spectroscopy, DOI: tbd.

Short description of code:
    1. Processes raw NIR spectra from Bruker OPUS (*.0*, *.1*, etc.) and creates a *.csv* with all spectral information. Data are loaded from *.csv* if available in folder, unless 
        'opus_processing_override = True'.
    2. Pre-processes NIR spectra with nippy (further documentation at https://github.com/UEF-BBC/nippy) and export results as *.csv*.
    3. Performs multivariate analysis on pre-processed data (semi-NMF and principal component analysis) and exports results as *.csv*.
    4. Plots results.

Code assumes the following folder structure:

Main folder (base path)
    experiment ID (e.g. experiment- or ELN number)
        OPUS Spectra
            Spectrum.0
            Spectrum.1
            Spectrum.2
            .
            .
            .
            Spectrum.i
        References
            Reference spectrum 1.0
            Reference spectrum 2.0

Required input: 
    1.  Base path
    2.  Experiment ID to analyze (subfolder in base path folder)
    3.  Settings (see SETTINGS section below, lines 73-101)

Optional input:
    1.  Reference spectra (OPUS file format)

Output:
    1.  Concatenated and pre-processed spectra as *.csv*
    2.  Multivariate analysis output as *.csv*
    3.  Plots of raw- and pre-processed spectra (including references, if available) and overview of multivariate analysis results

Required packages:
    1.  brukeropusreader 1.3.9 (conda install -c spectrocat brukeropusreader)
    2.  configparser 5.3.0 (pip install configparser)
    3.  nippy 1.0 (pip install git+https://github.com/UEF-BBC/nippy.git)
    4.  sklearn 1.0.2+

Required local functions:
    1.  opus_concat_v3.py
    2.  semi_nmf_v1.py

R. van Putten <rvanputt@its.jnj.com>
February 2023
"""

#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import configparser
import nippy
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from opus_concat_v3 import opus_concat
from semi_nmf_v1 import semi_nmf


#%% SETTINGS
'''Parameters for data import and multivariate analysis'''

# Data input
base_path = os.path.dirname(os.path.abspath(__file__))             # Main folder path. Must contain folder named as experiment ID
experiment_ID = 'Example data'                                     # Experiment to analyze
data_subfolder = 'OPUS Spectra'                                    # Experiment ID subfolder with raw spectra
references_subfolder = 'References'                                # Experiment ID subfolder with reference spectra
parallel_experiments = 6                                           # Number of parallel experiments. If >1, number of spectra for each experiment must be identical
opus_processing_override = False                                   # If true, reprocess raw OPUS files and re-generate output *.csv*. If false, will use existing *.csv* (if avai

# Pre-processing
savitzky_golay_window = 15                                         # Number of points in Savitzky-Golay filter window
savitzky_golay_polynomial = 3                                      # Order of polynomial used for Savitzky-Golay smoothing
savitzky_golay_derivative = 2                                      # Order of smoothed derivative returned after pre-processing
spec_range = np.array([6100, 6000])                                # Spectral range of interest. Enter high wavenumber to low: 12000-40

# Multivariate analysis
n_components = 2                                                   # Number of components to fit
W_initialization = True                                            # Reference spectra for semi-NMF initialization. Otherwise random initialization
np.random.seed(1)                                                  # Seed for random initialization
max_iter = 2000                                                    # Maximum number of iterations
seminmf_divergence_check = False                                   # Control divergence during semi-NMF analysis
multivariate_method = 'semi-NMF'                                   # Multivariate method for detailed analysis. Available options: semi-NMF and PCA

# Post-NMF data processing
plot_units = 'nm'                                                  # Units of wavelength/wavenumber plots. Available options: nm, cm-1
conversion_scaling = False                                         # Scale normalized NMF H data by final conversion
conversion = np.array([0.64, 0.85, 0.85, 0.84, 0.84, 0.46])        # Example data scaling to conversion


#%% DATA IMPORT
'''Import new data or load from existing file'''

# Set folder structure
experiment_folder = base_path + '\\' + experiment_ID
data_subfolder = experiment_folder + '\\' + data_subfolder
references_subfolder = experiment_folder + '\\' + references_subfolder

# Check if number of spectra supplied is multiple of number of experiments. If not, raise exception and stop execution
if (len(os.listdir(data_subfolder))/parallel_experiments).is_integer() != True:        
    print('Error during execution. Please check if number of spectra is identical for all experiments')
    sys.exit()   

# Create new file for data and save as *.csv*
if opus_processing_override == True or os.path.exists(experiment_folder + '\\' + experiment_ID + '_sorted_spectra.csv') == False:   
    spectra, wavenumbers, acq_datetime, acq_timedelta_s = opus_concat(data_subfolder, experiment_folder, experiment_ID, True)
    print('OPUS Reader: created new output')
    
# Data file already exists; load data
elif opus_processing_override == False and os.path.exists(experiment_folder + '\\' + experiment_ID + '_sorted_spectra.csv') == True:       
    spectra = np.loadtxt(open(experiment_folder + '\\' + experiment_ID + '_sorted_spectra.csv'), delimiter=',', skiprows=1)
    wavenumbers = np.loadtxt(open(experiment_folder + '\\' + experiment_ID + '_wavenumbers.csv'), delimiter=',', skiprows=1)
    acq_datetime = np.loadtxt(open(experiment_folder + '\\' + experiment_ID + '_acq_datetime.csv'), delimiter=',', skiprows=1, dtype='datetime64')
    acq_timedelta_s = np.loadtxt(open(experiment_folder + '\\' + experiment_ID + '_acq_timedelta_s.csv'), delimiter=',', skiprows=1)
    print('OPUS Reader: spectral data loaded from folder')

# Load spectra of reference compounds for W-matrix initialization
if os.path.exists(references_subfolder) == True and len(os.listdir(references_subfolder)) != 0:
    spectra_references = opus_concat(references_subfolder, references_subfolder, 'references', False)[0]

# Catch error when W initialization is on but no references are provided
if W_initialization == True and os.path.exists(references_subfolder) == False:
    if len(os.listdir(references_subfolder)) == 0:
        print('Error during execution. W-initialization is on, but no references provided')
        sys.exit()  
    print('Error during execution. W-initialization is on, but no references provided')
    sys.exit()     

# Convert units for convenient plotting
approx_acq_timedelta_s = np.zeros([int(len(acq_timedelta_s)/parallel_experiments),1])

for i in range(len(approx_acq_timedelta_s)):
    # Construct approximate acquisition time array from exact acquisition times for plotting
    approx_acq_timedelta_s[i,0] = acq_timedelta_s[i*parallel_experiments]

approx_acq_timedelta_min = approx_acq_timedelta_s/60
approx_acq_timedelta_h = approx_acq_timedelta_s/3600


#%% PRE-PROCESSING
'''Spectral pre-processing using nippy. Automatically generates nippy config file. See nippy documentation for details on usage'''

# Generate nippy config file
config = configparser.ConfigParser()
config.read('nippy_config.ini')
config.set('SAVGOL', 'filter_win', str(savitzky_golay_window))
config.set('SAVGOL', 'poly_order', str(savitzky_golay_polynomial))
config.set('SAVGOL', 'deriv_order', str(savitzky_golay_derivative))
config.set('TRIM', 'bins', str(spec_range[1]) + '-' + str(spec_range[0]))

with open('nippy_config.ini', 'w') as configfile:
    config.write(configfile)

# Pre-process spectra
sys.stdout = open(os.devnull, 'w') # Block nippy print to terminal
preprocessed_spectra = nippy.nippy(wavenumbers, spectra, nippy.read_configuration('nippy_config.ini'))[0][1]
preprocessed_wavenumbers = nippy.nippy(wavenumbers, spectra, nippy.read_configuration('nippy_config.ini'))[0][0]

if os.path.exists(references_subfolder) == True and len(os.listdir(references_subfolder)) != 0:
    preprocessed_references = nippy.nippy(wavenumbers, spectra_references, nippy.read_configuration('nippy_config.ini'))[0][1]
    preprocessed_references_wavenumbers = nippy.nippy(wavenumbers, spectra_references, nippy.read_configuration('nippy_config.ini'))[0][0]
    np.savetxt(experiment_folder + '\\' + experiment_ID + '_preprocessed_references.csv', preprocessed_references, delimiter=',')
    np.savetxt(experiment_folder + '\\' + experiment_ID + '_preprocessed_references_wavenumbers.csv', preprocessed_references_wavenumbers, delimiter=',')
sys.stdout = sys.__stdout__ # Re-enable printing to terminal

# Convert units for convenient plotting
wavenumbers_nm = 1e7/wavenumbers
preprocessed_wavenumbers_nm = 1e7/preprocessed_wavenumbers

# Save intermediate output in experiment ID folder
np.savetxt(experiment_folder + '\\' + experiment_ID + '_preprocessed_spectra.csv', preprocessed_spectra, delimiter=',')
np.savetxt(experiment_folder + '\\' + experiment_ID + '_preprocessed_wavenumbers.csv', preprocessed_wavenumbers, delimiter=',')
np.savetxt(experiment_folder + '\\' + experiment_ID + '_preprocessed_wavenumbers_nm.csv', preprocessed_wavenumbers_nm, delimiter=',')

print('Pre-processing complete. Starting multivariate analysis')


#%% MULTIVARIATE ANALYSIS
'''Multivariate analysis on preprocessed spectra using semi-NMF or sklearn PCA. See sklearn documentation for details on usage'''

# Initialize W and H for semi-NMF
W0 = np.random.rand(np.shape(preprocessed_spectra)[0], n_components)
H0 = np.random.rand(n_components, np.shape(preprocessed_spectra)[1])

if W_initialization == True:
    W0 = preprocessed_references

# semi-NMF
semiNMF_W, semiNMF_H = semi_nmf(preprocessed_spectra, W0, H0, max_iter, seminmf_divergence_check)
semiNMF_residuals = preprocessed_spectra - np.matmul(semiNMF_W, semiNMF_H)

# sklearn PCA
pca = PCA(n_components)
PCA_W = pca.fit_transform(preprocessed_spectra)
PCA_H = pca.components_
PCA_residuals = preprocessed_spectra - pca.inverse_transform(PCA_W)

# Save intermediate output in experiment ID folder
np.savetxt(experiment_folder + '\\' + experiment_ID + '_semiNMF_W.csv', semiNMF_W, delimiter=',')
np.savetxt(experiment_folder + '\\' + experiment_ID + '_semiNMF_H.csv', semiNMF_H, delimiter=',')
np.savetxt(experiment_folder + '\\' + experiment_ID + '_semiNMF_residuals.csv', semiNMF_residuals, delimiter=',')
np.savetxt(experiment_folder + '\\' + experiment_ID + '_PCA_W.csv', PCA_W, delimiter=',')
np.savetxt(experiment_folder + '\\' + experiment_ID + '_PCA_H.csv', PCA_H, delimiter=',')
np.savetxt(experiment_folder + '\\' + experiment_ID + '_PCA_residuals.csv', PCA_residuals, delimiter=',')

# Select results for detailed analysis

if multivariate_method == 'semi-NMF':
    W = semiNMF_W
    H = np.transpose(semiNMF_H)
    residuals = semiNMF_residuals

if multivariate_method == 'PCA':
    W = PCA_W
    H = np.transpose(PCA_H)
    residuals = PCA_residuals

elif multivariate_method != 'PCA' and multivariate_method != 'semi-NMF':
    print('Error during execution. Please check if correct multivariate method is provided.')
    sys.exit() 

#%% PROCESS RESULTS
'''Processes results from multivariate analysis: break into separate experiments (if parallel_experiments >1) and scale H-matrix'''

# Compute number of spectra for each experiment and initialize empty arrays
pts_per_exp = int(len(H)/parallel_experiments)                                                                                  
H_separate = np.zeros([int(len(H)/parallel_experiments), parallel_experiments*n_components])      
H_separate_norm = np.zeros([int(len(H)/parallel_experiments), parallel_experiments*n_components])
H_separate_scaled = np.zeros([int(len(H)/parallel_experiments), parallel_experiments*n_components])
H_separate_fit = np.zeros([int(len(H)/parallel_experiments), parallel_experiments*n_components])

# Reshape combined H matrix into components for individual experiments
for i in range(parallel_experiments):   
    H_separate[:,i*n_components:(1+i)*n_components] = H[i*pts_per_exp:(1+i)*pts_per_exp,:]

# Normalize columns
for i in range(H_separate.shape[1]):
    for j in range(len(H_separate)):
        H_separate_norm[j,i] = (H_separate[j,i] - min(H_separate[:,i])) / (max(H_separate[:,i]) - min(H_separate[:,i]))
                
# Scale results to final conversion
if conversion_scaling == True:
    for i in range(parallel_experiments):
        for j in range(n_components):

            # Component increases with time
            if H_separate_norm[-1,i*n_components+j] > H_separate_norm[0,i*n_components+j]:
                H_separate_scaled[:,i*n_components+j] = H_separate_norm[:,j+i*n_components]*conversion[i]

            # Component decreases with time
            else:                
                H_separate_scaled[:,i*n_components+j] = H_separate_norm[:,j+i*n_components]*conversion[i] + (1-conversion[i])
            

#%% EXPORT NMF RESULTS
'''Export best W matrix, normalised W matrix, and scaled W matrix to *.csv*'''

print('Analysis complete. Exporting results')

# Save output in experiment ID folder
np.savetxt(experiment_folder + '\\' + experiment_ID + '_H_invididual.csv', H_separate, delimiter=',')
np.savetxt(experiment_folder + '\\' + experiment_ID + '_H_individual_norm.csv', H_separate_norm, delimiter=',')
np.savetxt(experiment_folder + '\\' + experiment_ID + '_H_individual_scaled.csv', H_separate_scaled, delimiter=',')


#%% VISUALIZE RESULTS

# Set x-axes to desired unit
x_axis = preprocessed_wavenumbers
x_label = 'Wavenumber [cm$^{-1}$]'
x_lower = spec_range[0]
x_upper = spec_range[1]

if plot_units == 'nm':
    x_axis = preprocessed_wavenumbers_nm
    x_label = 'Wavelength [nm]'
    x_lower = 1e7/spec_range[0]
    x_upper = 1e7/spec_range[1]

print('Generating figures')
plt.close('all')

colors = np.array(['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','lime','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','black','lime'])
alpha = 0.8
linewidth = 2
markersize = 2

# Overview of raw- and preprocessed spectra
plt.figure(figsize=(12,8))
plt.suptitle(experiment_ID, fontsize=14)
plt.subplot(1,2,1)    
if plot_units == 'nm':
    plt.plot(1e7/wavenumbers, spectra, linewidth=linewidth)
    plt.xlim(x_lower, x_upper)
else:
    plt.plot(wavenumbers, spectra, linewidth=linewidth)
    plt.xlim(x_lower, x_upper)
plt.title('Raw spectra')
plt.xlabel(x_label)
plt.ylabel('Absorbance [a.u.]')
plt.grid(which='major',axis='both')
plt.subplot(1,2,2)
plt.plot(x_axis, preprocessed_spectra, linewidth=linewidth)
plt.xlim(x_lower, x_upper)
plt.title('Pre-processed spectra')
plt.xlabel(x_label)
plt.ylabel('[a.u.]')
plt.grid(which='major',axis='both')
plt.tight_layout()
plt.savefig(experiment_folder + '\\' + experiment_ID + '_raw_preprocessed_spectra' + '.png')

# Overview of raw- and preprocessed reference spectra if references are provided
if os.path.exists(references_subfolder) == True and len(os.listdir(references_subfolder)) != 0:
    plt.figure(figsize=(12,8))
    plt.suptitle(experiment_ID, fontsize=14)
    plt.subplot(1,2,1)    
    if plot_units == 'nm':
        plt.plot(1e7/wavenumbers, spectra_references, linewidth=linewidth)
        plt.xlim(x_lower, x_upper)
    else:
        plt.plot(wavenumbers, spectra_references, linewidth=linewidth)
        plt.xlim(x_lower, x_upper)
    plt.title('Raw reference spectra')
    plt.xlabel(x_label)
    plt.ylabel('Absorbance [a.u.]')
    plt.grid(which='major',axis='both')
    plt.legend(os.listdir(references_subfolder))
    plt.subplot(1,2,2)
    plt.plot(x_axis, preprocessed_references, linewidth=linewidth)
    plt.xlim(x_lower, x_upper)
    plt.title('Pre-processed reference spectra')
    plt.xlabel(x_label)
    plt.ylabel('[a.u.]')
    plt.grid(which='major',axis='both')
    plt.legend(os.listdir(references_subfolder))
    plt.tight_layout()
    plt.savefig(experiment_folder + '\\' + experiment_ID + '_raw_preprocessed_reference_spectra' + '.png')

# Overview of solutions and residuals for semi-NMF and PCA
plt.figure(figsize=(12,8))
plt.suptitle(experiment_ID, fontsize=14)

plt.subplot(3,2,1)
plt.plot(x_axis, semiNMF_W)
plt.title('semi-NMF - W matrix')
plt.xlim(x_lower, x_upper)
plt.grid(which='major',axis='both')
plt.xlabel(x_label)

plt.subplot(3,2,3)
plt.plot(np.transpose(semiNMF_H))
plt.title('semi-NMF - H matrix')
plt.grid(which='major',axis='both')
plt.xlabel('Spectrum # [-]')

plt.subplot(3,2,5)
plt.plot(x_axis, semiNMF_residuals)
plt.title('semi-NMF - residuals')
plt.xlim(x_lower, x_upper)
plt.grid(which='major',axis='both')
plt.xlabel(x_label)

plt.subplot(3,2,2)
plt.plot(x_axis, PCA_W)
plt.title('PCA - loadings')
plt.xlim(x_lower, x_upper)
plt.grid(which='major',axis='both')
plt.xlabel(x_label)

plt.subplot(3,2,4)
plt.plot(np.transpose(PCA_H))
plt.title('PCA - scores')
plt.grid(which='major',axis='both')
plt.xlabel('Spectrum # [-]')

plt.subplot(3,2,6)
plt.plot(x_axis, PCA_residuals)
plt.title('PCA - residuals')
plt.xlim(x_lower, x_upper)
plt.grid(which='major',axis='both')
plt.xlabel(x_label)

plt.tight_layout()
plt.savefig(experiment_folder + '\\' + experiment_ID + '_multivariate_results_overview' + '.png')

# If references provided, overlay of normalized references with W-matrix solution of chosen multivariate algorithm
if os.path.exists(references_subfolder) == True and len(os.listdir(references_subfolder)) != 0:
    plt.figure(figsize=(12,8))
    plt.suptitle(experiment_ID, fontsize=14)
    scaler = MinMaxScaler()
    preprocessed_references_scaled = scaler.fit_transform(preprocessed_references)
    if plot_units == 'nm':
        plt.plot(1e7/preprocessed_wavenumbers, preprocessed_references_scaled, '--', linewidth=linewidth, alpha=alpha)
        plt.xlim(x_lower, x_upper)
    else:
        plt.plot(preprocessed_wavenumbers, preprocessed_references_scaled, '--', linewidth=linewidth, alpha=alpha)
        plt.xlim(x_lower, x_upper)
    if multivariate_method == 'PCA':
        plt.title('PCA - loadings + references')
    else:
        plt.title(multivariate_method + ' - W matrix + references')
    plt.plot(x_axis, scaler.fit_transform(W))
    plt.legend(os.listdir(references_subfolder) + ['Component 1', 'Component 2', 'Component 3', 'Component 4'])
    plt.xlim(x_lower, x_upper)
    plt.grid(which='major',axis='both')
    plt.xlabel(x_label)
    plt.xlim(x_lower, x_upper)
    plt.grid(which='major',axis='both')
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.savefig(experiment_folder + '\\' + experiment_ID + '_' + multivariate_method + '_References_overlay' + '.png')

# Detail of selected multivariate solution 
plt.figure(figsize=(12,8))
plt.suptitle(experiment_ID, fontsize=14)
plt.subplot(3,1,1)
plt.plot(x_axis, W)
if multivariate_method == 'PCA':
    plt.title('PCA - loadings')
else:
    plt.title(multivariate_method + ' - W matrix')
plt.xlim(x_lower, x_upper)
plt.grid(which='major',axis='both')
plt.xlabel(x_label)
plt.subplot(3,1,2)
plt.plot(H)
if multivariate_method == 'PCA':
    plt.title('PCA - scores')
else:
    plt.title(multivariate_method + ' - H matrix')
plt.grid(which='major',axis='both')
plt.xlabel('Spectrum # [-]')
plt.subplot(3,1,3)
plt.plot(x_axis, residuals)
plt.title(multivariate_method + ' - residuals')
plt.xlim(x_lower, x_upper)
plt.grid(which='major',axis='both')
plt.xlabel(x_label)
plt.tight_layout()
plt.savefig(experiment_folder + '\\' + experiment_ID + '_' + multivariate_method + '_multivariate_results_overview' + '.png')

# H matrix, indvididual traces, scaled to conversion if required

if conversion_scaling == True:
    plt.figure(figsize=(12,8))
    plt.suptitle(experiment_ID, fontsize=14) 
    for i in range(n_components):
        plt.subplot(1,n_components,i+1)
        for j in range(parallel_experiments):                   
            plt.plot(approx_acq_timedelta_h, H_separate_scaled[:,j*n_components+i], 'o', markersize=markersize, color=colors[j], alpha=alpha)
            plt.xlabel('Approximate time [h]')
            plt.ylabel('Reaction progress [-]')
        plt.legend(['Reaction 1','Reaction 2','Reaction 3','Reaction 4','Reaction 5','Reaction 6'])               
        plt.tight_layout()
        plt.savefig(experiment_folder + '\\' + experiment_ID + '_H_matrix_scaled_comp_' + str(i+1) + '.png')

if conversion_scaling == False:
    plt.figure(figsize=(12,8))
    plt.suptitle(experiment_ID, fontsize=14) 
    for i in range(n_components):
        plt.subplot(1,n_components,i+1)
        for j in range(parallel_experiments):                   
            plt.plot(approx_acq_timedelta_h, H_separate[:,j*n_components+i], 'o', markersize=markersize, color=colors[j], alpha=alpha)
            plt.xlabel('Approximate time [h]')
            plt.ylabel('Reaction progress [-]')
        plt.legend(['Reaction 1','Reaction 2','Reaction 3','Reaction 4','Reaction 5','Reaction 6'])               
        plt.tight_layout()
        plt.savefig(experiment_folder + '\\' + experiment_ID + '_H_matrix_comp_' + str(i+1) + '.png')

plt.show()