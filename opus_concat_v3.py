# -*- coding: utf-8 -*-
"""
opus_concat_v3.py

Sequentially opens OPUS files in specified directory, concatenates spectra in DataFrame, extracts acquisition time, 
and returns concatenated spectra, wavenumbers, and acquisition times as numpy arrays and *.csv*.

Required input: 
    1.  folder_spectra (path to folder that contains to-process spectra)
    2.  folder_output (path to folder where output will be stored)
    3.  name_output (output file name, e.g., ELN- or experiment number)

Optional input:
    1.  csv_export (boolean, controls whether output is saved as *.csv* or not. Default behavior = True)

Output (numpy array and *.csv*):
    1.  Concatenated spectra of files in folder
    2.  Wavenumbers of spectra in folder
    3.  Acquisition datetime of spectra in folder
    4.  Acquisition time of each spectrum since the experiment was started, in s

Required packages:
    1.  brukeropusreader 1.3.9 (conda install -c spectrocat brukeropusreader)

R. van Putten
<rvanputt@its.jnj.com>

February 2023
"""

def opus_concat(folder_spectra, folder_output, name_output, csv_export = True):
    
    #%% IMPORT MODULES
    
    import os
    import re
    import pandas as pd
    import numpy as np
    from brukeropusreader import read_file
    from datetime import datetime    

    #%% IMPORT SPECTRA AND BUILD DATAFRAME
    
    first = True # Create 'first' variable to specify first loop iteration
    
    if len(os.listdir(folder_spectra)) > 20:
        print('Processing', len(os.listdir(folder_spectra)), 'spectra. This may take a while...')
    
    for file in os.listdir(folder_spectra):
        '''Loops over files in folder to construct DataFrame'''
        
        if first:
            '''Copy wavenumbers and spectral data in first iteration'''
            
            first = False # Turn off 'first' designation
            
            opus_data = read_file(os.path.join(folder_spectra, file))
            wavenumbers = opus_data.get_range('AB')
            transmittance = opus_data['AB_(1)'][0:len(wavenumbers)]
            acq_datetime = opus_data['AB Data Parameter']['DAT'] + ' ' + opus_data['AB Data Parameter']['TIM']
            timeseries = np.array(np.datetime64(datetime.strptime(acq_datetime[0:18], '%d/%m/%Y %H:%M:%S')))
            df = pd.DataFrame(transmittance, columns=[str(file)])
          
        else:
            '''Concatenate each spectrum to existing DataFrame'''
            
            opus_data = read_file(os.path.join(folder_spectra, file))
            transmittance = opus_data['AB_(1)'][0:len(wavenumbers)]
            acq_datetime = opus_data['AB Data Parameter']['DAT'] + ' ' + opus_data['AB Data Parameter']['TIM']
            timeseries_tmp = np.array(np.datetime64(datetime.strptime(acq_datetime[0:18], '%d/%m/%Y %H:%M:%S')))
            
            df_tmp = pd.DataFrame(transmittance, columns=[str(file)])
            df = pd.concat([df, df_tmp], axis=1)
            timeseries = np.append(timeseries, timeseries_tmp)
    
    #%% SORT COLUMNS IN CORRECT ORDER
    
    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        def atoi(text):
            return int(text) if text.isdigit() else text
    
        return [atoi(c) for c in re.split('(\d+)', text)]
    
    df = df.reindex(columns=sorted(df.columns, key=natural_keys))
    acq_datetime = np.sort(timeseries)
    
    #%% COMPUTE TIMEDELTA ARRAY - DELTA TIME FROM FIRST SPECTRUM IN SECONDS
    
    acq_timedelta_s = np.empty([len(acq_datetime),1])
    
    for i in range(len(acq_datetime)):
        acq_timedelta_s[i] = ((acq_datetime[i] - acq_datetime[0]).astype('timedelta64[s]'))
        
    #%% EXPORT DATA
    
    if(csv_export == True):
        '''Export processed data as *.csv*'''
        
        print('Saving files as *.CSV*')
        
        df.to_csv(folder_output + '\\' + name_output + '_sorted_spectra.csv', index=None, header=True)
        pd.DataFrame(wavenumbers).to_csv(folder_output + '\\' + name_output + '_wavenumbers.csv', index=None, header=['Wavenumber [cm-1]'])
        pd.DataFrame(acq_datetime).to_csv(folder_output + '\\' + name_output + '_acq_datetime.csv', index=None, header=['Acquisition datetime'])
        pd.DataFrame(acq_timedelta_s).to_csv(folder_output + '\\' + name_output + '_acq_timedelta_s.csv', index=None, header=['Acquisition timedelta [s]'])

    else:
        pass

    return df.to_numpy(), wavenumbers, acq_datetime, acq_timedelta_s