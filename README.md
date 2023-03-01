# RIPPLY-NIR-NMF
Real-time parallel progress analysis of organic reactions using near-infrared spectroscopy and semi-non-negative matrix factorization

# Introduction
RIPPLY-NIR-NMF enables convenient pre-processing and multivariate analysis (semi-NMF and PCA) of (timeseries of) near-infrared spectra. Spectra in Bruker OPUS file format can be used directly.

Code accompanies published work: R. van Putten, K. De Smet, L. Lefort, "RIPPLY: Real-tIme Parallel Progress anaLYsis of organic reactions using near-infrared spectroscopy, DOI: tbd.

Short description of code:
1. Processes raw NIR spectra from Bruker OPUS (*.0*, *.1*, etc.) and creates a *.csv* with all spectral information. Data are loaded from *.csv* if available in folder, unless 'opus_processing_override = True'.
2. Pre-processes NIR spectra with nippy (further documentation at https://github.com/UEF-BBC/nippy) and export results as *.csv*.
3. Performs multivariate analysis on pre-processed data (semi-NMF and principal component analysis) and exports results as *.csv*.
4. Plots results.

Code assumes the following folder structure:

```
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
```

# Usage
1. Place all files in single folder, extract example data, and run code to process.
2. Settings can be changed on lines 73-101 (see in-line comments for detail).

# Requirements
```
brukeropusreader 1.3.9 (conda install -c spectrocat brukeropusreader)
configparser 5.3.0 (pip install configparser)
nippy 1.0 (pip install git+https://github.com/UEF-BBC/nippy.git)
sklearn 1.0.2+
```

# Repository contents
1. `nir_nmf_v3.py`: main script to perform data import, pre-processing, multivariate analysis, and visualization.
2. `opus_concat_v3.py`: local function (based on `brukeropusreader`) to concatenate all spectra into a single data matrix.
3. `semi_nmf_v1.py`: local function to perform multivariate semi-NMF analysis.
4. `nippy_config.ini`: config file for nippy.
5. `Example data`: folder containing 3960 example spectra and 2 reference spectra in Bruker OPUS file format.
