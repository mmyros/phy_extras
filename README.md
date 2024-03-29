# phy_extras
A collection of phy enhancements building on Allen Institute's spikesorting metrics among others
# Features
1) Trace view filtering and referencing
![trace](videos/trace_toggle.gif)

2) Auto-split via K-means, GAC, or time 
![split](videos/gac_split.gif)

# Installation
1) Install development versions of phy and phylib if you haven't yet:
```bash
# Conda install dependencies 
conda install -y numpy matplotlib scipy h5py dask cython pillow colorcet pyopengl requests qtconsole tqdm joblib click toolz scikit-learn
pip install git+https://github.com/mmyros/ssm.git
```
2) do one of the two things: 
- Copy `dotphy` directory to your home directory and rename it `.phy` 
- OR, on Mac or Linux, make a symbolic link: `ln -s ~/phy_extras/dotphy/  ~/.phy` 
(assuming you downloaded this repository to your home directory)

# Usage
Next time phy is launched, new columns should show up in cluster metrics. 
Also, the following keyboard shortcuts should work:

1. `Alt+R` to cycle through raw data and waveform views: 
raw, highpass, mean+highpass, median+highpass, highpass+median, highpass+mean   
2. `s` for Scrub outliers using Local Outlier Factor 
(also implemented are Robust covariance, One-Class SVM, and Isolation Forest. See `plugins/custom_split.py`)
4. `t` for Time split using gaussian hierarchical model       
1. `x` for k-means split
3. `d` for GAC split (Due to Swindale lab, gradient ascent clustering (GAC) algorithm, a variant of the
mean-shift algorithm)

Finally, the file `cluster_metrics.csv` will appear and contain the calculated cluster quality metrics

`plugins/spike_io.py` can be run standalone from the terminal or Anaconda prompt. It will generate `cluster_metrics.csv`
independently of phy
# Clustering algorithms tested
Tested on Debian Linux with:
- Matlab-based Kilosort 1 and 
- python-based Pykilosort 2.
 
Should also work with Matlab-based Kilosort 2 but untested 

# Warning
Splitting routines are meant to be used as exploratory tools to enhance understanding of clustering. 
I hope you find them useful to refine your core clustering algorithm's parameters

