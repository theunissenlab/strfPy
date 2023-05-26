Python Code for STRF Estimation
-------------------------------

This code base is a python version of the direct fit code of STRFLab.
This code estimates STRFs by performing ridge regression in the Fourier Domain for the temporal dimension.
Ridge regression in the Fourier domain is faster and more robust than Ridge in both spatial and temporal dimension since
the fourier componnents are the PCs is temporal statistics are stationary.

The algorithm also performs a modified version of lasso regression (thus elastic net) also for optimizing speed.



