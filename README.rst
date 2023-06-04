Python Code for STRF and MRF Estimation
-------------------------------

For STRF Estimation
This code base is a python version of the direct fit code of STRFLab.
This code estimates STRFs by performing ridge regression in the Fourier Domain for the temporal dimension.
Ridge regression in the Fourier domain is faster and more robust than Ridge in both spatial and temporal dimension since
the fourier componnents are the PCs is temporal statistics are stationary.

The algorithm also performs a modified version of lasso regression (thus elastic net) also for optimizing speed.

For MRF Estimation
The modulation receptive field is based on a segmentation of the sound stimulus into chunks of variable sizes.
Each chunk is characterized by its modulation power spectrum.  Cross-validate ridge regression is then used to predict the response in the chunks as described by the coefficients of a PCA.

