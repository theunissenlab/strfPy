% This exercise uses fake data from a natural movie to pracice obtaining an
% STRF using the least mean square analytical solution
%% preliminary stuff: get the directory we're in
% and add the proper subdirectories to the path
cpath = which('directfit_tutorial');
[rootDir, name, ext] = fileparts(cpath);
spath = fullfile(rootDir, 'strflab');
addpath(genpath(spath));
dfpath = fullfile(rootDir, 'direct_fit');
addpath(dfpath);
vpath = fullfile(rootDir, 'validation');
addpath(vpath);
ppath = fullfile(rootDir, 'preprocessing');
addpath(ppath);
dataDir = fullfile(rootDir, '../../', 'data'); %contains stim/response pairs
stimsDir = fullfile(dataDir, 'all_stims'); %contains the .wav files

%% load a 20x20x20000 natural movie
load (strcat(dataDir,'/mov.mat'));
% let's only take a part of it 
tlength = 15000;
rawStim = single(mov(1:10,1:10,1:tlength));  % single converts to single precision
rawStim = rawStim - mean(mean(mean(rawStim))); % Subtract mean.

%% Exercise 1. Using subplot plot the first 10 images.  Then plot images
% 10, 100, 1000, 10000.  Using these pictures comment on 
% the temporal and spatial correlations.

figure(1);
for i=1:10
    subplot(1, 10, i);
    imagesc(rawStim(:,:,i));
    axis square;
end


figure(2);
for i=1:4
    subplot(1, 4, i);
    imagesc(rawStim(:,:,i*10))
    axis square;
end
%% let's create some fake data for a simple cell V1 with a 2D Gabor filter
% First we are going to make a Gabor filter
gparams = [.5 .5 0 5.5 0 .0909 .3 0]';
[gabor, gabor90] = make3dgabor([10 10 1], gparams); 

%% Exercise 2. Plot the Gabor filter.
figure(3);
subplot(2, 1, 1);
imagesc(gabor);
axis square;
subplot(2, 1, 2);
imagesc(gabor90);
axis square;

%% Convolve the Gabor filter with the stimulus 
% and add Gaussian noise to get a response with an SNR of 0.5

SNR = 0.5;
gabor_vector = reshape(gabor, [10*10 1]);
rawStim_vector = reshape(rawStim, [10*10 tlength]);
resp = dotdelay(gabor_vector, rawStim_vector);
resp_pow = var(resp);
resp = resp + sqrt(resp_pow/SNR)*randn(tlength,1);


%% Exercise 3.  Recover the filter using the normal equation.  
% First cross-correlate the response and the stimulus.  In this case there
% is no time component. This is also the spike-triggered average or STA
cross_stim_response = (rawStim_vector*resp)./tlength; 

% Plot the STA and compare to the filter
figure(4);
subplot(2, 1, 1);
imagesc(reshape(cross_stim_response, [10 10]));
axis square;

subplot(2, 1, 2);
imagesc(gabor);
axis square;

%% Now calculate the stimulus auto-correlation and image it. Explain what you see?
mean_auto_corr = zeros(100, 100);
for it=1:tlength;
        auto_corr = rawStim_vector(:,it)*rawStim_vector(:,it)';
        mean_auto_corr = mean_auto_corr + auto_corr;
end
mean_auto_corr = mean_auto_corr./tlength;

figure(5);
imagesc(mean_auto_corr);
colormap('gray');
title('Auto Correlation');

%% Now normalize the cross-correlation by the auto-correlation to recover the filter
% Try first using the matrix division operator \
recovfiler = mean_auto_corr\cross_stim_response;
figure(6);
subplot(2,1,1);
imagesc(reshape(recovfiler,[10 10]));
colormap('gray');
title('Filter with full normalization');
axis image;
subplot(2,1,2);
imagesc(gabor);
colormap('gray');
title('Gabor Filter');
axis image;

%% Excercise 4. Now we are going to try regularizing.
% Find the solution using PCA regression also called subspace
% regression.  Also display the eigenvectors of the stimulus
% auto-correlation.
[u s v_svd] = svd(mean_auto_corr);
figure(7);
plot(diag(s));

% Let's try dividing in supspace of 10, 20, 50
sinv_10 = zeros(size(s));
for i=1:10;
    sinv_10(i,i) = 1./s(i,i);
end
myfilter_10 = v_svd*sinv_10*(u'*cross_stim_response);

sinv_20 = zeros(size(s));
for i=1:20;
    sinv_20(i,i) = 1./s(i,i);
end
myfilter_20 = v_svd*sinv_20*(u'*cross_stim_response);

sinv_50 = zeros(size(s));
for i=1:50;
    sinv_50(i,i) = 1./s(i,i);
end
myfilter_50 = v_svd*sinv_50*(u'*cross_stim_response);

figure(8);
subplot(1,3,1);
imagesc(reshape(myfilter_10, [10 10]));
colormap('gray');
title('10D normalization');
axis image;

subplot(1,3,2);
imagesc(reshape(myfilter_20, [10 10]));
colormap('gray');
title('20D normalization');
axis image;

subplot(1,3,3);
imagesc(reshape(myfilter_50, [10 10]));
colormap('gray');
title('50D normalization');
axis image;



%% Repeat the previous section using eigen value decompostion. The eig command in matlab
% does not necessarily sort the eigenvalues - use sort() to get then in
% descending order.

[v_eigen d] = eig(mean_auto_corr);
[ds, idx] = sort(diag(d), 'descend');
n_size = length(ds);
v_eigen_sorted = zeros(n_size, n_size); % Pre-allocate for speed
for i=1:n_size
    v_eigen_sorted(:,i) = v_eigen(:,idx(i));
end
d = diag(ds);   % Makes a new diagonal matrix


figure(9);
plot(diag(d));

% Let's try dividing in supspace of 10, 20, 50
dinv_10 = zeros(size(d));
for i=1:10;
    dinv_10(i,i) = 1./d(i,i);
end
myfilter_10 = v_eigen_sorted*dinv_10*(v_eigen_sorted'*cross_stim_response);

dinv_20 = zeros(size(d));
for i=1:20;
    dinv_20(i,i) = 1./d(i,i);
end
myfilter_20 = v_eigen_sorted*dinv_20*(v_eigen_sorted'*cross_stim_response);

dinv_50 = zeros(size(s));
for i=1:50;
    dinv_50(i,i) = 1./s(i,i);
end
myfilter_50 = v_eigen_sorted*dinv_50*(v_eigen_sorted'*cross_stim_response);

figure(10);
subplot(1,3,1);
imagesc(reshape(myfilter_10, [10 10]));
colormap('gray');
title('10D normalization');
axis image;

subplot(1,3,2);
imagesc(reshape(myfilter_20, [10 10]));
colormap('gray');
title('20D normalization');
axis image;

subplot(1,3,3);
imagesc(reshape(myfilter_50, [10 10]));
colormap('gray');
title('50D normalization');
axis image;




%% Excercise 5. Ridge Solution.
% Find the solution using ridge regression. Use values of lambda of 0, 100,
% 1000, 10000.  Hint. Use the code you wrote for PCA regression.



