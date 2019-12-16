clc; clear;

addpath('datasets/', 'tools/', 'SMCE/');

%% Load dataset
load('UMST_FaceDataset_32x32.mat');
X = im2double(X);
gtruth = Label(:);

% Best param k for LLMC and LCR
k_LLMC = 12;
k_LCR = 18;

%% Baseline1: LLMC
W_LLMC = LLMC(X, k_LLMC, k_LLMC-1);
[grp_LLMC, missrate_LLMC] = SpectralClustering(0.5 * (abs(W_LLMC) + abs(W_LLMC')), gtruth);
acc_LLMC = 1 - missrate_LLMC;

%% Baseline2: SMCE
% verbose = true if want to see the sparse optimization information
verbose = false;

% set the parameters of the SMCE algorithm
lambda = 10;
KMax = 14;
n = max(gtruth);

% run SMCE on the data
[~,clusters,missrate_SMCE] = smce(X,lambda,KMax,n,gtruth,verbose);
acc_SMCE = 1 - missrate_SMCE;

%% Our method: LCR+d
% Estimate intrinsic dim
intrinsic_dim = id_estimate(X, 20, 0.95);

W_LCR = LCR(X, k_LCR, intrinsic_dim);
[grp_LCR, missrate_LCR] = SpectralClustering(0.5 * (W_LCR + W_LCR'), gtruth);
acc_LCR = 1 - missrate_LCR;

%% Print results
fprintf(1, 'Accuracy result on UMIST using LLMC: %.2f\n', acc_LLMC * 100);
fprintf(1, 'Accuracy result on UMIST using SMCE: %.2f\n', acc_SMCE * 100);
fprintf(1, 'Accuracy result on UMIST using LCR+d: %.2f\n', acc_LCR * 100);