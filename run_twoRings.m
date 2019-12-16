clc; clear;

addpath('datasets/', 'tools/', 'SMCE/');

%% Load dataset
load('twoRings.mat');
N = size(X,2);

% Param k
k = 5;

%% LLMC
W_LLMC = LLMC(X, k, k-1);
[grp_LLMC, missrate_LLMC] = SpectralClustering(0.5 * (abs(W_LLMC) + abs(W_LLMC')), gtruth);
acc_LLMC = 1 - missrate_LLMC;

%% SMCE
verbose = false;
lambda = 10;
KMax = 5;
n = max(gtruth);

% run SMCE on the data
[W_SMCE, clusters,missrate_SMCE] = smce(X,lambda,KMax,n,gtruth,verbose);
acc_SMCE = 1 - missrate_SMCE;

%% Our method: LCR
W_LCR = LCR(X, k, k-1);
[grp_LCR, missrate_LCR] = SpectralClustering(0.5 * (W_LCR + W_LCR'), gtruth);
acc_LCR = 1 - missrate_LCR;

%% Print results
fprintf(1, 'Misclassified points on twoRings using LLMC: %d\n', missrate_LLMC * N);
fprintf(1, 'Misclassified points on twoRings using SMCE: %d\n', missrate_SMCE * N);
fprintf(1, 'Misclassified points on twoRings using LCR: %d\n\n', missrate_LCR * N);

idx = [54, 56, 126, 127, 128];
disp("Point 55's COEFFICIENTS using LLMC:");
disp(W_LLMC(idx, 55));
disp("Point 55's COEFFICIENTS using SMCE:");
disp(W_SMCE(idx, 55));
disp("Point 55's COEFFICIENTS using LCR:");
disp(W_LCR(idx, 55));