clc; clear;

addpath('datasets/');

%% Generate 2 Trefoil-Knots
[Y,x,gtruth] = manifoldGen('2trefoils');
N = size(Y,2);

% plot the original data points
figure(1)
colorr = jet(N);
for j = 1:N
    plot3(x(1,j),x(2,j),x(3,j),'o','color',colorr(j,:),'MarkerFaceColor',colorr(j,:),'MarkerSize',9)
    hold on
end
axis equal
title('Trefoil-knots embedded in R^{100}','fontsize',16)
set(gcf,'Renderer','Painters')

%% SMCE
lambda = 10; KMax = 50; n = max(gtruth);
verbose = false;

% run SMCE on the data
[~,clusters,missrate_SMCE] = smce(x,lambda,KMax,n,gtruth,verbose);
acc_SMCE = 1 - missrate_SMCE;

%% LLMC
k = 5;
intrinsic_dim = 1;

W_LLMC = LLMC(x, k, k-1);
[grp_LLMC, missrate_LLMC] = SpectralClustering(0.5 * (abs(W_LLMC) + abs(W_LLMC')), gtruth);
acc_LLMC = 1 - missrate_LLMC;

W_LLMC_pruning = LLMC(x, k, intrinsic_dim);
[grp_LLMC_pruning, missrate_LLMC_pruning] = SpectralClustering(0.5 * (abs(W_LLMC_pruning) + abs(W_LLMC_pruning')), gtruth);
acc_LLMC_pruning = 1 - missrate_LLMC_pruning;

%% Our method: LCR
W_LCR = LCR(x, k, intrinsic_dim);
[grp_LCR, missrate_LCR] = SpectralClustering(0.5 * (W_LCR + W_LCR'), gtruth);
acc_LCR = 1 - missrate_LCR;

% without pruning step: intrinsic_dim = k-1
W_LCR_no_pruning = LCR(x, k, k-1);
[grp_LCR_no_pruning, missrate_LCR_no_pruning] = SpectralClustering(0.5 * (W_LCR_no_pruning + W_LCR_no_pruning'), gtruth);
acc_LCR_no_pruning = 1 - missrate_LCR_no_pruning;

%% Print results
fprintf(1, 'Accuracy result on twoTrefoilknots using SMCE: %.2f\n', acc_SMCE * 100);
fprintf(1, 'Accuracy result on twoTrefoilknots using LLMC: %.2f\n', acc_LLMC * 100);
fprintf(1, 'Accuracy result on twoTrefoilknots using LLMC+d: %.2f\n', acc_LLMC_pruning * 100);
fprintf(1, 'Accuracy result on twoTrefoilknots using LCR: %.2f\n', acc_LCR_no_pruning * 100);
fprintf(1, 'Accuracy result on twoTrefoilknots using LCR+d: %.2f\n', acc_LCR * 100);