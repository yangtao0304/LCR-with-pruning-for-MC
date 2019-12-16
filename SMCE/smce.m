%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [W, clusters,missrate] = smce(Y,lambda,KMax,n,gtruth,verbose)

% solve the sparse optimization program
W = smce_optimization(Y,lambda,KMax,verbose);
W = processC(W,0.95);

% symmetrize the adjacency matrices
Wsym = max(abs(W),abs(W)');

% perform clustering
[clusters,missrate] = smce_clustering(Wsym,n,gtruth);