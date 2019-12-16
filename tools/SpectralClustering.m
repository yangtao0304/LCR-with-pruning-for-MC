function [grp,missrate] = SpectralClustering(W,gtruth)
% Inputs:
% W = symmetric affinity weight
% gtruth = ground-truth vector of memberships of points to the n manifolds
% Outputs:
% grp = clustering results
% missrate = clustering error rate corresponding to gtruth

% kmeans parameter settings
MAXiter = 1000;
REPlic = 20;

N = size(W,1);
n = max(gtruth);

% cluster the data using the normalized symmetric Laplacian 
D = diag( 1./sqrt(sum(W,1)+eps) );
L = eye(N) - D * W * D;
[~,~,V] = svd(L,'econ');
Yn = V(:,end:-1:end-n+1);
for i = 1:N
    Yn(i,:) = Yn(i,1:n) ./ norm(Yn(i,1:n)+eps);
end

% compute the misclassification rate
if n > 1
    grp = kmeans(Yn(:,1:n),n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    grp = bestMap(gtruth,grp);
    missrate = sum(gtruth(:) ~= grp(:)) / length(gtruth);
else 
    grp = ones(1,N);
    missrate = 0;
end
