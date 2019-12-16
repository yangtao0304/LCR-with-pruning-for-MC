function [intrinsic_dim] = id_estimate(X, k, ratio)
[~,n] = size(X);

fprintf(1,'-->Finding %d nearest neighbours.\n', k);
X2 = sum(X.^2,1);

% Compute the affinity matrix for the extrinsic neighborhood graph
D = repmat(X2,n,1) + repmat(X2',1,n) -2*(X'*X);
[~,index] = sort(D);
neighborhood = index(2:(1+k),:);

local_cov_eigens = [];
for i = 1:n
    C = X(:,neighborhood(:,i));
    [~,S1,~] = svd(C'*C);
    local_cov_eigens = [local_cov_eigens diag(S1)];
end

local_cov_eigens_mean = mean(local_cov_eigens, 2);

sum_total = sum(local_cov_eigens_mean);
for intrinsic_dim = 1:k
    cur_ratio = sum(local_cov_eigens_mean(1:intrinsic_dim)) / sum_total;
    if cur_ratio > ratio
        break
    end
end