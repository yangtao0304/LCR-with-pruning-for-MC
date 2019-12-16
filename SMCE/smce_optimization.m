%--------------------------------------------------------------------------
% This function solves the optimization function of SMCE for the given
% data points
% X: DxN matrix of N data points in the D-dimensional space
% lambda: regularization parameter of the SMCE optimization program
% KMax = maximum neighborhood size to select the sparse neighbors from
% verbose: ture if want to see the optimization information, else false
% W: NxN sparse matrix of weights obtained by the SMCE algorithm 
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function W = smce_optimization(X,lambda,KMax,verbose)

N = size(X,2);

if (nargin < 2)
    lambda = 10;
end
if (nargin < 3)
    KMax = N - 1;
end

if (KMax > N-1) || (KMax < 1)
    KMax = N-1;
end

X2 = sum(X.^2,1);
Dist = sqrt( repmat(X2,N,1)+repmat(X2',1,N)-2*(X'*X) );

W = zeros(N,N); % weight matrix used for clustering and dim. reduction

% solving the SMCE optimization program for data points
for i = 1:N
    [~,ids] = sort(Dist(:,i),'ascend');
    ids = ids(1:KMax);
    
    y = X(:,ids(1));
    Y = X(:,ids);
    Y(:,1) = [];
    Y = Y - repmat(y,1,KMax-1);
    v = Dist(ids,i);
    v(1) = [];
    for j = 1:KMax-1
        Y(:,j) = Y(:,j) ./ v(j);
    end
    
    if (verbose)
        fprintf('Point %4.0f, ',i);
    end
    % solving the Lasso-type optimization using ADMM framework
    c = admm_vec_func(Y,v./sum(v),lambda,verbose);
    
    W(ids(2:KMax),i) = abs(c./v) / sum(abs(c./v));
end
