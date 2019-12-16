function [W] = LCR(X, k, intrinsic_dim)

% Inputs:
% X = shape of [data point dimension, the number of the points]
% k = the number of nearest neighbors to set
% intrinsic_dim = estimated intrinsic dimension
%
% Output:
% W = nonnegative constraint based affinity weight

[d,n] = size(X);
fprintf(1,'Running on %d points in %d dimensions\n', n, d);

% Find k neighbors
fprintf(1,'-->Finding %d nearest neighbours.\n', k);

X2 = sum(X.^2,1);

D = repmat(X2,n,1) + repmat(X2',1,n) -2*(X'*X);
[~,index] = sort(D);
neighborhood = index(2:(1+k),:);

% Compute the affinity matrix for LCR
fprintf(1,'-->Solving for reconstruction weights.\n');

% Based on Lagrange Multiplier method for Equality Constrained LS Problem:
w = zeros(k, n);
Aeq =ones(1,k);
beq =1;
lb =zeros(k,1);
ub =[];
for ii=1:n
    C =X(:,neighborhood(:,ii));
    b =X(:,ii);
    w(:,ii)= lsqlin(C,b,[],[],Aeq,beq,lb,ub);
end

% Pruning step by keeping only the leading 'intrinsic_dim + 1' coefficients
[~, ind] = sort(w,'descend');

W =zeros(n, n);

for ii=1:n
    t =ind(1:intrinsic_dim+1,ii);
    W(neighborhood(t,ii),ii) = w(t,ii);
end
