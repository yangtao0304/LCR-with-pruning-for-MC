function [W] = LLMC(X, k, intrinsic_dim)

[D,N] = size(X);
fprintf(1,'LLMC running on %d points in %d dimensions\n', N, D);

% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
fprintf(1,'-->Finding %d nearest neighbours.\n',k);

X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*(X'*X);

[~,index] = sort(distance);
neighborhood = index(2:(1+k),:);

% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
fprintf(1,'-->Solving for reconstruction weights.\n');

if(k>D)
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

w = zeros(k,N);
for i=1:N
   z = X(:,neighborhood(:,i))-repmat(X(:,i),1,k);   % shift ith pt to origin
   C = z'*z;                                        % local covariance
   C = C + eye(k,k)*tol*trace(C);                   % regularlization (K>D)
   w(:,i) = C\ones(k,1);                            % solve Cw=1
   w(:,i) = w(:,i)/sum(w(:,i));                     % enforce sum(w)=1
end

W = zeros(N,N);
% for ii=1:N
%     for t=1:k
%         W(neighborhood(t,ii),ii) = w(t,ii);
%     end
% end

[~, ind] = sort(W,'descend');

for ii=1:N
    t =ind(1:intrinsic_dim+1,ii);
    W(neighborhood(t,ii),ii) = w(t,ii);
end

% other possible regularizers for K>D
%   C = C + tol*diag(diag(C));                       % regularlization
%   C = C + eye(K,K)*tol*trace(C)*K;                 % regularlization
