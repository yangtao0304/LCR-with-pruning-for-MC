%--------------------------------------------------------------------------
% This function generates manifols 'sphere' or '2trefoils'
% D = dimension of the ambient space
% sigma = variance of the noise added to the data
% N = number of points in each manifold
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [Yn,Y,gtruth,x] = manifoldGen(manifoldType)


D = 100; % ambient space dimension
sigma = 0.001; % noise variance

if (strcmp(manifoldType,'2trefoils'))
    N = 100;
    gtruth = [1*ones(1,N) 2*ones(1,N)];
elseif (strcmp(manifoldType,'sphere'))
    N = 1000;
    gtruth = ones(1,N);
else
    error('Unknown Mnifold Type!')
end

if (strcmp(manifoldType,'2trefoils'))
        % Generate two trefoil-knots
        d = 3; Par = 3.8;
        t = linspace(0,2*pi,N+1);
        t(end) = [];
        Yg{1}(1,:) = (2+cos(3*t)).*cos(2*t);
        Yg{1}(2,:) = (2+cos(3*t)).*sin(2*t);
        Yg{1}(3,:) = sin(3*t);
        Yg{2}(1,:) = (2+cos(3*t)).*cos(2*t) + Par;
        Yg{2}(2,:) = (2+cos(3*t)).*sin(2*t);
        Yg{2}(3,:) = sin(3*t);
        Y = [Yg{1} Yg{2}];
        U = orth(randn(D,d));
        Yn = U * Y;
        Yn = Yn + sigma * randn(size(Yn));
        x = [cos(t) cos(t);sin(t) sin(t)];
        
elseif (strcmp(manifoldType,'sphere'))
        % Generate a random sphere
        d = 3;
        r = 2 * ( sort(rand(1,N)).^.99);
        theta = linspace(0,2*pi,N+1);
        theta(end) = [];
        p = randperm(N);
        P = zeros(N,N);
        for i = 1:N
            P(p(i),i) = 1;
        end
        theta = theta * P;
        xx = r .* cos(theta);
        yy = r .* sin(theta);        
        Y = [2*xx./(1+xx.^2+yy.^2);2*yy./(1+xx.^2+yy.^2);(-1+xx.^2+yy.^2)./(1+xx.^2+yy.^2)];
        U = orth(randn(D,d));
        Yn = U * Y;
        Yn = Yn + sigma * randn(size(Yn));
        x = [xx;yy];
end
