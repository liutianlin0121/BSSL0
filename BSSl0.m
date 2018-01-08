function x = BSSl0(A, y, p0, sigma_min, sigma_decrease_factor, mu, L)

% For given A x = y, we reconstruct binary signal x from given A and y using boxed re-weigthed smoothed l0 optimization.

% input data:
% (1) A is the sensing matrix.
% (2) y is the observation signal.
% (3) p0 is the prob that x_i being 0.

% input parameters:
% (1) sigma_min is a small positive number near 0 that sigma converses to.
% (2) L is the inner loop iteration number
% (3) mu is the gradient descent factor

% (c) Tianlin Liu 2016



% Initialization
A_pinv = pinv(A);
[~,N] = size(A);
x = A_pinv*y;
K = (1-p0)*N;

sigma = 2*max(abs(x));

M = log(sigma_min/sigma) / log(sigma_decrease_factor);

iter = 0;
% Main Loop
while sigma > sigma_min
    k = 1 + iter / M * K;
    iter = iter + 1;
    
    for i=1:L        
        delta = ConstrainedDelta(x,sigma,k,K)/k;
        x = x - mu*delta;
        x = x - A_pinv*(A*x-y);   % Projection
            
    end
    
    sigma = sigma * sigma_decrease_factor;
end


    
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function delta = ConstrainedDelta(s,sigma,k,K)
[N,~] = size(s);
w = ((k+1)/2 - ((k-1)/2)*sign(s).*sign(1-s));

if w == (k+1)/2,
    w = 1;
end

delta = (1 - K/N).*s.*(exp((-abs(s).^2)/(2*sigma^2))).*w + (K/N)*(s-1).*(exp((-abs(s-1).^2)/(2*sigma^2))).*w;