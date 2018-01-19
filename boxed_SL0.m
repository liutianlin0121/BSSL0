function s= boxed_SL0(A, x, K, sigma_min, sigma_decrease_factor, mu, L, true_s)

if nargin == 7
    ShowProgress = false;
elseif nargin == 8
    ShowProgress = true;
else
    error('Error in calling boxed_SL0 function');
end

% Initialization
A_pinv = pinv(A);
s = A_pinv*x;
sigma = 2*max(abs(s));
epsilon = 1e-5;

Iter = log(sigma_min/sigma) / log(sigma_decrease_factor);

j = 0;
% Main Loop
while sigma > sigma_min
    k = 1 + j / Iter * K;
    j = j + 1;
    for i=1:L        
        delta = ConstrainedDelta(s,sigma,k)/k;
        s = s - mu*delta;
        
        if(norm(A*s-x, 2) > epsilon)
            s = s - A_pinv*(A*s-x);   % Projection
        end
            
    end
    
    if ShowProgress
        fprintf('     sigma = %f, SNR = %f\n',sigma,estimate_SNR(s,true_s))
    end
    
    sigma = sigma * sigma_decrease_factor;
end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function delta = ConstrainedDelta(s,sigma,k)
epsilon = 1e-5;
delta = s.*exp(-abs(s).^2/sigma^2) .* ((k+1)/2 - ((k-1)/2)*sign(s+epsilon).*sign(1+epsilon-s));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SNR=estimate_SNR(estim_s,true_s)

err = true_s - estim_s;
SNR = 10*log10(sum(abs(true_s).^2)/sum(abs(err).^2));