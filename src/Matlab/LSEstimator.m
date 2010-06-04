function [theta_est xi_est] = LSEstimator(x,phi_unwrapped,Delta,varsigma,f_max,v_0,Ts_invGamma_phi_psi)

if size(x,1) > size(x,2)
    x = x';
end

L = size(x,1);                      % size of state vector
T = size(x,2);                      % number of time points
Delta_squared = Delta^2;    % initialise


for t=1:T       % for all time points
    
    % find v_approx = phi^T(r)x_t
    v_approx = phi_unwrapped'*x(:,t);
    
    % find the firing rate using sigmoid function
    f = f_max./(1+exp(varsigma*(v_0-v_approx)))*Delta^2;
    
    % find inner product with Psi
    for n=1:size(Ts_invGamma_phi_psi,1)
        q(:,t,n) = squeeze(Ts_invGamma_phi_psi(n,:,:))*f;
    end
end

% form the block matrices to find LS solution
X = [];
Z = [];
for t=1:T-1
    X = [X ; [squeeze(q(:,t,:)) squeeze(x(:,t))]];
    Z = [Z ; squeeze(x(:,t+1))]; 
end

% get parameter estimate
parameters_est = (X'*X)\X'*Z;
theta_est(1) = parameters_est(1);
theta_est(2) = parameters_est(2);
theta_est(3) = parameters_est(3);
xi_est = parameters_est(4);
% zeta_est = -(parameters_est(4)-1)/Ts
