function Q_X = Q(X,phi_unwrapped,Ts_invGamma_theta_phi_psi,Delta_squared,f_max,varsigma,v_0,xi)

% compute the approximated field
% phi^T(r)x_t
v_approx = X'*phi_unwrapped;

f = f_max./(1+exp(varsigma*(v_0-v_approx)))*Delta_squared;

% loop through sigma vector
temp = zeros(size(X));
for n=1:size(f,1)
    temp(:,n) = Ts_invGamma_theta_phi_psi*f(n,:)';
end

Q_X = temp + xi*X;