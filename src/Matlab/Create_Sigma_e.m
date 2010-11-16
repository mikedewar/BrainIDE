% create Sigma_e
% this is the covariance matrix of the reduced model

% find the convolution of phi and gamma
phi_gamma_coefficient = (pi*sigma_gamma^2*sigma_phi^2) / (sigma_gamma^2+sigma_phi^2);
phi_gamma_conv_var = sigma_gamma^2+sigma_phi^2;

% find the inner product of the resulting convolution with phi
a = pi / ( (phi_gamma_conv_var + sigma_phi^2) / (phi_gamma_conv_var * sigma_phi^2) );
b = 1 / (sigma_phi^2+phi_gamma_conv_var);
inner_prod = zeros(L,L);       % initialize for speed
for n=1:L
    for nn=1:L
        mu_n_minus_mu_m = mu_phi(:,n)-mu_phi(:,nn);
        inner_prod(n,nn) = gamma_weight*phi_gamma_coefficient*a*exp(-b*(mu_n_minus_mu_m)'*mu_n_minus_mu_m);
    end
end

Sigma_e = inv_Gamma*inner_prod*inv_Gamma';

