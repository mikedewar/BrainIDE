% define gamma analytically

% Gamma = [phi_1 x phi_1, phi_1 x phi_2, ..., phi_1 x phi_n
%                   phi_2 x phi_1, ...                      , phi_2 x phi_n
%                       ...
%                   phi_n x phi_1, ...                      , phi_n x phi_n]
% where x is inner product
a = pi*sigma_phi^2/2;
b = 1/(2*sigma_phi^2);

Gamma = zeros(L,L);       % initialize for speed
for n=1:L
    for nn=1:L
        mu_n_minus_mu_m = mu_phi(:,n)-mu_phi(:,nn);
        Gamma(n,nn) = a*exp(-b*(mu_n_minus_mu_m)'*mu_n_minus_mu_m);
    end
end
inv_Gamma = inv(Gamma);