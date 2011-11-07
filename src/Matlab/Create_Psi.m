

% now form the matrix
% these are the coefficients for the analytic convolution of psi and phi
psi_phi_coefficient(1) = pi*sigma_psi(1)^2*sigma_phi^2 / (sigma_psi(1)^2+sigma_phi^2);
psi_phi_coefficient(2) = pi*sigma_psi(2)^2*sigma_phi^2 / (sigma_psi(2)^2+sigma_phi^2);
psi_phi_coefficient(3) = pi*sigma_psi(3)^2*sigma_phi^2 / (sigma_psi(3)^2+sigma_phi^2);

% compute the convolution between phi and psi
for n=1:L 
    for nn=1:length(theta)
        % these guys here are used with the LS algorithm for estimating
        % theta and xi
        psi_phi = psi_phi_coefficient(nn)*...
            Define2DGaussian(mu_phi(1,n), mu_phi(2,n), sigma_psi(nn)^2+sigma_phi^2, 0,NPoints,SpaceMin,SpaceMax);

        psi_phi_basis(nn,n,:) = psi_phi(:);  
        theta_psi_phi_basis(nn,n,:) = theta(nn)*psi_phi_basis(nn,n,:);
        
    end
end

Ts_invGamma_theta_phi_psi = Ts*(Gamma\squeeze(theta_psi_phi_basis(1,:,:) ...
    + theta_psi_phi_basis(2,:,:) ...
    + theta_psi_phi_basis(3,:,:)));

Ts_invGamma_phi_psi(1,:,:) = Ts*(Gamma\squeeze(psi_phi_basis(1,:,:)));
Ts_invGamma_phi_psi(2,:,:) = Ts*(Gamma\squeeze(psi_phi_basis(2,:,:)));
Ts_invGamma_phi_psi(3,:,:) = Ts*(Gamma\squeeze(psi_phi_basis(3,:,:)));
