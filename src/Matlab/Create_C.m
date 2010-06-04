% create C matrix for getting observations in the reduced model


% define analytic inner product of sensors and basis functions
a = pi / ( (sigma_phi^2+sigma_y^2)/(sigma_phi^2*sigma_y^2) );
b = 1/(sigma_phi^2+sigma_y^2);

% now define the observation matrix
C = zeros(NSensors_xy^2,NBasisFunctions_xy^2 );
for n=1:NSensors_xy^2
    for nn=1:NBasisFunctions_xy^2
        mu_phi_minus_mu_y = mu_y(:,n)-mu_phi(:,nn);
        C(n,nn) = a*exp(-b*(mu_phi_minus_mu_y)'*mu_phi_minus_mu_y);
    end
end