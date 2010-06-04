% Create phi, the basis functions for the decomposition

% first get them in 2D format
phi = zeros(L,NPoints,NPoints);
for n=1:L
    phi(n,:,:) = Define2DGaussian(mu_phi(1,n), mu_phi(2,n), sigma_phi^2, 0,NPoints,SpaceMin,SpaceMax);
%     imagesc(r,r,squeeze(phi(n,:,:)))
%     axis xy
%     drawnow
end

% now in vector format
phi_unwrapped = zeros(L,NPoints^2);
for n=1:L
    temp = Define2DGaussian(mu_phi(1,n), mu_phi(2,n), sigma_phi^2, 0,NPoints,SpaceMin,SpaceMax);
    phi_unwrapped(n,:) = temp(:);
end

% Check against Parham's value
% ~~~~~~~~~~~~~~~~~~
% load Phi_values
% figure
% imagesc(Phi_values - phi_unwrapped)
% colorbar