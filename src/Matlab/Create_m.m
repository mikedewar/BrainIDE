% create the sensor kernel m


% first get them in 2D format
m = zeros(NSensors,NPoints*3,NPoints*3);
for n=1:NSensors
    m(n,:,:) = Define2DGaussian(mu_m(1,n), mu_m(2,n), sigma_m^2, 0, NPoints*3, 3*SpaceMin, 3*SpaceMax);
%     imagesc(r,r,squeeze(phi(n,:,:)))
%     axis xy
%     drawnow
end

% for n=1:NSensors
%     imagesc(squeeze(m(n,:,:)))
%     drawnow
% end

% now in vector format
m_unwrapped = zeros(NSensors,NPoints^2);
for n=1:NSensors
    temp = Define2DGaussian(mu_m(1,n), mu_m(2,n), sigma_m^2, 0,NPoints,SpaceMin,SpaceMax);
    m_unwrapped(n,:) = temp(:);
end
