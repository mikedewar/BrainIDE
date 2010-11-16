% set all the parameters for the model and the estimator

% if we are running a batch than we dont want to clear things
if exist('RunningBatch','var') == 0
%     clear
    close all
%     clc
end

% spatial parameters
% ~~~~~~~~~~~
Delta = 0.5;                          % space step for the spatial discretisation
Delta_squared = Delta^2;
SpaceMax = 10;                    % maximum space in mm
SpaceMin = -SpaceMax;         % minimum space in mm
NPoints = (SpaceMax-SpaceMin)/Delta+1;
r = linspace(SpaceMin,SpaceMax,NPoints);      % define space

EstimationSpaceMax = 10;
EstimationSpaceMin = -10;

% temporal parameters
% ~~~~~~~~~~~~~
Ts = 1e-3;          % sampling period (s)
T = 1000;            % maximum time (ms)

% kernel parameters
% ~~~~~~~~~~~
theta(1) = 100.0;           % local kernel amplitude
theta(2) = -80;             % surround kernel amplitude
theta(3) = 5;               % lateral kernel amplitude

sigma_psi(1) = 1.8;     % local kernel width
sigma_psi(2) = 2.4;     % surround kernel width
sigma_psi(3) = 6;       % lateral kernel width

psi_0 = Define2DGaussian(0,0, sigma_psi(1)^2, 0,NPoints,SpaceMin,SpaceMax);
psi_1 = Define2DGaussian(0,0, sigma_psi(2)^2, 0,NPoints,SpaceMin,SpaceMax);
psi_2 = Define2DGaussian(0,0, sigma_psi(3)^2, 0,NPoints,SpaceMin,SpaceMax);
w = theta(1)*psi_0 + theta(2)*psi_1 + theta(3)*psi_2;       % the kernel

W = fft2(w);                                                                       % the fft of the kernel
Ts_W = W*Ts;                        % FFT of kernel times the time step
% ~~~~~~~~~~~~~~~~~~~~~~

% ~~~~~~~~~~~~~~~~~~~~~~
% triangle field basis function parameters
% ~~~~~~~~~~~~~~~~~~~
% phi_x_spacing = 2.5;
% angle = 60;
% phi_y_spacing = phi_x_spacing*sin((angle/360)*2*pi);
% 
% mu_phi_y = EstimationSpaceMin:phi_y_spacing:EstimationSpaceMax;
% mu_phi_y = mu_phi_y + (EstimationSpaceMax - mu_phi_y(end))/2;
% 
% mu_phi_x1 = EstimationSpaceMin:phi_x_spacing:EstimationSpaceMax;
% mu_phi_x1 = mu_phi_x1 + (EstimationSpaceMax - mu_phi_x1(end))/2;
% mu_phi_x2 = mu_phi_x1(1:end-1) + phi_x_spacing/2;
% 
% sigma_phi = sqrt(2.5);
% 
% m = 1;
% for n=1:2:length(mu_phi_y)      % start with the first row
%     for nn = 1:length(mu_phi_x1)
%         mu_phi(:,m) = [mu_phi_x1(nn) ; mu_phi_y(n)];
%         m=m+1;
%     end
% end
% for n=2:2:length(mu_phi_y)      % start with the second row
%     for nn = 1:length(mu_phi_x2)
%         mu_phi(:,m) = [mu_phi_x2(nn) ; mu_phi_y(n)];
%         m=m+1;
%     end
% end
% L = m-1;
% ~~~~~~~~~~~~~~~~~~~~~~


% square basis function arrangement
% ~~~~~~~~~~~~~~~~~~~~~
NBasisFunctions_xy = 9;
L = NBasisFunctions_xy^2;                   % number of states and the number of basis functions
mu_phi_xy = linspace(-EstimationSpaceMax,EstimationSpaceMax,NBasisFunctions_xy);
sigma_phi = sqrt(2.5);

mu_phi = zeros(2,L);     % initialize for speed
mm=1;
for n=1:NBasisFunctions_xy
    for nn=1:NBasisFunctions_xy
        mu_phi(:,mm) = [mu_phi_xy(n) ; mu_phi_xy(nn)];
        mm=mm+1;
    end
end
% ~~~~~~~~~~~~~
% ~~~~~~~~~~~~~

% sensor parameters
% ~~~~~~~~~~~~
NSensors_xy = 14;%21;%14;
NSensors = NSensors_xy^2;
mu_m_xy = linspace(EstimationSpaceMin+0.25,EstimationSpaceMax-0.25,NSensors_xy);               % sensor centers

% sensor_indexes = (mu_y_xy+EstimationSpaceMax)/Delta +1;             % so we can easily get the observations from the filed filtered by the sensors
sigma_m = 0.9;                                                         % sensor width
% m = Define2DGaussian(0,0, sigma_m^2, 0,NPoints,SpaceMin,SpaceMax);
% M = fft2(m);                                            % fft of sensor kernel to get observations quickly
% define all sensor centers
mu_m = zeros(2,NSensors);     % initialize for speed
mm=1;
for n=1:NSensors_xy
    for nn=1:NSensors_xy
        mu_m(:,mm) = [mu_m_xy(n) ; mu_m_xy(nn)];
        mm=mm+1;
    end
end
Create_m


% observation noise characteristics
% ~~~~~~~~~~~~~~~~~~~~
sigma_varepsilon = 0.1;                                  
Sigma_varepsilon = sigma_varepsilon*eye(NSensors);        % observation covariance matrix

% sigmoid parameters
% ~~~~~~~~~~~~
f_max = 1;             % maximum firing rate
varsigma = 0.56;         % sigmoid slope
v_0 = 1.8;                    % firing threshold

% synaptic kernel parameter
% ~~~~~~~~~~~~~~~~
tau = 0.01;                   % s
zeta = 1/tau;                 % 
xi = 1-Ts*zeta;           % coefficient for the discrete time model

% disturbance paramters
% ~~~~~~~~~~~~~
sigma_gamma = 1.3;          % parameter for covariance of disturbance
gamma_weight = 0.1;            % variance of disturbance
SphericalBoundary

% m=1;
% Sigma_gamma = zeros(NPoints^2,NPoints^2);   % create disturbance covariance matrix
% for n=1:NPoints
%     for nn=1:NPoints
%         temp = gamma_weight*Define2DGaussian(r(n),r(nn), sigma_gamma^2, 0,NPoints,SpaceMin,SpaceMax);
%         Sigma_gamma(:,m) = temp(:);
%         m=m+1;
%     end
% end
    
% plot the true kernel for comparison
% ~~~~~~~~~~~~~~~~~~~~~
k0 = theta(1)*exp(-sigma_psi(1)^-2 *(r.*r));
k1 = theta(2)*exp(-sigma_psi(2)^-2 *(r.*r));
k2 = theta(3)*exp(-sigma_psi(3)^-2 *(r.*r));

r_continuous = linspace(SpaceMin,SpaceMax,10*NPoints);      % define space
k0_continuous = theta(1)*exp(-sigma_psi(1)^-2 *(r_continuous.*r_continuous));
k1_continuous = theta(2)*exp(-sigma_psi(2)^-2 *(r_continuous.*r_continuous));
k2_continuous  = theta(3)*exp(-sigma_psi(3)^-2 *(r_continuous.*r_continuous));

% MeanKernel = mean(k0_continuous+k1_continuous+k2_continuous)
% 
% figure
% plot(r_continuous,k0_continuous+k1_continuous+k2_continuous,'r','linewidth',4)
% hold on
% plot(r,k0+k1+k2,'k','linewidth',4)
% legend('continuous kernel','discrete kernel')
% drawnow
% 
% figure
% plot(r_continuous,k0_continuous)


% plot the sensors and the basis functions
% ~~~~~~~~~~~~~~~~~~~~~~~~
% figure
% MS = 4;
% LW = 0.5;
% plot_phi=linspace(0,2*pi,1000);
% for n=1:L
%         plot(mu_phi(1,n),mu_phi(2,n),'+k','markersize',MS)        % plot sensor center and widths        
%         hold on
%         x = sigma_phi*cos(plot_phi)+mu_phi(1,n);
%         y = sigma_phi*sin(plot_phi)+mu_phi(2,n);
%         plot(x,y,'k','linewidth',LW)
%         xlim([-SpaceMax,SpaceMax])
%         ylim([-SpaceMax SpaceMax])
% end
% axis square
% 
% for n=1:NSensors
%     plot(mu_m(1,n),mu_m(2,n),'xr','markersize',MS)        % plot sensor center and widths        
%     hold on
%     x = sigma_m*cos(plot_phi)+mu_m(1,n);
%     y = sigma_m*sin(plot_phi)+mu_m(2,n);
%     plot(x,y,'r','linewidth',LW)
%     xlim([-SpaceMax,SpaceMax])
%     ylim([-SpaceMax SpaceMax])
% end