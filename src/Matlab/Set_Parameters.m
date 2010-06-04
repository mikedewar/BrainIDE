% set all the parameters for the model and the estimator

% if we are running a batch than we dont want to clear things
if exist('RunningBatch') == 0
    clear
    close all
    clc
end

% spatial parameters
Delta = 0.5;                          % space step for the spatial discretisation
Delta_squared = Delta^2;
SpaceMax = 10;                    % maximum space in mm
SpaceMin = -SpaceMax;         % minimum space in mm
NPoints = (SpaceMax-SpaceMin)/Delta+1;
r = linspace(SpaceMin,SpaceMax,NPoints);      % define space

% temporal parameter
Ts = 1e-3;          % sampling period
T = 300;            % maximum time (ms)

% kernel parameters
theta(1) = 10;     % local kernel amplitude
theta(2) = -8;     % surround kernel amplitude
theta(3) = 0.5;       % lateral kernel amplitude
sigma_psi(1) = 1.8;     % local kernel width
sigma_psi(2) = 2.4;     % surround kernel width
sigma_psi(3) = 6;       % lateral kernel width
psi_0 = Define2DGaussian(0,0, sigma_psi(1)^2, 0,NPoints,SpaceMin,SpaceMax);
psi_1 = Define2DGaussian(0,0, sigma_psi(2)^2, 0,NPoints,SpaceMin,SpaceMax);
psi_2 = Define2DGaussian(0,0, sigma_psi(3)^2, 0,NPoints,SpaceMin,SpaceMax);
w = theta(1)*psi_0 + theta(2)*psi_1 + theta(3)*psi_2;       % the kernel
W = fft2(w);                                                                       % the fft of the kernel
Ts_W = W*Ts;                        % FFT of kernel times the time step

% field basis function parameters
NBasisFunctions_xy = 9;
L = NBasisFunctions_xy^2;                   % number of states and the number of basis functions
mu_phi_xy = linspace(-10,10,NBasisFunctions_xy);
sigma_phi = sqrt(2.5);

mu_phi = zeros(2,L);     % initialize for speed
m=1;
for n=1:NBasisFunctions_xy
    for nn=1:NBasisFunctions_xy
        mu_phi(:,m) = [mu_phi_xy(n) ; mu_phi_xy(nn)];
        m=m+1;
    end
end

% sensor parameters
NSensors_xy = 9;%14;
NSensors = NSensors_xy^2;
mu_y_xy = linspace(-10,10,NSensors_xy);%linspace(-9.75,9.75,NSensors_xy);               % sensor centers
sensor_indexes = (mu_y_xy+SpaceMax)/Delta +1;             % so we can easily get the observations from the filed filtered by the sensors
sigma_y = 0.9;                                                         % sensor width
m = Define2DGaussian(0,0, sigma_y^2, 0,NPoints,SpaceMin,SpaceMax);
M = fft2(m);                                            % fft of sensor kernel to get observations quickly
% define all sensor centers
mu_y = zeros(2,NSensors);     % initialize for speed
m=1;
for n=1:NSensors_xy
    for nn=1:NSensors_xy
        mu_y(:,m) = [mu_y_xy(n) ; mu_y_xy(nn)];
        m=m+1;
    end
end

% observation noise characteristics
sigma_varepsilon = 0.0001;                                  
Sigma_varepsilon = sigma_varepsilon*eye(NSensors);        % observation covariance matrix

% sigmoid parameters
f_max = 20;             % maximum firing rate
varsigma = 0.8;         % sigmoid slope
v_0 = 2;                    % firing threshold

% synaptic kernel parameter
zeta = 100;             % synaptic time constant (ms)
xi = 1-Ts*zeta;         % coefficient for the discrete time model

% disturbance paramters
sigma_gamma = 1.3;          % parameter for covariance of disturbance
gamma_weight = 0.1;            % variance of disturbance

m=1;
Sigma_gamma = zeros(NPoints^2,NPoints^2);   % create disturbance covariance matrix
for n=1:NPoints
    for nn=1:NPoints
        temp = gamma_weight*Define2DGaussian(r(n),r(nn), sigma_gamma^2, 0,NPoints,SpaceMin,SpaceMax);
        Sigma_gamma(:,m) = temp(:);
        m=m+1;
    end
end

    
% plot the true kernel for comparison
k0 = theta(1)*exp(-sigma_psi(1)^-2 *(r.*r));
k1 = theta(2)*exp(-sigma_psi(2)^-2 *(r.*r));
k2 = theta(3)*exp(-sigma_psi(3)^-2 *(r.*r));
figure,plot(r,k0+k1+k2,'k','linewidth',4),hold on,drawnow
