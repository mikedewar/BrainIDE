% 2D WC neural field model
% Dean Freestone 04/12/09

% 04/12/09, begin coding of model, DF
% 05/12/09, made code clearer and checked for errors, DF

% table of interesting parameters
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CalcGamma = 1;
if CalcGamma
    clear
    CalcGamma = 1;
end

clc
close all

UseBasisFunctions = 1;


% initialise receptor kinetics
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
alpha = 100;                                            % alpha = 1/tau, post-synaptic time constant, (Wendling, 2002, 100 for excitatory, 500 for inhibitory), Schiff = 3
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


% initialise activation function
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
beta = 0.56;                                            % slope of sigmoid, spikes/mV           (Wendling, 2002, 0.56 mV^-1)
nu = 1;                                                 % maximum firing rate, spikes/s         (Wendling, 2002, 2*e0 = 5, or nu = 5 s^-1)                           
threshold = 6;                                          % firing threshold, mV                  (Wendling, 2002, 6 mV), (Schiff, 2007, threshold = 0.24 (Heaviside))
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


%initialise connectivity basis functions
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N_ConnBasisFuns = 3;
sigma1 = 4;                                             % width of connectivity kernels
sigma2 = 6;                                                                                       % Schiff, 0.91
sigma3 = 15;

theta1 = 1.00;                                           % amplitudes of connectivity kernels 
theta2 = -0.9;                                                                                        % Schiff, -1.38
theta3 = 0.045;

if UseBasisFunctions
    theta = 0.001*[theta1 ; theta2; theta3];               % scaled for basis functions
else
    theta = 2200*[theta1 ; theta2; theta3];
end

sigma = 0.5*[sigma1 ; sigma2 ; sigma3];                 % scaled by two to simulate a half sized field for speed
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


% initialise field basis functions
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sigma_field = 2.5;                                       % width of the field basis function
field_basis_separation = 2;                             % distance wrt masses, must be > 1, diagonal distance
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


% initialise time properties
% ~~~~~~~~~~~~~~~~~~~~~~~~~~
Fs = 1e3;                                               % sampling rate, s^-1
Ts = 1/Fs;                                              % sampling period, seconds
t_end = .1;                                              % seconds
NSamples = t_end*Fs;
time = linspace(0,t_end,NSamples);
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~


% initialise space properties
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MassDensity = 1;                                        % masses per mm
SpaceStep = 1/MassDensity;
FieldWidth = 40;                                        % mm in each axis, should be even
N_masses_in_width = MassDensity*FieldWidth+1;
Space_x = -FieldWidth/2:1/MassDensity:FieldWidth/2;
Space_y = Space_x;


% initialize observation kernel
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SensorWidth = 3;        % diameter in mm
SensorSpacing = 10;     % in mm
BoundryEffectWidth = 4; % how much space to ignore

DefineObservationKernel
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


% initialise connectivity kernel
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
disp('building connectivity kernel')
tic
create_connectivity_kernel
toc
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if CalcGamma
    % initialise field basis functions
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if UseBasisFunctions
        disp('building field basis functions')
        tic
        create_field_basis_functions2
        toc
        % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


% construct Gamma matrix
% ~~~~~~~~~~~~~~~~~~~~~~
        disp('creating Gamma matrix')
        tic         
        Gamma = CreateGamma(phi);
        toc

    end
end


% Disturbance properties
% ~~~~~~~~~~~~~~~~~~~~~~
DisturbanceMean = 0*ones(N_field_basis_function,1);
SigmaDisturbance = 8000*eye(N_field_basis_function,N_field_basis_function);           % no basis decomposition
if UseBasisFunctions
    R = chol(Gamma^-1);               % use cholesky decomp
    DisturbanceCovariance = R*SigmaDisturbance*R';
    DisturbanceMean = R*DisturbanceMean;
else
    Disturbance = sqrt(SigmaDisturbance)*randn(NSamples,N_masses_in_width,N_masses_in_width);
end
% ~~~~~~~~~~~~~~~~~~~~~~


% predefine matrices for speed
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Firing_Convolved_With_Kernel = zeros(NSamples,N_masses_in_width,N_masses_in_width);
Firing_Rate = zeros(NSamples,N_masses_in_width,N_masses_in_width);

if UseBasisFunctions

    DistubanceWithBasis = zeros(size(phi,1),N_masses_in_width,N_masses_in_width);
    FieldLayer = zeros(size(phi,1),N_masses_in_width,N_masses_in_width);
end


% initialise field and states
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~
Initial_Variance = 3;
[x Field] = InitialiseStates(NSamples, N_masses_in_width, N_field_basis_function, phi, UseBasisFunctions, Initial_Variance);
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


% main loop for the simulation
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
disp('running simulation')
tic
figure('units','normalized','position',[0 0 1 1])
lambda = 1-Ts*alpha;
for t=2:NSamples
    
    f = Sigmoid_Firing_Rate(nu, beta, threshold, squeeze(Field(t-1,:,:)));
    Firing_Rate(t,:,:) = f;
    Firing_Convolved_With_Kernel(t-1,:,:) = Convolve_Kernel_With_Firing_Rate(N_masses_in_width, f, SpaceStep, ConnectivityKernel);

    if UseBasisFunctions  
        
        Int_Phi_Mult_With_Convolved_Firing = Mult_By_phi_And_Integrate(Firing_Convolved_With_Kernel(t-1,:,:), ...
            N_field_basis_function, SpaceStep, phi, N_masses_in_width);
        
        Disturbance = DisturbanceMean+DisturbanceCovariance*randn(N_field_basis_function,1);
        
        for Field_Basis_Index=1:N_field_basis_function
            Disturbance_with_Phi(Field_Basis_Index) = sum(sum(Disturbance(Field_Basis_Index)*squeeze(phi(Field_Basis_Index,:,:)),1))*SpaceStep^2;
        end
        
        x(:,t) = Ts*Gamma\Int_Phi_Mult_With_Convolved_Firing + lambda*x(:,t-1) + Ts^2*Disturbance_with_Phi';
        
        Field(t,:,:) = Create_Field_From_States(phi, x(:,t));
        
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    else
        
        Field(t,:,:) = squeeze(Ts*UpdatedField(t-1,:,:) + lambda*Field(t-1,:,:) + Ts*Disturbance(t-1,:,:));
    end
    
% generate observations
% ~~~~~~~~~~~~~~~~~~~~~
    Noise_Covariance_Coefficient = 0.2;
    y(:,t) = Get_Observations(Noise_Covariance_Coefficient, NSensors, N_masses_in_width, Field(t,:,:), SpaceStep, m);

        
    % plot results
%     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
    subplot(231)
    imagesc(Space_x,Space_y,squeeze(Field(t,:,:)))
    title('$v_{t+1}(r\prime)$', 'interpreter','latex')
    axis square
    colorbar
    
    subplot(232)
    imagesc(Space_x,Space_y,squeeze(Firing_Rate(t-1,:,:)))
    title('$f(v_t(r\prime))$', 'interpreter','latex')
    axis square
    colorbar
    
    subplot(233)
    imagesc(Space_x,Space_y,squeeze(Firing_Convolved_With_Kernel(t-1,:,:)))
    axis square
    title('$\int_\Omega k(r-r\prime)f(v_t(r))dr\prime$', 'interpreter','latex')
    colorbar
    
    subplot(212)
    plot(y')
    drawnow
    time(t)
    
end
toc 
        