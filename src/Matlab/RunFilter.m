
% if we are running a batch than we dont want to clear things
if exist('RunningBatch') == 0
    clear
    close all
    clc
    load Parameters
end

% first create all the constants that we can do analytically
Create_phi
Create_Gamma
Create_Psi
Create_C
Create_Sigma_e

% initialize state sequence
P  = 2*Sigma_e;
x = (mvnrnd(zeros(1,L),Sigma_e,T))';

% use initial state sequence to get parameter estimate
[theta xi] = LSEstimator(x,phi_unwrapped,Delta,varsigma,f_max,v_0,Ts_invGamma_phi_psi);
disp(['Iteration 0, theta = ' num2str(theta) ', xi = ' num2str(xi)])          

% use estiamted parameters to create the Psi matrix
Create_Psi

% set filter parameters
alpha   = 1e-3;
kappa = 3-L;
beta = 2;
lambda = alpha^2*(L+kappa)-L;

% calculate the weigths
Wm = [lambda/(L+lambda) ; 1/(2*(L+lambda))*ones(2*L,1)];
Wc = [(lambda/(L+lambda)) + (1-alpha^2+beta) ; 1/(2*(L+lambda))*ones(2*L,1)];

% for calculating the sigma matrix
sqrt_L_plus_lambda = sqrt(L+lambda);

tic
NIterations = 3;
for Iteration = 1:NIterations
    for t=1:T-1

        % create the matrix of sigma vectors
        X = GetSigmaMatrix(x(:,t),P,sqrt_L_plus_lambda);

        % propagate the sigma matrix through Q
        Q_X = Q(X,phi_unwrapped,Ts_invGamma_theta_phi_psi,Delta_squared,f_max,varsigma,v_0,xi);

        % propagate the sigma matrix through the state equation, weigth and
        % get the predicted state and covariance
        [x_minus P_minus] = PredictStateAndCovariance(Wm,Wc,Q_X,Sigma_e);

        % use observation to correct the state and covariance prediction
        y_t_plus_1 = y(t+1,:)';
        [x(:,t) P] = GetFilteredStateAndCovariance(P_minus,C,Sigma_varepsilon,x_minus,y_t_plus_1);
      
    end
    
    % use state sequence to estimate parameters
    [theta xi] = LSEstimator(x,phi_unwrapped,Delta,varsigma,f_max,v_0,Ts_invGamma_phi_psi);
    disp(['Iteration ' num2str(Iteration) ', theta = ' num2str(theta) ', xi = ' num2str(xi)])
    
    % update Psi with estimated paraters
    Create_Psi
end
toc

k0 = theta(1)*exp(-sigma_psi(1)^-2 *(r.*r));
k1 = theta(2)*exp(-sigma_psi(2)^-2 *(r.*r));
k2 = theta(3)*exp(-sigma_psi(3)^-2 *(r.*r));
plot(r,k0+k1+k2,'r'),hold on,drawnow

disp(['true ratio = ' num2str(10/-8) ', estimated ratio = ' num2str(theta(1)/theta(2))])
resultsfilename = ['Results' SaveTime '.mat'];
save(resultsfilename,'theta','xi','x')
