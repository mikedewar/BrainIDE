
% if we are running a batch than we dont want to clear things
if exist('RunningBatch','var') == 0
%     clear
%     close all
%     clc
%     load Parameters
end

% first create all the constants that we can do analytically
Create_phi
Create_Gamma
Create_Psi
Create_C
Create_Sigma_e

% initialize state sequence
P_f = zeros(L,L,T);
x_f = 20*mvnrnd(zeros(1,L),Sigma_e,T)';
P_f(:,:,1)  = cov(x_f');

% use initial state sequence to get parameter estimate
[theta xi] = LSEstimator(x_f,phi_unwrapped,Delta,varsigma,f_max,v_0,Ts_invGamma_phi_psi);
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
NIterations = 5;
for Iteration = 1:NIterations
    
    disp('running the forward iteration (filtering)')
    for t=1:T-1

        % create the matrix of sigma vectors
        X_f_t = GetSigmaMatrix(x_f(:,t),P_f(:,:,t),sqrt_L_plus_lambda);

        % propagate the sigma matrix through Q
        X_f_minus_t_plus_1 = Q(X_f_t, phi_unwrapped, Ts_invGamma_theta_phi_psi, ...
            Delta_squared, f_max, varsigma, v_0, xi);

        % propagate the sigma matrix through the state equation, weigth and
        % get the predicted state and covariance
        [x_f_minus P_f_minus] = PredictStateAndCovariance(Wm,Wc,X_f_minus_t_plus_1,Sigma_e);

        % use observation to correct the state and covariance prediction
        y_t_plus_1 = y(t+1,:)';
        [x_f(:,t+1) P_f(:,:,t+1)] = GetFilteredStateAndCovariance(P_f_minus, C, Sigma_varepsilon, ...
            x_f_minus, y_t_plus_1);
      
    end
    
    x_b = x_f;
    P_b = P_f;
    
    % run the smoother
    disp('running the backward iteration (smoothing)')
    for n=1:T-1
        
        X_b_minus_t = GetSigmaMatrix(x_b(:,T-n),P_b(:,:,T-n),sqrt_L_plus_lambda);
        
        X_b_minus_t_plus_1 = Q(X_b_minus_t, phi_unwrapped, ...
            Ts_invGamma_theta_phi_psi, Delta_squared, f_max,varsigma, v_0, xi);
        
         [x_b_minus_t_plus_1 P_b_minus_t_plus_1] = PredictStateAndCovariance(Wm, Wc, X_b_minus_t_plus_1, Sigma_e);
         
         % calculate cross covariance matrix
         M_t_plus_1 = CalcCrossCovariance(Wc,X_b_minus_t, x_f(:,T-n), X_b_minus_t_plus_1, x_b_minus_t_plus_1);
        
         % get smoothed state and smoothed covaraince
         [P_b(:,:,T-n) x_b(:,T-n)] = GetSmoothedStateAndCovariance(M_t_plus_1, P_b_minus_t_plus_1, ...
             x_f(:,T-n), x_b(:,(T-n)+1), x_b_minus_t_plus_1, P_b(:,:,(T-n)+1), P_f(:,:,T-n));
        
    end
        
    % use state sequence to estimate parameters
    disp('Running LS parameter estimation')
    [theta xi] = LSEstimator(x_b,phi_unwrapped,Delta,varsigma,f_max,v_0,Ts_invGamma_phi_psi);
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
if exist('RunningBatch') == 1
    theta_save(Realisation,:) = theta;
    xi_save(Realisation) = xi;
end
save(resultsfilename,'theta','xi','x_f')
