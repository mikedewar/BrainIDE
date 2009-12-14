% 1 d example from paper
clear
close all
clc

LW = 5;

x(1) = 0;                   % define initial state

Q = 2;                     % process disturbance variance
R = 1;                      % measure noise variance
N_Samples = 100;             % number of time points

% get true state and measurements
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for k=2:N_Samples
    [x(k) z(k)] = UpdateStateAndObservation(x(k-1), k-1 , Q, R);
end
plot(1:N_Samples,x,'g','linewidth',LW)
hold on

% run particle filter to estimate state
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Particle_Variance = 10;
N_Particles = 100;
x_p_prior = sqrt(Particle_Variance)*randn(1,N_Particles);                            % subscript p means particle
% x_p_prior = sqrt(Particle_Variance)*randn(1,1)*ones(1,N_Particles);          % subscript p means particle
x_p = x_p_prior;


for k=2:N_Samples
    for Particle_Index = 1: N_Particles

        [x_p_update(k,Particle_Index) z_p(Particle_Index)] = UpdateStateAndObservation(x_p(k-1,Particle_Index), k-1 , Q, R);
        w(Particle_Index) = StateWeight2(z(k), R, z_p(Particle_Index));

    end
    
    
    [x_new w_new] = Resampler(x_p_update(k,:),w);
%     x_p(k,:) = mean(x_new)*ones(1,N_Particles);
    x_p(k,:) = x_new;
    
end

plot(mean(x_p,2),'k','linewidth',LW)
boxplot(x_p')
% legend('true state','prediction','x_mike')
for Particle_Index = 1: N_Particles
    plot(1:N_Samples,x_p(:,Particle_Index),'.')
end


legend('True State','Mean Particles', 'Particles')
