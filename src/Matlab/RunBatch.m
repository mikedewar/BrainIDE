% Run batch job
clc
clear
close all

Set_Parameters

RunningBatch = 1;               % this is a flag for the scripts
NRealisations = 3;
for Realisation=1:NRealisations
    tic
    disp(['Running realisation: ' num2str(Realisation) ' of ' num2str(NRealisations)])
    disp('running GenerateData script')
    GenerateData
    disp('running RunFilter script')
    RunFilter
    t_realisation = toc;
    disp(['time for generating data and estimation for realisation = ' num2str(t_realisation)])
end

save('ResultsForAllRealisations.mat','theta_save','xi_save',...
    'Delta','SpaceMax','Ts','T','sigma_psi','sigma_phi','NBasisFunctions_xy',...
    'mu_phi_xy','NSensors_xy','mu_m_xy','sigma_m','sigma_varepsilon',...
    'f_max','varsigma','v_0','zeta','sigma_gamma','gamma_weight')

% plot the final distributions
% ~~~~~~~~~~~~~~~
figure,subplot(141), hist(theta_save(:,1)),xlabel('\theta_0')
subplot(142), hist(theta_save(:,2)),xlabel('\theta_1')
subplot(143), hist(theta_save(:,3)),xlabel('\theta_2')
subplot(144), hist(xi_save),xlabel('\xi')