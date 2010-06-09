% Run batch job
clc
clear
close all

RunningBatch = 1;
NRealisations = 50;
for Realisation=1:NRealisations
    tic
    disp(['Running realisation: ' num2str(Realisation) ' of ' num2str(NRealisations)])
    disp('generating data')
    GenerateData
    RunFilter
    t_realisation = toc
end
save('ResultsForAllRealisations.mat','theta_save','xi_save',...
    'Delta','SpaceMax','Ts','T','sigma_psi','sigma_phi','NBasisFunctions_xy',...
    'mu_phi_xy','NSensors_xy','mu_y_xy','sigma_y','sigma_varepsilon',...
    'f_max','varsigma','v_0','zeta','sigma_gamma','gamma_weight')