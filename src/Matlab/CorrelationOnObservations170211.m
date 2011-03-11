% this script runs the correlation analysis on the field. It does not take
% into account the observation function. The motivation was to get the
% analysis working on a simple example first. In this script you can
% experiment on


clc
close all
clear

% for plotting
% ~~~~~~~
FS_Label = 10;          % fontsize for the axis label
FS_Tick = 8;                % fontsize for the ticks
MS = 10;                     % marker size
LW = 1;
plotwidth = 8.3;        % cm
plotheight = 6.5;

% spatial parameters
% ~~~~~~~~~~~
Delta = 0.5;                          % space step for the spatial discretisation
Delta_squared = Delta^2;
SpaceMax = 10;                    % maximum space in mm
SpaceMin = -SpaceMax;         % minimum space in mm
NPoints = (SpaceMax-SpaceMin)/Delta+1;
r = linspace(SpaceMin,SpaceMax,NPoints);      % define space

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

w_conj = ifft2(conj(fft2(w)));

psi_0 = Define2DGaussian(0,0, sigma_psi(1)^2, 0,81,2*SpaceMin,2*SpaceMax);
psi_1 = Define2DGaussian(0,0, sigma_psi(2)^2, 0,81,2*SpaceMin,2*SpaceMax);
psi_2 = Define2DGaussian(0,0, sigma_psi(3)^2, 0,81,2*SpaceMin,2*SpaceMax);
w_tau = theta(1)*psi_0 + theta(2)*psi_1 + theta(3)*psi_2;       % the kernel

% sensor parameters
% ~~~~~~~~~~~~
sigma_m = 0.9;                                                         % sensor width
m = Define2DGaussian(0,0,sigma_m^2,0,NPoints,SpaceMin,SpaceMax);        % sensor kernel
sigma_varepsilon = 0.0;       % observation noise                            

% sigmoid parameters
% ~~~~~~~~~~~~
f_max = 1;             % maximum firing rate
varsigma = 0.56;         % sigmoid slope
v_0 = 1.8;                    % firing threshold

% disturbance paramters
% ~~~~~~~~~~~~~
sigma_gamma = 1.3;              % parameter for covariance of disturbance
gamma_weight = 0.1;            % variance of disturbance

Torus = true;
if Torus == true
    SphericalBoundary                   % to get Sigma_gamma on a torus
else
    k=1;
    Sigma_gamma = zeros(NPoints^2,NPoints^2);   % create disturbance covariance matrix
    for n=1:NPoints
        for nn=1:NPoints
            temp = gamma_weight*Define2DGaussian(r(n),r(nn), sigma_gamma^2, 0,NPoints,SpaceMin,SpaceMax);
            Sigma_gamma(:,k) = temp(:);
            k=k+1;
        end
    end
end

% synaptic kernel parameter
% ~~~~~~~~~~~~~~~~
tau = 0.01;                   % seconds
zeta = 1/tau;                 % 
xi = 1-Ts*zeta;           % coefficient for the discrete time model

tic

N_realizations = 1;
R_vv = zeros(N_realizations,81,81);
R_vv_plus_1 = zeros(N_realizations,81,81);
w_est_FFT = zeros(N_realizations,81,81);

for n=1:N_realizations
    disp(['realization number: ' num2str(n)])
    
    % define disturance and observation noise
    % ~~~~~~~~~~~~~~~~~~~~~~~~
    e = mvnrnd(zeros(1,NPoints^2),Sigma_gamma,T);       % process disturbance
    varepsilon = mvnrnd(zeros(1,NPoints^2),sigma_varepsilon^2*eye(NPoints^2),T);
%     varepsilon = sigma_varepsilon*randn(T,NPoints,NPoints);     % observation noise

    % field inital condition
    % ~~~~~~~~~~~~
    max_field_init = gamma_weight;
    min_field_init = -gamma_weight;
    v = zeros(T,NPoints,NPoints);
    InitialCondition = min_field_init + (max_field_init - min_field_init)*rand(NPoints,NPoints);
    future_field = InitialCondition;

    fft_y = zeros(T,NPoints,NPoints);       % initialize for speed
    y = zeros(T,NPoints,NPoints);       % initialize for speed

    future_obs = squeeze(y(1,:,:));
    future_obs_full = zeros(81,81);
    xcorr = zeros(T-1,81,81);           % initialize for speed
    autocorr = xcorr;                       % initialize for speed
    obs_noise_autocorr = xcorr;
    extra_bit1 = xcorr;
    extra_bit2 = xcorr;
    extra_bit3 = xcorr;
%     extra_bit4 = xcorr;
    extra_bit5 = xcorr;
    extra_bit6 = xcorr;
    extra_bit7 = xcorr;
    extra_bit8 = xcorr;

    left_overs = xcorr;
    %     w_est_FFT = xcorr;
    
    % main loop
    for t=1:T-1
        current_field = future_field;
%         f = 1./(1+exp(varsigma*(v_0-current_field)));           % calc firing rate using sigmoid
        f = varsigma * current_field;
        if Torus == true
            f = padarray(f,size(f),'circular');                                     % pad array for toroidal boundary
        end
        % conv firing rate with spatial kernel
        g = Ts*conv2(w,f,'same')*Delta_squared;                 % conv connectivity kerenl with firing rates
        
        disturbance =  reshape(e(t,:,:),NPoints,NPoints);       
        
        future_field = g + xi*current_field + disturbance;  % update field        
        
        v(t+1,:,:) = future_field;
        
        current_obs = future_obs;
        current_obs_full = future_obs_full;

%         obs_noise = squeeze(varepsilon(t,:,:));
        obs_noise =  reshape(varepsilon(t,:,:),NPoints,NPoints);
        
        if Torus == true
            future_field_pad = padarray(future_field,size(future_field),'circular');                                     % pad array for toroidal boundary
            future_obs = conv2(m,future_field_pad,'same') * Delta_squared + obs_noise; 
        else
            future_obs = conv2(m,future_field,'same') * Delta_squared + obs_noise; 
            future_obs_full = conv2(m,future_field) * Delta_squared;
        end
        
        fft_y(t+1,:,:) = fft2(future_obs - mean(mean(future_obs,1)));
        
        autocorr(t,:,:) = xcorr2(current_obs, current_obs);
        xcorr(t,:,:) = xcorr2(future_obs, current_obs);
        
        autocorr_trim = squeeze(autocorr(t,41-20:41+20,41-20:41+20));
        
        extra_bit1(t,:,:) = xcorr2(conv2(m,padarray(disturbance,size(disturbance),'circular'),'same')*Delta^2,current_obs);
        extra_bit2(t,:,:) = xcorr2(conv2(m,xi*padarray(current_field,size(current_field),'circular'),'same')*Delta^2,current_obs);
        extra_bit3(t,:,:) = xcorr2( conv2(m,padarray(g,size(g),'circular') ,'same')*Delta^2,current_obs);        
%         extra_bit4(t,:,:) = Ts* xcorr2(conv2(m, conv2(w,f)*Delta^2,'same')*Delta^2 ,current_obs ); % should be the same as extra_bit3        
        extra_bit5(t,:,:) = Ts* xcorr2( conv2(w, conv2(m,f)*Delta^2,'same')*Delta^2,current_obs ); % should be the same as extra_bit3

        extra_bit6(t,:,:) = varsigma*Ts* xcorr2( conv2(w, padarray(current_obs,size(current_obs),'circular'),'same')*Delta^2, current_obs ); % should be the same as extra_bit3
        
        extra_bit_temp = varsigma*Ts* conv2( w,xcorr2(current_obs))*Delta^2; % should be the same as extra_bit3
        MidPoint = 61;
        OS = 40;
        extra_bit7(t,:,:)= extra_bit7(MidPoint-OS:MidPoint+OS,MidPoint-OS:MidPoint+OS);
        
        
%         plot(squeeze(extra_bit6(t,1,:)),'k')
%         hold on
%         plot(squeeze(extra_bit7(t,1,:)),'g')
%         hold off
%         drawnow
% extra_bit5 is not the same as extra bit4, but it should be due to the associativity and comutativity properties. 
%         a = conv2(m, conv2(w,f)*Delta^2,'same')*Delta^2;
%         b = conv2(w, conv2(m,f)*Delta^2,'same')*Delta^2;
        
% autocorr_trim = squeeze(autocorr(t,41-20:41+20,41-20:41+20));
% xcorr_trim = squeeze(xcorr(t,41-20:41+20,41-20:41+20));
% extra_bit1_trim = squeeze(extra_bit1(t,41-20:41+20,41-20:41+20));
% extra_bit6_trim = squeeze(extra_bit6(t,41-20:41+20,41-20:41+20));


        left_overs = xcorr(t,:,:) - xi*autocorr(t,:,:) - extra_bit7(t,:,:) - extra_bit1(t,:,:);
        
%         left_overs = xcorr_trim - xi*autocorr_trim - extra_bit8 - extra_bit1_trim;

% ok, removed the edge effects and these guys are the same
%         subplot(121),imagesc(squeeze(mean(extra_bit1,1))),axis square
%         colorbar
%         subplot(122),imagesc(squeeze(mean(autocorr,1))),axis square
%         colorbar
%         drawnow
% 
%         obs_noise_autocorr(t,:,:) = xcorr2(obs_noise, obs_noise);
        
    end

    mean_left_overs = squeeze(mean(left_overs,1));
    figure,imagesc(mean_left_overs),colorbar
    
    R_yy = squeeze(mean( autocorr(2:end,:,:) ,1));
    R_yy_plus_1 = squeeze(mean( xcorr(2:end,:,:) ,1));
    
    mean_extra_bit1 = squeeze(mean( extra_bit1(2:end,:,:) ,1));
%     mean_extra_bit2 = squeeze(mean( extra_bit2(2:end,:,:) ,1));
%     mean_extra_bit3 = squeeze(mean( extra_bit3(2:end,:,:) ,1));

    mean_obs_noise = squeeze(mean( obs_noise_autocorr(2:end,:,:) ,1));    
    
    mean_spatial_freq = squeeze(mean(fft_y(2:end,:,:),1));
    
    LHS = R_yy_plus_1 - xi*(R_yy-mean_obs_noise) - mean_extra_bit1;
    
%     %%
%     R_yy_conv_mat = convmtx2(R_yy-mean_obs_noise,81,81);
%     LHS_vect = LHS(:);
%     tic
%     w_vect = linsolve(full(R_yy_conv_mat)',LHS_vect);
%     toc
%     
%     %%
%     w_est_linsolve = reshape(w_vect,161,161)*xi/(Ts*varsigma*Delta^2);
%     figure,imagesc(w_est_linsolve)
% 
%     FFT_w_est_linsolve = fft2(w_est_linsolve);
%     
%     figure,imagesc(abs(FFT_w_est_linsolve))
% 
%     shifted_FFT_w_est_linsolve = fftshift(FFT_w_est_linsolve);
%     NSamps = 30;
%     MidPoint = 81;
%     temp1 = shifted_FFT_w_est_linsolve(MidPoint-NSamps:MidPoint+NSamps,MidPoint-NSamps:MidPoint+NSamps);
%     temp2 = zeros(size(FFT_w_est_linsolve));
%     temp2(MidPoint-NSamps:MidPoint+NSamps,MidPoint-NSamps:MidPoint+NSamps) = temp1;
%     FFT_w_est_linsolve_thresh = ifftshift(temp2);
%     
%     figure,imagesc(abs((FFT_w_est_linsolve_thresh)))
% 
%     w_est_linsolve_thresh = ifft2(FFT_w_est_linsolve_thresh);
%     
%     figure,imagesc(w_est_linsolve_thresh)
    
    %%
    S_yy = fft2(R_yy-mean_obs_noise); 
    S_LHS = fft2(LHS);               %auto and noise free
    
    %%
    
    ratio = S_LHS./S_yy;
    
    filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/FieldSpatialFreq.pdf';
    figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
        'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
    nu_field = linspace(0,2*Delta*2,size(mean_spatial_freq,1));
    CLIM = [-30 40];
    SpatialFreq_dB = 20*log10(abs(mean_spatial_freq*Delta^2));
    imagesc(nu_field,nu_field,SpatialFreq_dB,CLIM);
    xlabel('Spatial Freq','fontsize',FS_Label)
    ylabel('Spatial Freq','fontsize',FS_Label)
    set(gca,'xtick',[0 1 2],'ytick',[0 1 2],'fontsize',FS_Tick)
    axis square
    axis xy
    colorbar
    colormap hot
    drawnow
    
    filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/FieldSpatialFreqCrossSection.pdf';
    figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
        'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
    nu_field = linspace(0,2*Delta*2,size(mean_spatial_freq,1));
    plot(nu_field,SpatialFreq_dB(1,:),'k');
    hold on
    plot(nu_field,SpatialFreq_dB(:,1),'r');
    hold off
    xlabel('Spatial Freq','fontsize',FS_Label)
    ylabel('Power (dB)','fontsize',FS_Label)
    set(gca,'xtick',[0 1 2],'ytick',[-30 0 40],'fontsize',FS_Tick)
    ylim(CLIM)
    axis square
    drawnow
    
    filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/FFTKernelEstimateFull.pdf';
    figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
        'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
    nu = linspace(0,2*Delta*2,size(ratio,1));
    CLIM = [0 0.07];
    imagesc(nu,nu,abs(ratio)/ (varsigma*Ts));
    xlabel('Spatial Freq','fontsize',FS_Label)
    ylabel('Spatial Freq','fontsize',FS_Label)
    set(gca,'xtick',[0 1 2],'ytick',[0 1 2],'fontsize',FS_Tick)
    axis square
    axis xy
    colorbar
    colormap hot
    drawnow
    
    shifted_ratio = fftshift(ratio);
    
    figure,imagesc(abs(shifted_ratio))
    
    NSamps = 13;
%     temp1 = shifted_ratio(41-NSamps:41+NSamps,41-NSamps:41+NSamps);
%     temp2 = zeros(size(ratio));
%     temp2(41-NSamps:41+NSamps,41-NSamps:41+NSamps) = temp1;
    temp1 = shifted_ratio;
    temp1(41-NSamps:41+NSamps,41-NSamps:41+NSamps) = 0;
    temp2 = temp1;
    ratio = ifftshift(temp2);
    
    filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/FFTKernelEstimateThreshold.pdf';
    figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
        'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
    nu = linspace(0,2*Delta*2,size(ratio,1));
    disp(['cutoff freq: ' num2str(nu(NSamps)) ' cycles / mm'])
    imagesc(nu,nu,abs(ratio)/ (varsigma*Ts));
    xlabel('Spatial Freq','fontsize',FS_Label)
    ylabel('Spatial Freq','fontsize',FS_Label)
    set(gca,'xtick',[0 1 2],'ytick',[0 1 2],'fontsize',FS_Tick)
    axis square
    axis xy
    colorbar
    colormap hot
    drawnow

    filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/FFTKernel.pdf';
    figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
        'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
    nu = linspace(0,2*Delta*2,size(ratio,1));
    imagesc(nu,nu,Delta^2*abs(fft2(w_tau)));
    xlabel('Spatial Freq','fontsize',FS_Label)
    ylabel('Spatial Freq','fontsize',FS_Label)
    set(gca,'xtick',[0 1 2],'ytick',[0 1 2],'fontsize',FS_Tick)   
    axis square
    axis xy
    colorbar
    colormap hot
    drawnow
    
    w_est_FFT(n,:,:) = fftshift( ifft2(ratio) ) / (Delta^2*varsigma*Ts);
    
%%

end

%%
mean_w_est = squeeze(mean(w_est_FFT,1));

disp('generated data')
toc

figure
imagesc(mean_obs_noise)
axis square
title('auto corr of obs noise')
colorbar
r_xcor = linspace(-20,20,81);

filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/KernelEstimate.pdf';
figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
    'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
imagesc(r_xcor,r_xcor,mean_w_est)
xlabel('Space','fontsize',FS_Label)
ylabel('Space','fontsize',FS_Label)
xlim([-10,10])
ylim([-10,10])
set(gca,'xtick',[-10 0 10],'ytick',[-10 0 10],'fontsize',FS_Tick)
axis square
axis xy
colorbar
colormap hot
drawnow

% figure
% imagesc(r_xcor,r_xcor,mean_w_est),axis square,title('FFT'),colorbar
% xlim([-10,10])
% ylim([-10,10])

filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/Kernel.pdf';
figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
    'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
imagesc(r,r,w)    
xlabel('Space','fontsize',FS_Label)
ylabel('Space','fontsize',FS_Label)
set(gca,'xtick',[-10 0 10],'ytick',[-10 0 10],'fontsize',FS_Tick)
axis square
axis xy
colorbar
colormap hot
drawnow


r_deconv_mat = linspace(-40,40,161);

filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/Kernel.pdf';
figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
    'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
% plot(r,w(21,:)/max(w(21,:)),'k'),hold on
% plot(r_xcor,mean_w_est(41,:)/max(mean_w_est(41,:)),'r')
plot(r,w(21,:),'k'),hold on
plot(r_xcor,mean_w_est(41,:),'r')
xlim([-10,10])
% ylim([-0.5 1.1])
leg = legend('$w(\mathbf{r})$','$\hat{w}(\mathbf{r})$');
set(leg,'interpreter','latex','box','off')
xlabel('Space','fontsize',FS_Label)
ylabel('Connection Strength','fontsize',FS_Label)
set(gca,'xtick',[-10 0 10],'ytick',[-5 0 25.0],'fontsize',FS_Tick)
