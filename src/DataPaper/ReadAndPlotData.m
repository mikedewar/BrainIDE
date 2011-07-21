% this script reads in the micro data and plot the times series and the
% temporal and spatial frequency properties.

clc
clear
close all

% hi guys, when you use this uncomment your name. You also need to add
% where you have saved the data and where you want to save the figures
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
User = 'Dean';                   
%  User = 'Parham';
% User = 'Mike';
% User = 'Ken';

if strcmp(User,'Dean')
    DataFileLocation = '/Users/dean/Dropbox/iEEG Data Boston/MG29_Seizure22_LFP_Clip_Dean.mat';
    TemporalFreqFig = '/Users/dean/Projects/BrainIDE/ltx/data_paper/Figures/TemporalFreq.eps';
    SpatialFreqFig = '/Users/dean/Projects/BrainIDE/ltx/data_paper/Figures/SpatialFreq.eps';
    SpatialFreqCrossSectFig = '/Users/dean/Projects/BrainIDE/ltx/data_paper/Figures/SpatialFreqCrossSection.eps';
    FieldObservationsFig = '/Users/dean/Projects/BrainIDE/ltx/data_paper/Figures/FieldObservations.eps';
    LFPsFig = '/Users/dean/Projects/BrainIDE/ltx/data_paper/Figures/LFPs.eps';
    CrossCorrFig2D = '/Users/dean/Projects/BrainIDE/ltx/data_paper/Figures/CrossCorr2D.eps';
    CrossCorrFig1D = '/Users/dean/Projects/BrainIDE/ltx/data_paper/Figures/CrossCorr1D.eps';
    HomoCrossCorrFig = '/Users/dean/Projects/BrainIDE/ltx/data_paper/Figures/HomoTestCrossCorr.eps';
    
elseif strcmp(User,'Parham')
    DataFileLocation = '/home/parham/Dropbox/iEEG Data Boston/MG29_Seizure22_LFP_Clip_Dean.mat';                  % file location
    TemporalFreqFig = '/home/parham/Desktop/Figures/TemporalFreq.eps';                   % figure save location from here down.
    SpatialFreqFig = '/home/parham/Desktop/SpatialFreq.eps';
    SpatialFreqCrossSectFig = '/home/parham/Desktop/SpatialFreqCrossSection.eps';
    FieldObservationsFig = '/home/parham/Desktop/FieldObservations.eps';
    LFPsFig = '/home/parham/Desktop/Figures/LFPs.eps';
    
elseif strcmp(User,'Ken')   
    DataFileLocation = '';
    TemporalFreqFig = '';
    SpatialFreqFig = '';
    SpatialFreqCrossSectFig = '';
    FieldObservationsFig = '';
    LFPsFig = '';
    
elseif strcmp(User,'Mike')
    DataFileLocation = '';
    TemporalFreqFig = '';
    SpatialFreqFig = '';
    SpatialFreqCrossSectFig = '';
    FieldObservationsFig = '';
    LFPsFig = '';
end

FS = 10;
FS2 = 12;

TwoColumnWidth = 17.35;     % PLoS figure width cm
OneColumnWidth = 8.3; % cm
Fs = 30e3;                              % sampling rate in Hz of the raw data
FsDec = 1e3;                          % this is the sampling rate that we will use
TsDec = 1/FsDec;                    % the decimated sampling period
DecimationFactor = Fs/FsDec;

ElectrodeSpacing = 0.4;             % mm
SpatialFreqMax = 1/ElectrodeSpacing;

SzStart = 8;        % seconds
SzEnd = 34;         % seconds

% read in neuroport data from .mat file
% ~~~~~~~~~~~~~~~~~~~~~~
Data = open(DataFileLocation);
Data = Data.whole_data';

% map the rows of the data matrix to the channels
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DataMap = [0 2 1 3 4 6 8 10 14 0 ...
    65 66 33 34 7 9 11 12 16 18 ...
    67 68 35 36 5 17 13 23 20 22 ...
    69 70 37 38 48 15 19 25 27 24 ...
    71 72 39 40 42 50 54 21 29 26 ...
    73 74 41 43 44 46 52 62 31 28 ...
    75 76 45 47 51 56 58 60 64 30 ...
    77 78 82 49 53 55 57 59 61 32 ...
    79 80 84 86 87 89 91 94 63 95 ...
    0 81 83 85 88 90 92 93 96 0];
MappedData = Data;
for n=1:100
    if n==1 || n==10 || n==91 || n==100
        MappedData(:,n) = zeros(size(Data,1),1);
    else
        MappedData(:,n) = Data(:,DataMap(n));
    end
end
clear Data      % save some RAM

%%
% these are the channels that have loads of noise and will be useless
BadChannels = [13 15 19 24 26 35 42 46 47 54 56  57 67 81 85 87];
% get ride of the noisey guys
for n=1:length(BadChannels)
    MappedData(:,BadChannels(n)) = zeros(size(MappedData,1),1);
end

%%
% now need to re-reference the data
% ~~~~~~~~~~~~~~~~~~~~~
CAR = repmat(mean(MappedData,2),1,100);
CARData = MappedData-CAR;                       % CAR = common average reference
clear MappedData
clear CAR;

%%
% re-assign the bad channels and the corners to nans so we can interpolate
ZeroChannels = [1 10 13 15 19 24 26 35 42 46 47 54 56  57 67 81 85 87 91 100];
% get ride of the noisey guys
for n=1:length(ZeroChannels)
    CARData(:,ZeroChannels(n)) = nan(size(CARData,1),1);
end

%%
% low-pass filter data
Fc = 100;           % Cut-off frequency in Hz
Wn = Fc/(Fs/2);
FilterOrder = 2;
[b a] = butter(FilterOrder,Wn);
FiltData = filter(b,a,CARData);
clear CARData

%%
% downsample to FsDec
FiltData = downsample(FiltData,DecimationFactor);
t = TsDec*(0:size(FiltData,1)-1);

% subplot(133),semilogx(f,mean(TemporalFFT3,2))
% xlim([0 FMax])
% ylim([60 YMax])
% axis square
% set(gca,'fontsize',FS,'fontname','arial')
% xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
% ylabel('Power (dB)','fontsize',FS,'fontname','arial')
% title(gca,'\bf C','fontsize',FS2,'fontname','arial','position',[0.1 125])

%
% here reshape the data so each time point is 10x10 matrix
%  NPoints2DFFT = 30;
%  SpatialFreq = linspace(0,SpatialFreqMax,NPoints2DFFT);
% 
MatrixData = zeros(size(FiltData,1),10,10);
% DataFFT = zeros(size(FiltData,1),NPoints2DFFT,NPoints2DFFT);
 for n=1:size(FiltData,1)
     MatrixDataTemp = reshape(FiltData(n,:),10,10);
     MatrixDataTemp = inpaint_nans(MatrixDataTemp);            % interpolate missing data points
     MatrixData(n,:,:) = MatrixDataTemp;
 
    % MeanObservation = mean(mean(MatrixDataTemp));
    % DataFFT(n,:,:) = 20*log10(abs(fft2(MatrixDataTemp-MeanObservation,NPoints2DFFT,NPoints2DFFT )));
 end
clear FiltData

%%
% calc and plot the 2D cross-correlation
disp('finding cross correlations')

% EndSample = SzEnd*FsDec;
EndSample = SzStart*FsDec;

autocorr = zeros(EndSample,2*size(MatrixData,2)-1,2*size(MatrixData,2)-1);
xcorr = autocorr;
future_obs = zeros(size(MatrixData,2),size(MatrixData,3));
FFT_padding = 3;
fft_obs  = zeros(EndSample,FFT_padding*size(MatrixData,2),FFT_padding*size(MatrixData,2));
for n=1:EndSample           %size(MatrixData,1)-1   
    current_obs = future_obs;
    mean_current_obs = mean(mean(current_obs,2),1);
    future_obs = squeeze(MatrixData(n+1,:,:));
    fft_obs(n,:,:) = 20*log10( abs( fft2(current_obs-mean_current_obs, ...
        FFT_padding*size(MatrixData,2), FFT_padding*size(MatrixData,3)) ) );
%     clims = [0 70];
%     imagesc(squeeze(fft_obs(n,:,:)),clims)
%     axis xy
%     axis square
%     drawnow
%     pause
    autocorr(n,:,:) = xcorr2(current_obs, current_obs);
    xcorr(n,:,:) = xcorr2(future_obs, current_obs);
end

%%
% for plotting
% ~~~~~~~
FS_Label = 10;          % fontsize for the axis label
FS_Tick = 8;                % fontsize for the ticks
MS = 10;                     % marker size
LW = 1;
plotwidth = 8.3;        % cm
plotheight = 6.5;

Delta = ElectrodeSpacing;
varsigma = 0.56;
xi = 0.9;

% for n=1:size(fft_obs,1)
%     clims = [0 70];
%     imagesc(squeeze(fft_obs(n,:,:)),clims)    
%     drawnow
% end

mean_FFT = squeeze(mean(fft_obs(2:end,:,:),1));
filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/FFTObs.pdf';
figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
    'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
nu = linspace(0,1/Delta,size(mean_FFT,1));
CLIM = [30 70];
imagesc(nu,nu,mean_FFT)%,CLIM);
xlabel('Spatial Freq','fontsize',FS_Label)
ylabel('Spatial Freq','fontsize',FS_Label)
set(gca,'fontsize',FS_Tick)
axis square
axis xy
colorbar
% colormap hot
drawnow

R_yy = squeeze(mean( autocorr(2:end,:,:) ,1));
R_yy_plus_1 = squeeze(mean( xcorr(2:end,:,:) ,1));

mean_obs_noise = zeros(size(R_yy));
sigma_varepsilon = 0;
mean_obs_noise(floor(size(R_yy,1)/2)+1,floor(size(R_yy,2)/2)+1) = sigma_varepsilon;


% figure
% imagesc(R_yy)
% 
% figure
% imagesc(R_yy_plus_1)
% 
% figure
% imagesc(R_yy_plus_1 - xi*(R_yy-mean_obs_noise))

LHS = (R_yy_plus_1 - xi*(R_yy-mean_obs_noise)) ;
DeconvMode = 3;
if DeconvMode == 1
    R_yy_conv_mat = convmtx2(R_yy-mean_obs_noise,19,19);
    LHS_vect = LHS(:);
    w_vect = linsolve(full(R_yy_conv_mat)',LHS_vect);
    w_est = reshape(w_vect,37,37)*xi/(TsDec*varsigma);
    
    figure
    clim = [-5 10];
    imagesc(w_est,clim)
    colorbar
    axis xy
    axis square
    
    %%
    plot_fft_w_est = fft2(w_est-mean(mean(w_est)));      % need to demean
    fft_w_est = fft2(w_est);
    shifted_fft_w_est = fftshift(fft_w_est);
    
    figure
    clim = [140 160];
    clim = [40 45];
    nu = linspace(0,1/Delta,size(plot_fft_w_est,1));
    imagesc(nu,nu,20*log10(abs(plot_fft_w_est)),clim)
    axis square
    axis xy
    colorbar
    
    NSampsX = 17;
    NSampsY = 17;
    MidPoint = 19;
    OS = 0;
    temp1 = shifted_fft_w_est(MidPoint-NSampsX-OS:MidPoint+NSampsX, MidPoint-NSampsY-OS:MidPoint+NSampsY);
    
    temp2 = zeros(size(shifted_fft_w_est));
    temp2(MidPoint-NSampsX-OS:MidPoint+NSampsX,MidPoint-NSampsY-OS:MidPoint+NSampsY) = temp1;
%     Taper = 0.25;
%     TwoDWindow = tukeywin(size(temp2,1), Taper)*tukeywin(size(temp2,2), Taper)';
%     temp2 = temp2.*TwoDWindow;
%     temp2 = TwoDWindow;

    figure
    imagesc(abs(temp2))
    
    fft_w_est_thresh = ifftshift(temp2);
    
    figure
    imagesc(abs(fft_w_est_thresh),clim)
    w_est_thresh = (real(ifft2(fft_w_est_thresh)));
    figure,imagesc(w_est_thresh)
    axis square
    colorbar
    
    figure
    plot(w_est_thresh(19,:))
    %%
elseif DeconvMode == 2
    R_yy_conv_mat = convmtx2(R_yy,19,19);
    LHS_vect = LHS(:);
    inv_conv_mat = pinv(full(R_yy_conv_mat));        % took 2757 s
    w_vect = inv_conv_mat' * LHS_vect;
    w_est = reshape(w_vect,37,37)*xi/(TsDec*varsigma);
elseif DeconvMode == 3
    S_yy = fft2(R_yy-mean_obs_noise,19,19); 
    S_LHS = fft2(LHS,19,19);               %auto and noise free
    ratio = S_LHS ./ S_yy;
    shifted_ratio = fftshift(ratio);
    NSamps = 4;
    MidPoint = 10;
    OS = 0;
    temp1 = shifted_ratio(MidPoint-NSamps-OS:MidPoint+NSamps, MidPoint-NSamps-OS:MidPoint+NSamps);
    temp2 = zeros(size(ratio));
    temp2(MidPoint-NSamps-OS:MidPoint+NSamps,MidPoint-NSamps-OS:MidPoint+NSamps) = temp1;
    ratio_thresh = ifftshift(temp2);
    w_est = fftshift( (ifft2(ratio)) ) / (Delta^2*varsigma*TsDec);

    filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/FFTKernelEstimateFull.pdf';
    figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
        'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
    nu = linspace(0,2*Delta*2,size(ratio,1));
    imagesc(nu,nu,abs(ratio)/ (varsigma*TsDec));
    xlabel('Spatial Freq','fontsize',FS_Label)
    ylabel('Spatial Freq','fontsize',FS_Label)
    set(gca,'xtick',[0 1 2],'ytick',[0 1 2],'fontsize',FS_Tick)
    axis square
    axis xy
    colorbar
    colormap hot
    drawnow

    filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/FFTKernelEstimateThreshold.pdf';
    figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
        'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
    nu = linspace(0,2*Delta*2,size(ratio,1));
    disp(['cutoff freq: ' num2str(nu(NSamps)) ' cycles / mm'])
    % CLIM = [880 895];
    imagesc(nu,nu,abs(ratio_thresh)/ (varsigma*TsDec))%,CLIM);
    xlabel('Spatial Freq','fontsize',FS_Label)
    ylabel('Spatial Freq','fontsize',FS_Label)
    set(gca,'xtick',[0 Delta*2 2*Delta*2],'ytick',[0 Delta*2 2*Delta*2],'fontsize',FS_Tick)
    axis square
    axis xy
    colorbar
    colormap hot
    drawnow
end

%%
filename = '/Users/dean/Projects/BrainIDE/ltx/EMBCCorrelationAnalysisPaper/ltx/figures/KernelEstimate.pdf';
figure('units','centimeters','position',[0 0 plotwidth plotheight],'filename',filename,...
    'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
if DeconvMode == 1
    CLIMS = [-5 5];
    r = linspace(-7.2,7.2,size(w_est,1));
    imagesc(r,r,w_est,CLIMS)
elseif DeconvMode == 2
%     r = 0:Delta:(size(w_est,1)-1)*Delta;
%     r = r - r(floor(length(r)/2)+1);
    r = linspace(-7.2,7.2,size(w_est,1));
    CLIMS = [-5 5];
    imagesc(r,r,w_est,CLIMS)
elseif DeconvMode == 3
    r = linspace(-3.6,3.6,size(w_est,1));

   CLIMS = [-40 60];
   imagesc(r,r,w_est,CLIMS)
end

xlabel('Space','fontsize',FS_Label)
ylabel('Space','fontsize',FS_Label)
xlim([-3.6 3.6])
ylim([-3.6 3.6])
set(gca,'fontsize',FS_Tick,'xtick',[-3.6 -1.8 0 1.8 3.6],'ytick',[-3.6 -1.8 0 1.8 3.6])
axis square
axis xy
colorbar
colormap hot
drawnow

%%
% CrossCorrMeanPreSeizure = squeeze(mean(CrossCor(1:SzStart*FsDec,:,:),1)); 
% % CrossCorrMeanPreSeizure=CrossCorrMeanPreSeizure/max(max(CrossCorrMeanPreSeizure));
% CrossCorrMeanSeizure = squeeze(mean(CrossCor(SzStart*FsDec+1:SzEnd*FsDec,:,:),1)); 
% % CrossCorrMeanSeizure=CrossCorrMeanSeizure/max(max(CrossCorrMeanSeizure));
% CrossCorrMeanPostSeizure = squeeze(mean(CrossCor(SzEnd*FsDec:end,:,:),1)); 
% % CrossCorrMeanPostSeizure=CrossCorrMeanPostSeizure/max(max(CrossCorrMeanPostSeizure));
% 
% %% 
% % the plots for the xcorrs
% SpactialLocation = linspace(-1.8,1.8,size(CrossCorrMeanPreSeizure,1));
% HeightOffset = 1;
% HeigthScale = 0.5;
% ColorbarLims = [-0.1 0.3];
% figure('units','centimeters','position',[2,2,TwoColumnWidth,6],'renderer','painters',...
%     'filename',CrossCorrFig2D)
% subplot(131)
% imagesc(SpactialLocation,SpactialLocation,log10(1+CrossCorrMeanPreSeizure),ColorbarLims)
% xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% title('\bf A','fontsize',FS2,'fontname','arial','position',[-2.5 2.8])
% axis square
% CB = colorbar('units','centimeters','location','northoutside');
% Pos = get(CB,'position');
% set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% 
% subplot(132)
% imagesc(SpactialLocation,SpactialLocation,log10(1+CrossCorrMeanSeizure),ColorbarLims)
% xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% title('\bf B','fontsize',FS2,'fontname','arial','position',[-2.5 2.8])
% axis square
% CB = colorbar('units','centimeters','location','northoutside');
% Pos = get(CB,'position');
% set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% 
% subplot(133)
% imagesc(SpactialLocation,SpactialLocation,log10(1+CrossCorrMeanPostSeizure),ColorbarLims)
% xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% title('\bf C','fontsize',FS2,'fontname','arial','position',[-2.5 2.8])
% axis square
% CB = colorbar('units','centimeters','location','northoutside');
% Pos = get(CB,'position');
% set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% 
% %%
% % check for homogeneity 
% disp('finding cross correlations to test for homogeniety')
% NElectrodes = 6;
% % HomoCrossCor1 = zeros(size(MatrixData,1), 2*NElectrodes-1, 2*NElectrodes-1);
% % HomoCrossCor2 = zeros(size(MatrixData,1), 2*NElectrodes-1, 2*NElectrodes-1);
% % HomoCrossCor3 = zeros(size(MatrixData,1), 2*NElectrodes-1, 2*NElectrodes-1);
% % HomoCrossCor4 = zeros(size(MatrixData,1), 2*NElectrodes-1, 2*NElectrodes-1);
% HomoCrossCor1 = zeros(size(MatrixData,1), 15, 15);
% HomoCrossCor2 = zeros(size(MatrixData,1), 15, 15);
% HomoCrossCor3 = zeros(size(MatrixData,1), 15,15);
% HomoCrossCor4 = zeros(size(MatrixData,1), 15,15);
% 
% for n=1:size(MatrixData,1)-1  
% %     f1 = 1./(1+exp(varsigma*(v0 - squeeze(MatrixData(n,1:NElectrodes,1:NElectrodes)))));
% %     f2 = 1./(1+exp(varsigma*(v0 - squeeze(MatrixData(n,end-NElectrodes+1:end,end-NElectrodes+1:end)))));
% %     f3 = 1./(1+exp(varsigma*(v0 - squeeze(MatrixData(n,1:NElectrodes,end-NElectrodes+1:end)))));
% %     f4 = 1./(1+exp(varsigma*(v0 - squeeze(MatrixData(n,end-NElectrodes+1:end,1:NElectrodes)))));
%     
%     f1 = squeeze(MatrixData(n,1:NElectrodes,1:NElectrodes));
%     f2 = squeeze(MatrixData(n,end-NElectrodes+1:end,end-NElectrodes+1:end));
%     f3 = squeeze(MatrixData(n,1:NElectrodes,end-NElectrodes+1:end));
%     f4 = squeeze(MatrixData(n,end-NElectrodes+1:end,1:NElectrodes));    
%     
% %     HomoCrossCor1(n,:,:) = normxcorr2(squeeze(MatrixData(n+1,1:NElectrodes,1:NElectrodes)), f1);  % top left corner
% %     HomoCrossCor2(n,:,:) = normxcorr2(squeeze(MatrixData(n+1,end-NElectrodes+1:end,end-NElectrodes+1:end)), f2);   % bottom right corner
% %     HomoCrossCor3(n,:,:) = normxcorr2(squeeze(MatrixData(n+1,1:NElectrodes,end-NElectrodes+1:end)), f3);  % bottom left corner
% %     HomoCrossCor4(n,:,:) = normxcorr2(squeeze(MatrixData(n+1,end-NElectrodes+1:end,1:NElectrodes)), f4);        % top right corner
% %     
%     HomoCrossCor1(n,:,:) = normxcorr2(f1,squeeze(MatrixData(n+1,:,:)));         % top left corner
%     HomoCrossCor2(n,:,:) = normxcorr2(f2,squeeze(MatrixData(n+1,:,:)));         % bottom right corner
%     HomoCrossCor3(n,:,:) = normxcorr2(f3,squeeze(MatrixData(n+1,:,:)));         % bottom left corner
%     HomoCrossCor4(n,:,:) = normxcorr2(f4,squeeze(MatrixData(n+1,:,:)));         % top right corner
% end
% HomoCrossCorrMean1 = squeeze(mean(HomoCrossCor1,1)); 
% % HomoCrossCorrMean1 = HomoCrossCorrMean1/max(max(HomoCrossCorrMean1));
% HomoCrossCorrMean2 = squeeze(mean(HomoCrossCor2,1)); 
% % HomoCrossCorrMean2 = HomoCrossCorrMean2/max(max(HomoCrossCorrMean2));
% HomoCrossCorrMean3 = squeeze(mean(HomoCrossCor3,1)); 
% % HomoCrossCorrMean3 = HomoCrossCorrMean3/max(max(HomoCrossCorrMean3));
% HomoCrossCorrMean4 = squeeze(mean(HomoCrossCor4,1)); 
% % HomoCrossCorrMean4 = HomoCrossCorrMean4/max(max(HomoCrossCorrMean4));
% 
% %%
% % make all the plots to check the homogeneity
% HeightOffset = 0.9;     % this is for the colorbar positioning
% HeigthScale = 0.6;      % so is this
% figure('units','centimeters','position',[2,2,OneColumnWidth,OneColumnWidth],...
%      'filename',HomoCrossCorrFig)
% 
% % index the space over which the correlation is made
% Space1 = linspace(-2,ElectrodeSpacing*(NElectrodes-1)-2,size(HomoCrossCorrMean1,2));
% Space2 = linspace((10-NElectrodes+1)*ElectrodeSpacing-2,2,size(HomoCrossCorrMean1,2));
% ColorLims = [-0.5 1];
% subplot(223)
% imagesc(Space1, Space1,log10(1+HomoCrossCorrMean1))%,ColorLims)
% xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% title('\bf C','fontsize',FS2,'fontname','arial','position',[-2.5 0.25])
% axis square
% CB = colorbar('units','centimeters','location','northoutside');
% Pos = get(CB,'position');
% set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% 
% subplot(222),imagesc(Space2,Space1,log10(1+HomoCrossCorrMean2))%,ColorLims)
% xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% % ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% title('\bf D','fontsize',FS2,'fontname','arial','position',[-0.5 0.25])
% axis square
% CB = colorbar('units','centimeters','location','northoutside');
% Pos = get(CB,'position');
% set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% 
% subplot(224),imagesc(Space1,Space2,log10(1+HomoCrossCorrMean3))%,ColorLims)
% % xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% % ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% % title('\bf A','fontsize',FS2,'fontname','arial','position',[-2.5 2.8])
% % axis square
% % CB = colorbar('units','centimeters','location','northoutside');
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% % 
% % subplot(132)
% % imagesc(SpactialLocation,SpactialLocation,log10(1+CrossCorrMeanSeizure),ColorbarLims)
% % xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% % ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% % title('\bf B','fontsize',FS2,'fontname','arial','position',[-2.5 2.8])
% % axis square
% % CB = colorbar('units','centimeters','location','northoutside');
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% % 
% % subplot(133)
% % imagesc(SpactialLocation,SpactialLocation,log10(1+CrossCorrMeanPostSeizure),ColorbarLims)
% % xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% % ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% % title('\bf C','fontsize',FS2,'fontname','arial','position',[-2.5 2.8])
% % axis square
% % CB = colorbar('units','centimeters','location','northoutside');
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% % 
% % %%
% % % check for homogeneity 
% % disp('finding cross correlations to test for homogeniety')
% % NElectrodes = 6;
% % % HomoCrossCor1 = zeros(size(MatrixData,1), 2*NElectrodes-1, 2*NElectrodes-1);
% % % HomoCrossCor2 = zeros(size(MatrixData,1), 2*NElectrodes-1, 2*NElectrodes-1);
% % % HomoCrossCor3 = zeros(size(MatrixData,1), 2*NElectrodes-1, 2*NElectrodes-1);
% % % HomoCrossCor4 = zeros(size(MatrixData,1), 2*NElectrodes-1, 2*NElectrodes-1);
% % HomoCrossCor1 = zeros(size(MatrixData,1), 15, 15);
% % HomoCrossCor2 = zeros(size(MatrixData,1), 15, 15);
% % HomoCrossCor3 = zeros(size(MatrixData,1), 15,15);
% % HomoCrossCor4 = zeros(size(MatrixData,1), 15,15);
% % 
% % for n=1:size(MatrixData,1)-1  
% % %     f1 = 1./(1+exp(varsigma*(v0 - squeeze(MatrixData(n,1:NElectrodes,1:NElectrodes)))));
% % %     f2 = 1./(1+exp(varsigma*(v0 - squeeze(MatrixData(n,end-NElectrodes+1:end,end-NElectrodes+1:end)))));
% % %     f3 = 1./(1+exp(varsigma*(v0 - squeeze(MatrixData(n,1:NElectrodes,end-NElectrodes+1:end)))));
% % %     f4 = 1./(1+exp(varsigma*(v0 - squeeze(MatrixData(n,end-NElectrodes+1:end,1:NElectrodes)))));
% %     
% %     f1 = squeeze(MatrixData(n,1:NElectrodes,1:NElectrodes));
% %     f2 = squeeze(MatrixData(n,end-NElectrodes+1:end,end-NElectrodes+1:end));
% %     f3 = squeeze(MatrixData(n,1:NElectrodes,end-NElectrodes+1:end));
% %     f4 = squeeze(MatrixData(n,end-NElectrodes+1:end,1:NElectrodes));    
% %     
% % %     HomoCrossCor1(n,:,:) = normxcorr2(squeeze(MatrixData(n+1,1:NElectrodes,1:NElectrodes)), f1);  % top left corner
% % %     HomoCrossCor2(n,:,:) = normxcorr2(squeeze(MatrixData(n+1,end-NElectrodes+1:end,end-NElectrodes+1:end)), f2);   % bottom right corner
% % %     HomoCrossCor3(n,:,:) = normxcorr2(squeeze(MatrixData(n+1,1:NElectrodes,end-NElectrodes+1:end)), f3);  % bottom left corner
% % %     HomoCrossCor4(n,:,:) = normxcorr2(squeeze(MatrixData(n+1,end-NElectrodes+1:end,1:NElectrodes)), f4);        % top right corner
% % %     
% %     HomoCrossCor1(n,:,:) = normxcorr2(f1,squeeze(MatrixData(n+1,:,:)));         % top left corner
% %     HomoCrossCor2(n,:,:) = normxcorr2(f2,squeeze(MatrixData(n+1,:,:)));         % bottom right corner
% %     HomoCrossCor3(n,:,:) = normxcorr2(f3,squeeze(MatrixData(n+1,:,:)));         % bottom left corner
% %     HomoCrossCor4(n,:,:) = normxcorr2(f4,squeeze(MatrixData(n+1,:,:)));         % top right corner
% % end
% % HomoCrossCorrMean1 = squeeze(mean(HomoCrossCor1,1)); 
% % % HomoCrossCorrMean1 = HomoCrossCorrMean1/max(max(HomoCrossCorrMean1));
% % HomoCrossCorrMean2 = squeeze(mean(HomoCrossCor2,1)); 
% % % HomoCrossCorrMean2 = HomoCrossCorrMean2/max(max(HomoCrossCorrMean2));
% % HomoCrossCorrMean3 = squeeze(mean(HomoCrossCor3,1)); 
% % % HomoCrossCorrMean3 = HomoCrossCorrMean3/max(max(HomoCrossCorrMean3));
% % HomoCrossCorrMean4 = squeeze(mean(HomoCrossCor4,1)); 
% % % HomoCrossCorrMean4 = HomoCrossCorrMean4/max(max(HomoCrossCorrMean4));
% % 
% % %%
% % % make all the plots to check the homogeneity
% % HeightOffset = 0.9;     % this is for the colorbar positioning
% % HeigthScale = 0.6;      % so is this
% % figure('units','centimeters','position',[2,2,OneColumnWidth,OneColumnWidth],...
% %      'filename',HomoCrossCorrFig)
% % 
% % % index the space over which the correlation is made
% % Space1 = linspace(-2,ElectrodeSpacing*(NElectrodes-1)-2,size(HomoCrossCorrMean1,2));
% % Space2 = linspace((10-NElectrodes+1)*ElectrodeSpacing-2,2,size(HomoCrossCorrMean1,2));
% % ColorLims = [-0.5 1];
% % subplot(223)
% % imagesc(Space1, Space1,log10(1+HomoCrossCorrMean1))%,ColorLims)
% % xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% % ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% % title('\bf C','fontsize',FS2,'fontname','arial','position',[-2.5 0.25])
% % axis square
% % CB = colorbar('units','centimeters','location','northoutside');
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% % 
% % subplot(222),imagesc(Space2,Space1,log10(1+HomoCrossCorrMean2))%,ColorLims)
% % xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% % % ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% % title('\bf D','fontsize',FS2,'fontname','arial','position',[-0.5 0.25])
% % axis square
% % CB = colorbar('units','centimeters','location','northoutside');
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% % 
% % subplot(224),imagesc(Space1,Space2,log10(1+HomoCrossCorrMean3))%,ColorLims)
% % % xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% % ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% % title('\bf A','fontsize',FS2,'fontname','arial','position',[-2.5 2.25])
% % axis square
% % CB = colorbar('units','centimeters','location','northoutside');
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% % 
% % subplot(221),imagesc(Space2,Space2,HomoCrossCorrMean4)%,ColorLims)
% % % xlabel('Space (mm)','fontsize',FS,'fontname','arial')
% % % ylabel('Space (mm)','fontsize',FS,'fontname','arial')
% % title('\bf B','fontsize',FS2,'fontname','arial','position',[-0.5 2.25])
% % axis square
% % CB = colorbar('units','centimeters','location','northoutside');
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
% % 
% % %%
% % % plot the results of the fft analysis
% % 
% % figure('units','centimeters','position',[2,2,TwoColumnWidth,6],'renderer','painters',...
% %     'filename',SpatialFreqFig)
% % ColorBarLims = [30, 65];
% % HeightOffset = 1;
% % HeigthScale = 0.5;
% % MeanFFT1 = squeeze( mean( DataFFT(1:SzStart*FsDec,:,:),1));
% % subplot(131),imagesc( SpatialFreq,SpatialFreq,MeanFFT1,ColorBarLims )
% % axis square
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial','xtick',[0 0.5 1 1.5 2 2.5],'ytick',[0 0.5 1 1.5 2 2.5])
% % CB = colorbar('units','centimeters','location','northoutside');
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % xlabel('\nu (Hz)','fontsize',FS,'fontname','arial')
% % ylabel('\nu (Hz)','fontsize',FS,'fontname','arial')
% % title('\bf A','fontsize',FS2,'fontname','arial','position',[-0.5 2.8])
% % 
% % MeanFFT2 = squeeze( mean( DataFFT(SzStart*FsDec+1:SzEnd*FsDec,:,:),1));
% % subplot(132),imagesc( SpatialFreq,SpatialFreq,MeanFFT2,ColorBarLims )
% % axis square
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial','xtick',[0 0.5 1 1.5 2 2.5],'ytick',[0 0.5 1 1.5 2 2.5])
% % CB = colorbar('units','centimeters','location','northoutside');%[Pos(1) Pos(2)+Pos(4)+.1 Pos(3) 1])
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % xlabel('\nu (Hz)','fontsize',FS,'fontname','arial')
% % ylabel('\nu (Hz)','fontsize',FS,'fontname','arial')
% % title('\bf B','fontsize',FS2,'fontname','arial','position',[-0.5 2.8])
% % 
% % MeanFFT3 = squeeze( mean( DataFFT(SzEnd*FsDec+1:end,:,:),1));
% % subplot(133),imagesc( SpatialFreq,SpatialFreq,MeanFFT3,ColorBarLims )
% % axis square
% % set(gca,'fontsize',FS,'YDir','normal','fontname','arial','xtick',[0 0.5 1 1.5 2 2.5],'ytick',[0 0.5 1 1.5 2 2.5])
% % CB = colorbar('units','centimeters','location','northoutside');%[Pos(1) Pos(2)+Pos(4)+.1 Pos(3) 1])
% % Pos = get(CB,'position');
% % set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
% % xlabel('\nu (Hz)','fontsize',FS,'fontname','arial')
% % ylabel('\nu (Hz)','fontsize',FS,'fontname','arial')
% % title('\bf C','fontsize',FS2,'fontname','arial','position',[-0.5 2.8])
% % 
% % %%
% % % % plot cross section of the spatial frequencies
% % % % ~~~~~~~~~~~~~~~~~~~~~~~~~~
% % % MinPower = 25;
% % % MaxPower = 65;
% % % figure('units','centimeters','position',[2,2,TwoColumnWidth,6],...
% % %     'filename',SpatialFreqCrossSectFig)
% % % FFTCrossSect1 = diag(MeanFFT1);
% % % subplot(131),plot(SpatialFreq,FFTCrossSect1)
% % % axis square
% % % xlim([SpatialFreq(1) SpatialFreq(end)])
% % % ylim([MinPower MaxPower])
% % % set(gca,'fontsize',FS,'fontname','arial')
% % % xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
% % % ylabel('Power (dB)','fontsize',FS,'fontname','arial')
% % % title(gca,'\bf A','fontsize',FS2,'fontname','arial','position',[0 67])
% % % ThreeDBPoint = max(FFTCrossSect1(2:floor(length(FFTCrossSect1)/2)))-3;
% % % I = find(FFTCrossSect1(2:floor(length(FFTCrossSect1)/2))<ThreeDBPoint,1,'first')+1;
% % % hold on
% % % plot(SpatialFreq(I),FFTCrossSect1(I),'r.')
% % % text(SpatialFreq(I),FFTCrossSect1(I),['\bullet\leftarrow \nu_c \approx ' num2str(round(100*SpatialFreq(I))/100) ' Hz'],'fontsize',FS)
% % % hold off
% % % 
% % % FFTCrossSect2 = diag(MeanFFT2);
% % % subplot(132),plot(SpatialFreq,FFTCrossSect2)
% % % axis square
% % % xlim([SpatialFreq(1) SpatialFreq(end)])
% % % ylim([MinPower MaxPower])
% % % set(gca,'fontsize',FS,'fontname','arial')
% % % xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
% % % ylabel('Power (dB)','fontsize',FS,'fontname','arial')
% % % title(gca,'\bf B','fontsize',FS2,'fontname','arial','position',[0 67])
% % % ThreeDBPoint = max(FFTCrossSect2(2:floor(length(FFTCrossSect2)/2)))-3;
% % % I = find(FFTCrossSect2(2:floor(length(FFTCrossSect2)/2))<ThreeDBPoint,1,'first')+1;
% % % hold on
% % % plot(SpatialFreq(I),FFTCrossSect2(I),'r.')
% % % text(SpatialFreq(I),FFTCrossSect2(I),['\bullet\leftarrow \nu_c \approx ' num2str(round(100*SpatialFreq(I))/100) ' Hz'],'fontsize',FS)
% % % hold off
% % % 
% % % FFTCrossSect3 = diag(MeanFFT3);
% % % subplot(133),plot(SpatialFreq,FFTCrossSect3)
% % % axis square
% % % xlim([SpatialFreq(1) SpatialFreq(end)])
% % % ylim([MinPower MaxPower])
% % % set(gca,'fontsize',FS,'fontname','arial')
% % % xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
% % % ylabel('Power (dB)','fontsize',FS,'fontname','arial')
% % % title(gca,'\bf C','fontsize',FS2,'fontname','arial','position',[0 67])
% % % ThreeDBPoint = max(FFTCrossSect3(2:floor(length(FFTCrossSect3)/2)))-3;
% % % I = find(FFTCrossSect3(2:floor(length(FFTCrossSect3)/2))<ThreeDBPoint,1,'first')+1;
% % % hold on
% % % plot(SpatialFreq(I),FFTCrossSect3(I),'r.')
% % % text(SpatialFreq(I),FFTCrossSect3(I),['\bullet\leftarrow \nu_c \approx ' num2str(round(100*SpatialFreq(I))/100) ' Hz'],'fontsize',FS)
% % % hold off
% % % 
% % % %%
% % % 
% this bit plot the observed field
% ~~~~~~~~~~~~~~~~~~
% climit = 80;
% SampleStep = 2;
% SampleStart = 40390;
% figure('units','centimeters','position',[0 0 30 30])
% for n=SampleStart:SampleStep:size(FiltData,1)
%     imagesc(squeeze(MatrixData(n,:,:) ))
%     shading('interp')
% %     title(['Sample = ' num2str(n)])
% %     colorbar
%     axis off
%     axis square
%     drawnow
%     pause
% end
%%
HeightOffset = 1;
HeigthScale = 0.5;
figure('units','centimeters','position',[2,2,TwoColumnWidth,TwoColumnWidth],'renderer','painters',...
    'filename',FieldObservationsFig)
climit = 50;
StartSample = floor(SzEnd*FsDec)+120;
StartTime = SzEnd+120/FsDec
SampleStep = 10;
PlotSample = StartSample;

subplot(331),imagesc(squeeze(MatrixData(PlotSample,:,:) ),[-climit climit])
axis square
set(gca,'fontsize',FS,'fontname','arial','units','centimeters')
PlotSample = PlotSample + SampleStep;
title('\bf A','fontsize',FS2,'fontname','arial','position',[-0.5 0.8])  % 11

subplot(332),imagesc(squeeze(MatrixData(PlotSample,:,:) ),[-climit climit])
axis square
title('\bf B','fontsize',FS2,'fontname','arial','position',[-0.5 0.8])
CB = colorbar('units','centimeters','location','northoutside');%[Pos(1) Pos(2)+Pos(4)+.1 Pos(3) 1])
Pos = get(CB,'position');
set(CB, 'position', [Pos(1) Pos(2)+1.3*HeightOffset Pos(3) 2*HeigthScale*Pos(4)] )
set(gca,'fontsize',FS,'fontname','arial')

PlotSample = PlotSample + SampleStep;
subplot(333),imagesc(squeeze(MatrixData(PlotSample,:,:) ),[-climit climit])
axis square
set(gca,'fontsize',FS,'fontname','arial')
title('\bf C','fontsize',FS2,'fontname','arial','position',[-0.5 0.8])

PlotSample = PlotSample + SampleStep;
subplot(334),imagesc(squeeze(MatrixData(PlotSample,:,:) ),[-climit climit])
axis square
set(gca,'fontsize',FS,'fontname','arial')
title('\bf D','fontsize',FS2,'fontname','arial','position',[-0.5 0.8])
ylabel('Space (Electrode Index)','fontsize',FS,'fontname','arial')

PlotSample = PlotSample + SampleStep;
subplot(335),imagesc(squeeze(MatrixData(PlotSample,:,:) ),[-climit climit])
axis square
set(gca,'fontsize',FS,'fontname','arial')
title('\bf E','fontsize',FS2,'fontname','arial','position',[-0.5 0.8])

PlotSample = PlotSample + SampleStep;
subplot(336),imagesc(squeeze(MatrixData(PlotSample,:,:) ),[-climit climit])
axis square
set(gca,'fontsize',FS,'fontname','arial')
title('\bf F','fontsize',FS2,'fontname','arial','position',[-0.5 0.8])

PlotSample = PlotSample + SampleStep;
subplot(337),imagesc(squeeze(MatrixData(PlotSample,:,:) ),[-climit climit])
axis square
set(gca,'fontsize',FS,'fontname','arial')
title('\bf G','fontsize',FS2,'fontname','arial','position',[-0.5 0.8])

PlotSample = PlotSample + SampleStep;
subplot(338),imagesc(squeeze(MatrixData(PlotSample,:,:) ),[-climit climit])
axis square
set(gca,'fontsize',FS,'fontname','arial')
title('\bf H','fontsize',FS2,'fontname','arial','position',[-0.5 0.8])
xlabel('Space (Electrode Index)','fontsize',FS,'fontname','arial')

PlotSample = PlotSample + SampleStep;
subplot(339),imagesc(squeeze(MatrixData(PlotSample,:,:) ),[-climit climit])
axis square
set(gca,'fontsize',FS,'fontname','arial')
title('\bf I','fontsize',FS2,'fontname','arial','position',[-0.5 0.8])

% %%
% plot channels as a time series
% ~~~~~~~~~~~~~~~~~~
% PlotOffset  = 500;                          % this just puts some space between channels
% OffsetMatrix = PlotOffset*(1:100);
% OffsetMatrix = repmat(OffsetMatrix,size(FiltData,1),1);
% OffsetData = OffsetMatrix+FiltData;
% 
% StartChannel = 1;
% EndChannel = 100;
% StartTime = 7.5;
% EndTime = 9;
% plot(t(StartTime*FsDec:EndTime*FsDec),OffsetData(StartTime*FsDec:EndTime*FsDec,StartChannel:EndChannel),'k','linewidth',0.5)
% axis tight
% axis off
% 
% figure('units','centimeters','position',[2,2,TwoColumnWidth,6],...
%     'filename',LFPsFig)
% StartChannel = 2;
% EndChannel = 9;
% plot(t,OffsetData(:,StartChannel:EndChannel),'k','linewidth',0.5)
% axis tight
% axis off
% 
% draw on the scale bars
% ~~~~~~~~~~~~~~
% Pos = get(gca,'position');
% TimeTotal = t(end);
% PlotWidth = Pos(3);
% LineTime = 5;        % seconds
% LineLength = PlotWidth*LineTime/TimeTotal;
% 
% MaxAmp = max(OffsetData(:,EndChannel));
% MinAmp = min(OffsetData(:,StartChannel));
% AmpRange = MaxAmp - MinAmp;
% 
% LineAmp = 500;
% PlotHeight = Pos(4);
% AmpLength = PlotHeight*LineAmp/AmpRange;
% 
% ScaleBarXOffset = 0;
% ScaleBarYOffset = -0.05;
% annotation('line',[Pos(2) Pos(2)+LineLength]+ScaleBarXOffset,[Pos(1) Pos(1)]+ScaleBarYOffset,'linewidth',3)
% annotation('line',[Pos(2) Pos(2)]+ScaleBarXOffset,[Pos(1) Pos(1)+AmpLength]+ScaleBarYOffset,'linewidth',3)
% 
% TextBoxWidth = 0.1;
% TextBoxHeight = 0.05;
% annotation('textbox',[Pos(2)+0.02 Pos(1)-0.1 TextBoxWidth TextBoxHeight],...
%     'string',[num2str(LineTime) ' s'],'fontsize',FS,'fontname','arial','linestyle','none')
% annotation('textbox',[Pos(2)-0.085 Pos(1) TextBoxWidth TextBoxHeight],...
%     'string',[num2str(LineAmp) ' \muV'],'fontsize',FS,'fontname','arial','linestyle','none')
% ~~~~~~~~~~~~~~~
% 
% draw on the seizure markersSzStart
% ~~~~~~~~~~~~~~~~~~~~~~
% LineSzStart = PlotWidth*SzStart/TimeTotal;
% LineSzEnd = PlotWidth*SzEnd/TimeTotal;
% annotation('line',[LineSzStart LineSzStart]+Pos(1),[Pos(1) Pos(1)+Pos(3)],'linewidth',2,'color','red','linestyle',':')
% annotation('line',[LineSzEnd LineSzEnd]+Pos(1),[Pos(1) Pos(1)+Pos(3)],'linewidth',2,'color','blue','linestyle','--')
