% this script reads in the micro data and plot the times series and the
% temporal and spatial frequency properties.

clc
clear
close all
FS = 10;
FS2 = 12;

TwoColumnWidth = 17.35;     % PLoS figure width cm
Fs = 30e3;                              % sampling rate in Hz
FsDec = 5e3;                          % this is the sampling rate that we will use
TsDec = 1/FsDec;                    % the decimated sampling period
DecimationFactor = Fs/FsDec;

ElectrodeSpacing = 0.4;     % mm
SpatialFreqMax = 1/ElectrodeSpacing;

SzStart = 8;        % seconds
SzEnd = 34;         % seconds

% read in neuroport data from .mat file
DataFileLocation = '/Users/dean/Dropbox/iEEG Data Boston/MG29_Seizure22_LFP_Clip_Dean.mat';
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
        MappedData(:,n) = Data(:,DataMap(n)) - mean(Data(:,DataMap(n)));
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
% downsample to 1 kHz
FiltData = downsample(FiltData,DecimationFactor);
t = TsDec*(0:size(FiltData,1)-1);

%%
% get indexes for good channels
m=1;
for n=1:size(FiltData,2)
    if ~ismember(n,ZeroChannels)
        GoodChannels(m) = n;
        m=m+1;
    end
end

%%
NFFTPoints = 150*1024;
YMax = 122;
FMax = 100;
figure('units','centimeters','position',[2,2,TwoColumnWidth,6])
TemporalFFT1 = 20*log10(abs(fft(FiltData(1:SzStart*FsDec,GoodChannels),NFFTPoints,1)));
TemporalFFT2 = 20*log10(abs(fft(FiltData(SzStart*FsDec+1:SzEnd*FsDec,GoodChannels),NFFTPoints,1)));
TemporalFFT3 = 20*log10(abs(fft(FiltData(SzEnd*FsDec+1:end,GoodChannels),NFFTPoints,1)));

f = linspace(0,FsDec,NFFTPoints);
subplot(131),semilogx(f,mean(TemporalFFT1,2))
xlim([0 FMax])
ylim([60 YMax])
axis square
set(gca,'fontsize',FS,'fontname','arial')
xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
ylabel('Power (dB)','fontsize',FS,'fontname','arial')
title('\bf A','fontsize',FS2,'fontname','arial','position',[0.1 125])

subplot(132),semilogx(f,mean(TemporalFFT2,2))
xlim([0 FMax])
ylim([60 YMax])
axis square
set(gca,'fontsize',FS,'fontname','arial')
xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
ylabel('Power (dB)','fontsize',FS,'fontname','arial')
title('\bf B','fontsize',FS2,'fontname','arial','position',[0.1 125])

subplot(133),semilogx(f,mean(TemporalFFT3,2))
xlim([0 FMax])
ylim([60 YMax])
axis square
set(gca,'fontsize',FS,'fontname','arial')
xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
ylabel('Power (dB)','fontsize',FS,'fontname','arial')
title(gca,'\bf C','fontsize',FS2,'fontname','arial','position',[0.1 125])

%%
% here reshape the data so each time point is 10x10 matrix
NPoints2DFFT = 30;
SpatialFreq = linspace(0,SpatialFreqMax,NPoints2DFFT);

MatrixData = zeros(size(FiltData,1),10,10);
DataFFT = zeros(size(FiltData,1),NPoints2DFFT,NPoints2DFFT);
for n=1:size(FiltData,1)
    MatrixDataTemp = reshape(FiltData(n,:),10,10);
    MatrixDataTemp = inpaint_nans(MatrixDataTemp);            % interpolate missing data points
    MeanObservation = mean(mean(MatrixDataTemp));
    DataFFT(n,:,:) = 20*log10(abs(fft2(MatrixDataTemp-MeanObservation,NPoints2DFFT,NPoints2DFFT )));
    MatrixData(n,:,:) = MatrixDataTemp;
end

%%
% figure,imagesc(squeeze(mean(abs(DataFFT),1)))
figure('units','centimeters','position',[2,2,TwoColumnWidth,6],'renderer','opengl')
ColorLims = [30, 65];
HeightOffset = 1;
HeigthScale = 0.5;
MeanFFT1 = squeeze( mean( DataFFT(1:SzStart*FsDec,:,:),1));
subplot(131),imagesc( SpatialFreq,SpatialFreq,MeanFFT1 )
axis square
set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
CB = colorbar('units','centimeters','location','northoutside');
Pos = get(CB,'position');
set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
xlabel('Hz','fontsize',FS,'fontname','arial')
ylabel('Hz','fontsize',FS,'fontname','arial')
title('\bf A','fontsize',FS2,'fontname','arial','position',[-0.5 2.8])

MeanFFT2 = squeeze( mean( DataFFT(SzStart*FsDec+1:SzEnd*FsDec,:,:),1));
subplot(132),imagesc( SpatialFreq,SpatialFreq,MeanFFT2 )
axis square
set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
CB = colorbar('units','centimeters','location','northoutside');%[Pos(1) Pos(2)+Pos(4)+.1 Pos(3) 1])
Pos = get(CB,'position');
set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
xlabel('Hz','fontsize',FS,'fontname','arial')
ylabel('Hz','fontsize',FS,'fontname','arial')
title('\bf B','fontsize',FS2,'fontname','arial','position',[-0.5 2.8])

MeanFFT3 = squeeze( mean( DataFFT(SzEnd*FsDec+1:end,:,:),1));
subplot(133),imagesc( SpatialFreq,SpatialFreq,MeanFFT3 )
axis square
set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
CB = colorbar('units','centimeters','location','northoutside');%[Pos(1) Pos(2)+Pos(4)+.1 Pos(3) 1])
Pos = get(CB,'position');
set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
xlabel('Hz','fontsize',FS,'fontname','arial')
ylabel('Hz','fontsize',FS,'fontname','arial')
title('\bf C','fontsize',FS2,'fontname','arial','position',[-0.5 2.8])

%%
% plot cross section of the spatial frequencies
% ~~~~~~~~~~~~~~~~~~~~~~~~~~
MinPower = 25;
MaxPower = 65;
figure('units','centimeters','position',[2,2,TwoColumnWidth,6])
FFTCrossSect1 = diag(MeanFFT1);
subplot(131),plot(SpatialFreq,FFTCrossSect1)
axis square
xlim([SpatialFreq(1) SpatialFreq(end)])
ylim([MinPower MaxPower])
set(gca,'fontsize',FS,'fontname','arial')
xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
ylabel('Power (dB)','fontsize',FS,'fontname','arial')
title(gca,'\bf A','fontsize',FS2,'fontname','arial','position',[0 67])
ThreeDBPoint = max(FFTCrossSect1(2:floor(length(FFTCrossSect1)/2)))-3;
I = find(FFTCrossSect1(2:floor(length(FFTCrossSect1)/2))<ThreeDBPoint,1,'first')+1;
hold on
plot(SpatialFreq(I),FFTCrossSect1(I),'r.')
text(SpatialFreq(I),FFTCrossSect1(I),['\bullet\leftarrow \nu_c \approx ' num2str(round(100*SpatialFreq(I))/100) ' Hz'],'fontsize',FS)
hold off

FFTCrossSect2 = diag(MeanFFT2);
subplot(132),plot(SpatialFreq,FFTCrossSect2)
axis square
xlim([SpatialFreq(1) SpatialFreq(end)])
ylim([MinPower MaxPower])
set(gca,'fontsize',FS,'fontname','arial')
xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
ylabel('Power (dB)','fontsize',FS,'fontname','arial')
title(gca,'\bf B','fontsize',FS2,'fontname','arial','position',[0 67])
ThreeDBPoint = max(FFTCrossSect2(2:floor(length(FFTCrossSect2)/2)))-3;
I = find(FFTCrossSect2(2:floor(length(FFTCrossSect2)/2))<ThreeDBPoint,1,'first')+1;
hold on
plot(SpatialFreq(I),FFTCrossSect2(I),'r.')
text(SpatialFreq(I),FFTCrossSect2(I),['\bullet\leftarrow \nu_c \approx ' num2str(round(100*SpatialFreq(I))/100) ' Hz'],'fontsize',FS)
hold off

FFTCrossSect3 = diag(MeanFFT3);
subplot(133),plot(SpatialFreq,FFTCrossSect3)
axis square
xlim([SpatialFreq(1) SpatialFreq(end)])
ylim([MinPower MaxPower])
set(gca,'fontsize',FS,'fontname','arial')
xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
ylabel('Power (dB)','fontsize',FS,'fontname','arial')
title(gca,'\bf C','fontsize',FS2,'fontname','arial','position',[0 67])
ThreeDBPoint = max(FFTCrossSect3(2:floor(length(FFTCrossSect3)/2)))-3;
I = find(FFTCrossSect3(2:floor(length(FFTCrossSect3)/2))<ThreeDBPoint,1,'first')+1;
hold on
plot(SpatialFreq(I),FFTCrossSect3(I),'r.')
text(SpatialFreq(I),FFTCrossSect3(I),['\bullet\leftarrow \nu_c \approx ' num2str(round(100*SpatialFreq(I))/100) ' Hz'],'fontsize',FS)
hold off

%%

% this bit plot the observed field
% ~~~~~~~~~~~~~~~~~~
% climit = 80;
% SampleStep = 2;
% SampleStart = 80000;
% figure
% for n=SampleStart:SampleStep:size(FiltData,1)
%     imagesc(squeeze(MatrixData(n,:,:) ),[-climit climit])
% %     shading('interp')
%     title(['Sample = ' num2str(n)])
% %     colorbar
%     drawnow
%     
% end
%%
figure('units','centimeters','position',[2,2,TwoColumnWidth,TwoColumnWidth],'renderer','opengl')
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

%%
% plot channels as a time series
% ~~~~~~~~~~~~~~~~~~
PlotOffset  = 500;                          % this just puts some space between channels
OffsetMatrix = PlotOffset*(1:100);
OffsetMatrix = repmat(OffsetMatrix,size(FiltData,1),1);
OffsetData = OffsetMatrix+FiltData;
figure('units','normalized','position',[0 0 1 1])

figure('units','centimeters','position',[2,2,TwoColumnWidth,6])
StartChannel = 2;
EndChannel = 6;
plot(t,OffsetData(:,StartChannel:EndChannel),'k')
axis tight
axis off

% draw on the scale bars
% ~~~~~~~~~~~~~~
Pos = get(gca,'position');
TimeTotal = t(end);
PlotWidth = Pos(3);
LineTime = 5;        % seconds
LineLength = PlotWidth*LineTime/TimeTotal;

MaxAmp = max(OffsetData(:,EndChannel));
MinAmp = min(OffsetData(:,StartChannel));
AmpRange = MaxAmp - MinAmp;

LineAmp = 500;
PlotHeight = Pos(4);
AmpLength = PlotHeight*LineAmp/AmpRange;

ScaleBarXOffset = 0;
ScaleBarYOffset = -0.05;
annotation('line',[Pos(2) Pos(2)+LineLength]+ScaleBarXOffset,[Pos(1) Pos(1)]+ScaleBarYOffset,'linewidth',3)
annotation('line',[Pos(2) Pos(2)]+ScaleBarXOffset,[Pos(1) Pos(1)+AmpLength]+ScaleBarYOffset,'linewidth',3)

TextBoxWidth = 0.1;
TextBoxHeight = 0.05;
annotation('textbox',[Pos(2)+0.02 Pos(1)-0.1 TextBoxWidth TextBoxHeight],...
    'string',[num2str(LineTime) ' s'],'fontsize',FS,'fontname','arial','linestyle','none')
annotation('textbox',[Pos(2)-0.085 Pos(1) TextBoxWidth TextBoxHeight],...
    'string',[num2str(LineAmp) ' \muV'],'fontsize',FS,'fontname','arial','linestyle','none')
% ~~~~~~~~~~~~~~~

% draw on the seizure markersSzStart
% ~~~~~~~~~~~~~~~~~~~~~~
LineSzStart = PlotWidth*SzStart/TimeTotal;
LineSzEnd = PlotWidth*SzEnd/TimeTotal;
annotation('line',[LineSzStart LineSzStart]+Pos(1),[Pos(1) Pos(1)+Pos(3)],'linewidth',1.5,'color','red','linestyle',':')
annotation('line',[LineSzEnd LineSzEnd]+Pos(1),[Pos(1) Pos(1)+Pos(3)],'linewidth',1.5,'color','blue','linestyle','--')
