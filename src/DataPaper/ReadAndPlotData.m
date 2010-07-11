clc
clear
close all
FS = 10;
FS2 = 12;

TwoColumnWidth = 17.35;     % PLoS figure width cm
Fs = 30e3;          % sampling rate in Hz
FsDec = 5e3;
TsDec = 1/FsDec;
DecimationFactor = Fs/FsDec;

ElectrodeSpacing = 0.4;     % mm
SpatialFreqMax = 1/ElectrodeSpacing;
SpatialFreq = linspace(0,SpatialFreqMax,10);

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

% these are the channels that have loads of noise and will be useless
BadChannels = [13 15 19 24 26 35 42 46 47 54 56  57 67 81 85 87];
% get ride of the noisey guys
for n=1:length(BadChannels)
    MappedData(:,BadChannels(n)) = zeros(size(MappedData,1),1);
end

% now need to re-reference the data
% ~~~~~~~~~~~~~~~~~~~~~
CAR = repmat(mean(MappedData,2),1,100);
CARData = MappedData-CAR;                       % CAR = common average reference
clear MappedData

% re-assign the bad channels and the corners to nans so we can interpolate
ZeroChannels = [1 10 13 15 19 24 26 35 42 46 47 54 56  57 67 81 85 87 91 100];
% get ride of the noisey guys
for n=1:length(ZeroChannels)
    CARData(:,ZeroChannels(n)) = nan(size(CARData,1),1);
end

% low-pass filter data
Fc = 100;           % Cut-off frequency in Hz
Wn = Fc/(Fs/2);
FilterOrder = 2;
[b a] = butter(FilterOrder,Wn);
FiltData = filter(b,a,CARData);
clear CARData

% downsample to 1 kHz
FiltData = downsample(FiltData,DecimationFactor);
t = TsDec*(0:size(FiltData,1)-1);

% get indexes for good channels
m=1;
for n=1:size(FiltData,2)
    if ~ismember(n,ZeroChannels)
        GoodChannels(m) = n;
        m=m+1;
    end
end

NFFTPoints = 10*1024;
YMax = 105;
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
title('\bf A','fontsize',FS2,'fontname','arial','position',[0.2 100 17])

subplot(132),semilogx(f,mean(TemporalFFT2,2))
xlim([0 FMax])
ylim([60 YMax])
axis square
set(gca,'fontsize',FS,'fontname','arial')
xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
ylabel('Power (dB)','fontsize',FS,'fontname','arial')
title('\bf B','fontsize',FS2,'fontname','arial','position',[0.2 100 17])

subplot(133),semilogx(f,mean(TemporalFFT3,2))
xlim([0 FMax])
ylim([60 YMax])
axis square
set(gca,'fontsize',FS,'fontname','arial')
xlabel('Frequency (Hz)','fontsize',FS,'fontname','arial')
ylabel('Power (dB)','fontsize',FS,'fontname','arial')
title(gca,'\bf C','fontsize',FS2,'fontname','arial','position',[0.2 100 17])


% here reshape the data so each time point is 10x10 matrix
MatrixData = zeros(size(FiltData,1),10,10);
DataFFT = zeros(size(FiltData,1),10,10);
for n=1:size(FiltData,1)
    MatrixDataTemp = reshape(FiltData(n,:),10,10);
    MatrixDataTemp = inpaint_nans(MatrixDataTemp);            % interpolate missing data points
    MeanObservation = mean(mean(MatrixDataTemp));
    DataFFT(n,:,:) = 20*log10(abs(fft2(MatrixDataTemp-MeanObservation )));
    MatrixData(n,:,:) = MatrixDataTemp;
end

% figure,imagesc(squeeze(mean(abs(DataFFT),1)))
figure('units','centimeters','position',[2,2,TwoColumnWidth,6])
ColorLims = [30, 65];
HeightOffset = 1;
HeigthScale = 0.5;
subplot(131),imagesc( SpatialFreq,SpatialFreq,squeeze( mean( DataFFT(1:SzStart*FsDec,:,:),1)) )
axis square
set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
CB = colorbar('units','centimeters','location','northoutside');%[Pos(1) Pos(2)+Pos(4)+.1 Pos(3) 1])
Pos = get(CB,'position');
set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
xlabel('Hz','fontsize',FS,'fontname','arial')
ylabel('Hz','fontsize',FS,'fontname','arial')
title('\bf A','fontsize',FS2,'fontname','arial','position',[-0.5 2.8])

subplot(132),imagesc( SpatialFreq,SpatialFreq,squeeze( mean( DataFFT(SzStart*FsDec+1:SzEnd*FsDec,:,:),1)) )
axis square
set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
CB = colorbar('units','centimeters','location','northoutside');%[Pos(1) Pos(2)+Pos(4)+.1 Pos(3) 1])
Pos = get(CB,'position');
set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
xlabel('Hz','fontsize',FS,'fontname','arial')
ylabel('Hz','fontsize',FS,'fontname','arial')
title('\bf B','fontsize',FS2,'fontname','arial','position',[-0.5 2.8])

subplot(133),imagesc( SpatialFreq,SpatialFreq,squeeze( mean( DataFFT(SzEnd*FsDec+1:end,:,:),1)) )
axis square
set(gca,'fontsize',FS,'YDir','normal','fontname','arial')
CB = colorbar('units','centimeters','location','northoutside');%[Pos(1) Pos(2)+Pos(4)+.1 Pos(3) 1])
Pos = get(CB,'position');
set(CB, 'position', [Pos(1) Pos(2)+HeightOffset Pos(3) HeigthScale*Pos(4)] )
xlabel('Hz','fontsize',FS,'fontname','arial')
ylabel('Hz','fontsize',FS,'fontname','arial')
title('\bf C','fontsize',FS2,'fontname','arial','position',[-0.5 2.8])

% this bit plot the observed field
% ~~~~~~~~~~~~~~~~~~
climit = 80;
SampleStep = 1;
for n=1:SampleStep:size(FiltData,1)
    imagesc(squeeze(MatrixData(n,:,:) ),[-climit climit])
%     shading('interp')
    title(['Sample = ' num2str(n)])
%     colorbar
    drawnow
end

% plot channels as a time series
% ~~~~~~~~~~~~~~~~~~
PlotOffset  = 500;                          % this just puts some space between channels
OffsetMatrix = PlotOffset*(1:100);
OffsetMatrix = repmat(OffsetMatrix,size(FiltData,1),1);
OffsetData = OffsetMatrix+FiltData;
figure('units','normalized','position',[0 0 1 1])

% WindowSize = 50000;
% WindowStep = WindowSize/2;
% for n=WindowSize+1:WindowStep:size(OffsetData,1)
%     plot(OffsetData(n-WindowSize:n,:))
%     drawnow
%     pause
% end

figure('units','centimeters','position',[2,2,TwoColumnWidth,6])
StartChannel = 2;
EndChannel = 6;
plot(t,OffsetData(:,StartChannel:EndChannel),'k')
axis tight
axis off

% draw on the scale bars
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

% draw on the seizure markersSzStart
LineSzStart = PlotWidth*SzStart/TimeTotal;
LineSzEnd = PlotWidth*SzEnd/TimeTotal;
annotation('line',[LineSzStart LineSzStart]+Pos(1),[Pos(1) Pos(1)+Pos(3)],'linewidth',1.5,'color','red','linestyle',':')
annotation('line',[LineSzEnd LineSzEnd]+Pos(1),[Pos(1) Pos(1)+Pos(3)],'linewidth',1.5,'color','blue','linestyle','--')

xlabel('Time (s)','fontsize',FS,'fontname','arial')
ylabel('Amplitude','fontsize',FS,'fontname','arial')