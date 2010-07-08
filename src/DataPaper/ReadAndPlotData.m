clc
clear
close all

% read in neuroport data from .mat file
DataFile = '/Users/dean/Dropbox/iEEG Data Boston/MG29_Seizure22_LFP_Clip_Dean.mat';
Data = open('/Users/dean/Dropbox/iEEG Data Boston/MG29_Seizure22_LFP_Clip_Dean.mat');
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

% here reshape the data so each time point is 10x10 matrix
MatrixData = zeros(size(CARData,1),10,10);
for n=1:size(CARData,1)
    MatrixData(n,:,:) = reshape(CARData(n,:),10,10);
end

% this bit plot the observed field
% ~~~~~~~~~~~~~~~~~~
climit = 80;
for n=1:size(CARData,1)
    pcolor(squeeze(MatrixData(n,:,:) ))
%     shading('interp')
    caxis(gca,[-climit climit])
    title(['Sample = ' num2str(n)])
    drawnow
end

% plot channels as a time series
% ~~~~~~~~~~~~~~~~~~
PlotOffset  = 200;                          % this just puts some space between channels
OffsetMatrix = PlotOffset*(1:100);
OffsetMatrix = repmat(OffsetMatrix,size(CARData,1),1);
OffsetData = OffsetMatrix+CARData;
figure('units','normalized','position',[0 0 1 1])
% WindowSize = 50000;
% WindowStep = WindowSize/2;
% for n=WindowSize+1:WindowStep:size(OffsetData,1)
%     plot(OffsetData(n-WindowSize:n,:))
%     drawnow
%     pause
% end
plot(OffsetData)