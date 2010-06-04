% Run batch job
clc
clear
close all

RunningBatch = 1;
NRealisations = 80;
for Realisation=1:NRealisations
    tic
    disp(['Running realisation: ' num2str(Realisation) ' of ' num2str(NRealisations)])
    disp('generating data')
    GenerateData
    RunFilter
    t_realisation = toc
end