clc
close all

GenerateData

Fs = 1e3;
f = linspace(0,Fs/2,size(y,1));

for n=1:NSensors
    FFTy(n,:) = abs(fft(detrend(y(:,n))));
    
end

FFTave = mean(FFTy,1);

figure
IndexMax = 200;
FS = 16;
semilogx(f(1:IndexMax),20*log10(FFTave(1:IndexMax)),'k')
set(gca,'fontsize',FS)
xlim([0 100])
xlabel('Frequency (Hz)','fontsize',FS)
ylabel('Power (dB)','fontsize',FS)