% This script runs the derivation for estimation of kernel support from the paper!


for realizationindex=1:50
    
    tic
    disp(['realization number: ' num2str(realizationindex)])
    GenerateData

    clear xcorr_mat
    usefield = false;
    if usefield
        
        % xcorr_mat = zeros(T-1,163,163);
        % autocorr_mat = xcorr_mat;
        xcorr_mat = zeros(T-1,81,81);
        autocorr_mat = xcorr_mat;
        
    else
        
        xcorr_mat = zeros(T-1,27,27);
        autocorr_mat = xcorr_mat;      

    end

    disp('calculating correlations')
    for t=2:T-1
        if usefield

            v_t = squeeze(v(t,:,:));
            v_t_plus_1 = squeeze(v(t+1,:,:));
%             xcorr_mat(t,:,:) = normxcorr2(v_t,v_t_plus_1);
%             autocorr_mat(t,:,:) = normxcorr2(v_t,v_t);
            xcorr_mat(t,:,:) = xcorr2(v_t,v_t_plus_1);
            autocorr_mat(t,:,:) = xcorr2(v_t,v_t);            
        else

            y_t = squeeze(y_matrix(t,:,:));
            y_t_plus_1 = squeeze(y_matrix(t+1,:,:));
%             xcorr_mat(t,:,:) = normxcorr2(y_t,y_t_plus_1);
%             autocorr_mat(t,:,:) = normxcorr2(y_t,y_t);
            xcorr_mat(t,:,:) = xcorr2(y_t,y_t_plus_1);
            autocorr_mat(t,:,:) = xcorr2(y_t,y_t);
            
        end    

    end
    
    disp('calculating mean correlations')
    meanxcorr_all(realizationindex,:,:)  = squeeze(mean(xcorr_mat,1));
    meanautocorr_all(realizationindex,:,:)  = squeeze(mean(autocorr_mat,1));
    toc
    
end

R_yyplus1 = squeeze(mean(meanxcorr_all,1));
R_yy = squeeze(mean(meanautocorr_all,1));

delta_sigma_varepsilon2 = zeros(size(R_yy));
delta_sigma_varepsilon2( floor(size(R_yy,1)/2)+1, floor(size(R_yy,2)/2)+1) = sigma_varepsilon^2;

R_zz = R_yy - delta_sigma_varepsilon2;

ZeroPadFactor = 1;
Syyplus1 = fft2(R_yyplus1, ZeroPadFactor*size(R_yyplus1,1), ZeroPadFactor*size(R_yyplus1,2));
Szz = fft2(R_zz, ZeroPadFactor*size(R_yy,1), ZeroPadFactor*size(R_yy,2));

Numerator = Syyplus1 - xi*Szz;
Denominator = Szz;

w_temp = ifft2( Numerator ./ Denominator) / (Ts*varsigma);%, size(meanxcorr,1), size(meanxcorr,2) );
w_est = ifftshift( w_temp );

xscale = 0:1.5:size(w_est,1)*1.5-1;
figure
imagesc(xscale-21,xscale-21, w_est)
% surf(xscale-22.5,xscale-21, w_est,'edgecolor','none' )
axis square

%%
filename = '/Users/dean/Projects/BrainIDE/ltx/data_paper/Figures/KernelWidthEstimation2.pdf';
figure('units','centimeters','position',[0 0 8 3],'filename',filename,...
    'papersize',[8+1, 3+1],'paperorientation','landscape','renderer','painters')
plot(xscale-21,diag(w_est)/max(diag(w_est)),'k')
hold
plot(r,diag(w)/max(diag(w)),'r')
xlim([-10 10])
ylim([-0.4 1.1])
box off
xlabel('Space')

%%

disp('plotting correlations')
plotheight = 7;
plotwidth = 9;

NoiseCovariance = gamma_weight*Define2DGaussian(0,0, sigma_gamma^2, 0, 41, -10,10);
figure('units','centimeters','position',[0 0 plotwidth plotheight],...
    'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
imagesc(NoiseCovariance)
title('Disturbance Covariance Function')
colorbar
axis square

figure('units','centimeters','position',[0 0 plotwidth plotheight],...
    'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
imagesc(R_yyplus1)
title('Time-Averaged Cross-Correlation')
colorbar
axis square

figure('units','centimeters','position',[0 0 plotwidth plotheight],...
    'papersize',[plotheight, plotwidth],'paperorientation','landscape','renderer','painters')
imagesc(r,r,w)
title('Connectivity Kernel')
colorbar
axis square
