% generate noise with set covariance on a sphere
% clc
% clear
% close all

% spatial parameters
% ~~~~~~~~~~~
% Delta = 0.5;                          % space step for the spatial discretisation
Delta_squared = Delta^2;
SpaceMaxPeriodicField = 30;                    % maximum space in mm
SpaceMinPeriodicField = -30;         % minimum space in mm
NPointsInPeriodicField = (SpaceMaxPeriodicField-SpaceMinPeriodicField)/Delta+1;
NPointsInField = (NPointsInPeriodicField-1)/3 + 1;
r = linspace(SpaceMinPeriodicField/3,SpaceMaxPeriodicField/3,NPointsInField);      % define space

% temporal parameters
% ~~~~~~~~~~~~~
% Ts = 1e-3;          % sampling period (s)
% T = 500;            % maximum time (ms)

% disturbance paramters
% ~~~~~~~~~~~~~
% sigma_gamma = 1.3;          % parameter for covariance of disturbance
% gamma_weight = 0.1;            % variance of disturbance

% ~~~~~~~~~~~~~~~~~~~~~~~~~~
% ~~~~~~~~~~~~~~~~~~~~~~~~~~
FirstThirdEnd = (NPointsInPeriodicField-1)/3;
SecondThirdEnd = 2*FirstThirdEnd+1;

mm=1;
Sigma_gamma = zeros(NPointsInField^2,NPointsInField^2);   % create disturbance covariance matrix
% figure
template = zeros(NPointsInField,NPointsInField);
for n=1:NPointsInField
    for nn=1:NPointsInField
        temp = gamma_weight*Define2DGaussian(r(n),r(nn), sigma_gamma^2, 0, NPointsInPeriodicField, SpaceMinPeriodicField,SpaceMaxPeriodicField);
        
        topleft =[zeros(1,FirstThirdEnd+1) ; [zeros(FirstThirdEnd,1) temp(1:FirstThirdEnd,1:FirstThirdEnd)]];         
        top = [zeros(1,FirstThirdEnd+1) ;temp(1:FirstThirdEnd,FirstThirdEnd+1:SecondThirdEnd)];
        topright = [zeros(1,FirstThirdEnd+1) ; [temp(1:FirstThirdEnd,SecondThirdEnd+1:end) zeros(FirstThirdEnd,1)]]; 
        
        left = [zeros(FirstThirdEnd+1,1) temp(FirstThirdEnd+1:SecondThirdEnd,1:FirstThirdEnd)];
        middle = temp(FirstThirdEnd+1:SecondThirdEnd,FirstThirdEnd+1:SecondThirdEnd);
        right = [temp(FirstThirdEnd+1:SecondThirdEnd,SecondThirdEnd+1:end) zeros(FirstThirdEnd+1,1)];
        
        bottomleft = [zeros(FirstThirdEnd+1,1) [temp(SecondThirdEnd+1:end,1:FirstThirdEnd) ; zeros(1,FirstThirdEnd)]];
        bottom = [temp(SecondThirdEnd+1:end,FirstThirdEnd+1:SecondThirdEnd) ; zeros(1,FirstThirdEnd+1)];
        bottomright = [[temp(SecondThirdEnd+1:end,SecondThirdEnd+1:end) zeros(FirstThirdEnd,1)] ; zeros(1,FirstThirdEnd+1)];
        
        temp2 = middle + topleft + top + topright + left + right + bottom + bottomleft + bottomright;
        
        Sigma_gamma(:,mm) = temp2(:);
        mm=mm+1;
        
%         clim = [-200 -1];         % for log
%         clim = [0 0.08];
% 
%         subplot(3,3,1),imagesc(topleft,clim)
%         subplot(3,3,2),imagesc(top,clim)
%         subplot(3,3,3),imagesc(topright,clim)
%         
%         subplot(3,3,4),imagesc(left,clim)
%         subplot(3,3,5),imagesc(middle,clim)
%         subplot(3,3,6),imagesc(right,clim)
%         
%         subplot(3,3,7),imagesc(bottomleft,clim)
%         subplot(3,3,8),imagesc(bottom,clim)
%         subplot(3,3,9),imagesc(bottomright,clim)
% % 
%  %       imagesc(log10(temp2))
%         drawnow
%         clim = [0 0.08];
% % 
%         subplot(3,3,1),imagesc(topleft,clim)
%         subplot(3,3,2),imagesc(top,clim)
%         subplot(3,3,3),imagesc(topright,clim)
%         
%         subplot(3,3,4),imagesc(left,clim)
%         subplot(3,3,5),imagesc(middle,clim)
%         subplot(3,3,6),imagesc(right,clim)
%         
%         subplot(3,3,7),imagesc(bottomleft,clim)
%         subplot(3,3,8),imagesc(bottom,clim)
%         subplot(3,3,9),imagesc(bottomright,clim)
% % 
% %         imagesc(log10(temp2))
%         drawnow
     end
 end

    %end
%end

% m=1;
% Sigma_gamma = zeros(NPointsInPeriodicField^2,NPointsInPeriodicField^2);   % create disturbance covariance matrix
% for n=1:NPointsInPeriodicField
%     for nn=1:NPointsInPeriodicField
%         temp = gamma_weight*Define2DGaussian(r(n),r(nn), sigma_gamma^2, 0,NPointsInPeriodicField,SpaceMinPeriodicField,SpaceMaxPeriodicField);
%         Sigma_gamma(:,m) = temp(:);
%         m=m+1;
%     end
% end
% ~~~~~~~~~~~~~~~~~~~~~~~~~~
% ~~~~~~~~~~~~~~~~~~~~~~~~~~


% figure
% % subplot(121)
% imagesc(Sigma_gamma),axis square
% % subplot(122)
% % surf(Sigma_gamma-Sigma_gamma','edgecolor','none'),axis square
% % max(max(Sigma_gamma-Sigma_gamma',[],1))
% % [T,P] = cholcov(Sigma_gamma);

%#######################################
% e = mvnrnd(zeros(1,NPointsInField^2),Sigma_gamma,T);
% for t=1:T-1
%     e_square = reshape(e(t,:,:),NPointsInField,NPointsInField);
%     e_pad = padarray(e_square,size(e_square),'circular');
%     imagesc(e_pad)
%     axis square
%     drawnow
%     pause
%     
% end
    