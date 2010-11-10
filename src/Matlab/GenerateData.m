for ppp=1:1
    % implementation of the neural field model to generate data

    % if we are running a batch than we dont want to clear things
    if exist('RunningBatch') == 0
        clear
        close all
        clc

        % run the script that sets all the parameters
        Set_Parameters
        plotdata = 1;
    %     load Parameters
    else
        plotdata = 0;
    end

    % create the observation disturbance
    varepsilon = mvnrnd(zeros(1,NSensors),Sigma_varepsilon,T);

    % DisturbanceFileName = ['e_' num2str(T) '_' num2str(100*Delta) '_' num2str(100*SpaceMax) '_' ...
    %     num2str(100*sigma_gamma) '_' num2str(100*gamma_weight) '.mat'];

    % if exist(DisturbanceFileName) == 2
    %     load(DisturbanceFileName)
    % else
    
    disp('generating disturbance signal')
    e = mvnrnd(zeros(1,NPoints^2),Sigma_gamma,T);
    
    %     save(DisturbanceFileName,'e')
    % end

    disp('generating data')
    % initialize field
    v = zeros(T,NPoints,NPoints);
    v(1,:,:) = Define2DGaussian(0,0, sigma_phi^2, 0,NPoints,SpaceMin,SpaceMax);

    % generate data
    y = zeros(T,NSensors_xy^2);         % initialise obseravation for speed
    y_matrix = zeros(T,NSensors_xy,NSensors_xy);
    for t=1:T-1
        
        f = f_max./(1+exp(varsigma*(v_0-squeeze(v(t,:,:)))));           % calc firing rate using sigmoid
        f = padarray(f,size(f),'circular');
        
        % conv firing rate with spatial kernel
        g = Ts*conv2(w,f,'same')*Delta_squared;
        
%         F = fft2(f);                                                                                % take FFT of firing for conv with kernel
%         g = ifftshift(ifft2(F.*Ts_W,'symmetric'))*Delta_squared;            % conv kernel with firing rate 
        
        v(t+1,:,:) = g + xi*squeeze(v(t,:,:)) + reshape(e(t,:,:),NPoints,NPoints);  % update field
        V = fft2(squeeze(v(t+1,:,:)));                                                  % take FFT of field for conv with sensor
        
        m_conv_v = ifftshift(ifft2(M.*V,'symmetric'))*Delta_squared;    % conv with sensor 

        y_matrix(t+1,:,:) = m_conv_v(sensor_indexes,sensor_indexes) ...
            + reshape(varepsilon(t+1,:,:),NSensors_xy,NSensors_xy);     % take indexes of field conv with sensor for observations
        y_temp = squeeze(y_matrix(t+1,:,:));
        y(t+1,:) = y_temp(:);                                                                 % observations as a vector

    end

    if plotdata == 1
        disp('plotting data')
        figure
        for t=1:T
%             temp1 = squeeze(v(t,:,:));
%             temp2 = padarray(temp1,size(temp1),'circular');
%             imagesc(temp2);
            imagesc(r,r,squeeze(v(t,:,:)))                          % plot field to check things are working
            title(['Time = ' num2str(t) ' ms'])
            axis xy
            axis square
            colorbar
            drawnow
        end
    end

    if exist('RunningBatch') == 1
        disp('saving generated data and parameters')
        SaveTime = datestr(now,30);                                                     % save parameters
        datefilename = ['Parameters' SaveTime '.mat'];
        save(datefilename,'y','Delta','SpaceMax','Ts','T','theta',...
            'sigma_psi','sigma_phi','L',...
            'mu_phi_x1','mu_phi_x2','mu_phi_y','NSensors_xy','mu_y_xy','sigma_y','sigma_varepsilon',...
            'f_max','varsigma','v_0','zeta','sigma_gamma','gamma_weight')
    end 

    max1 = squeeze(max(v,[],2));
    max_v_t(ppp) = mean(max(max1,[],2));
    min1 = squeeze(min(v,[],2));
    min_v_t(ppp) = mean(min(min1,[],2));
    
end

mean(max_v_t)
mean(min_v_t)