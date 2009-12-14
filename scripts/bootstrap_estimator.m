% bootstrp 12/12/2009 Parham and Dean
clc

% define initial state, zero mean and large std dev
initial_state_variance = Initial_Variance*eye(N_field_basis_function);
Init_State_Mean = zeros(N_field_basis_function,1);

initial_x_estimate = mvnrnd(Init_State_Mean,initial_state_variance,1)';
Observation_Noise_Covariance=0.2*eye(NSensors);


% create initial field from initial states
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for FieldBasisIndex=1:size(phi,1) 
    Field_Layer_Init(FieldBasisIndex,:,:) = initial_x_estimate(FieldBasisIndex)*phi(FieldBasisIndex,:,:);
end
Field_Init = squeeze(sum(Field_Layer_Init,1));

N_particles = 50; 
for Particle_Index=1: N_particles

    f = Sigmoid_Firing_Rate(nu, beta, threshold, Field_Init);                           % compute firing rate
    Firing_Convolved_With_Kernel = Convolve_Kernel_With_Firing_Rate(N_masses_in_width, f, SpaceStep, ConnectivityKernel);
    Int_Phi_Mult_With_Convolved_Firing = Mult_By_phi_And_Integrate(Firing_Convolved_With_Kernel, N_field_basis_function, SpaceStep, phi, N_masses_in_width);
    x_particles(:,Particle_Index) = Ts*Gamma\Int_Phi_Mult_With_Convolved_Firing + lambda*initial_x_estimate + R*randn(N_field_basis_function,1)*SpaceStep^2;
                                                                                                            %~~~~~
                                                                                                            %scaled
    Field_particle(Particle_Index,:,:) = Create_Field_From_States(phi, x_particles(:,Particle_Index)); 
    y_particle(:,Particle_Index) = Get_Observations(Noise_Covariance_Coefficient, NSensors, N_masses_in_width, Field(t,:,:), SpaceStep, m);
        
% multivariate distribution of mean of true measurement and std deviation of observation noise
    w(Particle_Index) = StateWeight(y(:,2), Observation_Noise_Covariance, y_particle(:,Particle_Index), NSensors);
    
end

% define weights for particle filter
Initial_Weights = ones(1,N_particles)/N_particles;
