% define initial states and predefine matrices 

function [x Field] = InitialiseStates(N_Samples, N_Masses_In_Width, N_Field_Basis_Functions, phi, Use_Basis_Functions, Initial_Variance)

Field = 0.5*randn(N_Samples,N_Masses_In_Width,N_Masses_In_Width);

if Use_Basis_Functions
    x = zeros(N_Field_Basis_Functions,N_Samples);
    initial_state_variance = Initial_Variance*eye(N_Field_Basis_Functions);
    Init_State_Mean = zeros(N_Field_Basis_Functions,1);
    x(:,1) = mvnrnd(Init_State_Mean,initial_state_variance,1)';

    for FieldBasisIndex=1:N_Field_Basis_Functions 
        FieldLayer(FieldBasisIndex,:,:) = x(FieldBasisIndex,1)*phi(FieldBasisIndex,:,:);
    end
    
    Field(1,:,:) = sum(FieldLayer,1);
    
else
    x = [];
    Field = Initial_Variance*randn(N_Samples, N_Masses_In_Width, N_Masses_In_Width);
end