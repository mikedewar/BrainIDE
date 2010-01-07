% create Gamma matrix for field decomposition state-space model

function Gamma = CreateGamma(phi)

N_Field_Basis_Functions = size(phi,1);

Gamma = zeros(N_Field_Basis_Functions,N_Field_Basis_Functions);
        
for n=1:N_Field_Basis_Functions
    for nn=1:N_Field_Basis_Functions
        phi_i_phi_j = squeeze( phi(n,:,:).*phi(nn,:,:) );
        Gamma(n,nn) = sum(sum(phi_i_phi_j,1));
    end
end