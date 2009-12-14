

function Int_Phi_Mult_With_Convolved_Firing = Mult_By_phi_And_Integrate(Firing_Convolved_With_Kernel,N_Field_Basis_Functions, Space_Step, phi, N_Masses_In_Width)

    Mult_With_Basis_Function = zeros(N_Field_Basis_Functions, N_Masses_In_Width, N_Masses_In_Width);

    for Field_Basis_Index=1:N_Field_Basis_Functions
        Mult_With_Basis_Function(Field_Basis_Index,:,:) = squeeze(Firing_Convolved_With_Kernel) .* squeeze(phi(Field_Basis_Index,:,:));
    end

    IntOver_x = squeeze(sum(Mult_With_Basis_Function,2));
    Int_Phi_Mult_With_Convolved_Firing = squeeze(sum(IntOver_x,2))*Space_Step^2;
    
end