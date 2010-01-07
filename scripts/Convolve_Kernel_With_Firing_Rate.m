

function Firing_Convolved_With_Kernel = Convolve_Kernel_With_Firing_Rate(N_Masses_In_Width, f, Space_Step, Connectivity_Kernel)

    Firing_Convolved_With_Kernel = zeros(N_Masses_In_Width,N_Masses_In_Width);

    for x_space_index=1:N_Masses_In_Width                                     % sum over the x direction
        for y_space_index=1:N_Masses_In_Width                                 % summing over the y direction

            Translated_Connectivity_Kernel = Connectivity_Kernel((N_Masses_In_Width:end)-(x_space_index-1),(N_Masses_In_Width:end)-(y_space_index-1));      % translates kernel around the space
            Firing_Convolved_With_Kernel(x_space_index,y_space_index) = sum( sum( Translated_Connectivity_Kernel.*f, 1 ))*Space_Step^2;

        end
    end

end