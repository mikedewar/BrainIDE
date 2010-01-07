% construct connectivity kernel
% needs to be twice the size of the field so we can centre it here and
% translate it to cover all points

Kernel_Space_x = -FieldWidth:1/MassDensity:FieldWidth;          % need to add one so the size is always odd to get a center point
Kernel_Space_y = Kernel_Space_x;

Kernel_width = length(Kernel_Space_x);

psi = zeros(N_ConnBasisFuns,Kernel_width,Kernel_width);

for x_space_index=1:Kernel_width                                     % sum over the x direction
    for y_space_index=1:Kernel_width
        
        for basis_fun_index=1:N_ConnBasisFuns
                        
            psi(basis_fun_index,x_space_index,y_space_index) = theta(basis_fun_index)...
                *exp(-((Kernel_Space_x(x_space_index)).^2./(2*sigma(basis_fun_index).^2)...
                + (Kernel_Space_y(y_space_index)).^2./(2*sigma(basis_fun_index).^2)));
            
        end
    end
end

ConnectivityKernel = squeeze(sum(psi,1));


% plot results
% ~~~~~~~~~~~~
figure('name','connectivity kernel','units','normalized','position',[0 0 1 1])

surf(ConnectivityKernel,'EdgeColor','none')
title(['Full Kernel, integrates to:' num2str(sum(sum(ConnectivityKernel,1)))])
xlabel('Space, x dimension')
ylabel('Space, y dimension')
zlabel('Connectivity Strength')
colorbar
drawnow
