

% define outer grid
% ~~~~~~~~~~~~~~~~~
x_center_outer = linspace(-FieldWidth/2,FieldWidth/2,N_masses_in_width/(2*field_basis_separation/sqrt(2)));
y_center_outer = x_center_outer;

% define inner grid
% ~~~~~~~~~~~~~~~~~
distance_between_centers = abs(x_center_outer(2)-x_center_outer(1));
x_center_inner = linspace(-FieldWidth/2+distance_between_centers/2,FieldWidth/2-distance_between_centers/2,N_masses_in_width/(2*field_basis_separation/sqrt(2))-1);
y_center_inner = x_center_inner;

% combine the vectors of centers
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_centers = [x_center_outer x_center_inner];
y_centers = [y_center_outer y_center_inner];


% plot basis function grid
% ~~~~~~~~~~~~~~~~~~~~~~~~
figure
angle = linspace(-pi,pi,100);
BasisFunctionIndex = 1;
for x_center_outer_index=1:length(x_center_outer)
    for y_center_outer_index=1:length(y_center_outer)
        plot(x_center_outer(x_center_outer_index), y_center_outer(y_center_outer_index),'+')
        hold on
        plot(sigma_field*cos(angle)+x_center_outer(x_center_outer_index),sigma_field*sin(angle)+y_center_outer(y_center_outer_index),'k')
        
        
        for x_space_index=1:N_masses_in_width                                     % sum over the x direction
            for y_space_index=1:N_masses_in_width
                
                phi(BasisFunctionIndex,x_space_index,y_space_index) = exp(-((x_center_outer(x_center_outer_index) - Space_x(x_space_index)).^2./(sigma_field.^2)...
                    + (y_center_outer(y_center_outer_index) - Space_y(y_space_index)).^2./(sigma_field.^2)));
                
            end
        end
        BasisFunctionIndex = BasisFunctionIndex+1;
    end
end


for x_center_inner_index=1:length(x_center_inner)
    for y_center_inner_index=1:length(y_center_inner)
        plot(x_center_inner(x_center_inner_index), y_center_inner(y_center_inner_index),'+')
        hold on
        plot(sigma_field*cos(angle)+x_center_inner(x_center_inner_index),sigma_field*sin(angle)+y_center_inner(y_center_inner_index),'k')
        
        for x_space_index=1:N_masses_in_width                                     % sum over the x direction
            for y_space_index=1:N_masses_in_width
                
                phi(BasisFunctionIndex,x_space_index,y_space_index) = exp(-((x_center_inner(x_center_inner_index) - Space_x(x_space_index)).^2./(2*sigma_field.^2)...
                    + (y_center_inner(y_center_inner_index) - Space_y(y_space_index)).^2./(sigma_field.^2)));
                
            end
        end
        BasisFunctionIndex = BasisFunctionIndex+1;
    end
end
axis square

N_field_basis_function = BasisFunctionIndex-1


figure
imagesc(Space_x,Space_y,squeeze(sum(phi,1)))
drawnow
