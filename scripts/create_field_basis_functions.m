% create field basis functions
% Dean Freestone 6/12/09
% ~~~~~~~~~~~~~~~~~~~~~~

% for a staggered grid we need ...
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


x_separation = field_basis_separation/sqrt(2);              % field_basis_separation = diagonal distance between basis functions
y_separation = 2*field_basis_separation/sqrt(2);

FieldBasisCenters_x = -FieldWidth/2:x_separation:FieldWidth/2;
FieldBasisCenters_x = FieldBasisCenters_x + (FieldWidth/2-FieldBasisCenters_x(end))/2;      % centre basis function

FieldBasisCenters_y = -FieldWidth/2+y_separation/2:y_separation:FieldWidth/2;
FieldBasisCenters_y = FieldBasisCenters_y + (FieldWidth/2-FieldBasisCenters_y(end))/2;      % centre basis function

N_field_basis_function = length(FieldBasisCenters_x)* length(FieldBasisCenters_y);

% plot basis function grid
% ~~~~~~~~~~~~~~~~~~~~~~~~
figure
angle = linspace(-pi,pi,100);
for Centre_index_x = 1:length(FieldBasisCenters_x)
    
    for Centre_index_y = 1:length(FieldBasisCenters_y)

        if (floor(Centre_index_x/2) - Centre_index_x/2) == 0                                            % will be even
            plot(FieldBasisCenters_x(Centre_index_x), FieldBasisCenters_y(Centre_index_y),'+')
            hold on
            plot(sigma_field*cos(angle)+FieldBasisCenters_x(Centre_index_x),sigma_field*sin(angle)+FieldBasisCenters_y(Centre_index_y),'k')
        else                                                                                            % it is odd
            plot(FieldBasisCenters_x(Centre_index_x), FieldBasisCenters_y(Centre_index_y)-y_separation/2,'r+')
            plot(sigma_field*cos(angle)+FieldBasisCenters_x(Centre_index_x),sigma_field*sin(angle)+FieldBasisCenters_y(Centre_index_y)-y_separation/2,'k')
            hold on
        end
        
    end
end
axis square

% create basis functions
% ~~~~~~~~~~~~~~~~~~~~~~
n=1;
for Centre_index_x = 1:length(FieldBasisCenters_x)
    for Centre_index_y = 1:length(FieldBasisCenters_y)
        
        if (floor(Centre_index_x/2) - Centre_index_x/2) == 0                                            % will be even
            OS = y_separation/2;
        else
            OS = 0;
        end
        for x_space_index=1:N_masses_in_width                                     % sum over the x direction
            for y_space_index=1:N_masses_in_width
                
                phi(n,x_space_index,y_space_index) = exp(-((FieldBasisCenters_x(Centre_index_x) - Space_x(x_space_index)).^2./(2*sigma_field.^2)...
                    + ((FieldBasisCenters_y(Centre_index_y)- OS) - Space_y(y_space_index)).^2./(2*sigma_field.^2)));
                
            end
        end
        
        n=n+1;
    end
end

% double check field basis function spacing
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% figure
% for n=1:size(phi,1)
%     imagesc(squeeze(phi(n,:,:)))
%     drawnow
%     pause(0.1)
% end

figure
imagesc(Space_x,Space_y,squeeze(sum(phi,1)))
drawnow
