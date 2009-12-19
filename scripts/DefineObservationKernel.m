
% define observation kernels


ObservationCentre = (-FieldWidth/2 + BoundryEffectWidth):SensorSpacing:(FieldWidth/2 - BoundryEffectWidth);
NSensors = length(ObservationCentre)^2;
RightDistance = abs((FieldWidth/2 - BoundryEffectWidth) - ObservationCentre(end));              % to edge of the usable field

ObservationCentre = ObservationCentre + RightDistance/2;        % recenter grid

angle = linspace(-pi,pi,100);
ObservationFunctionIndex = 1;

for CenterIndex_x=1:length(ObservationCentre)
    for CenterIndex_y=1:length(ObservationCentre)
        
        plot(ObservationCentre(CenterIndex_x),ObservationCentre(CenterIndex_y),'+')
        hold on
        plot(SensorWidth*cos(angle)+ObservationCentre(CenterIndex_x),SensorWidth*sin(angle)+ObservationCentre(CenterIndex_y),'k')
        
        for x_space_index=1:N_masses_in_width                                     % sum over the x direction
            for y_space_index=1:N_masses_in_width
                
                m(ObservationFunctionIndex,x_space_index,y_space_index) = exp(-((ObservationCentre(CenterIndex_x) - Space_x(x_space_index)).^2./(SensorWidth.^2)...
                    + (ObservationCentre(CenterIndex_y) - Space_y(y_space_index)).^2./(SensorWidth.^2)));
                
            end
        end
        ObservationFunctionIndex = ObservationFunctionIndex+1;
    end
end

figure
imagesc(Space_x,Space_y,squeeze(sum(m,1)))