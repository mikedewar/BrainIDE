

function y = Get_Observations(Noise_Covariance_Coefficient, N_Sensors, N_Masses_In_Width, Field, Space_Step, m)

    Observation_Noise_Covariance = Noise_Covariance_Coefficient*eye(N_Sensors);
    Observation_Noise_mean = zeros(N_Sensors,1);
    Observation_Noise = mvnrnd(Observation_Noise_mean,Observation_Noise_Covariance,1)';
    
    y = zeros(N_Sensors,1);
    
    for Sensor_Index=1:N_Sensors
        for x_space_index=1:N_Masses_In_Width                                     % sum over the x direction
            for y_space_index=1:N_Masses_In_Width
                
                y(Sensor_Index) = sum(sum(squeeze( m(Sensor_Index,:,:).*Field ),1)) * Space_Step^2;
                
            end
        end
    end

    y = y + Observation_Noise;
    
end