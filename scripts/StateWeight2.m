function w = StateWeight2(Observation, Observation_Noise_Covariance, Observation_Particle)
    
    Coef = 1/(2*pi*Observation_Noise_Covariance)^(1/2);
    
    ParticleMinusObser = (Observation_Particle - Observation)^2;
    InverseNoise = 1/(2*Observation_Noise_Covariance);
        
    ExponentialPart = exp(-InverseNoise*ParticleMinusObser);
    w = Coef*ExponentialPart;

end