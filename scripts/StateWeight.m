
function w = StateWeight(Observation, Observation_Noise_Covariance, Observation_Particle, NSensors)
    
    Coef = (1/(2*pi)^(NSensors/2))*(1/sqrt((det(Observation_Noise_Covariance))));
    
    ParticleMinusObser = Observation - Observation_Particle;
    InverseNoise = Observation_Noise_Covariance^-1;
        
    ExponentialPart = exp(-0.5*ParticleMinusObser'*InverseNoise*ParticleMinusObser);
    w = Coef*ExponentialPart;

end