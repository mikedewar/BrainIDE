GaussianCentre1 = [-15 -10];             % x and y coordinate
GaussianCentre2 = [-0 -0];
sigma_init_cond = 5;
InitGaussian1 = zeros(N_masses_in_width,N_masses_in_width);
InitGaussian2 = zeros(N_masses_in_width,N_masses_in_width);
for x_space_index=1:N_masses_in_width                                     % sum over the x direction
    for y_space_index=1:N_masses_in_width
        InitGaussian1(x_space_index,y_space_index) = exp(-((GaussianCentre1(1) - Space_x(x_space_index)).^2./(2*sigma_init_cond.^2)...
            + (GaussianCentre1(2) - Space_y(y_space_index)).^2./(2*sigma_init_cond.^2)));
        InitGaussian2(x_space_index,y_space_index) = exp(-((GaussianCentre2(1) - Space_x(x_space_index)).^2./(2*sigma_init_cond.^2)...
            + (GaussianCentre2(2) - Space_y(y_space_index)).^2./(2*sigma_init_cond.^2)));
    end
end

subplot(121)
imagesc(Space_x,Space_y,InitGaussian1)
axis square

subplot(122)
imagesc(Space_x,Space_y,InitGaussian2)
axis square