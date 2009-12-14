
fig=figure('name','Field and Firing Rates','units','normalized','position',[0.1 0.1 .8 .8]);
aviobj = avifile('WC_Simulation3.avi');

for n=1:2:NSamples
    subplot(231)
    h=imagesc(Space_x,Space_y,squeeze(Field(n,:,:)));
    axis off
    axis square
    colorbar
    title(['$v_t(r\prime)$, Time = ' num2str(floor(n*Ts*1e3)) ' ms'], 'interpreter','latex')
    
    subplot(232)
    hh=imagesc(Space_x,Space_y,squeeze(Firing_Rate(n,:,:)));
    title('$f(v_t(r\prime))$', 'interpreter','latex')
    axis off
    axis square
    colorbar
    
    subplot(233)
    hhh=imagesc(Space_x,Space_y,squeeze(Firing_Convolved_With_Kernel(n,:,:)));
    title('$\int_\Omega k(r-r\prime)f(v_t(r))dr\prime$', 'interpreter','latex')
    axis off
    axis square
    colorbar
    
    subplot(212)
    plot(y')
    F = getframe(fig);
    aviobj = addframe(aviobj,F);
end

close(fig)
aviobj = close(aviobj);

% fig=figure;
% aviobj = avifile('WC_Simulation_FiringRate.avi');
% 
% for n=1:5:5000
%     
%     h=imagesc(squeeze(FiringRate(n,:,:)));
%     axis off
%     colorbar
%     title(['Time = ' num2str(floor(n*Ts*1e3)) ' ms, $f(v_t(r\prime))$'], 'interpreter','latex')
%     F = getframe(fig);
%     aviobj = addframe(aviobj,F);
% end
% 
% close(fig)
% aviobj = close(aviobj);