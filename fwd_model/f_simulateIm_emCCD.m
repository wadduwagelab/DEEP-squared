function [Xhat XhatADU] = f_simulateIm_emCCD(X0,emhist,pram)

  Xdark   = pram.cam_dXdt_dark * pram.cam_t_exp;                          % [e-]    Dark noise
  
  try 
    Xhat        = poissrnd(X0 + Xdark);                                   % [e-]    Poisson shot noise
  catch
    Xhat        = gpuArray(poissrnd(gather(X0) + Xdark));                 % [e-]    poissrnd doesnt work on GPU for numbers >14        
  end
  
  %% Em-process - approximate method with pre simulated distribution 
  N_reps            = size(emhist,2);
  max_input_photons = size(emhist,1);
  
  Xhat(Xhat>max_input_photons) = max_input_photons;
  Xhat_temp         = Xhat;
  for i = 1:max_input_photons
    i_inds          = find(Xhat_temp(:)==i); 
    Xhat(i_inds)    = emhist(i,randi(N_reps,size(i_inds)));
  end
  
  %% EM-process - direct methord
%   fprintf('%0.4d/%0.4d',0,pram.cam_N_gainStages)
%   for i=1:pram.cam_N_gainStages 
%     fprintf('\b\b\b\b\b\b\b\b\b%0.4d/%0.4d',i,pram.cam_N_gainStages)
%     Xhat  = Xhat + binornd(Xhat,pram.cam_Brnuli_alpha);                 % [e-]    Add binomial noise due to each step in the Em process
%   end
%   fprintf('\n')
  
  %% Effective read noise and output images
  Xhat    = (Xhat + normrnd(0,pram.cam_sigma_rd))/pram.cam_EMgain;        % [e-]    Input (to EM register) image with noise
  XhatADU = Xhat * pram.cam_EMgain * pram.cam_ADCfactor + pram.cam_bias;  % [ADU]   simulated image in ADU
end