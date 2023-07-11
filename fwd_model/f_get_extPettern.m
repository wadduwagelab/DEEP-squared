
function [E Y_exp X_refs pram] = f_get_extPettern(pram)

  switch pram.pattern_typ
    case 'dmd_sim_rnd'                      
      E     = single(rand([pram.Ny pram.Nx pram.Nt])>0.5); % for DMDs
      Y_exp = [];
      X_refs= [];
      pram  = pram;
    case 'dmd_sim_Hadamard'                 
      H     = hadamard(pram.Ny*pram.Nx);
      A     = (H+1)/2;
      E     = single(reshape(A,pram.Ny,pram.Nx,[])); % for DMDs
      E     = E(:,:,1:pram.Nt);
      
      Y_exp = [];
      X_refs= [];
      pram  = pram;
    case 'dmd_exp_tfm_beads_7sls_20201219'  
      load('./_extPatternsets/dmd_exp_tfm_beads_7sls_20201219.mat')
      
      y_inds  = size(Data.Ex,1)-pram.Ny+1:size(Data.Ex,1);      % select the lower left coner as it's brighter
      x_inds  = 1:pram.Nx;
      t_inds  = 20 + (1:pram.Nt);
      
      E                   = single(Data.Ex           (y_inds,x_inds,:));
      Y_exp.beads1        = single(Data.beads1_7sls  (y_inds,x_inds,:));
      Y_exp.beads2        = single(Data.beads2_7sls  (y_inds,x_inds,:));
      X_refs.beads1_sf_wf0= single(Data.beads1_sf_wf0(y_inds,x_inds,:));
      X_refs.beads2_sf_wf0= single(Data.beads2_sf_wf0(y_inds,x_inds,:));

      X_refs.beads1_avg0  = mean(Y_exp.beads1,3);
      X_refs.beads2_avg0  = mean(Y_exp.beads2,3);
      
      E                   = E           (:,:,t_inds);
      Y_exp.beads1        = Y_exp.beads1(:,:,t_inds);
      Y_exp.beads2        = Y_exp.beads2(:,:,t_inds);
      
      X_refs.beads1_avg   = mean(Y_exp.beads1,3);
      X_refs.beads2_avg   = mean(Y_exp.beads2,3);
      
      % normalize E
      E                   = E - min(E(:));
      E                   = E / max(E(:));
      
      pram.maxcount         = max([X_refs.beads1_avg0(:); X_refs.beads2_avg0(:)]);
      pram.dx               = Data.pram_ex.dx0;
      pram.cam_bias         = Data.pram_beads.bias;
      pram.cam_ADCfactor    = Data.pram_beads.ADCfactor;
      pram.cam_EMgain       = Data.pram_beads.EMgain;
      pram.cam_t_exp        = Data.pram_beads.t_exp / 1e3;    % [s]  
      
      pram.cam_sigma_rd     = 3;                              % [e-]        Read noise      
      pram.cam_dXdt_dark    = 0.005;                          % [e-/px/s]   Dark current
      pram.cam_Brnuli_alpha = 0.01;                           %             Probability of a multiplication event in an Em gain stage (=1-2% in Ref2)
      pram.cam_N_gainStages = round(log(pram.cam_EMgain)/log(1+pram.cam_Brnuli_alpha)); 
                                                              %             Number of Em-gain stages              
    case 'dmd_exp_tfm_mouse_20201224_sf'    
      load('./_extPatternsets/dmd_exp_tfm_mouse_20201224.mat')
      
      pram.z0_um  = -1;                                   % [um]        surface is 0 um -ve is below the surface
      pram.Nx     = min(size(Data.Ex,1),size(Data.Ex,2)); %             make square image
      pram.Ny     = pram.Nx;
      x_inds      = 1:pram.Nx;    
      y_inds      = 1:pram.Ny;
      t_inds      = 20 + (1:pram.Nt);
      
      %% Y_exp, E, select y_inds, x_inds
      E                           = single(Data.Ex            (y_inds,x_inds,:));
      Y_exp.anml1_r1_sf           = single(Data.anml1_r1_sf   (y_inds,x_inds,:));
      Y_exp.anml1_r2_sf           = single(Data.anml1_r2_sf   (y_inds,x_inds,:));
      %% X_refs.xx_avg0
      X_refs.anml1_r1_sf_avg0     = mean(Y_exp.anml1_r1_sf   ,3);
      X_refs.anml1_r2_sf_avg0     = mean(Y_exp.anml1_r2_sf   ,3);
      %% X_refs.xx_wf0
      X_refs.anml1_r1_sf_wf0      = single(Data.anml1_r1_sf_wf0   (y_inds,x_inds,:));
      X_refs.anml1_r2_sf_wf0      = single(Data.anml1_r2_sf_wf0   (y_inds,x_inds,:));
      %% seletc the t_inds patterns
      E                           = E                   (:,:,t_inds);
      Y_exp.anml1_r1_sf           = Y_exp.anml1_r1_sf   (:,:,t_inds);
      Y_exp.anml1_r2_sf           = Y_exp.anml1_r2_sf   (:,:,t_inds);
      %% X_refs.xx_avg
      X_refs.anml1_r1_sf_avg      = mean(Y_exp.anml1_r1_sf   ,3);
      X_refs.anml1_r2_sf_avg      = mean(Y_exp.anml1_r2_sf   ,3);
      
      %%      
      % normalize E
      E                           = E - min(E(:));
      E                           = E / max(E(:));
      
      pram.maxcount               = max([X_refs.anml1_r1_sf_avg0(:);...
                                         X_refs.anml1_r2_sf_avg0(:)]);

      pram.dx                     = Data.pram.dx0;
      pram.cam_bias               = Data.pram.bias;
      pram.cam_ADCfactor          = Data.pram.ADCfactor;
      pram.cam_EMgain             = Data.pram.EMgain_lt300um;
      pram.cam_t_exp              = Data.pram.t_exp / 1e3;          % [s]  
      
      pram.cam_sigma_rd           = 3;                              % [e-]        Read noise      
      pram.cam_dXdt_dark          = 0.005;                          % [e-/px/s]   Dark current
      pram.cam_Brnuli_alpha       = 0.01;                           %             Probability of a multiplication event in an Em gain stage (=1-2% in Ref2)
      pram.cam_N_gainStages       = round(log(pram.cam_EMgain)/log(1+pram.cam_Brnuli_alpha)); 
                                                                    %             Number of Em-gain stages
    case 'dmd_exp_tfm_mouse_20201224_100um' 
      load('./_extPatternsets/dmd_exp_tfm_mouse_20201224.mat')
      
      pram.z0_um                  = -100;                                 % [um]        surface is 0 um -ve is below the surface
      pram.Nx                     = min(size(Data.Ex,1),size(Data.Ex,2)); %             make square image
      pram.Ny                     = pram.Nx;
      x_inds                      = 1:pram.Nx;    
      y_inds                      = 1:pram.Ny;
      t_inds                      = 20 + (1:pram.Nt);
      
      %% Y_exp, E, select y_inds, x_inds
      E                           = single(Data.Ex            (y_inds,x_inds,:));      
      Y_exp.anml1_r1_100um        = single(Data.anml1_r1_100um(y_inds,x_inds,:));
      Y_exp.anml1_r2_100um        = single(Data.anml1_r2_100um(y_inds,x_inds,:));
      Y_exp.anml2_r1_100um        = single(Data.anml2_r1_100um(y_inds,x_inds,:));
      %% X_refs.xx_avg0
      X_refs.anml1_r1_100um_avg0  = mean(Y_exp.anml1_r1_100um,3);
      X_refs.anml1_r2_100um_avg0  = mean(Y_exp.anml1_r2_100um,3);
      X_refs.anml2_r1_100um_avg0  = mean(Y_exp.anml2_r1_100um,3);
      %% X_refs.xx_wf0
      X_refs.anml1_r1_100um_wf0   = single(Data.anml1_r1_100um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r2_100um_wf0   = single(Data.anml1_r2_100um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r1_100um_wf0   = single(Data.anml2_r1_100um_wf0(y_inds,x_inds,:));
      %% seletc the t_inds patterns
      E                           = E                   (:,:,t_inds);
      Y_exp.anml1_r1_100um        = Y_exp.anml1_r1_100um(:,:,t_inds);
      Y_exp.anml1_r2_100um        = Y_exp.anml1_r2_100um(:,:,t_inds);
      Y_exp.anml2_r1_100um        = Y_exp.anml2_r1_100um(:,:,t_inds);
      %% X_refs.xx_avg
      X_refs.anml1_r1_100um_avg   = mean(Y_exp.anml1_r1_100um,3);
      X_refs.anml1_r2_100um_avg   = mean(Y_exp.anml1_r2_100um,3);
      X_refs.anml2_r1_100um_avg   = mean(Y_exp.anml2_r1_100um,3);
      
      %%      
      % normalize E
      E                           = E - min(E(:));
      E                           = E / max(E(:));
      
      pram.maxcount               = max([X_refs.anml1_r1_100um_avg0(:);...
                                         X_refs.anml1_r2_100um_avg0(:);...
                                         X_refs.anml2_r1_100um_avg0(:)]);

      pram.dx                     = Data.pram.dx0;
      pram.cam_bias               = Data.pram.bias;
      pram.cam_ADCfactor          = Data.pram.ADCfactor;
      pram.cam_EMgain             = Data.pram.EMgain_lt300um;
      pram.cam_t_exp              = Data.pram.t_exp / 1e3;    % [s]  
      
      pram.cam_sigma_rd           = 3;                              % [e-]        Read noise      
      pram.cam_dXdt_dark          = 0.005;                          % [e-/px/s]   Dark current
      pram.cam_Brnuli_alpha       = 0.01;                           %             Probability of a multiplication event in an Em gain stage (=1-2% in Ref2)
      pram.cam_N_gainStages       = round(log(pram.cam_EMgain)/log(1+pram.cam_Brnuli_alpha)); 
                                                                    %             Number of Em-gain stages      
    case 'dmd_exp_tfm_mouse_20201224_200um' 
      load('./_extPatternsets/dmd_exp_tfm_mouse_20201224.mat')
      
      pram.z0_um                  = -200;                                 % [um]        surface is 0 um -ve is below the surface      
      pram.Nx                     = min(size(Data.Ex,1),size(Data.Ex,2)); %             make square image
      pram.Ny                     = pram.Nx;
      x_inds                      = 1:pram.Nx;    
      y_inds                      = 1:pram.Ny;
      t_inds                      = 20 + (1:pram.Nt);
      
      %% Y_exp, E, select y_inds, x_inds
      E                           = single(Data.Ex            (y_inds,x_inds,:));
      Y_exp.anml1_r1_200um        = single(Data.anml1_r1_200um(y_inds,x_inds,:));
      Y_exp.anml1_r2_200um        = single(Data.anml1_r2_200um(y_inds,x_inds,:));
      Y_exp.anml2_r1_200um        = single(Data.anml2_r1_200um(y_inds,x_inds,:));
      %% X_refs.xx_avg0
      X_refs.anml1_r1_200um_avg0  = mean(Y_exp.anml1_r1_200um,3);
      X_refs.anml1_r2_200um_avg0  = mean(Y_exp.anml1_r2_200um,3);
      X_refs.anml2_r1_200um_avg0  = mean(Y_exp.anml2_r1_200um,3);
      %% X_refs.xx_wf0
      X_refs.anml1_r1_200um_wf0   = single(Data.anml1_r1_200um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r2_200um_wf0   = single(Data.anml1_r2_200um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r1_200um_wf0   = single(Data.anml2_r1_200um_wf0(y_inds,x_inds,:));
      %% seletc the t_inds patterns
      E                           = E                   (:,:,t_inds);
      Y_exp.anml1_r1_200um        = Y_exp.anml1_r1_200um(:,:,t_inds);
      Y_exp.anml1_r2_200um        = Y_exp.anml1_r2_200um(:,:,t_inds);
      Y_exp.anml2_r1_200um        = Y_exp.anml2_r1_200um(:,:,t_inds);
      %% X_refs.xx_avg
      X_refs.anml1_r1_200um_avg   = mean(Y_exp.anml1_r1_200um,3);
      X_refs.anml1_r2_200um_avg   = mean(Y_exp.anml1_r2_200um,3);
      X_refs.anml2_r1_200um_avg   = mean(Y_exp.anml2_r1_200um,3);
      
      %%      
      % normalize E
      E                           = E - min(E(:));
      E                           = E / max(E(:));
      
      pram.maxcount               = max([X_refs.anml1_r1_200um_avg0(:);...
                                         X_refs.anml1_r2_200um_avg0(:);...
                                         X_refs.anml2_r1_200um_avg0(:)]);

      pram.dx                     = Data.pram.dx0;
      pram.cam_bias               = Data.pram.bias;
      pram.cam_ADCfactor          = Data.pram.ADCfactor;
      pram.cam_EMgain             = Data.pram.EMgain_lt300um;
      pram.cam_t_exp              = Data.pram.t_exp / 1e3;    % [s]  
      
      pram.cam_sigma_rd           = 3;                              % [e-]        Read noise      
      pram.cam_dXdt_dark          = 0.005;                          % [e-/px/s]   Dark current
      pram.cam_Brnuli_alpha       = 0.01;                           %             Probability of a multiplication event in an Em gain stage (=1-2% in Ref2)
      pram.cam_N_gainStages       = round(log(pram.cam_EMgain)/log(1+pram.cam_Brnuli_alpha)); 
                                                                    %             Number of Em-gain stages
    case 'dmd_exp_tfm_mouse_20201224_300um' 
      load('./_extPatternsets/dmd_exp_tfm_mouse_20201224.mat')

      pram.z0_um                  = -300;                                 % [um]        surface is 0 um -ve is below the surface      
      pram.Nx                     = min(size(Data.Ex,1),size(Data.Ex,2)); %             make square image
      pram.Ny                     = pram.Nx;
      x_inds                      = 1:pram.Nx;    
      y_inds                      = 1:pram.Ny;
      t_inds                      = 20 + (1:pram.Nt);
      
      %% Y_exp, E, select y_inds, x_inds
      E                           = single(Data.Ex            (y_inds,x_inds,:));
      Y_exp.anml1_r1_300um        = single(Data.anml1_r1_300um(y_inds,x_inds,:));
      Y_exp.anml1_r2_300um        = single(Data.anml1_r2_300um(y_inds,x_inds,:));
      Y_exp.anml2_r1_300um        = single(Data.anml2_r1_300um(y_inds,x_inds,:));
      Y_exp.anml2_r2_300um        = single(Data.anml2_r2_300um(y_inds,x_inds,:));
      %% X_refs.xx_avg0
      X_refs.anml1_r1_300um_avg0  = mean(Y_exp.anml1_r1_300um,3);
      X_refs.anml1_r2_300um_avg0  = mean(Y_exp.anml1_r2_300um,3);
      X_refs.anml2_r1_300um_avg0  = mean(Y_exp.anml2_r1_300um,3);
      X_refs.anml2_r2_300um_avg0  = mean(Y_exp.anml2_r2_300um,3);
      %% X_refs.xx_wf0
      X_refs.anml1_r1_300um_wf0   = single(Data.anml1_r1_300um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r2_300um_wf0   = single(Data.anml1_r2_300um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r1_300um_wf0   = single(Data.anml2_r1_300um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r2_300um_wf0   = single(Data.anml2_r2_300um_wf0(y_inds,x_inds,:));
      %% seletc the t_inds patterns
      E                           = E                   (:,:,t_inds);
      Y_exp.anml1_r1_300um        = Y_exp.anml1_r1_300um(:,:,t_inds);
      Y_exp.anml1_r2_300um        = Y_exp.anml1_r2_300um(:,:,t_inds);
      Y_exp.anml2_r1_300um        = Y_exp.anml2_r1_300um(:,:,t_inds);
      Y_exp.anml2_r2_300um        = Y_exp.anml2_r2_300um(:,:,t_inds);
      %% X_refs.xx_avg
      X_refs.anml1_r1_300um_avg   = mean(Y_exp.anml1_r1_300um,3);
      X_refs.anml1_r2_300um_avg   = mean(Y_exp.anml1_r2_300um,3);
      X_refs.anml2_r1_300um_avg   = mean(Y_exp.anml2_r1_300um,3);
      X_refs.anml2_r2_300um_avg   = mean(Y_exp.anml2_r2_300um,3);
      
      %%      
      % normalize E
      E                           = E - min(E(:));
      E                           = E / max(E(:));
      
      pram.maxcount               = max([X_refs.anml1_r1_300um_avg0(:);...
                                         X_refs.anml1_r2_300um_avg0(:);...
                                         X_refs.anml2_r1_300um_avg0(:);...
                                         X_refs.anml2_r2_300um_avg0(:)]);

      pram.dx                     = Data.pram.dx0;
      pram.cam_bias               = Data.pram.bias;
      pram.cam_ADCfactor          = Data.pram.ADCfactor;
      pram.cam_EMgain             = Data.pram.EMgain_gt300um;
      pram.cam_t_exp              = Data.pram.t_exp / 1e3;          % [s]  
      
      pram.cam_sigma_rd           = 3;                              % [e-]        Read noise      
      pram.cam_dXdt_dark          = 0.005;                          % [e-/px/s]   Dark current
      pram.cam_Brnuli_alpha       = 0.01;                           %             Probability of a multiplication event in an Em gain stage (=1-2% in Ref2)
      pram.cam_N_gainStages       = round(log(pram.cam_EMgain)/log(1+pram.cam_Brnuli_alpha));
                                                                    %             Number of Em-gain stages
    case 'dmd_exp_tfm_mouse_20201224_350um' 
      load('./_extPatternsets/dmd_exp_tfm_mouse_20201224.mat')
      
      pram.z0_um                  = -350;                                 % [um]        surface is 0 um -ve is below the surface      
      pram.Nx                     = min(size(Data.Ex,1),size(Data.Ex,2)); %             make square image
      pram.Ny                     = pram.Nx;
      x_inds                      = 1:pram.Nx;    
      y_inds                      = 1:pram.Ny;
      t_inds                      = 20 + (1:pram.Nt);
      
      %% Y_exp, E, select y_inds, x_inds
      E                           = single(Data.Ex            (y_inds,x_inds,:));
      Y_exp.anml1_r1_350um        = single(Data.anml1_r1_350um(y_inds,x_inds,:));
      Y_exp.anml1_r2_350um        = single(Data.anml1_r2_350um(y_inds,x_inds,:));
      Y_exp.anml2_r1_350um        = single(Data.anml2_r1_350um(y_inds,x_inds,:));
      Y_exp.anml2_r2_350um        = single(Data.anml2_r2_350um(y_inds,x_inds,:));
      %% X_refs.xx_avg0
      X_refs.anml1_r1_350um_avg0  = mean(Y_exp.anml1_r1_350um,3);
      X_refs.anml1_r2_350um_avg0  = mean(Y_exp.anml1_r2_350um,3);
      X_refs.anml2_r1_350um_avg0  = mean(Y_exp.anml2_r1_350um,3);
      X_refs.anml2_r2_350um_avg0  = mean(Y_exp.anml2_r2_350um,3);
      %% X_refs.xx_wf0
      X_refs.anml1_r1_350um_wf0   = single(Data.anml1_r1_350um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r2_350um_wf0   = single(Data.anml1_r2_350um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r1_350um_wf0   = single(Data.anml2_r1_350um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r2_350um_wf0   = single(Data.anml2_r2_350um_wf0(y_inds,x_inds,:));
      %% seletc the t_inds patterns
      E                           = E                   (:,:,t_inds);
      Y_exp.anml1_r1_350um        = Y_exp.anml1_r1_350um(:,:,t_inds);
      Y_exp.anml1_r2_350um        = Y_exp.anml1_r2_350um(:,:,t_inds);
      Y_exp.anml2_r1_350um        = Y_exp.anml2_r1_350um(:,:,t_inds);
      Y_exp.anml2_r2_350um        = Y_exp.anml2_r2_350um(:,:,t_inds);
      %% X_refs.xx_avg
      X_refs.anml1_r1_350um_avg   = mean(Y_exp.anml1_r1_350um,3);
      X_refs.anml1_r2_350um_avg   = mean(Y_exp.anml1_r2_350um,3);
      X_refs.anml2_r1_350um_avg   = mean(Y_exp.anml2_r1_350um,3);
      X_refs.anml2_r2_350um_avg   = mean(Y_exp.anml2_r2_350um,3);
      
      %%      
      % normalize E
      E                           = E - min(E(:));
      E                           = E / max(E(:));
      
      pram.maxcount               = max([X_refs.anml1_r1_350um_avg0(:);...
                                         X_refs.anml1_r2_350um_avg0(:);...
                                         X_refs.anml2_r1_350um_avg0(:);...
                                         X_refs.anml2_r2_350um_avg0(:)]);

      pram.dx                     = Data.pram.dx0;
      pram.cam_bias               = Data.pram.bias;
      pram.cam_ADCfactor          = Data.pram.ADCfactor;
      pram.cam_EMgain             = Data.pram.EMgain_gt300um;
      pram.cam_t_exp              = Data.pram.t_exp / 1e3;    % [s]  
      
      pram.cam_sigma_rd           = 3;                              % [e-]        Read noise      
      pram.cam_dXdt_dark          = 0.005;                          % [e-/px/s]   Dark current
      pram.cam_Brnuli_alpha       = 0.01;                           %             Probability of a multiplication event in an Em gain stage (=1-2% in Ref2)
      pram.cam_N_gainStages       = round(log(pram.cam_EMgain)/log(1+pram.cam_Brnuli_alpha)); 
                                                                    %             Number of Em-gain stages
    case 'dmd_exp_tfm_mouse_20201224_400um' 
      load('./_extPatternsets/dmd_exp_tfm_mouse_20201224.mat')

      pram.z0_um                  = -400;                                 % [um]        surface is 0 um -ve is below the surface      
      pram.Nx                     = min(size(Data.Ex,1),size(Data.Ex,2)); %             make square image
      pram.Ny                     = pram.Nx;
      x_inds                      = 1:pram.Nx;    
      y_inds                      = 1:pram.Ny;
      t_inds                      = 20 + (1:pram.Nt);
      
      %% Y_exp, E, select y_inds, x_inds
      E                           = single(Data.Ex            (y_inds,x_inds,:));
      Y_exp.anml2_r2_400um        = single(Data.anml2_r2_400um(y_inds,x_inds,:));           
      %% X_refs.xx_avg0
      X_refs.anml2_r2_400um_avg0  = mean(Y_exp.anml2_r2_400um,3);
      %% X_refs.xx_wf0
      X_refs.anml2_r2_400um_wf0   = single(Data.anml2_r2_400um_wf0(y_inds,x_inds,:));
      %% seletc the t_inds patterns
      E                           = E                   (:,:,t_inds);
      Y_exp.anml2_r2_400um        = Y_exp.anml2_r2_400um(:,:,t_inds);
      %% X_refs.xx_avg
      X_refs.anml2_r2_400um_avg   = mean(Y_exp.anml2_r2_400um,3);
      
      %%      
      % normalize E
      E                           = E - min(E(:));
      E                           = E / max(E(:));
      
      pram.maxcount               = max(X_refs.anml2_r2_400um_avg0(:));

      pram.dx                     = Data.pram.dx0;
      pram.cam_bias               = Data.pram.bias;
      pram.cam_ADCfactor          = Data.pram.ADCfactor;
      pram.cam_EMgain             = Data.pram.EMgain_gt300um;
      pram.cam_t_exp              = Data.pram.t_exp / 1e3;    % [s]  
      
      pram.cam_sigma_rd           = 3;                              % [e-]        Read noise      
      pram.cam_dXdt_dark          = 0.005;                          % [e-/px/s]   Dark current
      pram.cam_Brnuli_alpha       = 0.01;                           %             Probability of a multiplication event in an Em gain stage (=1-2% in Ref2)
      pram.cam_N_gainStages       = round(log(pram.cam_EMgain)/log(1+pram.cam_Brnuli_alpha)); 
                                                                    %             Number of Em-gain stages
    case 'dmd_exp_tfm_mouse_20201224_all'   
      load('./_extPatternsets/dmd_exp_tfm_mouse_20201224.mat')
      
      pram.Nx = min(size(Data.Ex,1),size(Data.Ex,2));         %             make square image
      pram.Ny = pram.Nx;
      x_inds  = 1:pram.Nx;    
      y_inds  = 1:pram.Ny;
      t_inds  = 20 + (1:pram.Nt);
      
      %% Y_exp, E, select y_inds, x_inds
      E                           = single(Data.Ex            (y_inds,x_inds,:));
      Y_exp.anml1_r1_sf           = single(Data.anml1_r1_sf   (y_inds,x_inds,:));
      Y_exp.anml1_r1_100um        = single(Data.anml1_r1_100um(y_inds,x_inds,:));
      Y_exp.anml1_r1_200um        = single(Data.anml1_r1_200um(y_inds,x_inds,:));
      Y_exp.anml1_r1_300um        = single(Data.anml1_r1_300um(y_inds,x_inds,:));
      Y_exp.anml1_r1_350um        = single(Data.anml1_r1_350um(y_inds,x_inds,:));
      Y_exp.anml1_r2_sf           = single(Data.anml1_r2_sf   (y_inds,x_inds,:));
      Y_exp.anml1_r2_100um        = single(Data.anml1_r2_100um(y_inds,x_inds,:));
      Y_exp.anml1_r2_200um        = single(Data.anml1_r2_200um(y_inds,x_inds,:));
      Y_exp.anml1_r2_300um        = single(Data.anml1_r2_300um(y_inds,x_inds,:));
      Y_exp.anml1_r2_350um        = single(Data.anml1_r2_350um(y_inds,x_inds,:));
      Y_exp.anml2_r1_100um        = single(Data.anml2_r1_100um(y_inds,x_inds,:));
      Y_exp.anml2_r1_200um        = single(Data.anml2_r1_200um(y_inds,x_inds,:));
      Y_exp.anml2_r1_300um        = single(Data.anml2_r1_300um(y_inds,x_inds,:));
      Y_exp.anml2_r1_350um        = single(Data.anml2_r1_350um(y_inds,x_inds,:));
      Y_exp.anml2_r2_300um        = single(Data.anml2_r2_300um(y_inds,x_inds,:));
      Y_exp.anml2_r2_350um        = single(Data.anml2_r2_350um(y_inds,x_inds,:));
      Y_exp.anml2_r2_400um        = single(Data.anml2_r2_400um(y_inds,x_inds,:));           
      %% X_refs.xx_avg0
      X_refs.anml1_r1_sf_avg0     = mean(Y_exp.anml1_r1_sf   ,3);
      X_refs.anml1_r1_100um_avg0  = mean(Y_exp.anml1_r1_100um,3);
      X_refs.anml1_r1_200um_avg0  = mean(Y_exp.anml1_r1_200um,3);
      X_refs.anml1_r1_300um_avg0  = mean(Y_exp.anml1_r1_300um,3);
      X_refs.anml1_r1_350um_avg0  = mean(Y_exp.anml1_r1_350um,3);
      X_refs.anml1_r2_sf_avg0     = mean(Y_exp.anml1_r2_sf   ,3);
      X_refs.anml1_r2_100um_avg0  = mean(Y_exp.anml1_r2_100um,3);
      X_refs.anml1_r2_200um_avg0  = mean(Y_exp.anml1_r2_200um,3);
      X_refs.anml1_r2_300um_avg0  = mean(Y_exp.anml1_r2_300um,3);
      X_refs.anml1_r2_350um_avg0  = mean(Y_exp.anml1_r2_350um,3);
      X_refs.anml2_r1_100um_avg0  = mean(Y_exp.anml2_r1_100um,3);
      X_refs.anml2_r1_200um_avg0  = mean(Y_exp.anml2_r1_200um,3);
      X_refs.anml2_r1_300um_avg0  = mean(Y_exp.anml2_r1_300um,3);
      X_refs.anml2_r1_350um_avg0  = mean(Y_exp.anml2_r1_350um,3);
      X_refs.anml2_r2_300um_avg0  = mean(Y_exp.anml2_r2_300um,3);
      X_refs.anml2_r2_350um_avg0  = mean(Y_exp.anml2_r2_350um,3);
      X_refs.anml2_r2_400um_avg0  = mean(Y_exp.anml2_r2_400um,3);
      %% X_refs.xx_wf0
      X_refs.anml1_r1_sf_wf0      = single(Data.anml1_r1_sf_wf0   (y_inds,x_inds,:));
      X_refs.anml1_r1_100um_wf0   = single(Data.anml1_r1_100um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r1_200um_wf0   = single(Data.anml1_r1_200um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r1_300um_wf0   = single(Data.anml1_r1_300um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r1_350um_wf0   = single(Data.anml1_r1_350um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r2_sf_wf0      = single(Data.anml1_r2_sf_wf0   (y_inds,x_inds,:));
      X_refs.anml1_r2_100um_wf0   = single(Data.anml1_r2_100um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r2_200um_wf0   = single(Data.anml1_r2_200um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r2_300um_wf0   = single(Data.anml1_r2_300um_wf0(y_inds,x_inds,:));
      X_refs.anml1_r2_350um_wf0   = single(Data.anml1_r2_350um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r1_100um_wf0   = single(Data.anml2_r1_100um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r1_200um_wf0   = single(Data.anml2_r1_200um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r1_300um_wf0   = single(Data.anml2_r1_300um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r1_350um_wf0   = single(Data.anml2_r1_350um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r2_300um_wf0   = single(Data.anml2_r2_300um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r2_350um_wf0   = single(Data.anml2_r2_350um_wf0(y_inds,x_inds,:));
      X_refs.anml2_r2_400um_wf0   = single(Data.anml2_r2_400um_wf0(y_inds,x_inds,:));
      %% seletc the t_inds patterns
      E                           = E                   (:,:,t_inds);
      Y_exp.anml1_r1_sf           = Y_exp.anml1_r1_sf   (:,:,t_inds);
      Y_exp.anml1_r1_100um        = Y_exp.anml1_r1_100um(:,:,t_inds);
      Y_exp.anml1_r1_200um        = Y_exp.anml1_r1_200um(:,:,t_inds);
      Y_exp.anml1_r1_300um        = Y_exp.anml1_r1_300um(:,:,t_inds);
      Y_exp.anml1_r1_350um        = Y_exp.anml1_r1_350um(:,:,t_inds);
      Y_exp.anml1_r2_sf           = Y_exp.anml1_r2_sf   (:,:,t_inds);
      Y_exp.anml1_r2_100um        = Y_exp.anml1_r2_100um(:,:,t_inds);
      Y_exp.anml1_r2_200um        = Y_exp.anml1_r2_200um(:,:,t_inds);
      Y_exp.anml1_r2_300um        = Y_exp.anml1_r2_300um(:,:,t_inds);
      Y_exp.anml1_r2_350um        = Y_exp.anml1_r2_350um(:,:,t_inds);
      Y_exp.anml2_r1_100um        = Y_exp.anml2_r1_100um(:,:,t_inds);
      Y_exp.anml2_r1_200um        = Y_exp.anml2_r1_200um(:,:,t_inds);
      Y_exp.anml2_r1_300um        = Y_exp.anml2_r1_300um(:,:,t_inds);
      Y_exp.anml2_r1_350um        = Y_exp.anml2_r1_350um(:,:,t_inds);
      Y_exp.anml2_r2_300um        = Y_exp.anml2_r2_300um(:,:,t_inds);
      Y_exp.anml2_r2_350um        = Y_exp.anml2_r2_350um(:,:,t_inds);
      Y_exp.anml2_r2_400um        = Y_exp.anml2_r2_400um(:,:,t_inds);
      %% X_refs.xx_avg
      X_refs.anml1_r1_sf_avg      = mean(Y_exp.anml1_r1_sf   ,3);
      X_refs.anml1_r1_100um_avg   = mean(Y_exp.anml1_r1_100um,3);
      X_refs.anml1_r1_200um_avg   = mean(Y_exp.anml1_r1_200um,3);
      X_refs.anml1_r1_300um_avg   = mean(Y_exp.anml1_r1_300um,3);
      X_refs.anml1_r1_350um_avg   = mean(Y_exp.anml1_r1_350um,3);
      X_refs.anml1_r2_sf_avg      = mean(Y_exp.anml1_r2_sf   ,3);
      X_refs.anml1_r2_100um_avg   = mean(Y_exp.anml1_r2_100um,3);
      X_refs.anml1_r2_200um_avg   = mean(Y_exp.anml1_r2_200um,3);
      X_refs.anml1_r2_300um_avg   = mean(Y_exp.anml1_r2_300um,3);
      X_refs.anml1_r2_350um_avg   = mean(Y_exp.anml1_r2_350um,3);
      X_refs.anml2_r1_100um_avg   = mean(Y_exp.anml2_r1_100um,3);
      X_refs.anml2_r1_200um_avg   = mean(Y_exp.anml2_r1_200um,3);
      X_refs.anml2_r1_300um_avg   = mean(Y_exp.anml2_r1_300um,3);
      X_refs.anml2_r1_350um_avg   = mean(Y_exp.anml2_r1_350um,3);
      X_refs.anml2_r2_300um_avg   = mean(Y_exp.anml2_r2_300um,3);
      X_refs.anml2_r2_350um_avg   = mean(Y_exp.anml2_r2_350um,3);
      X_refs.anml2_r2_400um_avg   = mean(Y_exp.anml2_r2_400um,3);
      
      %%      
      % normalize E
      E                     = E - min(E(:));
      E                     = E / max(E(:));
      
      pram.maxcount         = max([X_refs.anml1_r1_100um_avg0(:);...
                                   X_refs.anml1_r1_200um_avg0(:);...
                                   X_refs.anml1_r1_300um_avg0(:);...
                                   X_refs.anml1_r1_350um_avg0(:);...
                                   X_refs.anml1_r2_100um_avg0(:);...
                                   X_refs.anml1_r2_100um_avg0(:);...
                                   X_refs.anml1_r2_300um_avg0(:);...
                                   X_refs.anml1_r2_350um_avg0(:);...
                                   X_refs.anml2_r1_100um_avg0(:);...
                                   X_refs.anml2_r1_200um_avg0(:);...
                                   X_refs.anml2_r1_300um_avg0(:);...
                                   X_refs.anml2_r1_350um_avg0(:);...
                                   X_refs.anml2_r2_300um_avg0(:);...
                                   X_refs.anml2_r2_350um_avg0(:);...
                                   X_refs.anml2_r2_400um_avg0(:)]);

      pram.dx                         = Data.pram.dx0;
      pram.cam_bias                   = Data.pram.bias;
      pram.cam_ADCfactor              = Data.pram.ADCfactor;
      pram.cam_EMgain_lt300um         = Data.pram.EMgain_lt300um;
      pram.cam_EMgain_gt300um         = Data.pram.EMgain_gt300um;
      pram.cam_t_exp                  = Data.pram.t_exp / 1e3;    % [s]  
      
      pram.cam_sigma_rd               = 3;                              % [e-]        Read noise      
      pram.cam_dXdt_dark              = 0.005;                          % [e-/px/s]   Dark current
      pram.cam_Brnuli_alpha           = 0.01;                           %             Probability of a multiplication event in an Em gain stage (=1-2% in Ref2)
      pram.cam_N_gainStages_lt300um   = round(log(pram.cam_EMgain_lt300um)/log(1+pram.cam_Brnuli_alpha)); 
      pram.cam_N_gainStages_gt300um   = round(log(pram.cam_EMgain_gt300um)/log(1+pram.cam_Brnuli_alpha)); 
                                                                        %             Number of Em-gain stages
  end

end

function H = subf_get_hadamard_patterns(Ny,Nx,Nt)

  ny      = sqrt(Nt/cos(pi/6));                     % from tesselation nx = ny*cos(pi/6) and Area = Nt = nx*ny  
  ny      = round(ny);
  nx      = floor(Nt/ny);

  hadmat  = hadamard(Nt);
% hadmat(:,1)  = [1:Nt]';                           % to check the that the tesselation is right
  
  h       = reshape(hadmat(1:nx*ny,:),[ny nx Nt]);  % one block
  h2      = [h circshift(h,round(size(h,1)/2),1)];  % two blocks for triangulization

  H       = repmat(h2,[ceil(Ny/ny) ceil(Nx/(2*nx)) 1]);
  H       = H(1:Ny,1:Nx,:);
end

