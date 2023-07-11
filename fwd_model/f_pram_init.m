% notes: 
%   The parameters: NA 1.0; Size of camera pixel on sample 330nm; exc/em wavelength: 800nm/590nm. [Cheng 2020-12-24]


function pram = f_pram_init()

  %% names
  pram.sim2dOr3d    = '3D';                               % {'2D','3D'}
  pram.mic_typ      = 'DMD';                              % {'DMD','WGD'}
  pram.pattern_typ  = 'dmd_exp_tfm_mouse_20201224_400um'; % {'dmd_sim_rnd',
                                                          %  'dmd_exp_tfm_beads_7sls_20201219'
                                                          %  'dmd_exp_tfm_mouse_20201224_sf'
                                                          %  'dmd_exp_tfm_mouse_20201224_100um'
                                                          %  'dmd_exp_tfm_mouse_20201224_200um'
                                                          %  'dmd_exp_tfm_mouse_20201224_300um'
                                                          %  'dmd_exp_tfm_mouse_20201224_350um'
                                                          %  'dmd_exp_tfm_mouse_20201224_400um'
                                                          %  'dmd_exp_tfm_mouse_20201224_all'}
  pram.dataset      = 'mouse_400um';                      % {'minist',
                                                          %  'andrewCells_fociW3_63x_maxProj',
                                                          %  'andrewCells_dapi_20x_maxProj',
                                                          %  'beads',
                                                          %  'mouse_sf',
                                                          %  'mouse_100um',
                                                          %  'mouse_200um',
                                                          %  'mouse_300um',
                                                          %  'mouse_350um',
                                                          %  'mouse_400um'}
  pram.psf_typ      = 'MC';                               % {'MC','gaussian',...}
    
  %% optical properties of the tissue
  pram.mus        = 200;                                  % [cm^-1]   scattering coefficient of tissue
  pram.g          = 0.90;                                 % [AU]      anisotropy of scattering of tissue
  pram.nt         = 1.33;                                 % [AU]      refractive index of tissue  
  pram.nm         = 1.33;                                 % [AU]      refractive index of the medium (ex:water,air)  
  pram.sl         = (1/pram.mus)*10*1e3;                  % [um]      sacttering length  

  %% data size parameters
  pram.Nx      = 100;
  pram.Ny      = 100;
  pram.Nz      = 64;
  pram.Nc      = 1;
  pram.Nt      = 8;
  pram.Nb      = 1e4;                                     %           number of batches (instances)
  pram.dx      = 0.33;
  pram.dz      = 1;
  
  %% MIC and imaging parameters
  pram.lambda_ex  = 0.800;                                % [um]      excitation wavelength
  pram.lambda_em  = 0.590;                                % [um]      emission wavelength {0.606 }
  pram.NA         = 1;                                    % [AU]      numerical aperture of the objective
  pram.z0_um      = -7*pram.sl;                           % [um]      depth (z=0 is the surface and -ve is below)
    
  %% camera parameters <THIS IS OLD NEED UPDATING WITH THE NEW CAM MODEL>
  pram.binR             = 1;
  pram.cam_emhist_Nreps = 10000;
  
  %% run environment parameters  
  pram.useGPU   = gpuDeviceCount>0 ;
  
end




