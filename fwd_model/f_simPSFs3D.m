
function PSFs = f_simPSFs3D(pram)
  of        = cd('./_submodules/MC_LightScattering/');
  
  %% set prams
  mcls_pram           = f_praminit();
  mcls_pram.savepath  = [of '/_PSFs/'];
  mcls_pram.fNameStem = 'MC';
  mcls_pram.Nx        = pram.Nx*4+3;
  mcls_pram.Nz        = 100;
  mcls_pram.dx        = pram.dx/2;
  mcls_pram.z0_um     = pram.z0_um;
  mcls_pram.sl        = pram.sl;
  mcls_pram.mus       = pram.mus;
  mcls_pram.lambda_ex = pram.lambda_ex;
  mcls_pram.lambda_em = pram.lambda_em;
  mcls_pram.NA        = pram.NA;
  mcls_pram.Nphotons  = 1E6;
  mcls_pram.Nsims     = 32*4;
  mcls_pram.useGpu    = 1;
   
  %% simulate exPSF and emPSF
  of2 = cd('_supToolboxes/optical_PSF/');
  delete(gcp('nocreate'));
  parpool(4) % so that the gpu don't go out of memory 
  
    %% exPSF
  APSF_3D     = Efficient_PSF(mcls_pram.NA, mcls_pram.nm, mcls_pram.lambda_ex, mcls_pram.dx,mcls_pram.Nx-2,mcls_pram.Nx-2,100,100);
  PSF_3D      = abs(APSF_3D{1}).^2+abs(APSF_3D{2}).^2+abs(APSF_3D{3}).^2;
  PSF_3D_2p   = PSF_3D.^2; % 2021-04-13 check with Peter if this dependence is correct. 
    
  sum_at_focus= max(sum(sum(PSF_3D_2p,1),2));
  exPSF       = PSF_3D_2p/sum_at_focus;
  
  % select > 1% excitation efficiency compared to max as the axial rangee, Nz
  axial_sum   = squeeze(sum(sum(exPSF,1),2));
  half_Nz     = mcls_pram.Nz/2+1 - min(find(axial_sum>0.01));
  z_range     = mcls_pram.Nz/2+1-half_Nz : mcls_pram.Nz/2+1+half_Nz;
  z_range_um  = (z_range-half_Nz+1) * mcls_pram.dx;
  
  exPSF       = exPSF(:,:,z_range);
    
    %% emPSF
  APSF_3D     = Efficient_PSF(mcls_pram.NA, mcls_pram.nm, mcls_pram.lambda_em, mcls_pram.dx,mcls_pram.Nx-2,mcls_pram.Nx-2,100,100);
  PSF_3D      = abs(APSF_3D{1}).^2+abs(APSF_3D{2}).^2+abs(APSF_3D{3}).^2;
  sum_at_focus= max(sum(sum(PSF_3D,1),2));
  emPSF       = PSF_3D/sum_at_focus;
  emPSF       = emPSF(:,:,z_range);
  
  delete(gcp('nocreate'));
  cd(of2);

  %% simulate sPSF (saves to [mclm_pram.savepath mclm_pram.fNameStem '_sPSF.mat'])
  z_range_um  = ( -floor(size(emPSF,3)/2):floor(size(emPSF,3)/2) ) .* mcls_pram.dx + mcls_pram.z0_um;
  
  for i=1:length(z_range_um)
    i
    mcls_pram.fNameStem = sprintf('MC_%d',i);
    mcls_pram.z0_um     = z_range_um(i);  
    main(mcls_pram);
    load([mcls_pram.savepath mcls_pram.fNameStem '_sPSF.mat']);       % loads sPSF
    sPSF_3d(:,:,length(z_range_um)-i+1) = sPSF(2:end-1,2:end-1);
  end
  cd(of);  
  
  %% convolve emPSF and sPSF 
  %emConvSPSF  = gather(conv2(gpuArray(sPSF),gpuArray(emPSF),'same'));
    
  %% save PSFs
  PSFs.exPSF      = exPSF;
  PSFs.emPSF      = emPSF;
  PSFs.sPSF       = sPSF_3d;
  PSFs.pram       = mcls_pram;
        
  save([mcls_pram.savepath 'PSFs' datestr(datetime('now')) '.mat'],'PSFs','-v7.3'); % save sPSF
end
