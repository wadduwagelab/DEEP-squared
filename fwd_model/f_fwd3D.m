% next to do is to go through and modify for the 4th domention of X0 in puts


function [Yhat Xgt] = f_fwd3D(X0,E,PSFs,emhist,pram)
  % Input dimentionality
  %   X0          - [y,x,z]
  %   E           - [y,x,t=Nt]
  %   PSFs.xxPSF  - [y,x,z]

  % Dimentionality used within the function (here t = patterns,b = instances) 
  %   X0          - [y,x,z  ,t=1 ,b  ]
  %   E           - [y,x,z=1,t=Nt,b=1]
  %   PSFs.xxPSF  - [y,x,z  ,t=1 ,b=1]
  %   Y0          - [y,x,z=1,t=Nt,b  ]
  
  % Output dimentionality
  %   Yhat        - [y,x,t=Nt,b]
  %   Xgt         - [y,x,t=1 ,b]
    
  %% preprocess inputs (for size, resolution, and dimentionality, and gpu-use)

  rsf_x     = PSFs.pram.dx/pram.dx;
  rsf_y     = rsf_x;
  rsf_z     = PSFs.pram.dx/pram.dz;  

  exPSF     = single(imresize3(PSFs.exPSF,'Scale',[rsf_y,rsf_x,rsf_z],'Method','linear','Antialiasing',true));
  emPSF     = single(imresize3(PSFs.emPSF,'Scale',[rsf_y,rsf_x,rsf_z],'Method','linear','Antialiasing',true));
  sPSF      = single(imresize3(PSFs.sPSF ,'Scale',[rsf_y,rsf_x,rsf_z],'Method','linear','Antialiasing',true));
  
  E         = reshape(E,[size(E,1),size(E,2),1,size(E,3)]);
  E_gt      = single(ones(pram.Ny,pram.Nx,1,1));
  
  if pram.useGPU ==1
    X0      = gpuArray(X0);
    E       = gpuArray(E);
    emPSF   = gpuArray(emPSF);
    exPSF   = gpuArray(exPSF);
    sPSF    = gpuArray(sPSF);
  end
  X0        = padarray(X0,round([(size(exPSF,1)-pram.Ny)/2 (size(exPSF,2)-pram.Nx)/2 0 0 0]),0,'both');
  X0        = X0(1:size(exPSF,1),1:size(exPSF,2),:,:,:);
  
  %% fwd model - equation: Em(x,y,z,t) = emPSF **3d {sPSF **2d [(exPSF **3d E).*X0]}, here **=conv
  Eex_3D    = f_conv3nd(exPSF,E,'same');
  
  vol_Nz    = size(exPSF,3);
  vol_inits = [1:pram.dist:size(X0,3)-vol_Nz];
  
  
  Nb        = length(vol_inits);
  meanX0    = mean(X0(:));
  b_t       = 1;

  for b = 1:Nb
    X0_vol  = X0(:,:,vol_inits(b):vol_inits(b)+vol_Nz-1);
    if mean(X0_vol(:))>meanX0
      vol_inits_valid(b_t) = vol_inits(b);
      b_t   = b_t+1;
    end
  end
  if b_t > 1
    vol_inits = vol_inits_valid;
    Nb        = b_t - 1;
  

    for b = 1:Nb
       % b
      X0_vol  = X0(:,:,vol_inits(b):vol_inits(b)+vol_Nz-1);
      X0_vol  = imrotate(X0_vol,90*rem(b,4));
      X_ex    = Eex_3D .* X0_vol;
    
      for j=1:size(X_ex,4)                              % using the loop in the function is slow for some reason
        X_sctterd(:,:,:,j) = f_conv2nd(sPSF ,X_ex(:,:,:,j),'same');
      end
  
      for j=1:size(X_ex,4)                              % using the loop in the function is slow for some reason
        X_em(:,:,:,j)      = f_conv3nd(emPSF,X_sctterd(:,:,:,j),'same');
      end
    
      %% gt image (as in PS-TPM in the absence of scattering)
      Xgt_3D    = f_conv3nd(exPSF,X0_vol,'same');
  
      %% postprocess (cropping)
      y_range   = round(size(X_em,1)/2 - pram.Nx/2)+1:round(size(X_em,1)/2 + pram.Nx/2);
      x_range   = round(size(X_em,2)/2 - pram.Nx/2)+1:round(size(X_em,2)/2 + pram.Nx/2);


    %% images (Xgt,Y0)
      Xgt       = Xgt_3D(y_range,x_range,ceil(size(X_em,3)/2),:,:);
      Y0        = X_em  (y_range,x_range,ceil(size(X_em,3)/2),:,:);

    %% match experimental counts (refer to f_get_extPettern and f_read_data on the original data folder)
   
      Y0        = double(pram.maxcount*Y0/max(Y0(:)));
      Xgt       = double(pram.maxcount*Xgt/max(Xgt(:)));

    
      Y0_all  (:,:,:,:,b) = Y0;
      Xgt_all (:,:,:,:,b) = Xgt;
    end
  %% load emccd noise distributions for Yhat    
    [Yhat_all YhatADU]= f_simulateIm_emCCD(Y0_all,emhist,pram);
  
    %% change dims for output
    Xgt       = reshape(Xgt_all ,[pram.Ny pram.Nx 1       Nb]);
    Yhat      = reshape(Yhat_all,[pram.Ny pram.Nx pram.Nt Nb]);
  else
    Xgt = 0
    Yhat = 0  
  end 
end
