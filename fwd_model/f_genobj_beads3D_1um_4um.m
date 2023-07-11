
function X0 = f_genobj_beads3D_1um_4um(N_beads,pram)
  
  r_range     = [0.5/pram.dx 0.5/pram.dx 2/pram.dx];                            % bead radii
  Ny0         = pram.Ny;
  Nx0         = pram.Nx;
  Nz0         = round(pram.Nz*pram.dz/pram.dx);                     % for isotropic resolution on the object
    
  if isempty(N_beads)
    N_beads   = round((Ny0*Nx0*Nz0/max(r_range)^3)/100);            % 1/100 of grid maximum packing
  end
  % gives you x,y,z values of centroids
  centroids   = [randi([ceil(max(r_range)), Ny0-ceil(max(r_range))],N_beads,1),...
                 randi([ceil(max(r_range)), Nx0-ceil(max(r_range))],N_beads,1),...
                 randi([ceil(max(r_range)), Nz0-ceil(max(r_range))],N_beads,1)];
             
  radii       = r_range(randi([1,length(r_range)],N_beads,1));      %Size of the beads (can change)
  norm_int    = normrnd(1,0.1,N_beads,1);                           % normalized intensity

  % boost up small bead's intensity by 5x
  norm_int(radii == r_range(1)) = norm_int(radii == r_range(1))*5;  %intensity = 5
  
  X0          = single(zeros(Ny0,Nx0,Nz0));
  
  [Y,X,Z]     = meshgrid(1:Ny0,1:Nx0,1:Nz0);
    
  for i=1:N_beads
    % i
    beads_fll = (Y - centroids(i,1)).^2 + (X - centroids(i,2)).^2 + (Z - centroids(i,3)).^2 < radii(i)^2;
    X0        = X0 + beads_fll*norm_int(i);
  end

  X0          = imresize3(X0,[pram.Ny pram.Ny pram.Nz],'linear');
  X0          = X0(:,:,round(min(centroids(:,3))*pram.Nz/Nz0)+1:round(max(centroids(:,3))*pram.Nz/Nz0)-1);
  
    
end






